# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import logging
from functools import partial
from typing import Iterable

import modelopt.torch.distill as mtd
import torch
from megatron.core import parallel_state
from megatron.core.models.gpt import GPTModel
from megatron.core.pipeline_parallel.utils import is_pp_first_stage, is_pp_last_stage
from megatron.core.utils import (
    get_batch_on_this_cp_rank,
    get_model_config,
    is_te_min_version,
    unwrap_model,
)

from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.losses import masked_next_token_loss
from megatron.bridge.training.post_training.distillation import loss_func_kd
from megatron.bridge.training.state import GlobalState
from megatron.bridge.training.utils.packed_seq_utils import get_packed_seq_params
from megatron.bridge.training.utils.pg_utils import get_pg_collection


logger = logging.getLogger(__name__)


def _uses_packed_sequence_metadata(cfg: ConfigContainer) -> bool:
    """Return whether the dataset is expected to provide packed sequence metadata."""
    dataset_cfg = getattr(cfg, "dataset", None)
    packed_sequence_specs = getattr(dataset_cfg, "packed_sequence_specs", None)
    if packed_sequence_specs is not None:
        packed_sequence_size = getattr(packed_sequence_specs, "packed_sequence_size", None)
        return packed_sequence_size is None or packed_sequence_size > 0

    return getattr(dataset_cfg, "pack_sequences_in_batch", False)


def _middle_pp_stage_needs_batch(cfg: ConfigContainer) -> bool:
    """Return whether middle PP stages need batch metadata for attention."""
    dataset_cfg = getattr(cfg, "dataset", None)
    uses_custom_attention_mask = not getattr(dataset_cfg, "skip_getting_attention_mask_from_dataset", True)
    return uses_custom_attention_mask or _uses_packed_sequence_metadata(cfg)


def _partition_packed_batch_for_cp(batch: dict[str, torch.Tensor], cp_size: int) -> dict[str, torch.Tensor]:
    """Partition THD/packed batches across context-parallel ranks.

    Uses transformer_engine's `thd_get_partitioned_indices` to slice sequence
    dimension aligned with packed cu_seqlens. This avoids the generic
    `get_batch_on_this_cp_rank` slicing which assumes contiguous sequence tokens.
    """

    err_msg = "Please update Transformer Engine to >= 1.10 to use Context Parallel with THD format data"
    try:
        import transformer_engine_torch as tex

        if not is_te_min_version("1.10.0"):
            logger.error(err_msg)
            raise RuntimeError(err_msg)
    except ModuleNotFoundError as e:
        logger.error(err_msg)
        raise e

    cp_rank = parallel_state.get_context_parallel_rank()
    cu_seqlens = batch["cu_seqlens"]
    if cu_seqlens.dim() > 1 and cu_seqlens.size(0) != 1:
        raise ValueError("Packed THD batches expect micro-batch size 1 for context-parallel slicing (THD layout)")
    cu_seqlens = cu_seqlens.squeeze()
    cu_seqlens_unpadded = batch.get("cu_seqlens_unpadded")
    if cu_seqlens_unpadded is not None:
        batch["cu_seqlens_unpadded"] = cu_seqlens_unpadded.squeeze()

    skip_keys = {
        "cu_seqlens",
        "cu_seqlens_unpadded",
        "cu_seqlens_argmin",
        "cu_seqlens_unpadded_argmin",
        "max_seqlen",
        "token_count",
    }

    for key, val in batch.items():
        if val is None or key in skip_keys:
            continue
        index = tex.thd_get_partitioned_indices(cu_seqlens, val.size(1), cp_size, cp_rank)
        batch[key] = val.index_select(1, index)

    return batch


_PRINTED_MODEL_STRUCTURE_RANKS: set = set()


def _maybe_print_model_structure(model) -> None:
    """When MBRIDGE_PRINT_MODEL is set, print this rank's model structure once.

    Each distributed rank prints exactly once. Output is gated by the env var
    so it's opt-in for alignment debugging (mirrors STEPTRON_PRINT_MODEL on
    the SteptronOss side). Observer-only — does not change the model.
    """
    if not os.environ.get("MBRIDGE_PRINT_MODEL"):
        return

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    if rank in _PRINTED_MODEL_STRUCTURE_RANKS:
        return
    _PRINTED_MODEL_STRUCTURE_RANKS.add(rank)

    target = unwrap_model(model)
    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    pp_world = parallel_state.get_pipeline_model_parallel_world_size()
    header = (
        f"[MODEL] ===== Megatron-Bridge model structure "
        f"(rank={rank}, PP={pp_rank}/{pp_world}) ====="
    )
    print(header, flush=True)
    print(repr(target), flush=True)
    n_params = sum(p.numel() for p in target.parameters())
    n_trainable = sum(p.numel() for p in target.parameters() if p.requires_grad)
    print(f"[MODEL]   total params     : {n_params:,}", flush=True)
    print(f"[MODEL]   trainable params : {n_trainable:,}", flush=True)
    print("[MODEL] " + "=" * (len(header) - len("[MODEL] ")), flush=True)


def _print_embedding_debug_info(module, tensor: torch.Tensor) -> None:
    """Print embedding diagnostic info for cross-framework alignment debugging."""
    print("[ALIGN] ========== Embedding Debug Info (Megatron-Bridge) ==========", flush=True)
    print(f"[ALIGN] module type                : {type(module).__name__}", flush=True)

    cfg = getattr(module, "config", None)
    for attr in [
        "vocab_size",
        "max_sequence_length",
        "add_position_embedding",
        "num_tokentypes",
        "scatter_to_sequence_parallel",
        "reduce_scatter_embeddings",
    ]:
        val = getattr(module, attr, None)
        if val is None and cfg is not None:
            val = getattr(cfg, attr, "N/A")
        print(f"[ALIGN]   {attr:<40}: {val}", flush=True)
    for attr in [
        "hidden_size",
        "hidden_dropout",
        "fp32_residual_connection",
        "sequence_parallel",
        "embedding_init_method",
    ]:
        val = getattr(cfg, attr, "N/A") if cfg is not None else "N/A"
        print(f"[ALIGN]   config.{attr:<33}: {val}", flush=True)

    word_emb = getattr(module, "word_embeddings", None)
    if word_emb is not None and hasattr(word_emb, "weight"):
        w = word_emb.weight.data.detach().float()
        print(f"[ALIGN]   word_embeddings.weight shape  : {tuple(word_emb.weight.shape)}", flush=True)
        print(f"[ALIGN]   word_embeddings.weight dtype  : {word_emb.weight.dtype}", flush=True)
        print(
            f"[ALIGN]   word_embeddings.weight stats  : min={w.min():.6f}  max={w.max():.6f}  mean={w.mean():.6f}  std={w.std():.6f}",
            flush=True,
        )

    pos_emb = getattr(module, "position_embeddings", None)
    if pos_emb is not None and hasattr(pos_emb, "weight"):
        w = pos_emb.weight.data.detach().float()
        print(f"[ALIGN]   position_embeddings.weight shape: {tuple(pos_emb.weight.shape)}", flush=True)
        print(f"[ALIGN]   position_embeddings.weight dtype: {pos_emb.weight.dtype}", flush=True)
        print(
            f"[ALIGN]   position_embeddings.weight stats: min={w.min():.6f}  max={w.max():.6f}  mean={w.mean():.6f}  std={w.std():.6f}",
            flush=True,
        )

    t = tensor.detach().float()
    print(f"[ALIGN]   output shape                 : {tuple(tensor.shape)}", flush=True)
    print(f"[ALIGN]   output dtype                 : {tensor.dtype}", flush=True)
    print(
        f"[ALIGN]   output stats                 : min={t.min():.6f}  max={t.max():.6f}  mean={t.mean():.6f}  std={t.std():.6f}",
        flush=True,
    )
    print(
        f"[ALIGN]   output has_nan                : {torch.isnan(t).any().item()}  has_inf: {torch.isinf(t).any().item()}",
        flush=True,
    )
    print("[ALIGN] ==============================================================", flush=True)


def _build_intermediate_hooks(model: GPTModel, save_dir: str) -> list:
    """Register forward hooks to capture intermediate activations for layer-by-layer alignment.

    Saves per-layer: embedding output, each layer's attention output, each
    layer's MLP/MoE output, plus fine-grained tensors (QKV split, norms, RoPE,
    core_attention I/O, router probs, shared-expert output, ...). Tensors are
    in Megatron layout [S, B, H]. Triggered by env var
    MBRIDGE_SAVE_INTERMEDIATE_PATH pointing to an output directory.

    Implementation note: this is an observer-only path — every entry uses
    ``register_forward_hook`` / ``register_forward_pre_hook`` and never replaces
    a module's forward, so the production model logic (Step35DecoderLayer,
    standard MoELayer, TE submodules, ...) stays untouched. MoE-internal
    intermediates are captured by hooking ``mlp.router`` / ``mlp.shared_experts``
    directly because Megatron's ``MoELayer.forward`` invokes those via
    ``__call__`` (so forward hooks fire).

    Three orthogonal env-var switches further select WHICH activations get
    dumped (defaults below match the 5.4 alignment recipe shell):
      - DUMP_BLOCK_IO=1  : top-level block outputs — embedding, layer_NNN_attention,
                           layer_NNN_ffn.
      - DUMP_FINEGRAIN=1 : every other fine-grained tensor (norms, QKV splits,
                           RoPE cos/sin, core_attention I/O, router probs,
                           shared-expert output, w1/w2 outs, ...).
      - DUMP_LMHEAD=1    : post-decoder lm-head path — final_norm output, lm_head
                           logits (vocab TP-split). The cross-entropy labels /
                           per-token loss are written from the alignment block
                           in ``_forward_step_common``, not this builder.
    """
    dump_block_io = os.environ.get("DUMP_BLOCK_IO", "1") == "1"
    dump_finegrain = os.environ.get("DUMP_FINEGRAIN", "1") == "1"
    dump_lmhead = os.environ.get("DUMP_LMHEAD", "0") == "1"
    os.makedirs(save_dir, exist_ok=True)

    def make_hook(name: str):
        path = os.path.join(save_dir, f"{name}.pt")

        def hook(_module, _inp, out):
            if os.path.exists(path):
                print(
                    f"[ALIGN] dump skip {name} (file already exists, expected with multi-rank): {path}",
                    flush=True,
                )
                return
            tensor = out[0] if isinstance(out, (tuple, list)) else out
            if not isinstance(tensor, torch.Tensor):
                return
            torch.save(tensor.detach().cpu(), path)
            print(f"[ALIGN] intermediate saved {name}: shape={tuple(tensor.shape)}", flush=True)
            print(f"[ALIGN] intermediate {name}: {tensor}", flush=True)
            if name == "embedding":
                _print_embedding_debug_info(_module, tensor)
                word_emb = getattr(_module, "word_embeddings", None)
                if word_emb is not None and hasattr(word_emb, "weight"):
                    w_path = os.path.join(save_dir, "embedding_weight.pt")
                    if not os.path.exists(w_path):
                        w = word_emb.weight.data.detach().cpu()
                        torch.save(w, w_path)
                        wf = w.float()
                        print(f"[ALIGN] embedding_weight saved: shape={tuple(w.shape)}, dtype={w.dtype}", flush=True)
                        print(
                            f"[ALIGN] embedding_weight stats: min={wf.min():.6f}  max={wf.max():.6f}  mean={wf.mean():.6f}  std={wf.std():.6f}",
                            flush=True,
                        )
                        print(f"[ALIGN] embedding_weight: {w}", flush=True)

        return hook

    def make_input_hook(name: str):
        path = os.path.join(save_dir, f"{name}.pt")

        def hook(_module, inp, _out):
            if os.path.exists(path):
                print(
                    f"[ALIGN] dump skip {name} (file already exists, expected with multi-rank): {path}",
                    flush=True,
                )
                return
            if not inp:
                return
            tensor = inp[0]
            if not isinstance(tensor, torch.Tensor):
                return
            torch.save(tensor.detach().cpu(), path)
            print(f"[ALIGN] input saved {name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}", flush=True)
            print(f"[ALIGN] input {name}: {tensor}", flush=True)

        return hook

    def make_rmsnorm_hook(name: str):
        """RMSNorm-specific hook: saves output, weight, and config (eps, zero_centered_gamma, ...)."""
        out_path = os.path.join(save_dir, f"{name}.pt")
        weight_path = os.path.join(save_dir, f"{name}_weight.pt")

        def hook(_module, _inp, out):
            tensor = out[0] if isinstance(out, (tuple, list)) else out
            if isinstance(tensor, torch.Tensor) and not os.path.exists(out_path):
                torch.save(tensor.detach().cpu(), out_path)
                tf = tensor.detach().float()
                print(
                    f"[ALIGN] rmsnorm_out saved {name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}", flush=True
                )
                print(
                    f"[ALIGN] rmsnorm_out stats {name}: min={tf.min():.6f}  max={tf.max():.6f}  mean={tf.mean():.6f}  std={tf.std():.6f}",
                    flush=True,
                )
                print(f"[ALIGN] rmsnorm_out {name}: {tensor}", flush=True)

            weight = getattr(_module, "weight", None)
            if weight is not None and not os.path.exists(weight_path):
                w = weight.data.detach().cpu()
                torch.save(w, weight_path)
                wf = w.float()
                print(
                    f"[ALIGN] rmsnorm_weight saved {name}_weight: shape={tuple(w.shape)}, dtype={w.dtype}", flush=True
                )
                print(
                    f"[ALIGN] rmsnorm_weight stats {name}_weight: min={wf.min():.6f}  max={wf.max():.6f}  mean={wf.mean():.6f}  std={wf.std():.6f}",
                    flush=True,
                )
                print(f"[ALIGN] rmsnorm_weight {name}_weight: {w}", flush=True)
                print(
                    f"[ALIGN] rmsnorm_cfg {name}: module={type(_module).__name__}  "
                    f"eps={getattr(_module, 'eps', 'N/A')}  "
                    f"zero_centered_gamma={getattr(_module, 'zero_centered_gamma', 'N/A')}  "
                    f"sequence_parallel={getattr(_module, 'sequence_parallel', 'N/A')}",
                    flush=True,
                )

        return hook

    def make_qkv_canon_hook(attn_module, qkv_name: str, ln_name: str):
        """Capture every tensor exposed by the fused ``TELayerNormColumnParallelLinear``.

        ``linear_qkv.forward`` returns ``(mixed_qkv, bias, ln_output)`` (Megatron's
        SelfAttention.forward unpacks all three). This hook saves the canonical
        Q|K|V repack of the GQA-interleaved mixed_qkv, the fused-LN output,
        and the GEMM/LN weights, all named to match the SteptronOss alignment
        compare scripts.

        Step3.5 sets ``head_wise_attn_gate=True`` (config.head_wise_attn_gate),
        which fuses per-head scalar gate weights into the tail of linear_qkv;
        those are split out into ``{qkv_name}_gate.pt`` / ``_weight_gate.pt``
        so the canonical Q|K|V dumps stay gate-free.
        """
        path = os.path.join(save_dir, f"{qkv_name}.pt")
        q_path = os.path.join(save_dir, f"{qkv_name}_q.pt")
        k_path = os.path.join(save_dir, f"{qkv_name}_k.pt")
        v_path = os.path.join(save_dir, f"{qkv_name}_v.pt")
        input_path = os.path.join(save_dir, f"{qkv_name}_input.pt")
        weight_canon_path = os.path.join(save_dir, f"{qkv_name}_weight.pt")
        weight_q_path = os.path.join(save_dir, f"{qkv_name}_weight_q.pt")
        weight_k_path = os.path.join(save_dir, f"{qkv_name}_weight_k.pt")
        weight_v_path = os.path.join(save_dir, f"{qkv_name}_weight_v.pt")
        gate_act_path = os.path.join(save_dir, f"{qkv_name}_gate.pt")
        gate_weight_path = os.path.join(save_dir, f"{qkv_name}_weight_gate.pt")
        ln_path = os.path.join(save_dir, f"{ln_name}.pt")
        ln_weight_path = os.path.join(save_dir, f"{ln_name}_weight.pt")
        ln_eff_weight_path = os.path.join(save_dir, f"{ln_name}_effective_weight.pt")

        def hook(_module, inp, out):
            mixed_qkv = out[0] if isinstance(out, (tuple, list)) else out
            ln_output = out[2] if isinstance(out, (tuple, list)) and len(out) >= 3 else None
            if not isinstance(mixed_qkv, torch.Tensor):
                return

            nh = attn_module.num_attention_heads_per_partition
            ng = attn_module.num_query_groups_per_partition
            hn = attn_module.hidden_size_per_attention_head
            per_g = nh // ng

            has_gate = bool(getattr(attn_module.config, "head_wise_attn_gate", False))
            if has_gate:
                gate_size = nh
                qkv_part, gate_act = torch.split(
                    mixed_qkv, [mixed_qkv.size(-1) - gate_size, gate_size], dim=-1
                )
            else:
                qkv_part = mixed_qkv
                gate_act = None

            num_qkv_heads_per_group = per_g + 2
            new_tensor_shape = qkv_part.size()[:-1] + (
                ng,
                num_qkv_heads_per_group * hn,
            )
            qkv_view = qkv_part.view(*new_tensor_shape)
            split_arg_list = [per_g * hn, hn, hn]
            query, key, value = torch.split(qkv_view, split_arg_list, dim=3)

            S, B = qkv_part.shape[0], qkv_part.shape[1]
            q_part = query.reshape(S, B, ng * per_g * hn)
            k_part = key.reshape(S, B, ng * hn)
            v_part = value.reshape(S, B, ng * hn)
            canonical = torch.cat([q_part, k_part, v_part], dim=-1).contiguous()
            if not os.path.exists(path):
                torch.save(canonical.detach().cpu(), path)
                print(
                    f"[ALIGN] intermediate saved {qkv_name} (canonical Q|K|V): shape={tuple(canonical.shape)}",
                    flush=True,
                )
                print(f"[ALIGN] intermediate {qkv_name}: {canonical}", flush=True)
            for sub_name, sub_tensor, sub_path in (
                ("q", q_part, q_path),
                ("k", k_part, k_path),
                ("v", v_part, v_path),
            ):
                if os.path.exists(sub_path):
                    continue
                s = sub_tensor.contiguous().detach().cpu()
                torch.save(s, sub_path)
                sf = s.float()
                print(
                    f"[ALIGN] intermediate saved {qkv_name}_{sub_name}: shape={tuple(s.shape)}, dtype={s.dtype}",
                    flush=True,
                )
                print(
                    f"[ALIGN] intermediate stats {qkv_name}_{sub_name}: min={sf.min():.6f}  max={sf.max():.6f}  mean={sf.mean():.6f}  std={sf.std():.6f}",
                    flush=True,
                )
                print(f"[ALIGN] intermediate {qkv_name}_{sub_name}: {s}", flush=True)

            if has_gate and gate_act is not None and not os.path.exists(gate_act_path):
                ga = gate_act.detach().contiguous()
                torch.save(ga.cpu(), gate_act_path)
                gf = ga.float()
                print(f"[ALIGN] gate saved {qkv_name}_gate: shape={tuple(ga.shape)}, dtype={ga.dtype}", flush=True)
                print(
                    f"[ALIGN] gate stats {qkv_name}_gate: min={gf.min():.6f}  max={gf.max():.6f}  mean={gf.mean():.6f}  std={gf.std():.6f}",
                    flush=True,
                )
                print(f"[ALIGN] gate {qkv_name}_gate: {ga}", flush=True)

            if isinstance(ln_output, torch.Tensor) and not os.path.exists(ln_path):
                torch.save(ln_output.detach().cpu(), ln_path)
                lf = ln_output.detach().float()
                print(
                    f"[ALIGN] ln_output saved {ln_name}: shape={tuple(ln_output.shape)}, dtype={ln_output.dtype}",
                    flush=True,
                )
                print(
                    f"[ALIGN] ln_output stats {ln_name}: min={lf.min():.6f}  max={lf.max():.6f}  mean={lf.mean():.6f}  std={lf.std():.6f}",
                    flush=True,
                )
                print(f"[ALIGN] ln_output {ln_name}: {ln_output}", flush=True)

            if inp:
                inp_tensor = ln_output
                if isinstance(inp_tensor, torch.Tensor) and not os.path.exists(input_path):
                    torch.save(inp_tensor.detach().cpu(), input_path)
                    inp_f = inp_tensor.detach().float()
                    print(
                        f"[ALIGN] qkv_input saved {qkv_name}_input: shape={tuple(inp_tensor.shape)}, dtype={inp_tensor.dtype}",
                        flush=True,
                    )
                    print(
                        f"[ALIGN] qkv_input stats {qkv_name}_input: min={inp_f.min():.6f}  max={inp_f.max():.6f}  mean={inp_f.mean():.6f}  std={inp_f.std():.6f}",
                        flush=True,
                    )
                    print(f"[ALIGN] qkv_input {qkv_name}_input: {inp_tensor}", flush=True)

            weight = getattr(_module, "weight", None)
            if weight is not None and not os.path.exists(weight_canon_path):
                w = weight.data.detach().cpu()
                hidden = w.shape[-1]
                if has_gate:
                    w_qkv = w[: w.shape[0] - nh]
                    w_gate = w[w.shape[0] - nh :]
                    if not os.path.exists(gate_weight_path):
                        torch.save(w_gate, gate_weight_path)
                        wgf = w_gate.float()
                        print(
                            f"[ALIGN] gate_weight saved {qkv_name}_weight_gate: shape={tuple(w_gate.shape)}, dtype={w_gate.dtype}",
                            flush=True,
                        )
                        print(
                            f"[ALIGN] gate_weight stats {qkv_name}_weight_gate: min={wgf.min():.6f}  max={wgf.max():.6f}  mean={wgf.mean():.6f}  std={wgf.std():.6f}",
                            flush=True,
                        )
                        print(f"[ALIGN] gate_weight {qkv_name}_weight_gate: {w_gate}", flush=True)
                else:
                    w_qkv = w

                w_view = w_qkv.reshape(ng, per_g + 2, hn, hidden)
                wq = w_view[:, :per_g, :, :].reshape(ng * per_g * hn, hidden)
                wk = w_view[:, per_g, :, :].reshape(ng * hn, hidden)
                wv = w_view[:, per_g + 1, :, :].reshape(ng * hn, hidden)
                w_canon = torch.cat([wq, wk, wv], dim=0).contiguous()
                torch.save(w_canon, weight_canon_path)
                wcf = w_canon.float()
                print(
                    f"[ALIGN] qkv_weight saved {qkv_name}_weight (canonical Q|K|V): shape={tuple(w_canon.shape)}, dtype={w_canon.dtype}",
                    flush=True,
                )
                print(
                    f"[ALIGN] qkv_weight stats {qkv_name}_weight: min={wcf.min():.6f}  max={wcf.max():.6f}  mean={wcf.mean():.6f}  std={wcf.std():.6f}",
                    flush=True,
                )
                print(f"[ALIGN] qkv_weight {qkv_name}_weight: {w_canon}", flush=True)
                for sub_name, sub_w, sub_path in (
                    ("q", wq.contiguous(), weight_q_path),
                    ("k", wk.contiguous(), weight_k_path),
                    ("v", wv.contiguous(), weight_v_path),
                ):
                    if os.path.exists(sub_path):
                        continue
                    torch.save(sub_w, sub_path)
                    wsf = sub_w.float()
                    print(
                        f"[ALIGN] qkv_weight saved {qkv_name}_weight_{sub_name}: shape={tuple(sub_w.shape)}, dtype={sub_w.dtype}",
                        flush=True,
                    )
                    print(
                        f"[ALIGN] qkv_weight stats {qkv_name}_weight_{sub_name}: min={wsf.min():.6f}  max={wsf.max():.6f}  mean={wsf.mean():.6f}  std={wsf.std():.6f}",
                        flush=True,
                    )
                    print(f"[ALIGN] qkv_weight {qkv_name}_weight_{sub_name}: {sub_w}", flush=True)

            ln_weight = getattr(_module, "layer_norm_weight", None)
            if ln_weight is not None and not os.path.exists(ln_weight_path):
                w = ln_weight.data.detach().cpu()
                torch.save(w, ln_weight_path)
                wf = w.float()
                print(f"[ALIGN] ln_weight saved {ln_name}_weight: shape={tuple(w.shape)}, dtype={w.dtype}", flush=True)
                print(
                    f"[ALIGN] ln_weight stats {ln_name}_weight: min={wf.min():.6f}  max={wf.max():.6f}  mean={wf.mean():.6f}  std={wf.std():.6f}",
                    flush=True,
                )
                print(f"[ALIGN] ln_weight {ln_name}_weight: {w}", flush=True)

                zero_centered = bool(getattr(_module, "zero_centered_gamma", False))
                eff = (w + 1.0) if zero_centered else w.clone()
                torch.save(eff, ln_eff_weight_path)
                eff_f = eff.float()
                print(
                    f"[ALIGN] ln_effective_weight saved {ln_name}_effective_weight: "
                    f"shape={tuple(eff.shape)}, zero_centered_gamma={zero_centered}",
                    flush=True,
                )
                print(
                    f"[ALIGN] ln_effective_weight stats {ln_name}_effective_weight: min={eff_f.min():.6f}  max={eff_f.max():.6f}  mean={eff_f.mean():.6f}  std={eff_f.std():.6f}",
                    flush=True,
                )
                print(f"[ALIGN] ln_effective_weight {ln_name}_effective_weight: {eff}", flush=True)

                print(
                    f"[ALIGN] rmsnorm_cfg {ln_name}: module={type(_module).__name__}  "
                    f"eps={getattr(_module, 'eps', 'N/A')}  "
                    f"zero_centered_gamma={zero_centered}  "
                    f"sequence_parallel={getattr(_module, 'sequence_parallel', 'N/A')}  "
                    f"normalization={getattr(_module, 'normalization', 'N/A')}",
                    flush=True,
                )

        return hook

    def make_post_rope_hook(qkv_name: str):
        """forward_pre_hook on core_attention to capture Q/K/V after RoPE."""
        q_path = os.path.join(save_dir, f"{qkv_name}_q_post_rope.pt")
        k_path = os.path.join(save_dir, f"{qkv_name}_k_post_rope.pt")
        v_path = os.path.join(save_dir, f"{qkv_name}_v_post_rope.pt")

        def _dump(tensor, path, tag):
            if not isinstance(tensor, torch.Tensor) or os.path.exists(path):
                return
            torch.save(tensor.detach().cpu(), path)
            tf = tensor.detach().float()
            print(
                f"[ALIGN] {tag} saved {os.path.basename(path)[:-3]}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}",
                flush=True,
            )
            print(
                f"[ALIGN] {tag} stats {os.path.basename(path)[:-3]}: min={tf.min():.6f}  max={tf.max():.6f}  mean={tf.mean():.6f}  std={tf.std():.6f}",
                flush=True,
            )
            print(f"[ALIGN] {tag} {os.path.basename(path)[:-3]}: {tensor}", flush=True)

        def hook(_module, args, kwargs):
            query = args[0] if len(args) >= 1 else kwargs.get("query")
            key = args[1] if len(args) >= 2 else kwargs.get("key")
            value = args[2] if len(args) >= 3 else kwargs.get("value")
            _dump(query, q_path, "q_post_rope")
            _dump(key, k_path, "k_post_rope")
            _dump(value, v_path, "v_post_rope")

        return hook

    def make_core_attention_io_hook(name: str):
        """Capture core_attention's I/O at the boundary the SDPA actually sees."""
        q_path = os.path.join(save_dir, f"{name}_input_q.pt")
        k_path = os.path.join(save_dir, f"{name}_input_k.pt")
        v_path = os.path.join(save_dir, f"{name}_input_v.pt")
        out_path = os.path.join(save_dir, f"{name}_output.pt")

        def _dump(tensor, path, tag):
            if not isinstance(tensor, torch.Tensor) or os.path.exists(path):
                return
            torch.save(tensor.detach().cpu(), path)
            tf = tensor.detach().float()
            base = os.path.basename(path)[:-3]
            print(
                f"[ALIGN] {tag} saved {base}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}",
                flush=True,
            )
            print(
                f"[ALIGN] {tag} stats {base}: min={tf.min():.6f}  max={tf.max():.6f}  mean={tf.mean():.6f}  std={tf.std():.6f}",
                flush=True,
            )
            print(f"[ALIGN] {tag} {base}: {tensor}", flush=True)

        def pre_hook(_module, args, kwargs):
            query = args[0] if len(args) >= 1 else kwargs.get("query")
            key = args[1] if len(args) >= 2 else kwargs.get("key")
            value = args[2] if len(args) >= 3 else kwargs.get("value")
            _dump(query, q_path, "core_attn_in_q")
            _dump(key, k_path, "core_attn_in_k")
            _dump(value, v_path, "core_attn_in_v")

        def post_hook(_module, _inp, out):
            tensor = out[0] if isinstance(out, (tuple, list)) else out
            _dump(tensor, out_path, "core_attn_out")

        return pre_hook, post_hook

    def make_moe_router_hook(name: str):
        """Capture Megatron MoE TopKRouter's tuple output (probs, routing_map).

        Megatron's TopKRouter forward returns ``(probs, routing_map)`` — a
        post-activation score tensor of shape [T, E] and a per-token expert
        mask/ids tensor. We dump both for cross-framework diff.

        Note: in version 9 MoELayer uses production forward, so
        ``mlp.router.__call__`` is invoked and this hook fires. (Version 6's
        ``MoELayer_debug.forward`` deliberately bypassed submodule __call__ and
        emitted dumps inline — that path is not used here.)
        """
        probs_path = os.path.join(save_dir, f"{name}_probs.pt")
        map_path = os.path.join(save_dir, f"{name}_routing_map.pt")

        def _dump(tensor, path, tag):
            if not isinstance(tensor, torch.Tensor) or os.path.exists(path):
                return
            torch.save(tensor.detach().cpu(), path)
            base = os.path.basename(path)[:-3]
            print(
                f"[ALIGN] {tag} saved {base}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}",
                flush=True,
            )
            if tensor.is_floating_point():
                tf = tensor.detach().float()
                print(
                    f"[ALIGN] {tag} stats {base}: min={tf.min():.6f}  max={tf.max():.6f}  mean={tf.mean():.6f}  std={tf.std():.6f}",
                    flush=True,
                )
            print(f"[ALIGN] {tag} {base}: {tensor}", flush=True)

        def hook(_module, _inp, out):
            if isinstance(out, (tuple, list)) and len(out) >= 2:
                probs, routing_map = out[0], out[1]
            else:
                probs, routing_map = out, None
            _dump(probs, probs_path, "moe_router_probs")
            _dump(routing_map, map_path, "moe_router_map")

        return hook

    def make_rope_emb_hook(rope_name: str):
        """Dump RotaryEmbedding's forward output, plus derived cos/sin."""
        emb_path = os.path.join(save_dir, f"{rope_name}_emb.pt")
        cos_path = os.path.join(save_dir, f"{rope_name}_cos.pt")
        sin_path = os.path.join(save_dir, f"{rope_name}_sin.pt")

        def hook(_module, _inp, out):
            if not isinstance(out, torch.Tensor):
                return
            emb_2d = out
            while emb_2d.dim() > 2:
                emb_2d = emb_2d.squeeze(1)
            if not os.path.exists(emb_path):
                torch.save(emb_2d.detach().cpu(), emb_path)
                ef = emb_2d.detach().float()
                print(
                    f"[ALIGN] rope_emb saved {rope_name}_emb: shape={tuple(emb_2d.shape)}, dtype={emb_2d.dtype}",
                    flush=True,
                )
                print(
                    f"[ALIGN] rope_emb stats {rope_name}_emb: min={ef.min():.6f}  max={ef.max():.6f}  mean={ef.mean():.6f}  std={ef.std():.6f}",
                    flush=True,
                )
                print(f"[ALIGN] rope_emb {rope_name}_emb: {emb_2d}", flush=True)
            if not os.path.exists(cos_path):
                cos = emb_2d.detach().float().cos()
                torch.save(cos.cpu(), cos_path)
                print(
                    f"[ALIGN] rope_cos saved {rope_name}_cos: shape={tuple(cos.shape)}, dtype={cos.dtype}", flush=True
                )
                print(
                    f"[ALIGN] rope_cos stats {rope_name}_cos: min={cos.min():.6f}  max={cos.max():.6f}  mean={cos.mean():.6f}  std={cos.std():.6f}",
                    flush=True,
                )
                print(f"[ALIGN] rope_cos {rope_name}_cos: {cos}", flush=True)
            if not os.path.exists(sin_path):
                sin = emb_2d.detach().float().sin()
                torch.save(sin.cpu(), sin_path)
                print(
                    f"[ALIGN] rope_sin saved {rope_name}_sin: shape={tuple(sin.shape)}, dtype={sin.dtype}", flush=True
                )
                print(
                    f"[ALIGN] rope_sin stats {rope_name}_sin: min={sin.min():.6f}  max={sin.max():.6f}  mean={sin.mean():.6f}  std={sin.std():.6f}",
                    flush=True,
                )
                print(f"[ALIGN] rope_sin {rope_name}_sin: {sin}", flush=True)

        return hook

    gpt_model = unwrap_model(model)
    hooks = []

    if hasattr(gpt_model, "embedding"):
        if dump_block_io:
            hooks.append(gpt_model.embedding.register_forward_hook(make_hook("embedding")))
        if dump_finegrain and hasattr(gpt_model.embedding, "word_embeddings"):
            hooks.append(
                gpt_model.embedding.word_embeddings.register_forward_hook(make_input_hook("embedding_input_ids"))
            )

    decoder = getattr(gpt_model, "decoder", None) or getattr(gpt_model, "encoder", None)
    if decoder is not None and hasattr(decoder, "layers"):
        for i, layer in enumerate(decoder.layers):
            layer_number = layer.layer_number
            layer_idx = layer_number - 1
            if dump_finegrain and getattr(layer, "pre_mlp_layernorm", None) is not None and not isinstance(
                layer.pre_mlp_layernorm, torch.nn.Identity
            ):
                hooks.append(
                    layer.pre_mlp_layernorm.register_forward_hook(make_rmsnorm_hook(f"layer_{layer_idx:03d}_ffn_norm"))
                )
            if hasattr(layer, "self_attention"):
                attn = layer.self_attention
                if dump_block_io:
                    hooks.append(attn.register_forward_hook(make_hook(f"layer_{layer_idx:03d}_attention")))
                if dump_finegrain:
                    if hasattr(attn, "linear_qkv"):
                        hooks.append(
                            attn.linear_qkv.register_forward_hook(
                                make_qkv_canon_hook(
                                    attn,
                                    f"layer_{layer_idx:03d}_attention_qkv",
                                    f"layer_{layer_idx:03d}_attention_norm",
                                )
                            )
                        )
                    if getattr(attn, "q_layernorm", None) is not None:
                        hooks.append(
                            attn.q_layernorm.register_forward_hook(make_rmsnorm_hook(f"layer_{layer_idx:03d}_attention_qnorm"))
                        )
                    if getattr(attn, "k_layernorm", None) is not None:
                        hooks.append(
                            attn.k_layernorm.register_forward_hook(make_rmsnorm_hook(f"layer_{layer_idx:03d}_attention_knorm"))
                        )
                    if hasattr(attn, "core_attention"):
                        hooks.append(
                            attn.core_attention.register_forward_hook(make_hook(f"layer_{layer_idx:03d}_attention_core"))
                        )
                        hooks.append(
                            attn.core_attention.register_forward_pre_hook(
                                make_post_rope_hook(f"layer_{layer_idx:03d}_attention_qkv"),
                                with_kwargs=True,
                            )
                        )
                        _core_pre, _core_post = make_core_attention_io_hook(
                            f"layer_{layer_idx:03d}_attention_core"
                        )
                        hooks.append(
                            attn.core_attention.register_forward_pre_hook(_core_pre, with_kwargs=True)
                        )
                        hooks.append(attn.core_attention.register_forward_hook(_core_post))
                    if getattr(attn, "rotary_pos_emb", None) is not None:
                        hooks.append(
                            attn.rotary_pos_emb.register_forward_hook(
                                make_rope_emb_hook(f"layer_{layer_idx:03d}_attention_rope")
                            )
                        )
                    if hasattr(attn, "linear_proj"):
                        hooks.append(
                            attn.linear_proj.register_forward_hook(
                                make_input_hook(f"layer_{layer_idx:03d}_attention_preproj")
                            )
                        )
            if hasattr(layer, "mlp"):
                mlp = layer.mlp
                if dump_block_io:
                    hooks.append(mlp.register_forward_hook(make_hook(f"layer_{layer_idx:03d}_ffn")))
                if dump_finegrain:
                    # Dense MLP: layer.mlp == MLP (linear_fc1 + SwiGLU + linear_fc2)
                    if hasattr(mlp, "linear_fc1") and getattr(mlp, "linear_fc1", None) is not None:
                        hooks.append(
                            mlp.linear_fc1.register_forward_hook(
                                make_hook(f"layer_{layer_idx:03d}_ffn_w1_out")
                            )
                        )
                    if hasattr(mlp, "linear_fc2") and getattr(mlp, "linear_fc2", None) is not None:
                        hooks.append(
                            mlp.linear_fc2.register_forward_hook(
                                make_hook(f"layer_{layer_idx:03d}_ffn_w2_out")
                            )
                        )
                    # MoE path: Megatron's MoELayer.forward invokes router /
                    # shared_experts via __call__, so observer hooks fire here.
                    # (Version 6 deliberately skipped these because its
                    # MoELayer_debug.forward bypassed submodule __call__ and
                    # emitted dumps inline — but version 9 keeps production
                    # MoELayer.forward, so hooks are the correct attachment
                    # point and don't touch any model logic.)
                    router = getattr(mlp, "router", None)
                    if router is not None:
                        hooks.append(
                            router.register_forward_hook(
                                make_moe_router_hook(f"layer_{layer_idx:03d}_ffn_router")
                            )
                        )
                    shared_experts = getattr(mlp, "shared_experts", None)
                    if shared_experts is not None:
                        hooks.append(
                            shared_experts.register_forward_hook(
                                make_hook(f"layer_{layer_idx:03d}_ffn_shared_out")
                            )
                        )
                    experts = getattr(mlp, "experts", None)
                    if experts is not None:
                        hooks.append(
                            experts.register_forward_hook(
                                make_hook(f"layer_{layer_idx:03d}_ffn_experts_out")
                            )
                        )

    # ===== ALIGNMENT: post-decoder lm-head path =====
    if dump_lmhead:
        if decoder is not None and getattr(decoder, "final_layernorm", None) is not None and not isinstance(
            decoder.final_layernorm, torch.nn.Identity
        ):
            hooks.append(
                decoder.final_layernorm.register_forward_hook(make_rmsnorm_hook("final_norm"))
            )
        if hasattr(gpt_model, "output_layer") and gpt_model.output_layer is not None:
            hooks.append(gpt_model.output_layer.register_forward_hook(make_hook("lm_head_logits")))

    return hooks


def _load_align_batch(batch_path: str, is_first: bool, is_last: bool) -> dict:
    """Load a SteptronOss-saved batch and convert to Megatron-Bridge format.

    SteptronOss key mapping:
      input_ids  [1, S]  -> tokens      [1, S]
      position_id [S]    -> position_ids [1, S]
      labels     [1, S]  -> labels      [1, S]
      loss_masks [1, S]  -> loss_mask   [1, S]
      cu_seqlens [N+1]   -> cu_seqlens  [N+1]  (no padding; argmin = len)
      max_seq_len scalar -> max_seqlen  scalar
    """
    batch = torch.load(batch_path, map_location="cpu")

    cu_seqlens = batch.get("cu_seqlens")
    if cu_seqlens is not None:
        # SteptronOss cu_seqlens has no zero-padding; tell Megatron the full array is valid.
        cu_seqlens_argmin = torch.tensor([len(cu_seqlens)], dtype=torch.long)
        max_seqlen = batch.get("max_seq_len")
        if max_seqlen is not None:
            max_seqlen = max_seqlen.unsqueeze(0) if max_seqlen.dim() == 0 else max_seqlen
        else:
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().unsqueeze(0)
    else:
        cu_seqlens_argmin = None
        max_seqlen = None

    position_id = batch.get("position_id")
    if position_id is not None and position_id.dim() == 1:
        position_id = position_id.unsqueeze(0)  # [S] -> [1, S]

    result: dict = {"attention_mask": None}
    if is_first:
        result["tokens"] = batch["input_ids"].cuda(non_blocking=True)
        result["position_ids"] = position_id.cuda(non_blocking=True) if position_id is not None else None
    else:
        result["tokens"] = None
        result["position_ids"] = None
    if is_last:
        result["labels"] = batch["labels"].cuda(non_blocking=True)
        result["loss_mask"] = batch["loss_masks"].cuda(non_blocking=True)
    else:
        result["labels"] = None
        result["loss_mask"] = None
    if cu_seqlens is not None:
        result["cu_seqlens"] = cu_seqlens.cuda(non_blocking=True)
        result["cu_seqlens_argmin"] = cu_seqlens_argmin  # host tensor
        result["max_seqlen"] = max_seqlen  # host tensor
    else:
        result["cu_seqlens"] = None
        result["cu_seqlens_argmin"] = None
        result["max_seqlen"] = None
    return result


def get_batch_from_iterator(
    data_iterator: Iterable,
    use_mtp: bool = False,
    skip_getting_attention_mask_from_dataset: bool = True,
    *,
    is_first_pp_stage: bool,
    is_last_pp_stage: bool,
    include_full_batch_fields: bool = False,
) -> dict[str, torch.Tensor]:
    """Get a batch of data from the iterator.

    Args:
        data_iterator: The data iterator to get the batch from.
        use_mtp: Whether Multi-Token Prediction layers are enabled.
        skip_getting_attention_mask_from_dataset: If set, the dataset will pass a None attention mask.
        include_full_batch_fields: Whether to include all standard training tensors regardless of PP stage.

    Returns:
        dict[str, torch.Tensor]: A dictionary containing the batch data.
    """
    batch = next(data_iterator)

    required_device_keys = set()
    required_host_keys = set()

    if include_full_batch_fields:
        required_device_keys.update(("tokens", "labels", "loss_mask", "attention_mask", "position_ids"))
    elif not skip_getting_attention_mask_from_dataset:
        required_device_keys.add("attention_mask")

    if "cu_seqlens" in batch:
        required_device_keys.add("cu_seqlens")
        if "cu_seqlens_unpadded" in batch:
            required_device_keys.add("cu_seqlens_unpadded")
        required_host_keys.add("cu_seqlens_argmin")
        required_host_keys.add("max_seqlen")
        if "cu_seqlens_unpadded_argmin" in batch:
            required_host_keys.add("cu_seqlens_unpadded_argmin")

    if not include_full_batch_fields:
        if is_first_pp_stage or use_mtp:
            required_device_keys.update(("tokens", "position_ids"))
        if is_last_pp_stage:
            required_device_keys.update(("labels", "loss_mask"))

    _batch_required_keys = {}
    for key, val in batch.items():
        if key in required_device_keys:
            _batch_required_keys[key] = val.cuda(non_blocking=True) if val is not None else None
        elif key in required_host_keys:
            _batch_required_keys[key] = val.cpu() if val is not None else None
        else:
            _batch_required_keys[key] = None

    return _batch_required_keys


def get_batch(
    data_iterator: Iterable, cfg: ConfigContainer, use_mtp: bool = False, *, pg_collection
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Generate a batch.

    Args:
        data_iterator: Input data iterator
        cfg: Configuration container
        use_mtp: Whether Multi-Token Prediction layers are enabled

    Returns:
        tuple of tensors containing tokens, labels, loss_mask, attention_mask, position_ids,
        cu_seqlens, cu_seqlens_argmin, max_seqlen, cu_seqlens_unpadded, and
        cu_seqlens_unpadded_argmin
    """
    # ===== ALIGNMENT: load batch from SteptronOss if env var is set =====
    _align_batch_path = os.environ.get("MBRIDGE_LOAD_BATCH_PATH", "")
    print(f"[ALIGN] _align_batch_path: {_align_batch_path}")
    if _align_batch_path and os.path.exists(_align_batch_path):
        # is_first = is_pp_first_stage(pg_collection.pp)
        # is_last = is_pp_last_stage(pg_collection.pp)
        is_first = True
        is_last = True
        if (not is_first) and (not is_last):
            return None, None, None, None, None, None, None, None, None, None
        b = _load_align_batch(_align_batch_path, is_first, is_last)
        print(f"[ALIGN] loaded input batch: {b}")
        cp_size = pg_collection.cp.size()
        if cp_size > 1:
            b = _partition_packed_batch_for_cp(b, cp_size)
        else:
            b = get_batch_on_this_cp_rank(b, cp_group=pg_collection.cp)
        return (
            b.get("tokens"),
            b.get("labels"),
            b.get("loss_mask"),
            b.get("attention_mask"),
            b.get("position_ids"),
            b.get("cu_seqlens"),
            b.get("cu_seqlens_argmin"),
            b.get("max_seqlen"),
            None,
            None,
        )
    # ===== END ALIGNMENT =====

    # Determine pipeline stage role via process group collection
    is_first = is_pp_first_stage(pg_collection.pp)
    is_last = is_pp_last_stage(pg_collection.pp)
    is_middle = (not is_first) and (not is_last)
    include_full_batch_fields = is_middle and _middle_pp_stage_needs_batch(cfg)
    if is_middle and not include_full_batch_fields:
        return None, None, None, None, None, None, None, None, None, None

    batch = get_batch_from_iterator(
        data_iterator,
        use_mtp,
        getattr(cfg.dataset, "skip_getting_attention_mask_from_dataset", True),
        is_first_pp_stage=is_first,
        is_last_pp_stage=is_last,
        include_full_batch_fields=include_full_batch_fields,
    )

    cp_size = pg_collection.cp.size()
    has_packed = batch.get("cu_seqlens") is not None
    if has_packed and cp_size > 1:
        batch = _partition_packed_batch_for_cp(batch, cp_size)
    else:
        # slice batch along sequence dimension for context parallelism
        batch = get_batch_on_this_cp_rank(batch, cp_group=pg_collection.cp)

    return (
        batch["tokens"],
        batch["labels"],
        batch["loss_mask"],
        batch.get(
            "attention_mask"
        ),  # Attention_mask is optional for pre-training as a casual mask is generated automatically.
        batch["position_ids"],
        batch.get("cu_seqlens"),
        batch.get("cu_seqlens_argmin"),
        batch.get("max_seqlen"),
        batch.get("cu_seqlens_unpadded"),
        batch.get("cu_seqlens_unpadded_argmin"),
    )


def _forward_step_common(
    state: GlobalState, data_iterator: Iterable, model: GPTModel, return_schedule_plan: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward training step.

    Args:
        state: Global state for the run
        data_iterator: Input data iterator
        model: The GPT Model
        return_schedule_plan (bool): Whether to return the schedule plan instead of the output tensor

    Returns:
        tuple containing the output tensor and loss mask
    """
    timers = state.timers
    straggler_timer = state.straggler_timer

    config = get_model_config(model)
    pg_collection = get_pg_collection(model)
    use_mtp = (getattr(config, "mtp_num_layers", None) or 0) > 0

    timers("batch-generator", log_level=2).start()
    with straggler_timer(bdata=True):
        (
            tokens,
            labels,
            loss_mask,
            attention_mask,
            position_ids,
            cu_seqlens,
            cu_seqlens_argmin,
            max_seqlen,
            cu_seqlens_unpadded,
            cu_seqlens_unpadded_argmin,
        ) = get_batch(data_iterator, state.cfg, use_mtp, pg_collection=pg_collection)
    timers("batch-generator").stop()

    forward_args = {
        "input_ids": tokens,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

    # Add packed sequence support
    if cu_seqlens is not None:
        packed_seq_params = {
            "cu_seqlens": cu_seqlens,
            "cu_seqlens_argmin": cu_seqlens_argmin,
            "max_seqlen": max_seqlen,
            "cu_seqlens_unpadded": cu_seqlens_unpadded,
            "cu_seqlens_unpadded_argmin": cu_seqlens_unpadded_argmin,
        }
        # total_tokens drives seq_idx computation in PackedSeqParams.__post_init__,
        # which is only needed for Mamba/hybrid SSM layers. Skip it for pure
        # transformer models to avoid per-step CUDA overhead.
        if getattr(config, "is_hybrid_model", False):
            if tokens is not None:
                packed_seq_params["total_tokens"] = tokens.size(1)
            elif labels is not None:
                packed_seq_params["total_tokens"] = labels.size(1)
            else:
                packed_seq_params["total_tokens"] = getattr(config, "seq_length", None)
        forward_args["packed_seq_params"] = get_packed_seq_params(packed_seq_params)

    with straggler_timer:
        if return_schedule_plan:
            assert config.overlap_moe_expert_parallel_comm, (
                "overlap_moe_expert_parallel_comm must be enabled to return the schedule plan"
            )
            schedule_plan = model.build_schedule_plan(
                tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask
            )
            return schedule_plan, loss_mask
        else:
            # ===== ALIGNMENT: one-shot model structure print =====
            _maybe_print_model_structure(model)
            # ===== ALIGNMENT: register intermediate activation hooks =====
            _intermediate_dir = os.environ.get("MBRIDGE_SAVE_INTERMEDIATE_PATH", "")
            # MBRIDGE_DUMP_PP_RANK0_ONLY=1 (default) restricts the intermediate-activation
            # dump to PP rank 0 — useful while aligning one PP stage at a time.
            # Set to 0 once PP0 is matched to enable dumps on all PP stages.
            _dump_pp_rank0_only = os.environ.get("MBRIDGE_DUMP_PP_RANK0_ONLY", "1") == "1"
            _pp_rank = parallel_state.get_pipeline_model_parallel_rank()
            _dump_enabled = bool(_intermediate_dir) and (not _dump_pp_rank0_only or _pp_rank == 0)
            if _intermediate_dir:
                print(
                    f"[ALIGN] [rank={torch.distributed.get_rank()} pp_rank={_pp_rank}] "
                    f"intermediate activation hooks: "
                    f"{'registering' if _dump_enabled else 'SKIP (MBRIDGE_DUMP_PP_RANK0_ONLY=1)'}"
                )
            _ihooks = _build_intermediate_hooks(model, _intermediate_dir) if _dump_enabled else []
            # ===== END ALIGNMENT =====
            output_tensor = model(**forward_args)
            # ===== ALIGNMENT: remove intermediate hooks =====
            for _h in _ihooks:
                _h.remove()
            if _intermediate_dir:
                print(
                    f"[ALIGN] [rank={torch.distributed.get_rank()}] removed intermediate activation hooks"
                )
            # ===== ALIGNMENT: lm-head loss path dumps =====
            # Mirror SteptronOss NTPTrainerConfig.loss_func() exactly so the
            # per-token loss tensor matches in shape, dtype, and reduction
            # order:
            #   logits [B,S,V/tp] -> transpose -> [S,B,V/tp]
            #   labels [B,S]      -> transpose -> [S,B]
            #   losses = vocab_parallel_cross_entropy(logits.float(), labels) -> [S,B] fp32
            #   losses.transpose(1,0).contiguous()                            -> [B,S] fp32
            # Only PP last stage holds logits, so gate on is_pp_last_stage.
            # Run a labels=None forward to bypass GPTModel's internal CE and
            # obtain raw logits; idempotent file checks let the first forward
            # win across replay/recompute.
            if (
                os.environ.get("DUMP_LMHEAD", "0") == "1"
                and _intermediate_dir
                and is_pp_last_stage(pg_collection.pp)
            ):
                from megatron.core.tensor_parallel import vocab_parallel_cross_entropy

                os.makedirs(_intermediate_dir, exist_ok=True)
                with torch.no_grad():
                    _lmhead_logits = model(**{**forward_args, "labels": None})  # [B,S,V/tp]
                    _logits_sbv = _lmhead_logits.transpose(0, 1).contiguous()  # [S,B,V/tp]
                    _labels_sb = labels.transpose(0, 1).contiguous()  # [S,B]
                    _losses_sb = vocab_parallel_cross_entropy(_logits_sbv.float(), _labels_sb)  # [S,B] fp32
                    _lm_loss_per_token = _losses_sb.transpose(1, 0).contiguous()  # [B,S] fp32

                for _name, _tensor in (
                    ("labels", labels),
                    ("loss_mask", loss_mask),
                    ("lm_loss_per_token", _lm_loss_per_token),
                ):
                    if not isinstance(_tensor, torch.Tensor):
                        continue
                    _p = os.path.join(_intermediate_dir, f"{_name}.pt")
                    if os.path.exists(_p):
                        print(
                            f"[ALIGN] lmhead skip {_name} (file already exists, expected with multi-rank): {_p}",
                            flush=True,
                        )
                        continue
                    torch.save(_tensor.detach().cpu(), _p)
                    print(
                        f"[ALIGN] lmhead saved {_name}: shape={tuple(_tensor.shape)}, dtype={_tensor.dtype}",
                        flush=True,
                    )
                    print(f"[ALIGN] lmhead {_name}: {_tensor}", flush=True)
            # ===== END ALIGNMENT =====

    # ===== ALIGNMENT: save logits and compare with SteptronOss =====
    _align_output_path = os.environ.get("MBRIDGE_SAVE_OUTPUT_PATH", "")
    if _align_output_path and not os.path.exists(_align_output_path) and is_pp_last_stage(pg_collection.pp):
        _align_fwd = {**forward_args, "labels": None}
        with torch.no_grad():
            _align_logits = model(**_align_fwd)  # [B, S, V/tp] when labels=None
            print(f"[ALIGN] _align_logits.shape: {_align_logits.shape}")
            print(f"[ALIGN] _align_logits: {_align_logits}")
        torch.save(_align_logits.detach().cpu(), _align_output_path)
        print(
            f"[ALIGN] Megatron-Bridge logits saved to {_align_output_path}: "
            f"shape={tuple(_align_logits.shape)}, dtype={_align_logits.dtype}",
            flush=True,
        )

        # Compare with SteptronOss logits if available.
        _steptron_output_path = os.environ.get("STEPTRON_SAVE_OUTPUT_PATH", "")
        if _steptron_output_path and os.path.exists(_steptron_output_path):
            _steptron_logits = torch.load(_steptron_output_path, map_location="cpu")
            # SteptronOss saves [S, B, V/tp]; Megatron saves [B, S, V/tp] -> align shapes.
            if _steptron_logits.shape != _align_logits.cpu().shape:
                _steptron_logits = _steptron_logits.transpose(0, 1).contiguous()
            _mbridge_logits_cpu = _align_logits.detach().cpu().float()
            _steptron_logits_cpu = _steptron_logits.float()
            print(f"[ALIGN] _steptron_logits_cpu.shape: {_steptron_logits_cpu.shape}")
            print(f"[ALIGN] _steptron_logits_cpu: {_steptron_logits_cpu}")
            _match = torch.allclose(_mbridge_logits_cpu, _steptron_logits_cpu, atol=0.01)
            _max_diff = (_mbridge_logits_cpu - _steptron_logits_cpu).abs().max().item()
            _mean_diff = (_mbridge_logits_cpu - _steptron_logits_cpu).abs().mean().item()
            print(
                f"[ALIGN] Logits comparison (atol=0.01): match={_match} | "
                f"max_diff={_max_diff:.6f} | mean_diff={_mean_diff:.6f} | "
                f"mbridge_shape={tuple(_mbridge_logits_cpu.shape)} | "
                f"steptron_shape={tuple(_steptron_logits_cpu.shape)}",
                flush=True,
            )
        else:
            print(
                f"[ALIGN] SteptronOss logits not found at STEPTRON_SAVE_OUTPUT_PATH="
                f"'{_steptron_output_path}', skipping comparison.",
                flush=True,
            )
    # ===== END ALIGNMENT =====

    return output_tensor, loss_mask


def forward_step(
    state: GlobalState, data_iterator: Iterable, model: GPTModel, return_schedule_plan: bool = False
) -> tuple[torch.Tensor, partial]:
    """Forward training step.

    Args:
        state: Global state for the run
        data_iterator: Input data iterator
        model: The GPT Model
        return_schedule_plan (bool): Whether to return the schedule plan instead of the output tensor

    Returns:
        tuple containing the output tensor and the loss function
    """
    output, loss_mask = _forward_step_common(state, data_iterator, model, return_schedule_plan)

    loss_function = _create_loss_function(
        loss_mask,
        check_for_nan_in_loss=state.cfg.rerun_state_machine.check_for_nan_in_loss,
        check_for_spiky_loss=state.cfg.rerun_state_machine.check_for_spiky_loss,
    )

    return output, loss_function


def _create_loss_function(loss_mask: torch.Tensor, check_for_nan_in_loss: bool, check_for_spiky_loss: bool) -> partial:
    """Create a partial loss function with the specified configuration.

    Args:
        loss_mask: Used to mask out some portions of the loss
        check_for_nan_in_loss: Whether to check for NaN values in the loss
        check_for_spiky_loss: Whether to check for spiky loss values

    Returns:
        A partial function that can be called with output_tensor to compute the loss
    """
    return partial(
        masked_next_token_loss,
        loss_mask,
        check_for_nan_in_loss=check_for_nan_in_loss,
        check_for_spiky_loss=check_for_spiky_loss,
    )


def forward_step_modelopt(
    state: GlobalState, data_iterator: Iterable, model: GPTModel, return_schedule_plan: bool = False
) -> tuple[torch.Tensor, partial]:
    """Forward training step with ModelOpt required modifications.

    Args:
        state: Global state for the run
        data_iterator: Input data iterator
        model: The GPT Model
        return_schedule_plan (bool): Whether to return the schedule plan instead of the output tensor

    Returns:
        tuple containing the output tensor and the loss function
    """
    output, loss_mask = _forward_step_common(state, data_iterator, model, return_schedule_plan)

    loss_function = _create_loss_function_modelopt(
        loss_mask,
        model,
        check_for_nan_in_loss=state.cfg.rerun_state_machine.check_for_nan_in_loss,
        check_for_spiky_loss=state.cfg.rerun_state_machine.check_for_spiky_loss,
    )

    return output, loss_function


def _create_loss_function_modelopt(
    loss_mask: torch.Tensor, model: GPTModel, check_for_nan_in_loss: bool, check_for_spiky_loss: bool
) -> partial:
    """Create a partial loss function with the specified configuration.

    Kept here for backward compatibility with tests and callers that patch
    `megatron.bridge.training.gpt_step.masked_next_token_loss`.

    Args:
        loss_mask: Used to mask out some portions of the loss
        model: The GPT Model
        check_for_nan_in_loss: Whether to check for NaN values in the loss
        check_for_spiky_loss: Whether to check for spiky loss values

    Returns:
        A partial function that can be called with output_tensor to compute the loss
    """
    mnt_loss_func = partial(
        masked_next_token_loss,
        loss_mask,
        check_for_nan_in_loss=check_for_nan_in_loss,
        check_for_spiky_loss=check_for_spiky_loss,
    )
    unwrapped_model = unwrap_model(model)
    if isinstance(unwrapped_model, mtd.DistillationModel):
        return partial(loss_func_kd, loss_mask=loss_mask, original_loss_fn=mnt_loss_func, model=unwrapped_model)
    else:
        return mnt_loss_func
