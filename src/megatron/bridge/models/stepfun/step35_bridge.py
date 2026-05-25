# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import logging
import os
from functools import partial
from typing import Dict, Optional

import torch
import torch.nn.functional as F
import transformer_engine.pytorch as te
from megatron.core import parallel_state
from megatron.core.extensions.transformer_engine import (
    TEDotProductAttention,
    TELayerNormColumnParallelLinear,
    TERowParallelLinear,
    _get_extra_te_kwargs,
)
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.mappings import reduce_from_tensor_model_parallel_region
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.moe.moe_layer import MoELayer
from transformers import AutoConfig

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
    QKVGMapping,
)
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.stepfun.configuration_step35 import Step35Config
from megatron.bridge.models.stepfun.step35_provider import (
    Step35DecoderLayer,
    Step35ModelProvider,
    Step35SharedExpertMLP,
)


logger = logging.getLogger(__name__)


def _get_swiglu_limit(layer_id, limits):
    """Return the per-layer SwiGLU clip value, or ``None`` to mean "no clip".

    SteptronOss treats ``0.0`` the same as "no clip" (``if swiglu_limit:`` falls
    through), so this helper normalizes both ``None`` and ``0.0`` to ``None``.
    """
    if limits is None or layer_id is None or layer_id < 0 or layer_id >= len(limits):
        return None
    v = limits[layer_id]
    if v is None or float(v) == 0.0:
        return None
    return float(v)


def _swiglu_with_clip_after_silu(gate_up, limit):
    """SteptronOss-equivalent SwiGLU with optional clip.

    Mirrors ``steptronoss/model/common/feed_forward.py:20-26`` bit-for-bit:
        l, r = chunk(x, 2, dim=-1)
        l = silu(l)
        if limit:
            l = l.clamp(max=limit)
            r = r.clamp(-limit, limit)
        return l * r
    """
    l, r = torch.chunk(gate_up, 2, dim=-1)
    l = F.silu(l)
    if limit is not None:
        l = l.clamp(max=limit)
        r = r.clamp(min=-limit, max=limit)
    return l * r


def _swiglu_with_clip_before_silu(gate_up, limit):
    """Megatron-Core-equivalent SwiGLU with optional clip.

    Mirrors ``megatron/core/fusions/fused_bias_swiglu.py:51-58`` bit-for-bit:
        l, r = chunk(x, 2, dim=-1)
        if limit:
            l = l.clamp(max=limit)
            r = r.clamp(-limit, limit)
        l = silu(l)
        return l * r
    """
    l, r = torch.chunk(gate_up, 2, dim=-1)
    if limit is not None:
        l = l.clamp(max=limit)
        r = r.clamp(min=-limit, max=limit)
    l = F.silu(l)
    return l * r

print(f"MEGATRON_SWIGLU_WITH_CLIP_AFTER_SILU: {os.environ['MEGATRON_SWIGLU_WITH_CLIP_AFTER_SILU']}")
if os.environ.get("MEGATRON_SWIGLU_WITH_CLIP_AFTER_SILU","0") == "1":
    _swiglu_with_clip = _swiglu_with_clip_after_silu
    print("Using _swiglu_with_clip_after_silu")
else:
    _swiglu_with_clip = _swiglu_with_clip_before_silu
    print("Using _swiglu_with_clip_before_silu")

# Register the Step3.5 config with transformers AutoConfig.
# This allows AutoConfig.from_pretrained to resolve "step3p5" without requiring
# hub access (works in offline CI environments).
#
# The literal strings "step3p5" and "Step3p5ForCausalLM" are *external HF
# identifiers*: they come from the `model_type` and `architectures` fields in
# the config.json shipped on `stepfun-ai/Step-3.5-Flash`. They are intentionally
# NOT renamed to "step35" / "Step35ForCausalLM" — otherwise
# `AutoConfig.from_pretrained("stepfun-ai/Step-3.5-Flash")` would route to a
# different config class and the bridge resolution below would fail.
AutoConfig.register("step3p5", Step35Config, exist_ok=True)


class StackedExpertAutoMapping(AutoMapping):
    """Maps Megatron per-expert weight{i} ↔ HF stacked expert tensor[i].

    Step3.5 HF stores all experts in a single stacked tensor, e.g.
    ``model.layers.*.moe.down_proj.weight`` with shape ``[num_experts, H, I]``.
    Megatron creates individual per-expert tensors named ``weight0``, ``weight1``, …

    The ``megatron_param`` uses a trailing ``weight*`` wildcard to match these names;
    ``hf_param`` has one fewer wildcard (no expert index in the path).  During
    wildcard resolution ``_resolve_names`` resets ``capture_index`` to 0 for the HF
    side, so ``hf_param`` only consumes the layer-index capture and the expert-index
    capture is available to slice the stacked tensor in ``hf_to_megatron``.
    """

    is_grouped_export = True  # All per-expert tasks share the same HF stacked tensor.

    def _expert_idx(self) -> int:
        return int(self.megatron_param.rsplit("weight", 1)[-1])

    def hf_to_megatron(self, hf_weights: torch.Tensor, megatron_module) -> torch.Tensor:
        # hf_weights: [num_experts, H, I] — slice to this expert before delegating.
        return super().hf_to_megatron(hf_weights[self._expert_idx()], megatron_module)


class StackedExpertGatedMLPMapping(GatedMLPMapping):
    """GatedMLPMapping for per-expert Megatron weights backed by HF stacked tensors.

    HF stores all experts' gate/up projections as stacked tensors with shape
    [num_experts, I, H].  Megatron creates individual per-expert
    ``linear_fc1.weight{i}`` tensors (shape [2*I, H], gate+up fused).

    ``megatron_param`` uses a trailing ``weight*`` wildcard.  ``gate`` / ``up``
    each have one fewer wildcard (no expert index in the HF path).  During
    wildcard resolution ``_resolve_names`` resets ``capture_index`` for every
    dict key, so both gate/up only consume the layer-index capture.
    """

    is_grouped_export = True  # All per-expert tasks share the same HF stacked tensors.

    def _expert_idx(self) -> int:
        return int(self.megatron_param.rsplit("weight", 1)[-1])

    def hf_to_megatron(self, hf_weights: Dict[str, torch.Tensor], megatron_module) -> torch.Tensor:
        # hf_weights["gate"/"up"]: [num_experts, I, H] — slice to this expert.
        expert_idx = self._expert_idx()
        sliced = {
            "gate": hf_weights["gate"][expert_idx],
            "up": hf_weights["up"][expert_idx],
        }
        return super().hf_to_megatron(sliced, megatron_module)


class _MTPDenseLayerSpecsList(list):
    """List of per-decoder-layer specs that returns a dense spec on negative-index access.

    ``get_gpt_mtp_block_spec_for_backend`` reads ``spec.layer_specs[-1]`` to decide
    which layer type the MTP transformer sub-layers should use.  For Step3.5 the
    last decoder layer (layer 44) is MoE, but MTP layers 45-47 are NOT in
    ``moe_layers_enum`` and must be dense.

    Overriding ``__getitem__`` for negative indices intercepts only that single
    look-up while leaving normal forward iteration (used by ``TransformerBlock``
    to instantiate the 45 main decoder layers) completely unaffected — CPython's
    list iterator operates on the internal C array directly, bypassing
    ``__getitem__``.
    """

    def __init__(self, data, dense_mtp_spec):
        super().__init__(data)
        self._dense_mtp_spec = dense_mtp_spec

    def __getitem__(self, idx):
        if isinstance(idx, int) and idx < 0:
            return self._dense_mtp_spec
        return super().__getitem__(idx)


class TELayerNormColumnParallelLinear_debug(TELayerNormColumnParallelLinear):
    """Bit-for-bit reference of SteptronOss's ``attention_norm`` (RMSNorm) + ``wqkv`` (ColumnParallelLinear).

    Inherits parameter init, TP sharding, and sharded_state_dict from TE so the checkpoint stays
    compatible with the production class — only ``forward`` is replaced with the explicit fp32-norm /
    bf16-multiply / no-bias-linear sequence used by SteptronOss/RMSNorm so the two stacks produce
    identical activations down to the last bit.

    Reference: SteptronOss ``rms_norm.py`` (``rms_foward`` + ``RMSNorm.forward``):
        l2 = rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-5)
        y  = (x.float() * l2).type_as(x)
        return y * (weight + 1.0)        # use_zero_init=True ⇒ bias=1
    followed by ``ColumnParallelLinear`` with ``bias=False, gather_output=False``.
    """

    def forward(self, x):
        in_dtype = x.dtype
        x_fp32 = x.float()
        l2_norm_inv = torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.config.layernorm_epsilon)
        y = (x_fp32 * l2_norm_inv).to(in_dtype)

        ln_weight = self.layer_norm_weight
        if self.config.layernorm_zero_centered_gamma:
            ln_weight = ln_weight + 1.0

        ln_output = y * ln_weight
        out_hidden = torch.nn.functional.linear(ln_output, self.weight)

        ret_bias = None
        ret_ln_output = ln_output  # always return the ln_output
        return out_hidden, ret_bias, ret_ln_output


class TENorm_debug(te.RMSNorm):
    """Bit-for-bit reference of SteptronOss's ``q_norm`` / ``k_norm`` (RMSNorm over head_dim).

    Inherits parameter init / naming / sharded_state_dict from ``te.pytorch.RMSNorm`` (which is what
    ``TENorm`` builds in production for Step3.5's RMSNorm) so checkpoints stay compatible — only
    ``forward`` is replaced with the explicit fp32-norm / +1-gamma / fp32-multiply / cast-back
    sequence SteptronOss uses, so the two stacks produce identical qnorm/knorm outputs.

    Reference: SteptronOss ``rms_norm.py:55-62`` (``RMSNorm.forward``):
        weight = self.weight + self.bias   # bias == 1 when use_zero_init=True
        y = RMSNormFunction.apply(x.float()).type_as(x)
        return y * weight
    where ``RMSNormFunction.forward = x * rsqrt(x.pow(2).mean(-1) + 1e-5)``.

    The diff against stock TENorm/te.RMSNorm: TE's RMSNorm runs the multiply in input dtype (bf16),
    SteptronOss runs the entire normalization in fp32 and casts only the final ``y`` back.
    """

    def __init__(self, config, hidden_size: int, eps: float = 1e-5):
        super().__init__(
            normalized_shape=hidden_size,
            eps=eps,
            sequence_parallel=config.sequence_parallel,
            zero_centered_gamma=config.layernorm_zero_centered_gamma,
            **_get_extra_te_kwargs(config),
        )
        self.config = config
        self.eps_value = eps

    def forward(self, x):
        # Match SteptronOss ``RMSNorm.forward`` (rms_norm.py:55-62) bit-for-bit:
        #   y = RMSNormFunction.apply(x.float()).type_as(x)   # fp32 normalize → immediate cast back
        #   return y * (self.weight + self.bias)              # bf16 multiply (bias=1 when use_zero_init)
        # i.e. RMS normalize runs in fp32 but the `(γ+1) * y` multiply runs in bf16.
        in_dtype = x.dtype
        x_fp32 = x.float()
        l2_norm_inv = torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps_value)
        y = (x_fp32 * l2_norm_inv).to(in_dtype)

        weight = self.weight
        if self.config.layernorm_zero_centered_gamma:
            weight = weight + 1.0

        return y * weight


class TENorm_debug_mlp(TENorm_debug):
    """MoE-layer ``pre_mlp_layernorm`` variant of ``TENorm_debug``.

    Identical numerics — kept as a separate subclass purely for clarity so the
    layer's ``__repr__`` distinguishes the standalone RMSNorm before the MoE
    block (Megatron's equivalent of SteptronOss ``ffn_norm``) from the one
    used for ``q_layernorm`` / ``k_layernorm`` inside self-attention.

    Constructor signature follows ``TENorm`` (``config, hidden_size, eps``) so
    ``TransformerLayer.__init__`` can build it via ``submodules.pre_mlp_layernorm``
    without any changes.

    Used by ``_build_step35_layer_spec``: MoE layers carry a real RMSNorm at
    ``pre_mlp_layernorm`` while dense layers fold the LN into ``linear_fc1``
    (and therefore keep ``pre_mlp_layernorm = IdentityOp``). The swap only fires
    when the spec field is not IdentityOp.
    """

    def forward(self, x):
        in_dtype = x.dtype
        x_fp32 = x.float()
        l2_norm_inv = torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps_value)
        y = (x_fp32 * l2_norm_inv).to(in_dtype)

        weight = self.weight
        if self.config.layernorm_zero_centered_gamma:
            weight = weight + 1.0

        return y * weight, x


class TERowParallelLinear_debug(TERowParallelLinear):
    """Bit-for-bit reference of SteptronOss's ``wo`` (RowParallelLinear, bias=False).

    Inherits parameter init, TP sharding, and sharded_state_dict from TE so the checkpoint
    stays compatible with the production class — only ``forward`` is replaced with the
    explicit no-bias ``F.linear`` + TP all-reduce sequence used by SteptronOss
    ``RowParallelLinear`` so the two stacks produce identical ``linear_proj`` outputs down
    to the last bit.

    Reference: SteptronOss ``layers.py`` (``RowParallelLinear.forward``) +
    ``grouped_query_attention.py`` (``head_wise_attn_gate_function``):
        # input_is_parallel=True, bias=False, no sequence parallel.
        # head_wise_attn_gate is folded into the GEMM input via custom_pre_recompute_function.
        if gate is not None:
            attn_out = input.view(S, B, num_local_heads, head_dim)
            input    = (attn_out * gate.unsqueeze(-1).sigmoid()).view(S, B, -1)
        output_parallel = F.linear(input, self.weight)            # no bias add
        output          = reduce_from_tensor_model_parallel_region(output_parallel)
        return output, None

    Head-wise gate flow (A2 alignment):
        ``SelfAttention.forward`` checks ``self.linear_proj._apply_head_wise_gate_internally``;
        when True it skips the outer gate apply and instead assigns
        ``self.linear_proj._pending_head_wise_gate = head_wise_gate``. We read and clear that
        attribute on the next forward, then apply ``out * sigmoid(gate)`` exactly as the
        original attention.py block did (fp32 sigmoid → cast back to input dtype) — this
        keeps Megatron's gate semantics unchanged while moving the apply site *inside*
        linear_proj so the ``attention_preproj`` input hook captures the **pre-gate**
        tensor, matching SteptronOss's ``wo`` input hook bit-for-bit at the same
        semantic point.

    Notes:
        - Step3.5 sets ``bias=False`` on this layer, so we ignore ``self.bias`` entirely.
        - Sequence-parallel reduce-scatter is not implemented; the alignment runs use
          ``model.sequence_parallel=False``.
    """

    # Read by SelfAttention.forward: when True, attention.py skips its outer
    # head_wise_gate apply and forwards the gate tensor via attribute injection
    # so this class can apply it inside the forward — see class docstring.
    _apply_head_wise_gate_internally = True

    def forward(self, x):
        # Pop the gate injected by SelfAttention.forward (if any) and apply it
        # using bf16 sigmoid to match SteptronOss head_wise_attn_gate_function
        # bit-for-bit. (Megatron's original attention.py block runs sigmoid in
        # fp32 then casts back; SteptronOss runs sigmoid directly in the gate's
        # native dtype — typically bf16. We follow SteptronOss here.)
        gate = getattr(self, "_pending_head_wise_gate", None)
        if gate is not None:
            self._pending_head_wise_gate = None
            gate_states = gate.view(*gate.shape[:2], -1, 1)
            x = x.view(*gate_states.shape[:3], -1)
            x = x * gate_states.sigmoid()
            x = x.view(*gate_states.shape[:2], -1)

        output_parallel = torch.nn.functional.linear(x, self.weight)
        output = reduce_from_tensor_model_parallel_region(output_parallel, group=self._tp_group)
        return output, None


class TELayerNormColumnParallelLinear_debug_mlp(TELayerNormColumnParallelLinear_debug):
    """Dense-MLP variant of ``TELayerNormColumnParallelLinear_debug``.

    Dense MLP's ``linear_fc1`` call site is::
        intermediate_parallel, bias_parallel = self.linear_fc1(hidden_states)   # mlp.py:248
    which unpacks **2 elements**. The attention base class returns
    ``(out_hidden, bias, ln_output)`` to satisfy SelfAttention's
    ``mixed_qkv, _, _ = self.linear_qkv(...)`` (attention.py:1672), so here we
    strip the ``ln_output`` slot before returning. Numerical logic
    (fp32-norm → cast → bf16-multiply-γ → no-bias linear) is inherited
    unchanged from ``TELayerNormColumnParallelLinear_debug``.

    Used in ``_build_step35_layer_spec`` to swap dense layers' ``linear_fc1``
    so the ``layer_NNN_ffn_norm`` / ``ffn_input`` / downstream dumps match
    SteptronOss bit-for-bit.
    """

    def forward(self, x):
        out_hidden, ret_bias, _ = super().forward(x)
        return out_hidden, ret_bias


class TERowParallelLinear_debug_mlp(TERowParallelLinear):
    """Dense-MLP variant of ``TERowParallelLinear_debug`` — no head_wise_gate logic.

    Mirrors SteptronOss ``FeedForward.w2`` (RowParallelLinear, bias=False)::
        output_parallel = F.linear(x, self.weight)
        output          = reduce_from_tensor_model_parallel_region(output_parallel)
        return output, None

    Unlike the attention-side ``TERowParallelLinear_debug``, MLP's ``linear_fc2``
    never receives a head_wise_gate injection from upstream, so the gate path
    is removed entirely for clarity. Step3.5 sets ``bias=False``, so
    ``self.bias`` is ignored.
    """

    def forward(self, x):
        output_parallel = torch.nn.functional.linear(x, self.weight)
        output = reduce_from_tensor_model_parallel_region(output_parallel, group=self._tp_group)
        return output, None


@torch._dynamo.disable
def _maybe_save_sdpa_io(
    module,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask,
    is_causal: bool,
    dropout_p: float,
    out: torch.Tensor,
) -> None:
    """Dump SDPA inputs/output to ``MBRIDGE_SAVE_INTERMEDIATE_PATH`` for cross-framework diff.

    Files (per (layer, sdpa-call)):
        layer_NNN_attention_core_sdpa_callC_{q,k,v,output}.pt   tensors
        layer_NNN_attention_core_sdpa_callC_mask.pt             only when attn_mask is a Tensor
        layer_NNN_attention_core_sdpa_callC_meta.pt             dict with is_causal, dropout_p, shapes, dtypes

    Idempotent: existing files are not overwritten so backward recompute / multi-iter runs
    don't pollute the dump. ``module.layer_number`` is inherited from TEDotProductAttention
    (1-indexed); we save as ``(layer_number - 1)``.
    """
    # Fine-grained dump: gated by DUMP_FINEGRAIN so DUMP_BLOCK_IO-only runs
    # skip per-SDPA-call I/O (kept on by default for backwards compatibility).
    if os.environ.get("DUMP_FINEGRAIN", "1") != "1":
        return
    save_dir = os.environ.get("MBRIDGE_SAVE_INTERMEDIATE_PATH")
    if not save_dir:
        return
    # MBRIDGE_DUMP_PP_RANK0_ONLY=1 (default) limits the dump to PP rank 0 so
    # one PP stage can be aligned at a time. Set to 0 to dump on all PP ranks.
    if os.environ.get("MBRIDGE_DUMP_PP_RANK0_ONLY", "1") == "1":
        if parallel_state.get_pipeline_model_parallel_rank() != 0:
            return
    layer_number = getattr(module, "layer_number", None)
    if layer_number is None:
        return
    layer_id = int(layer_number) - 1

    call_idx = getattr(module, "_sdpa_call_counter", 0)
    module._sdpa_call_counter = call_idx + 1

    os.makedirs(save_dir, exist_ok=True)
    prefix = f"layer_{layer_id:03d}_attention_core_sdpa_call{call_idx}"

    def _save(tensor, name):
        path = os.path.join(save_dir, f"{prefix}_{name}.pt")
        if os.path.exists(path):
            print(
                f"[ALIGN] sdpa_io skip {prefix}_{name} (file already exists, expected with multi-rank): {path}",
                flush=True,
            )
            return
        torch.save(tensor.detach().cpu(), path)
        tf = tensor.detach().float()
        print(
            f"[ALIGN] sdpa_io saved {prefix}_{name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}",
            flush=True,
        )
        print(
            f"[ALIGN] sdpa_io stats {prefix}_{name}: min={tf.min():.6f}  max={tf.max():.6f}  mean={tf.mean():.6f}  std={tf.std():.6f}",
            flush=True,
        )
        print(f"[ALIGN] sdpa_io {prefix}_{name}: {tensor}", flush=True)

    _save(q, "q")
    _save(k, "k")
    _save(v, "v")
    if isinstance(attn_mask, torch.Tensor):
        _save(attn_mask, "mask")
    _save(out, "output")

    meta_path = os.path.join(save_dir, f"{prefix}_meta.pt")
    if not os.path.exists(meta_path):
        meta = {
            "is_causal": bool(is_causal),
            "dropout_p": float(dropout_p),
            "attn_mask_is_none": attn_mask is None,
            "attn_mask_dtype": str(attn_mask.dtype) if isinstance(attn_mask, torch.Tensor) else None,
            "attn_mask_shape": tuple(attn_mask.shape) if isinstance(attn_mask, torch.Tensor) else None,
            "q_shape": tuple(q.shape), "q_dtype": str(q.dtype),
            "k_shape": tuple(k.shape), "k_dtype": str(k.dtype),
            "v_shape": tuple(v.shape), "v_dtype": str(v.dtype),
            "out_shape": tuple(out.shape), "out_dtype": str(out.dtype),
        }
        torch.save(meta, meta_path)
        print(f"[ALIGN] sdpa_io meta {prefix}: {meta}", flush=True)


@torch._dynamo.disable
def _maybe_dump_moe_io(tensor, name: str) -> None:
    """Dump a MoE intermediate tensor to ``MBRIDGE_SAVE_INTERMEDIATE_PATH``.

    Used by ``MoELayer_debug.forward`` to expose router logits / routed combined
    output / shared expert output under the same ``layer_NNN_ffn_*`` names that
    SteptronOss schedules.py writes — so ``compare_intermediate_activations.py``
    can diff them 1:1.

    Idempotent (existing files skipped) and ``@torch._dynamo.disable`` so the
    file I/O doesn't trip dynamo trace.
    """
    if os.environ.get("DUMP_FINEGRAIN", "1") != "1":
        return
    save_dir = os.environ.get("MBRIDGE_SAVE_INTERMEDIATE_PATH")
    if not save_dir:
        return
    if os.environ.get("MBRIDGE_DUMP_PP_RANK0_ONLY", "1") == "1":
        if parallel_state.get_pipeline_model_parallel_rank() != 0:
            return
    if not isinstance(tensor, torch.Tensor):
        return
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{name}.pt")
    if os.path.exists(path):
        print(
            f"[ALIGN] moe dump skip {name} (file already exists, expected with multi-rank): {path}",
            flush=True,
        )
        return
    torch.save(tensor.detach().cpu(), path)
    print(
        f"[ALIGN] moe dump saved {name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}",
        flush=True,
    )
    if tensor.is_floating_point():
        tf = tensor.detach().float()
        print(
            f"[ALIGN] moe dump stats {name}: min={tf.min():.6f}  max={tf.max():.6f}  mean={tf.mean():.6f}  std={tf.std():.6f}",
            flush=True,
        )
    print(f"[ALIGN] moe dump {name}: {tensor}", flush=True)


class TEDotProductAttention_debug(TEDotProductAttention):
    """Bit-for-bit reference of SteptronOss ``AttentionCore.forward`` (PyTorch SDPA).

    Inherits TEDotProductAttention so layer construction (init args, CP/TP wiring,
    qk-clip stats) stays unchanged — only ``forward`` is replaced with the explicit
    SDPA sequence used by SteptronOss ``attention_core.py`` so the two stacks produce
    identical attention outputs.

    Reference (SteptronOss ``AttentionCore.forward``, attention_core.py:334-400):
      1. q,k,v come in BSHD; GQA-expand k/v along head dim via repeat_interleave;
      2. transpose to BHSD; call ``F.scaled_dot_product_attention`` with
         ``is_causal=self.causal and not use_mask`` (use_mask iff window>=0);
      3. transpose back to BSHD.

    Megatron passes q/k/v in sbhd (non-packed) or thd (packed) layout, so this
    wrapper handles the layout conversions before delegating to the same SDPA
    call. Returned tensor matches the original TEDotProductAttention output:
      - sbhd: [s, b, np_q * hn]
      - thd : [t, np_q * hn]

    Limitations (intentional, expand if needed):
      - ``attention_bias`` is not supported (Step3.5 doesn't use it).
      - ``num_splits`` is not supported (TE-specific kernel flag).
      - SWA (window_size > 0) builds an explicit boolean mask just like SteptronOss.
    """

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        attn_mask_type: AttnMaskType,
        attention_bias: Optional[torch.Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        num_splits: Optional[int] = None,
    ) -> torch.Tensor:
        assert attention_bias is None, "TEDotProductAttention_debug does not support attention_bias"
        assert num_splits is None, "TEDotProductAttention_debug does not support num_splits"

        # Reset per-forward sdpa call counter so dumps are idempotent across
        # forward/backward recompute and multi-iter runs.
        self._sdpa_call_counter = 0

        causal = attn_mask_type in (
            AttnMaskType.causal,
            AttnMaskType.padding_causal,
            AttnMaskType.causal_bottom_right,
        )

        # Per-layer SWA gating: ``self.config.window_size`` is a single global
        # value (e.g. [512, 0]) shared across all layers, but Step3.5 only wants
        # it on layers explicitly marked sliding. ``Step35DecoderLayer`` exposes
        # the resolved global 0-indexed layer_idx as ``self._layer_idx`` so we
        # can look up ``config.layer_types[layer_idx] == "sliding_attention"``.
        # If the layer is not SWA (or layer_idx is unavailable), force window=-1
        # so use_mask=False and SDPA falls back to the pure causal kernel.
        layer_idx = getattr(self, "_layer_idx", None)
        layer_types = getattr(self.config, "layer_types", None) or []
        is_swa_layer = (
            layer_idx is not None
            and 0 <= layer_idx < len(layer_types)
            and layer_types[layer_idx] == "sliding_attention"
        )

        if is_swa_layer:
            # SteptronOss sliding_window: -1 disables, >=0 enables (window on each side).
            # Megatron stores window as (left, right). Use the left window
            # (Step3.5 SWA is symmetric causal so right=0, left=W).
            window_size = getattr(self.config, "window_size", None)
            if isinstance(window_size, (list, tuple)) and len(window_size) >= 1:
                window = int(window_size[0])
            elif isinstance(window_size, int):
                window = int(window_size)
            else:
                window = -1
        else:
            window = -1
        use_mask = window >= 0

        if packed_seq_params is not None and getattr(packed_seq_params, "qkv_format", None) == "thd":
            return self._forward_thd(query, key, value, packed_seq_params, causal=causal, window=window, use_mask=use_mask)
        return self._forward_sbhd(query, key, value, causal=causal, window=window, use_mask=use_mask)

    @staticmethod
    def _expand_kv(k: torch.Tensor, v: torch.Tensor, np_q: int) -> tuple[torch.Tensor, torch.Tensor]:
        np_kv = k.shape[-2]
        if np_kv == np_q:
            return k, v
        assert np_q % np_kv == 0, f"np_q ({np_q}) must be divisible by np_kv ({np_kv})"
        repeat = np_q // np_kv
        return k.repeat_interleave(repeat, dim=-2), v.repeat_interleave(repeat, dim=-2)

    @staticmethod
    def _build_local_mask(q_len: int, k_len: int, window: int, causal: bool, device) -> torch.Tensor:
        # Match SteptronOss AttentionCore._build_local_mask bit-for-bit.
        q_idx = torch.arange(q_len, device=device).unsqueeze(1)
        k_idx = torch.arange(k_len, device=device).unsqueeze(0)
        if window < 0:
            allowed = k_idx <= q_idx if causal else torch.ones((q_len, k_len), dtype=torch.bool, device=device)
        else:
            if causal:
                allowed = (k_idx <= q_idx) & (k_idx >= (q_idx - window))
            else:
                allowed = (k_idx - q_idx).abs() <= window
        return allowed

    def _sdpa(self, q, k, v, *, is_causal: bool, attn_mask):
        # q/k/v: [b, h, s, d]
        dropout_p = self.config.attention_dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
        )
        _maybe_save_sdpa_io(self, q, k, v, attn_mask, is_causal, dropout_p, out)
        return out

    def _forward_sbhd(self, query, key, value, *, causal: bool, window: int, use_mask: bool):
        # query/key/value: [s, b, np, hn]
        s_q, b, np_q, hn = query.shape
        s_k = key.shape[0]

        # → [b, s, np, hn]
        q = query.transpose(0, 1)
        k = key.transpose(0, 1)
        v = value.transpose(0, 1)
        k, v = self._expand_kv(k, v, np_q)

        # → [b, h, s, d] for SDPA
        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)

        is_causal = causal and not use_mask
        attn_mask = None
        if use_mask:
            mask = self._build_local_mask(s_q, s_k, window, causal, device=q_t.device)
            attn_mask = mask.unsqueeze(0).unsqueeze(0)

        out = self._sdpa(q_t, k_t, v_t, is_causal=is_causal, attn_mask=attn_mask)
        # out: [b, h, s, d] → [s, b, h*d]
        out = out.transpose(1, 2).contiguous()  # [b, s, h, d]
        out = out.transpose(0, 1).contiguous()  # [s, b, h, d]
        return out.reshape(s_q, b, np_q * hn)

    def _forward_thd(self, query, key, value, packed_seq_params, *, causal: bool, window: int, use_mask: bool):
        # query/key/value: [t, np, hn]
        t_q, np_q, hn = query.shape
        cu_q = packed_seq_params.cu_seqlens_q
        cu_kv = packed_seq_params.cu_seqlens_kv
        if cu_q is None or cu_kv is None:
            raise RuntimeError("TEDotProductAttention_debug thd path requires cu_seqlens_q/cu_seqlens_kv")
        cu_q = cu_q.to(torch.int32)
        cu_kv = cu_kv.to(torch.int32)

        k, v = self._expand_kv(key, value, np_q)

        q_cu = cu_q.tolist()
        k_cu = cu_kv.tolist()
        outputs = []
        for b_idx in range(len(q_cu) - 1):
            q_start, q_end = q_cu[b_idx], q_cu[b_idx + 1]
            k_start, k_end = k_cu[b_idx], k_cu[b_idx + 1]
            q_seq = query[q_start:q_end].transpose(0, 1).unsqueeze(0)  # [1, h, q, d]
            k_seq = k[k_start:k_end].transpose(0, 1).unsqueeze(0)
            v_seq = v[k_start:k_end].transpose(0, 1).unsqueeze(0)

            is_causal = causal and not use_mask
            attn_mask = None
            if use_mask:
                mask = self._build_local_mask(q_seq.shape[-2], k_seq.shape[-2], window, causal, device=q_seq.device)
                attn_mask = mask.unsqueeze(0).unsqueeze(0)
            out = self._sdpa(q_seq, k_seq, v_seq, is_causal=is_causal, attn_mask=attn_mask)
            outputs.append(out.squeeze(0).transpose(0, 1))  # [q, h, d]

        out = torch.cat(outputs, dim=0)  # [t, h, d]
        return out.reshape(t_q, np_q * hn)


class MoELayer_debug(MoELayer):
    """SteptronOss-aligned MoE forward.

    Reuses ``MoELayer.__init__`` so router / experts / shared_experts are built
    with the production submodules (and the checkpoint loads unchanged). Only
    ``forward`` is replaced with a SteptronOss-style pipeline so the routed
    output matches ``MoeShareExpertFFN.forward`` (moe_share_expert_ffn.py:27) +
    ``MoEBlock.forward`` (moe_block.py:376) bit-for-bit.

    Pipeline (mirrors SteptronOss):
        x : [S, B, H]
        x = x.reshape(-1, H)                                # [T, H]
        logits = F.linear(x.float(), router.weight.float()) # [T, E] fp32
        gate_prob = sigmoid(logits)                          # Step3.5 enable_sigmoid_router=True
        sort + take top-K → topk_ids [T, K], topk_prob [T, K]
        token_weights = topk_prob / sum(topk_prob)           # norm_expert_weight=True
        # per-expert SwiGLU + weighted sum back to [T, H]
        for k in range(K):
            for e in range(E):
                mask = (topk_ids[:, k] == e)
                if not any: continue
                gate_up = F.linear(x[mask], w1[e])
                l, r   = gate_up.chunk(2, dim=-1)
                act    = silu(l) * r
                out_e  = F.linear(act, w2[e])
                routed[mask] += out_e * token_weights[mask, k]
        routed *= routed_scaling_factor                     # Step3.5: 3.0
        if use_shared_expert:
            shared, _ = shared_experts(hidden_states)        # Megatron's SharedExpertMLP
            output = routed + shared.reshape(-1, H)
        else:
            output = routed
        return output.reshape(S, B, H), None

    Limitations (good enough for the current 1-GPU alignment run):
      - EP = TP = ETP = 1 (no token-dispatch all-to-all; full token set on this rank).
      - O(K * E * T) naive loop; alignment runs once, performance is irrelevant.
      - aux-loss-free bias ignored (Step3.5 has ``router_bias_update_rate=0`` so
        the bias stays at zero; if a checkpoint with non-zero bias is ever
        loaded, this needs to be revisited).
      - Step3.5 constants (``_routed_scaling_factor``, ``_use_sigmoid_router``,
        ``_norm_expert_weight``) are class-level — adjust here if the recipe
        ever changes.
      - ``shared_experts`` is called as-is on the SBH tensor; its numerics must
        already match SteptronOss ``FeedForward`` (which is the case once dense
        MLP's linear_fc1/linear_fc2 are swapped to the ``_mlp`` debug variants).
    """

    _routed_scaling_factor: float = 3.0
    _use_sigmoid_router: bool = True
    _norm_expert_weight: bool = True

    def _extract_expert_weights(self):
        """Return ``(w1_list, w2_list)`` of length ``num_local_experts``.

        Handles both expert backends Step3.5 may use:
          - ``SequentialMLP``: ``experts.local_experts[i].linear_fc{1,2}.weight``
          - ``TEGroupedMLP`` / ``GroupedMLP``: ``experts.linear_fc{1,2}.weight{i}``
            (per-expert parameter names match ``StackedExpertAutoMapping``'s
            ``weight*`` wildcard, see step35_bridge.py top).
        """
        num = self.experts.num_local_experts
        if hasattr(self.experts, "local_experts"):
            w1s = [e.linear_fc1.weight for e in self.experts.local_experts]
            w2s = [e.linear_fc2.weight for e in self.experts.local_experts]
            return w1s, w2s
        fc1 = self.experts.linear_fc1
        fc2 = self.experts.linear_fc2
        if hasattr(fc1, "weight0"):
            w1s = [getattr(fc1, f"weight{i}") for i in range(num)]
            w2s = [getattr(fc2, f"weight{i}") for i in range(num)]
            return w1s, w2s
        raise RuntimeError(
            f"MoELayer_debug cannot extract per-expert weights from "
            f"experts={type(self.experts).__name__}; add a new branch in "
            f"_extract_expert_weights."
        )

    def forward(self, hidden_states, **kwargs):  # noqa: D401
        S, B, H = hidden_states.shape
        in_dtype = hidden_states.dtype
        x = hidden_states.reshape(-1, H)  # [T, H]
        T = x.shape[0]
        K = self.config.moe_router_topk
        E = self.config.num_moe_experts

        _layer_number = getattr(self, "layer_number", None)
        assert _layer_number is not None, (
            "MoELayer_debug.forward: self.layer_number is None. "
            "TransformerLayer must assign layer_number (1-indexed within PP stage) "
            "before MoELayer_debug runs — check that the MoE module is constructed "
            "as part of TransformerLayer, not in isolation."
        )
        _layer_id = int(_layer_number) - 1
        _prefix = f"layer_{_layer_id:03d}"

        # ----- Router (SteptronOss MoEGate + forward_router) ------------------
        router_logits = F.linear(x.float(), self.router.weight.float())  # [T, E]
        if _prefix is not None:
            _maybe_dump_moe_io(router_logits, f"{_prefix}_ffn_router_logits")

        if self._use_sigmoid_router:
            gate_prob = router_logits.sigmoid()
        else:
            gate_prob = router_logits.softmax(dim=-1)

        # aux-loss-free balance bias: sort key only — does NOT enter weights.
        expert_bias = getattr(self.router, "expert_bias", None)
        if isinstance(expert_bias, torch.Tensor):
            sort_key = gate_prob + expert_bias.to(gate_prob.dtype).unsqueeze(0)
        else:
            sort_key = gate_prob
        _sorted_key, _sorted_idx = sort_key.sort(dim=-1, descending=True, stable=True)
        topk_ids = _sorted_idx[:, :K].contiguous()       # [T, K]  int64
        topk_prob = gate_prob.gather(1, topk_ids)        # [T, K]  fp32

        token_weights = topk_prob
        _has_bias = isinstance(expert_bias, torch.Tensor)
        _eps = 1e-20 if _has_bias else 0.0
        if self._use_sigmoid_router:
            token_weights = token_weights / (token_weights.sum(dim=-1, keepdim=True) + _eps)
        elif self._norm_expert_weight:
            token_weights = token_weights / (topk_prob.sum(dim=-1, keepdim=True) + _eps)
        if _prefix is not None:
            _maybe_dump_moe_io(topk_ids, f"{_prefix}_ffn_router_topk_ids")
            _maybe_dump_moe_io(token_weights, f"{_prefix}_ffn_router_topk_weights")

        # ----- Routed experts (SteptronOss routed_grouped_ffn) -----------------
        from steptronoss.model.utils.moe_utils import (
            histogram as _st_histogram,
            index_compute as _st_index_compute,
            moe_scatter as _st_moe_scatter,
            grouped_gemm as _st_grouped_gemm,
            moe_weighted_gather as _st_moe_weighted_gather,
        )

        w1s, w2s = self._extract_expert_weights()
        w1_stacked = torch.stack(w1s, dim=0)  # [E, 2F, H]
        w2_stacked = torch.stack(w2s, dim=0)  # [E, H, F]

        token_expert_ids = topk_ids
        # Keep token_weights in fp32; MoEWeightedGather selects acc_dtype based on dtype.
        token_weights_in = token_weights

        if _prefix is not None:
            _maybe_dump_moe_io(x, f"{_prefix}_ffn_experts_x_input")
            _maybe_dump_moe_io(w1_stacked, f"{_prefix}_ffn_experts_w1")
            _maybe_dump_moe_io(w2_stacked, f"{_prefix}_ffn_experts_w2")
            _maybe_dump_moe_io(token_expert_ids, f"{_prefix}_ffn_experts_topk_ids_input")
            _maybe_dump_moe_io(token_weights_in, f"{_prefix}_ffn_experts_topk_weights_input")

        experts_histogram = _st_histogram(token_expert_ids, w1_stacked.shape[0])
        if _prefix is not None:
            _maybe_dump_moe_io(experts_histogram, f"{_prefix}_ffn_experts_histogram")

        if experts_histogram.numel() == 0 or int(experts_histogram.sum().item()) == 0:
            routed = x * token_weights_in.sum()
        else:
            batch_sizes = experts_histogram.long()
            scatter_index = _st_index_compute(token_expert_ids, experts_histogram)
            if _prefix is not None:
                _maybe_dump_moe_io(scatter_index, f"{_prefix}_ffn_experts_scatter_index")

            scattered = _st_moe_scatter(x, scatter_index)
            if _prefix is not None:
                _maybe_dump_moe_io(scattered, f"{_prefix}_ffn_experts_after_scatter")

            gemm1_out = _st_grouped_gemm(scattered, w1_stacked, batch_sizes=batch_sizes, trans_b=True)
            if _prefix is not None:
                _maybe_dump_moe_io(gemm1_out, f"{_prefix}_ffn_experts_after_gemm1")

            _routed_limit = _get_swiglu_limit(_layer_id, self.config.swiglu_limits)
            print(f"for debug, layer_number: {_layer_id}, in MoELayer_debug.forward, _routed_limit is {_routed_limit}")
            act_out = _swiglu_with_clip(gemm1_out, _routed_limit)
            if _prefix is not None:
                _maybe_dump_moe_io(act_out, f"{_prefix}_ffn_experts_after_act")

            gemm2_out = _st_grouped_gemm(act_out, w2_stacked, batch_sizes=batch_sizes, trans_b=True)
            if _prefix is not None:
                _maybe_dump_moe_io(gemm2_out, f"{_prefix}_ffn_experts_after_gemm2")

            routed = _st_moe_weighted_gather(gemm2_out, scatter_index, token_weights_in)

        if _prefix is not None:
            _maybe_dump_moe_io(routed, f"{_prefix}_ffn_experts_output")
            _maybe_dump_moe_io(routed, f"{_prefix}_ffn_expert_out")

        routed = routed * self._routed_scaling_factor
        print(f"for debug, layer_number: {_layer_id}, in MoELayer_debug.forward, self._routed_scaling_factor is {self._routed_scaling_factor}")

        if self.use_shared_expert and self.shared_experts is not None:
            if os.environ.get("MEGATRON_SWIGLU_WITH_CLIP_LIMITS_SHARED_EXPERT","0") == "1":
                _shared_limit = _get_swiglu_limit(_layer_id, self.config.swiglu_limits_shared)
                print(f"for debug, layer_number: {_layer_id}, in MoELayer_debug.forward, _shared_limit is {_shared_limit} from swiglu_limits_shared")
            else:
                _shared_limit = _get_swiglu_limit(_layer_id, self.config.swiglu_limits)
                print(f"for debug, layer_number: {_layer_id}, in MoELayer_debug.forward, _shared_limit is {_shared_limit} from swiglu_limits")
            if _shared_limit is not None:
                print(
                    f"[ALIGN][WARN] layer {_layer_id}: swiglu_limits_shared={_shared_limit} "
                    f"but MoELayer_debug currently routes the shared expert through Megatron's "
                    f"fused SharedExpertMLP without clip — numerical alignment with SteptronOss "
                    f"will diverge on this layer. Reimplement the shared-expert forward inline "
                    f"to apply _swiglu_with_clip before relying on this layer's dumps.",
                    flush=True,
                )
            shared_out = self.shared_experts(hidden_states)
            if _prefix is not None:
                _maybe_dump_moe_io(shared_out, f"{_prefix}_ffn_shared_out")
            output = routed.reshape(S, B, H) + shared_out
        else:
            output = routed.reshape(S, B, H)

        return output, None


def _format_spec(obj, indent=0):
    """Recursively format ModuleSpec / dataclasses / lists / dicts with one-line-per-element indenting."""
    import dataclasses

    pad = "  " * indent
    pad_inner = "  " * (indent + 1)

    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        cls_name = type(obj).__name__
        lines = [f"{cls_name}("]
        for f in dataclasses.fields(obj):
            v = getattr(obj, f.name)
            lines.append(f"{pad_inner}{f.name}={_format_spec(v, indent + 1)},")
        lines.append(f"{pad})")
        return "\n".join(lines)

    if isinstance(obj, (list, tuple)):
        if not obj:
            return "[]" if isinstance(obj, list) else "()"
        open_b, close_b = ("[", "]") if isinstance(obj, list) else ("(", ")")
        lines = [open_b]
        for i, v in enumerate(obj):
            lines.append(f"{pad_inner}[{i}] {_format_spec(v, indent + 1)},")
        lines.append(f"{pad}{close_b}")
        return "\n".join(lines)

    if isinstance(obj, dict):
        if not obj:
            return "{}"
        lines = ["{"]
        for k, v in obj.items():
            lines.append(f"{pad_inner}{k!r}: {_format_spec(v, indent + 1)},")
        lines.append(f"{pad}}}")
        return "\n".join(lines)

    return repr(obj)


def _build_step35_layer_spec(cfg, **kw):
    """Per-layer spec for Step3.5: dense for layers 0-2 and 45-47, MoE for 3-44.

    Also rewrites every main-decoder layer's ModuleSpec to use
    ``Step35DecoderLayer`` instead of the default ``TransformerLayer``. The
    custom layer reads ``cfg.layer_types`` at init time to determine whether
    the layer is a sliding-attention layer.

    Returns a TransformerBlockSubmodules whose layer_specs list is wrapped in
    _MTPDenseLayerSpecsList so that get_gpt_mtp_block_spec_for_backend receives
    a dense ModuleSpec (via layer_specs[-1]) for the MTP transformer sub-layers.
    """
    block_submodules = get_gpt_decoder_block_spec(cfg, use_transformer_engine=True, normalization="RMSNorm", **kw)
    # Swap the layer module class on every main-decoder spec. The dense MTP
    # spec below is used for MTP layers (which have their own 1-indexed
    # layer_number namespace) so the routed-expert FFN stays disabled even
    # when the last main decoder layer is MoE.
    for spec in block_submodules.layer_specs:
        spec.module = Step35DecoderLayer
        # Re-bind the shared-expert builder on MoE layers so the shared expert
        # honors ``activation_func_clamp_value_shared_expert``. Dense layers
        # have a plain MLP submodule (no ``shared_experts`` attribute) and are
        # skipped by the ``getattr`` guard.
        mlp_submodules = getattr(spec.submodules.mlp, "submodules", None)
        shared = getattr(mlp_submodules, "shared_experts", None)
        if shared is not None:
            mlp_submodules.shared_experts = partial(Step35SharedExpertMLP, **shared.keywords)
    dense_mtp_spec = get_gpt_layer_with_transformer_engine_spec(
        num_experts=None,
        moe_grouped_gemm=False,
        qk_layernorm=cfg.qk_layernorm,
    )
    dense_mtp_spec.module = Step35DecoderLayer
    block_submodules.layer_specs = _MTPDenseLayerSpecsList(block_submodules.layer_specs, dense_mtp_spec)

    if os.environ.get("USE_DEBUG_SUBMODULE", "0") == "1":
        for _spec in list(block_submodules.layer_specs) + [dense_mtp_spec]:
            _spec.submodules.self_attention.submodules.linear_qkv = TELayerNormColumnParallelLinear_debug
            _spec.submodules.self_attention.submodules.q_layernorm = TENorm_debug
            _spec.submodules.self_attention.submodules.k_layernorm = TENorm_debug
            _spec.submodules.self_attention.submodules.core_attention = TEDotProductAttention_debug
            _spec.submodules.self_attention.submodules.linear_proj = TERowParallelLinear_debug

            # MoE layers carry a standalone RMSNorm at pre_mlp_layernorm (Megatron's
            # equivalent of SteptronOss's ffn_norm). Dense layers keep it as
            # IdentityOp because the LN is fused into linear_fc1
            # (TELayerNormColumnParallelLinear_debug_mlp). Replace the MoE-side
            # RMSNorm with TENorm_debug_mlp so layer_NNN_ffn_norm matches
            # SteptronOss bit-for-bit (fp32 normalize → cast bf16 → bf16 (γ+1)*y).
            if _spec.submodules.pre_mlp_layernorm is not IdentityOp:
                _spec.submodules.pre_mlp_layernorm = TENorm_debug_mlp

            # Dense MLP swap (Step3.5 layers 0/1/2 + dense MTP spec). For dense layers
            # the RMSNorm is fused into linear_fc1 (TELayerNormColumnParallelLinear,
            # pre_mlp_layernorm is IdentityOp), so the layernorm-side alignment is
            # done by replacing linear_fc1 with its _mlp debug variant. linear_fc2
            # gets the gate-free row-parallel variant. MoE layer specs carry no
            # ``linear_fc1`` field (they use router/experts/shared_experts), so the
            # guard below silently skips them.
            mlp_subs = getattr(_spec.submodules.mlp, "submodules", None)
            if mlp_subs is not None and getattr(mlp_subs, "linear_fc1", None) is not None:
                mlp_subs.linear_fc1 = TELayerNormColumnParallelLinear_debug_mlp
                mlp_subs.linear_fc2 = TERowParallelLinear_debug_mlp

            # MoE layer module swap (Step3.5 layers 3/4/5). MoELayer_debug subclasses
            # MoELayer so __init__ still builds router/experts/shared_experts from
            # the production submodules (no checkpoint changes), but forward is
            # rewritten to mirror SteptronOss MoeShareExpertFFN+MoEBlock bit-for-bit.
            # Identified by ``mlp.module is MoELayer`` (dense layers and the dense
            # MTP spec use MLP and so are skipped).
            if getattr(_spec.submodules.mlp, "module", None) is MoELayer:
                _spec.submodules.mlp.module = MoELayer_debug

    print(f"for debug, rank: {torch.distributed.get_rank()}, block_submodules:")
    print(_format_spec(block_submodules, indent=1))

    return block_submodules


# ``source`` and ``model_type`` keep the legacy ``Step3p5ForCausalLM`` /
# ``"step3p5"`` spelling because those are the HF identifiers carried by
# ``stepfun-ai/Step-3.5-Flash``'s config.json (``architectures[0]`` and
# ``model_type``). The bridge registry looks the model up by exact string
# match on these, so they must stay in sync with HF — only the Python class
# name (``Step35Bridge``) follows the new ``Step35`` spelling.
@MegatronModelBridge.register_bridge(
    source="Step3p5ForCausalLM",
    target=GPTModel,
    provider=Step35ModelProvider,
    model_type="step3p5",
)
class Step35Bridge(MegatronModelBridge):
    """
    Megatron Bridge for Step3.5 Causal LM.

    This bridge handles the conversion between HuggingFace Step3p5ForCausalLM
    (the HF architecture name; preserved verbatim to match the upstream
    config.json) and Megatron-Core GPTModel formats. Step3.5 models use
    mixture of experts architecture with QK layernorm.

    Example:
        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained("stepfun-ai/Step-3.5-Flash")
        >>> provider = bridge.to_megatron_provider()
    """

    CONFIG_MAPPING = MegatronModelBridge.CONFIG_MAPPING + [
        ("num_attention_groups", "num_query_groups"),
        ("moe_num_experts", "num_moe_experts"),
        ("moe_top_k", "moe_router_topk"),
        ("share_expert_dim", "moe_shared_expert_intermediate_size"),
        ("share_expert_dims", "moe_shared_expert_intermediate_size"),
        ("use_head_wise_attn_gate", "head_wise_attn_gate"),
        ("attention_output_gate", "attention_output_gate"),
        ("layer_types", "layer_types"),
    ]

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> GPTModelProvider:
        """Convert HuggingFace Step3.5 config to GPTModelProvider."""
        provider = super().provider_bridge(hf_pretrained)

        hf_config = hf_pretrained.config

        provider.rotary_percents = hf_config.partial_rotary_factors
        # initialize the sliding_attention_setting with default values
        provider.sliding_attention_setting = {
            "window_size": [512, 0],
            "num_attention_heads": 96,
            "num_query_groups": 8,
            "kv_channels": 128,
        }
        # update the sliding_attention_setting with the values from the hf_config
        if hf_config.sliding_window is not None:
            provider.sliding_attention_setting["window_size"] = [hf_config.sliding_window, 0]
        if (
            hf_config.attention_other_setting
            and hf_config.attention_other_setting.get("attention_type", None) == "sliding_attention"
        ):
            provider.sliding_attention_setting["num_attention_heads"] = hf_config.attention_other_setting[
                "num_attention_heads"
            ]
            provider.sliding_attention_setting["num_query_groups"] = hf_config.attention_other_setting[
                "num_attention_groups"
            ]
            provider.sliding_attention_setting["kv_channels"] = hf_config.attention_other_setting["head_dim"]

        rope_theta = hf_config.rope_theta
        if isinstance(rope_theta, list):
            provider.rotary_base = rope_theta[0]  # for main model
            provider.rotary_base_per_layer = rope_theta  # for each transformer layer
        else:
            provider.rotary_base = rope_theta

        provider.normalization = "RMSNorm"
        provider.layernorm_zero_centered_gamma = hf_config.zero_centered
        provider.gated_linear_unit = True
        provider.add_bias_linear = False
        provider.add_qkv_bias = False
        provider.hidden_dropout = 0.0
        provider.attention_dropout = 0.0
        provider.qk_layernorm = hf_config.use_qk_norm
        if isinstance(hf_config.torch_dtype, str):
            if hf_config.torch_dtype == "bfloat16":
                provider.autocast_dtype = torch.bfloat16
            elif hf_config.torch_dtype == "float16":
                provider.autocast_dtype = torch.float16
            elif hf_config.torch_dtype == "float32":
                provider.autocast_dtype = torch.float32
            else:
                raise ValueError(f"Unknown torch dtype: {hf_config.torch_dtype}")
        elif isinstance(hf_config.torch_dtype, torch.dtype):
            provider.autocast_dtype = hf_config.torch_dtype
        else:
            raise ValueError(f"Unknown torch dtype: {hf_config.torch_dtype}")

        provider.moe_router_enable_expert_bias = hf_config.use_moe_router_bias
        provider.moe_router_score_function = hf_config.moe_router_activation
        provider.moe_router_topk_scaling_factor = hf_config.moe_router_scaling_factor
        provider.swiglu_limits = hf_config.swiglu_limits
        provider.swiglu_limits_shared = hf_config.swiglu_limits_shared
        if hf_config.need_fp32_gate:
            provider.moe_router_dtype = "fp32"

        provider.moe_grouped_gemm = True
        provider.moe_router_load_balancing_type = "aux_loss"
        provider.moe_aux_loss_coeff = 1e-3
        provider.moe_router_pre_softmax = False
        provider.moe_token_dispatcher_type = "alltoall"
        provider.moe_permute_fusion = True

        moe_layers_enum = getattr(hf_config, "moe_layers_enum", None)
        if moe_layers_enum is not None:
            moe_layer_freq = [0] * provider.num_layers
            if isinstance(moe_layers_enum, str):
                moe_layers = [int(layer) for layer in moe_layers_enum.split(",") if layer]
            else:
                moe_layers = [int(layer) for layer in moe_layers_enum]
            for idx in moe_layers:
                if idx < provider.num_layers:
                    moe_layer_freq[idx] = 1
            provider.moe_layer_freq = moe_layer_freq
            # _build_step35_layer_spec reads moe_layer_freq to produce per-layer dense/MoE
            # specs for the main decoder, and wraps layer_specs with _MTPDenseLayerSpecsList
            # so that get_gpt_mtp_block_spec_for_backend picks up a dense spec for MTP layers
            # (45-47 are not in moe_layers_enum).
            provider.transformer_layer_spec = _build_step35_layer_spec

        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        # Dictionary maps Megatron parameter names -> HF parameter names.
        # Supports wildcard (*) patterns for layer-specific parameters.
        param_mappings = {
            # Embedding and output
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
            "output_layer.weight": "lm_head.weight",
            "decoder.final_layernorm.weight": "model.norm.weight",
            # Pre-attention layernorm (standalone for MoE layers; fused into linear_qkv for dense layers)
            "decoder.layers.*.input_layernorm.weight": "model.layers.*.input_layernorm.weight",
            # Fused pre-attention layernorm weights (TELayerNormColumnParallelLinear).
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.layers.*.input_layernorm.weight",
            # Layernorm for q, k
            "decoder.layers.*.self_attention.q_layernorm.weight": "model.layers.*.self_attn.q_norm.weight",
            "decoder.layers.*.self_attention.k_layernorm.weight": "model.layers.*.self_attn.k_norm.weight",
            # Attention o projection
            "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
            # Pre-MLP layernorm (standalone for dense layers; fused into linear_fc1 for dense layers)
            "decoder.layers.*.pre_mlp_layernorm.weight": "model.layers.*.post_attention_layernorm.weight",
            "decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "model.layers.*.post_attention_layernorm.weight",
            # Dense MLP fc2 (layers 0–2)
            "decoder.layers.*.mlp.linear_fc2.weight": "model.layers.*.mlp.down_proj.weight",
            # Shared expert fc2 (runs alongside routed experts on MoE layers)
            "decoder.layers.*.mlp.shared_experts.linear_fc2.weight": "model.layers.*.share_expert.down_proj.weight",
            # MoE router
            "decoder.layers.*.mlp.router.weight": "model.layers.*.moe.gate.weight",
            # MoE router bias
            "decoder.layers.*.mlp.router.expert_bias": "model.layers.*.moe.router_bias",
        }

        mapping_list = []
        for megatron_param, hf_param in param_mappings.items():
            mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

        mapping_list.extend(
            [
                # QKV + per-head gate: merge Q, K, V (GQA-interleaved) and expand
                # the scalar g_proj rows into MCore's attention_output_gate layout.
                QKVGMapping(
                    megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
                    q="model.layers.*.self_attn.q_proj.weight",
                    k="model.layers.*.self_attn.k_proj.weight",
                    v="model.layers.*.self_attn.v_proj.weight",
                    g="model.layers.*.self_attn.g_proj.weight",
                ),
                # Dense MLP fc1 (gate+up concatenated; layers 0–2 and MTP layers 45–47)
                GatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.linear_fc1.weight",
                    gate="model.layers.*.mlp.gate_proj.weight",
                    up="model.layers.*.mlp.up_proj.weight",
                ),
                # MoE per-expert fc1: Megatron creates weight0…weightN; HF stores stacked [N, I, H].
                StackedExpertGatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.experts.linear_fc1.weight*",
                    gate="model.layers.*.moe.gate_proj.weight",
                    up="model.layers.*.moe.up_proj.weight",
                ),
                # Shared expert fc1 (gate+up concatenated). MCore names the shared
                # expert ``mlp.shared_experts`` (plural) — matches DeepSeek / GLM /
                # Sarvam bridges and is what TransformerLayerSubmodules expects.
                GatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.shared_experts.linear_fc1.weight",
                    gate="model.layers.*.share_expert.gate_proj.weight",
                    up="model.layers.*.share_expert.up_proj.weight",
                ),
                StackedExpertAutoMapping(
                    megatron_param="decoder.layers.*.mlp.experts.linear_fc2.weight*",
                    hf_param="model.layers.*.moe.down_proj.weight",
                ),
            ]
        )

        # MTP layer mappings (layers 45–47 in Step-3.5-Flash)
        if self.hf_config is None:
            logger.warning("No HF config found, skipping MTP mappings.")
            return MegatronMappingRegistry(*mapping_list)

        mtp_num_layers = getattr(self.hf_config, "num_nextn_predict_layers", 0)
        num_transformer_layers = self.hf_config.num_hidden_layers

        # Layer-specific param patterns to replicate for each MTP transformer sub-layer.
        # Step3.5 MTP layers are always dense (no MoE), so only dense-MLP and attention params.
        # g_proj weight/layernorm are merged into linear_qkv via QKVGMapping
        # below (parallels the main decoder mapping table above).
        mtp_layer_param_mappings = {
            "decoder.layers.*.input_layernorm.weight": "model.layers.*.input_layernorm.weight",
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.layers.*.input_layernorm.weight",
            "decoder.layers.*.pre_mlp_layernorm.weight": "model.layers.*.post_attention_layernorm.weight",
            "decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "model.layers.*.post_attention_layernorm.weight",
            "decoder.layers.*.self_attention.q_layernorm.weight": "model.layers.*.self_attn.q_norm.weight",
            "decoder.layers.*.self_attention.k_layernorm.weight": "model.layers.*.self_attn.k_norm.weight",
            "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
            "decoder.layers.*.mlp.linear_fc2.weight": "model.layers.*.mlp.down_proj.weight",
        }

        for mtp_layer in range(mtp_num_layers):
            hf_layer = mtp_layer + num_transformer_layers
            # Megatron may name the sub-layer "mtp_model_layer" or "transformer_layer".
            for layer_prefix in ("mtp_model_layer", "transformer_layer"):
                for megatron_param, hf_param in mtp_layer_param_mappings.items():
                    megatron_param_mtp = (
                        megatron_param.replace(".*", f".*.{layer_prefix}")
                        .replace("decoder", "mtp")
                        .replace(".*", f".{mtp_layer}")
                    )
                    hf_param_mtp = hf_param.replace("layers.*", f"layers.{hf_layer}")
                    mapping_list.append(AutoMapping(megatron_param=megatron_param_mtp, hf_param=hf_param_mtp))

                mapping_list.extend(
                    [
                        QKVGMapping(
                            megatron_param=f"mtp.layers.{mtp_layer}.{layer_prefix}.self_attention.linear_qkv.weight",
                            q=f"model.layers.{hf_layer}.self_attn.q_proj.weight",
                            k=f"model.layers.{hf_layer}.self_attn.k_proj.weight",
                            v=f"model.layers.{hf_layer}.self_attn.v_proj.weight",
                            g=f"model.layers.{hf_layer}.self_attn.g_proj.weight",
                        ),
                        GatedMLPMapping(
                            megatron_param=f"mtp.layers.{mtp_layer}.{layer_prefix}.mlp.linear_fc1.weight",
                            gate=f"model.layers.{hf_layer}.mlp.gate_proj.weight",
                            up=f"model.layers.{hf_layer}.mlp.up_proj.weight",
                        ),
                        AutoMapping(
                            megatron_param=f"mtp.layers.{mtp_layer}.{layer_prefix}.mlp.linear_fc2.weight",
                            hf_param=f"model.layers.{hf_layer}.mlp.down_proj.weight",
                        ),
                    ]
                )

            # MTP-specific normalization and projection layers
            mapping_list.extend(
                [
                    AutoMapping(
                        megatron_param=f"mtp.layers.{mtp_layer}.enorm.weight",
                        hf_param=f"model.layers.{hf_layer}.enorm.weight",
                    ),
                    AutoMapping(
                        megatron_param=f"mtp.layers.{mtp_layer}.hnorm.weight",
                        hf_param=f"model.layers.{hf_layer}.hnorm.weight",
                    ),
                    AutoMapping(
                        megatron_param=f"mtp.layers.{mtp_layer}.eh_proj.weight",
                        hf_param=f"model.layers.{hf_layer}.eh_proj.weight",
                    ),
                    # In Megatron, mtp use specific transformer.shared_head.norm different from main model,
                    # and share same transformer.shared_head.output.weight with main model
                    AutoMapping(
                        megatron_param=f"mtp.layers.{mtp_layer}.final_layernorm.weight",
                        hf_param=f"model.layers.{hf_layer}.transformer.shared_head.norm.weight",
                    ),
                ]
            )

        return MegatronMappingRegistry(*mapping_list)
