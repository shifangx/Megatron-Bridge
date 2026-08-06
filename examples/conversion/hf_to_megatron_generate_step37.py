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

"""Step-3.7 (stepfun) multimodal greedy generation from a Megatron checkpoint.

Unlike the Qwen-family ``hf_to_megatron_generate_vlm.py``, Step37Model.forward
consumes ``images: list[ImageForInsert]`` (NOT pixel_values/image_grid_thw).
This driver:

  1. Builds inputs with the SAME HF ``Step3VLProcessor`` + chat template that
     vLLM uses (so the token layout + image preprocessing match the vLLM
     reference dump under logs/dump_step3p7_vllm_serve/).
  2. Wraps the preprocessed pixels into an ``ImageForInsert`` via the model's
     own ``build_image_for_insert`` + ``compute_rope_args`` helpers.
  3. Registers forward hooks on every decoder layer to dump the per-layer
     residual-stream output as ``llm_layer{idx}`` (matching the vLLM tags).
  4. Runs greedy (temperature 0) generation.

Intermediate-activation dumps are controlled by the STEP3P7_DUMP_* env vars
(see modelling_step37/_dump.py).

Example:
  python -m torch.distributed.run --nproc_per_node=4 \
    examples/conversion/hf_to_megatron_generate_step37.py \
    --hf_model_path /path/Step-3.7-Flash \
    --megatron_model_path /path/Step-3.7-Flash_megatron_ckpt \
    --image_path /path/cats.jpg --prompt "What is in this picture?" \
    --tp 1 --pp 1 --ep 4 --max_new_tokens 100 --trust_remote_code
"""

import argparse
from io import BytesIO

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, GenerationConfig

from megatron.bridge import AutoBridge
from megatron.bridge.models.hf_pretrained.utils import is_safe_repo
from megatron.bridge.models.stepfun.data.flickr8k.multimodal_utils import (
    IMAGE_ITEM_TYPE,
    build_image_for_insert,
    compute_rope_args,
)
from megatron.bridge.models.stepfun.modelling_step37._dump import _max_llm_calls, dump_tensor
from megatron.bridge.utils.common_utils import (
    get_last_rank,
    maybe_initialize_distributed,
    print_rank_0,
    print_rank_last,
)

ENCODER_PATCH_SIZE = 14  # PE-G/14


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------
def _load_image(path: str) -> Image.Image:
    if path.startswith(("http://", "https://")):
        import requests

        return Image.open(BytesIO(requests.get(path, timeout=60).content)).convert("RGB")
    return Image.open(path).convert("RGB")


class SingleBatchIterator:
    """Yields one batch then stops (contract of get_forward_backward_func)."""

    def __init__(self, input_ids, position_ids, images):
        self.batch = dict(tokens=input_ids, position_ids=position_ids, images=images)
        self._yielded = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._yielded:
            raise StopIteration
        self._yielded = True
        return self.batch


def step37_forward_step(data_iterator, model, **kwargs):
    """Forward step for Step-3.7 generation: passes list[ImageForInsert]."""
    batch = next(data_iterator)
    forward_args = {
        "input_ids": batch["tokens"],
        "position_ids": batch["position_ids"],
        "attention_mask": None,
    }
    if batch.get("images"):
        forward_args["images"] = batch["images"]

    def loss_func(x, **kw):
        return x

    out = model(**forward_args)
    if isinstance(out, tuple):
        out = out[0]
    return out, loss_func


def register_llm_dump_hooks(model_chunks):
    """Register forward hooks that dump each decoder layer's residual-stream
    output as ``llm_layer{idx}`` (0-based), matching the vLLM reference."""
    handles = []
    for m in model_chunks:
        inner = m.module if hasattr(m, "module") else m
        lang = getattr(inner, "language_model", inner)
        decoder = getattr(lang, "decoder", None)
        if decoder is None or not hasattr(decoder, "layers"):
            continue
        for pos, layer in enumerate(decoder.layers):
            ln = getattr(layer, "layer_number", None)
            idx = (ln - 1) if isinstance(ln, int) else pos

            def make_hook(layer_idx):
                def hook(module, inputs, output):
                    h = output[0] if isinstance(output, tuple) else output
                    dump_tensor(
                        f"llm_layer{layer_idx}",
                        h,
                        enable_env="STEP3P7_DUMP_LLM",
                        max_calls=_max_llm_calls(),
                    )

                return hook

            handles.append(layer.register_forward_hook(make_hook(idx)))
    print_rank_0(f"[step37-gen] registered {len(handles)} LLM-layer dump hooks")
    return handles


def _hf_revision_kwargs(revision):
    return {"revision": revision} if revision is not None else {}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(args) -> None:
    maybe_initialize_distributed()
    tp, pp, ep, etp = args.tp, args.pp, args.ep, args.etp

    trust_remote = is_safe_repo(trust_remote_code=args.trust_remote_code, hf_path=args.hf_model_path)

    # ------------------------------------------------------------------ model
    bridge = AutoBridge.from_hf_pretrained(
        args.hf_model_path, trust_remote_code=trust_remote, **_hf_revision_kwargs(args.hf_revision)
    )
    print_rank_0(f"Loading Megatron model from: {args.megatron_model_path}")
    provider = bridge.to_megatron_provider(load_weights=False)
    provider.tensor_model_parallel_size = tp
    provider.pipeline_model_parallel_size = pp
    provider.expert_model_parallel_size = ep
    provider.expert_tensor_parallel_size = etp
    provider.pipeline_dtype = torch.bfloat16
    provider.init_model_with_meta_device = True
    provider.finalize()
    provider.initialize_model_parallel(seed=0)

    mp_overrides = {
        "tensor_model_parallel_size": tp,
        "pipeline_model_parallel_size": pp,
        "expert_model_parallel_size": ep,
        "expert_tensor_parallel_size": etp,
        "pipeline_dtype": torch.bfloat16,
    }
    model = bridge.load_megatron_model(
        args.megatron_model_path, mp_overrides=mp_overrides, wrap_with_ddp=False
    )

    def _disable_mtp(m):
        m.config.mtp_num_layers = None
        inner = m.module if hasattr(m, "module") else m
        lang = getattr(inner, "language_model", inner)
        if hasattr(lang, "mtp_process"):
            lang.mtp_process = False

    model = [m.cuda() for m in model]
    for m in model:
        m.eval()
        _disable_mtp(m)
        if hasattr(m, "config"):
            m.config.grad_scale_func = None

    register_llm_dump_hooks(model)

    # -------------------------------------------------------------- tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.hf_model_path, trust_remote_code=trust_remote, **_hf_revision_kwargs(args.hf_revision)
    )
    processor = AutoProcessor.from_pretrained(
        args.hf_model_path, trust_remote_code=trust_remote, **_hf_revision_kwargs(args.hf_revision)
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    im_start_id = tokenizer.convert_tokens_to_ids("<im_start>")
    patch_start_id = tokenizer.convert_tokens_to_ids("<patch_start>")

    # -------------------------------------------------------------- inputs
    # Build the prompt with the SAME chat template + processor vLLM uses.
    content = [{"type": "text", "text": args.prompt}]
    image_for_insert = None
    if args.image_path:
        content.append({"type": "image"})
    messages = [{"role": "user", "content": content}]
    prompt_str = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    if args.image_path:
        pil_image = _load_image(args.image_path)
        proc = processor(text=[prompt_str], images=[pil_image], return_tensors="pt")
        pixel_values = proc["pixel_values"]  # [1, 3, 728, 728]
        image_for_insert = build_image_for_insert(
            [(pixel_values[0], IMAGE_ITEM_TYPE)],
            patch_start_id=patch_start_id,
            image_start_id=im_start_id,
            rope_args_fn=lambda tensors: compute_rope_args(tensors, ENCODER_PATCH_SIZE),
            dtype=torch.bfloat16,
            to_cuda=True,
        )
    else:
        proc = processor(text=[prompt_str], return_tensors="pt")

    input_ids_raw = proc["input_ids"].cuda()
    # The chat template emits {{bos_token}} AND the tokenizer has
    # add_bos_token=true, so the processor yields a duplicated leading BOS
    # (190 tokens vs vLLM's 189). Strip the duplicate so the token layout —
    # and therefore the per-layer hidden states — line up with the vLLM dump.
    bos_id = tokenizer.bos_token_id
    if (
        bos_id is not None
        and input_ids_raw.size(1) >= 2
        and int(input_ids_raw[0, 0]) == bos_id
        and int(input_ids_raw[0, 1]) == bos_id
    ):
        input_ids_raw = input_ids_raw[:, 1:]
    prompt_length = input_ids_raw.size(1)
    print_rank_0(
        f"[step37-gen] prompt_length={prompt_length} "
        f"(#<im_patch>={(input_ids_raw == tokenizer.convert_tokens_to_ids('<im_patch>')).sum().item()})"
    )

    # -------------------------------------------------------------- generation
    try:
        gen_cfg = GenerationConfig.from_pretrained(args.hf_model_path, **_hf_revision_kwargs(args.hf_revision))
        stop_ids = gen_cfg.eos_token_id
    except OSError:
        stop_ids = tokenizer.eos_token_id
    if stop_ids is None:
        stop_ids = [tokenizer.eos_token_id]
    elif isinstance(stop_ids, int):
        stop_ids = [stop_ids]
    stop_tokens = set(stop_ids)

    generated_ids = input_ids_raw.clone()
    fwd_bwd_function = get_forward_backward_func()

    for step in range(args.max_new_tokens):
        with torch.no_grad():
            print_rank_0(f"Generation step {step}")
            real_seq_len = generated_ids.size(1)
            input_ids = generated_ids
            position_ids = (
                torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device)
                .unsqueeze(0)
                .expand_as(input_ids)
            )
            # Vision features are recomputed each step; the dump helper gates to
            # call0 (the prefill), so only the first forward writes vit_* / llm_*.
            iterator = SingleBatchIterator(input_ids, position_ids, image_for_insert)
            output = fwd_bwd_function(
                forward_step_func=step37_forward_step,
                data_iterator=iterator,
                model=model,
                num_microbatches=1,
                forward_only=True,
                seq_length=input_ids.size(1),
                micro_batch_size=1,
                collect_non_loss_data=True,
            )
            if isinstance(output, list) and len(output) > 0:
                output = output[0]

            if parallel_state.is_pipeline_last_stage():
                world_size = parallel_state.get_tensor_model_parallel_world_size()
                if world_size > 1:
                    gathered = [torch.zeros_like(output) for _ in range(world_size)]
                    dist.all_gather(gathered, output, group=parallel_state.get_tensor_model_parallel_group())
                    output = torch.cat(gathered, dim=2)
                last_pos = real_seq_len - 1
                next_token_ids = torch.argmax(output[:, last_pos], dim=-1, keepdim=True)
                if step < 5:
                    logits = output[0, last_pos, :]
                    top5_vals, top5_ids = torch.topk(logits, 5)
                    top5 = [tokenizer.decode([i]) for i in top5_ids]
                    print_rank_last(f"Step {step}: top5={list(zip(top5, top5_vals.tolist()))}")
                    print_rank_last(
                        f"Selected: '{tokenizer.decode([next_token_ids.item()])}' (id={next_token_ids.item()})"
                    )
            else:
                next_token_ids = torch.ones((1, 1), device=generated_ids.device, dtype=generated_ids.dtype)

            torch.distributed.broadcast(next_token_ids, get_last_rank())
            generated_ids = torch.cat([generated_ids, next_token_ids], dim=-1)
            if next_token_ids.item() in stop_tokens:
                break

    completion = tokenizer.decode(generated_ids[0, prompt_length:].tolist(), skip_special_tokens=True)
    print_rank_0("======== STEP-3.7 GENERATED TEXT ========")
    if args.image_path:
        print_rank_0(f"Image: {args.image_path}")
    print_rank_0(f"Prompt: {args.prompt}")
    print_rank_0(f"Completion: {completion}")
    print_rank_0("=========================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step-3.7 VLM generation from a Megatron checkpoint")
    parser.add_argument("--hf_model_path", type=str, required=True)
    parser.add_argument("--hf-revision", dest="hf_revision", default=None)
    parser.add_argument("--megatron_model_path", type=str, required=True)
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="What is in this picture?")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--ep", type=int, default=4)
    parser.add_argument("--etp", type=int, default=1)
    parser.add_argument("--trust_remote_code", action="store_true")
    args = parser.parse_args()
    main(args)

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
