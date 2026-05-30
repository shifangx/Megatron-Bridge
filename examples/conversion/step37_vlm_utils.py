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

"""Step3.7 inference adapter for ``hf_to_megatron_generate_vlm.py``.

The generic ``vlm_forward_step`` feeds HF-style VLM keys (``pixel_values`` …)
straight to the model, but ``Step37Model.forward`` speaks a different dialect:
``images: list[ImageForInsert]`` + ``packed_seq_params``. This module is the
missing translation layer. It mirrors the training-side contract implemented
in ``models/stepfun/step37_flickr8k_step.py`` + ``data/.../multimodal_utils.py``,
reusing the exact same building blocks so the inference path matches training.

Flow:
  1. ``process_step37_inputs``  — run ``Step3VLProcessor`` on (image, prompt),
     keeping ALL keys Step3.7 needs (``pixel_values`` / ``patch_pixel_values`` /
     ``input_ids`` with ``<im_start>`` / ``<patch_start>`` + ``<im_patch>`` spans).
  2. ``build_step37_images``    — raw pixels → ``list[ImageForInsert]`` (with
     per-image RoPE cu_seqlens) via ``build_image_for_insert``.
  3. ``build_step37_packed_seq_params`` — single-sequence ``cu_seqlens`` →
     ``PackedSeqParams`` for varlen FlashAttn (tail padding = its own sub-seq).
  4. ``Step37BatchIterator`` + ``step37_vlm_forward_step`` — feed
     ``Step37Model.forward`` natively.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import torch
from megatron.core.packed_seq_params import PackedSeqParams

from megatron.bridge.data.vlm_datasets.step37_flickr8k.multimodal_utils import (
    IMAGE_ITEM_TYPE,
    PATCH_ITEM_TYPE,
    build_image_for_insert,
    compute_rope_args,
)

# Vision encoder patch size (PE-G/14). Matches the flickr8k dataset config
# (``encoder_patch_size=14``) used by ``compute_rope_args``.
ENCODER_PATCH_SIZE = 14
IM_START_TOKEN = "<im_start>"
PATCH_START_TOKEN = "<patch_start>"


def process_step37_inputs(processor, image_path: Optional[str], prompt: str):
    """Run ``Step3VLProcessor`` on a single (image, prompt) pair.

    Mirrors ``transformers_step37.py``: builds a chat message, applies the
    chat template with ``tokenize=True`` and returns the full ``BatchFeature``
    (``input_ids`` / ``attention_mask`` / ``pixel_values`` / ``num_patches``
    and, when the image is large enough to be tiled, ``patch_pixel_values`` /
    ``patch_newline_mask``).

    Unlike the generic ``process_image_inputs`` (which only keeps the Qwen-style
    keys and drops ``num_patches`` / ``patch_pixel_values``), this keeps every
    key ``Step37Model`` needs.
    """
    content: list[dict[str, Any]] = []
    if image_path:
        content.append({"type": "image", "url": image_path})
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    return inputs


def build_step37_images(inputs, tokenizer, *, dtype: torch.dtype = torch.bfloat16) -> list:
    """Translate processor pixel tensors into ``list[ImageForInsert]``.

    ``pixel_values`` carries the whole-image tiles (728×728), inserted at the
    ``<im_start>`` placeholder; ``patch_pixel_values`` carries the sub-patches
    (504×504), inserted at ``<patch_start>``. Each image's per-patch RoPE
    cu_seqlens are attached via ``compute_rope_args`` (PE-G/14).

    Returns an empty list for text-only inputs (no ``pixel_values``).
    """
    pixel_values = inputs.get("pixel_values")
    if pixel_values is None:
        return []
    patch_pixel_values = inputs.get("patch_pixel_values")

    im_start_id = int(tokenizer.convert_tokens_to_ids(IM_START_TOKEN))
    patch_start_id = int(tokenizer.convert_tokens_to_ids(PATCH_START_TOKEN))

    # ``build_image_for_insert`` wants individual ``[3, H, W]`` tensors tagged
    # by item type; it groups + stacks them and builds one ImageForInsert per
    # type (patches first, then whole images), each scatter-inserted at its own
    # ``insert_start_token``.
    items: list[tuple[torch.Tensor, int]] = [(img, IMAGE_ITEM_TYPE) for img in pixel_values]
    if patch_pixel_values is not None and patch_pixel_values.shape[0] > 0:
        items += [(p, PATCH_ITEM_TYPE) for p in patch_pixel_values]

    return build_image_for_insert(
        items,
        patch_start_id=patch_start_id,
        image_start_id=im_start_id,
        rope_args_fn=lambda imgs: compute_rope_args(list(imgs), ENCODER_PATCH_SIZE),
        dtype=dtype,
        to_cuda=True,
    )


def build_step37_packed_seq_params(
    real_len: int, padded_len: int, device: torch.device
) -> PackedSeqParams:
    """Build ``PackedSeqParams`` for one (unpacked) prompt sequence.

    The prompt is a single sub-sequence ``[0, real_len)``; any tail padding
    added for TE/FP8 alignment becomes its own sub-sequence ``[real_len,
    padded_len)`` so it never attends to / from the real tokens.
    """
    if padded_len > real_len:
        cu_seqlens = torch.tensor([0, real_len, padded_len], dtype=torch.int32, device=device)
    else:
        cu_seqlens = torch.tensor([0, padded_len], dtype=torch.int32, device=device)
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    max_seqlen = int(seqlens.max().item())
    return PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens,
        max_seqlen_q=max_seqlen,
        cu_seqlens_kv=cu_seqlens,
        max_seqlen_kv=max_seqlen,
        cu_seqlens_q_padded=cu_seqlens,
        cu_seqlens_kv_padded=cu_seqlens,
    )


def pad_step37_input_ids(generated_ids: torch.Tensor, tp_size: int, pad_token_id: int):
    """Pad ``input_ids`` to a ``lcm(tp_size, 16)`` multiple (TE/FP8 friendly).

    Returns ``(input_ids, real_len, padded_len)``. Mirrors the padding in
    ``step37_flickr8k_step.forward_step`` (:119-133).
    """
    real_len = generated_ids.size(1)
    divisible_by = math.lcm(tp_size, 16)
    padded_len = math.ceil(real_len / divisible_by) * divisible_by
    if padded_len > real_len:
        input_ids = torch.nn.functional.pad(generated_ids, (0, padded_len - real_len), value=pad_token_id)
    else:
        input_ids = generated_ids
    return input_ids, real_len, padded_len


class Step37BatchIterator:
    """Single-batch iterator carrying Step3.7 forward kwargs."""

    def __init__(self, input_ids, images, packed_seq_params, attention_mask):
        self.batch = dict(
            input_ids=input_ids,
            images=images,
            packed_seq_params=packed_seq_params,
            attention_mask=attention_mask,
        )
        self._yielded = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._yielded:
            raise StopIteration
        self._yielded = True
        return self.batch


def step37_vlm_forward_step(data_iterator, model, **kwargs):
    """Forward step for Step3.7 — builds native ``Step37Model.forward`` kwargs.

    ``position_ids=None`` lets the decoder's per-layer RoPE derive positions
    from ``packed_seq_params``; ``images`` flows straight through (the vision
    tower runs inside ``Step37Model._encode_images_for_insert``).
    """
    batch = next(data_iterator)
    forward_args = {
        "input_ids": batch["input_ids"],
        "position_ids": None,
        "attention_mask": batch["attention_mask"],
        "images": batch["images"],
        "packed_seq_params": batch["packed_seq_params"],
    }

    def loss_func(x, **kw):
        return x

    model_output = model(**forward_args)
    if isinstance(model_output, tuple):
        output_tensor, _ = model_output
    else:
        output_tensor = model_output
    return output_tensor, loss_func


__all__ = [
    "process_step37_inputs",
    "build_step37_images",
    "build_step37_packed_seq_params",
    "pad_step37_input_ids",
    "Step37BatchIterator",
    "step37_vlm_forward_step",
]
