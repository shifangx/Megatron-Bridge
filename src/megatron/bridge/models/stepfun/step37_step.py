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

"""Forward-step function for Step-3.7 vision-language SFT training.

Modeled directly on ``megatron.bridge.models.qwen_vl.qwen3_vl_step.forward_step``.
The only differences are:

* Image batch keys: Step-3.7 uses fixed-resolution 728-square images (no dynamic
  grid). The batch may still carry ``pixel_values`` and ``image_grid_thw`` for
  compatibility with the qwen3-vl data path, but ``image_grid_thw`` is optional.
* Loss masking: identical to qwen3-vl (use the dataset-provided ``loss_mask``).
"""

from __future__ import annotations

import logging
import math
from functools import partial
from typing import Any, Iterable

import torch
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.pipeline_parallel.utils import is_pp_first_stage, is_pp_last_stage
from megatron.core.utils import get_batch_on_this_cp_rank, get_model_config

from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.losses import (
    create_masked_next_token_loss_function as _create_loss_function,
)
from megatron.bridge.training.state import GlobalState
from megatron.bridge.training.utils.padding_utils import (
    pad_or_truncate_2d_to_len,
    pad_or_truncate_attn_to_len,
    pad_or_truncate_pos_to_len,
)
from megatron.bridge.training.utils.pg_utils import get_pg_collection


logger = logging.getLogger(__name__)


def _get_batch_from_iterator(
    data_iterator: Iterable,
    skip_getting_attention_mask_from_dataset: bool = True,
    *,
    is_first_pp_stage: bool,
    is_last_pp_stage: bool,
) -> dict[str, Any]:
    """Pull one batch, hoist multimodal tensors onto CUDA, drop unused keys."""
    batch = next(data_iterator)

    required_device_keys: set[str] = set()
    required_host_keys: set[str] = set()

    if not skip_getting_attention_mask_from_dataset:
        required_device_keys.add("attention_mask")

    required_device_keys.add("visual_inputs")

    if "cu_seqlens" in batch:
        required_device_keys.add("cu_seqlens")
        required_host_keys.add("cu_seqlens_argmin")
        required_host_keys.add("max_seqlen")

    required_device_keys.update(("tokens", "input_ids", "position_ids"))
    if is_last_pp_stage:
        required_device_keys.update(("labels", "loss_mask"))

    out: dict[str, Any] = {}
    for key, val in batch.items():
        if key in required_device_keys:
            if key == "visual_inputs":
                if val is None:
                    out[key] = None
                else:
                    out[key] = val
                    for k, v in val.__dict__.items():
                        out[key].__dict__[k] = v.cuda(non_blocking=True) if v is not None else None
            else:
                out[key] = val.cuda(non_blocking=True) if val is not None else None
        elif key in required_host_keys:
            out[key] = val.cpu() if val is not None else None
        else:
            out[key] = None
    return out


def _get_batch(
    data_iterator: Iterable,
    cfg: ConfigContainer,
    *,
    is_first_pp_stage: bool,
    is_last_pp_stage: bool,
):
    batch = _get_batch_from_iterator(
        data_iterator,
        getattr(cfg.dataset, "skip_getting_attention_mask_from_dataset", True),
        is_first_pp_stage=is_first_pp_stage,
        is_last_pp_stage=is_last_pp_stage,
    )

    if "visual_inputs" in batch and batch.get("visual_inputs") is not None:
        multi_modal_inputs = batch.get("visual_inputs").normalized_for_model()
    else:
        multi_modal_inputs = {}

    return (
        batch.get("tokens") if batch.get("tokens") is not None else batch.get("input_ids"),
        batch.get("labels"),
        batch.get("loss_mask"),
        batch.get("attention_mask"),
        batch.get("position_ids"),
        multi_modal_inputs,
    )


def _pack_or_pad(
    tokens, labels, loss_mask, attention_mask, position_ids,
    pg_collection, *, use_fp8_padding: bool, force_to_pad_to_seq_len: bool, seq_length: int,
):
    batch_size, cur_len = tokens.shape
    device = tokens.device

    tp_size = pg_collection.tp.size()
    cp_size = pg_collection.cp.size()
    divisible_by = tp_size * cp_size * 2 if cp_size > 1 else tp_size
    divisible_by = math.lcm(divisible_by, 16) if use_fp8_padding else divisible_by

    target_len = math.ceil(cur_len / divisible_by) * divisible_by
    if force_to_pad_to_seq_len:
        target_len = seq_length

    tokens = pad_or_truncate_2d_to_len(tokens, target_len=target_len, max_cap=target_len, pad_value=0)
    labels = pad_or_truncate_2d_to_len(labels, target_len=target_len, max_cap=target_len, pad_value=-100)
    loss_mask = pad_or_truncate_2d_to_len(loss_mask, target_len=target_len, max_cap=target_len, pad_value=0)
    attention_mask = pad_or_truncate_attn_to_len(attention_mask, target_len=target_len, max_cap=target_len)
    position_ids = pad_or_truncate_pos_to_len(position_ids, target_len=target_len, max_cap=target_len)

    seqlens = torch.ones(batch_size, dtype=torch.int32, device=device) * target_len
    cu_seqlens = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    cu_seqlens[1:] = torch.cumsum(seqlens, dim=0)

    packed_seq_params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens,
        max_seqlen_q=seqlens.max().item(),
        cu_seqlens_kv=cu_seqlens,
        max_seqlen_kv=seqlens.max().item(),
        cu_seqlens_q_padded=cu_seqlens,
        cu_seqlens_kv_padded=cu_seqlens,
    )
    return tokens, labels, loss_mask, attention_mask, position_ids, packed_seq_params


def forward_step(
    state: GlobalState,
    data_iterator: Iterable,
    model,
    return_schedule_plan: bool = False,
) -> tuple[torch.Tensor, partial]:
    """Step-3.7 forward training step.

    Mirrors :func:`megatron.bridge.models.qwen_vl.qwen3_vl_step.forward_step`
    but produces a step37 forward call (image tokens spliced into the LLM
    embedding inside :class:`Step37Model.forward`).
    """
    timers = state.timers
    straggler_timer = state.straggler_timer

    pg_collection = get_pg_collection(model)
    is_first = is_pp_first_stage(pg_collection.pp)
    is_last = is_pp_last_stage(pg_collection.pp)

    config = get_model_config(model)

    timers("batch-generator", log_level=2).start()
    with straggler_timer(bdata=True):
        (
            tokens,
            labels,
            loss_mask,
            attention_mask,
            position_ids,
            multi_modal_inputs,
        ) = _get_batch(data_iterator, state.cfg, is_first_pp_stage=is_first, is_last_pp_stage=is_last)
    timers("batch-generator").stop()

    pack_sequences_in_batch = getattr(state.cfg.dataset, "pack_sequences_in_batch", False)

    tokens, labels, loss_mask, attention_mask, position_ids, packed_seq_params = _pack_or_pad(
        tokens, labels, loss_mask, attention_mask, position_ids,
        pg_collection,
        use_fp8_padding=True,
        force_to_pad_to_seq_len=pg_collection.pp.size() > 1 or pg_collection.ep.size() > 1,
        seq_length=config.seq_length,
    )

    forward_args = {
        "input_ids": tokens,
        "labels": labels,
        "loss_mask": loss_mask,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }
    original_tokens = tokens.clone()
    forward_args = get_batch_on_this_cp_rank(
        forward_args, is_hybrid_cp=False, cp_group=pg_collection.cp
    )
    forward_args["packed_seq_params"] = None
    forward_args["input_ids"] = original_tokens
    forward_args["position_ids"] = None

    if pack_sequences_in_batch:
        if forward_args["labels"] is not None:
            forward_args["labels"] = forward_args["labels"].reshape(1, -1)
        attention_mask_packed = torch.ones(
            original_tokens.shape[0], original_tokens.shape[1],
            dtype=torch.bool, device=original_tokens.device,
        )
        forward_args["attention_mask"] = attention_mask_packed
        if forward_args["loss_mask"] is not None:
            forward_args["loss_mask"] = forward_args["loss_mask"].reshape(1, -1)
        forward_args["packed_seq_params"] = packed_seq_params

    loss_mask = forward_args["loss_mask"]
    for k in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"):
        if k in multi_modal_inputs:
            forward_args[k] = multi_modal_inputs[k]

    check_for_nan = state.cfg.rerun_state_machine.check_for_nan_in_loss
    check_for_spike = state.cfg.rerun_state_machine.check_for_spiky_loss

    with straggler_timer:
        if return_schedule_plan:
            assert config.overlap_moe_expert_parallel_comm, (
                "overlap_moe_expert_parallel_comm must be enabled to return the schedule plan"
            )
            schedule_plan = model.build_schedule_plan(
                tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask
            )
            return schedule_plan, _create_loss_function(loss_mask, check_for_nan, check_for_spike)
        output_tensor = model(**forward_args)

    return output_tensor, _create_loss_function(loss_mask, check_for_nan, check_for_spike)
