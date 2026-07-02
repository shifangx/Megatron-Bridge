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

"""Step3.7 **BSHD** forward step — consumes the unpacked ``[B, S]`` batch
produced by :class:`Step37MockSFTDataProvider`.

This is the BSHD counterpart to ``step37_flickr8k_step`` (which drives the
packed *THD* pipeline). Here there is no ``cu_seqlens`` and no
``PackedSeqParams``: tokens are a plain ``[B, S]`` tensor and the attention
backend applies ordinary causal masking (``attention_mask=None``).
``list[ImageForInsert]`` is passed straight through to
``Step37Model.forward``.

Responsibilities:
  1. ``next(data_iterator)`` → BSHD dict (``input_ids`` / ``labels`` /
     ``loss_mask`` / ``images``).
  2. Move tensors + image pixels to CUDA (PP rank 0 carries the images).
  3. Call ``model(**forward_args)`` with ``packed_seq_params=None``.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Iterable

import torch
from megatron.core.models.gpt import GPTModel
from megatron.core.pipeline_parallel.utils import is_pp_first_stage, is_pp_last_stage

from megatron.bridge.training.losses import (
    create_masked_next_token_loss_function as _create_loss_function,
)
from megatron.bridge.training.state import GlobalState
from megatron.bridge.training.utils.pg_utils import get_pg_collection


def _images_to_cuda(images: list) -> list:
    """Move each ``ImageForInsert``'s raw pixels to CUDA in place."""
    for insert_image in images:
        if getattr(insert_image, "images", None) is not None:
            insert_image.images = insert_image.images.to("cuda", non_blocking=True)
    return images


def forward_step(
    state: GlobalState,
    data_iterator: Iterable,
    model: GPTModel,
    return_schedule_plan: bool = False,
) -> tuple[torch.Tensor, partial]:
    """BSHD forward step for the Step3.7 mock pipeline."""
    timers = state.timers
    straggler_timer = state.straggler_timer

    this_pg_collection = get_pg_collection(model)
    is_first = is_pp_first_stage(this_pg_collection.pp)
    is_last = is_pp_last_stage(this_pg_collection.pp)

    cfg = state.cfg

    timers("batch-generator", log_level=2).start()
    with straggler_timer(bdata=True):
        batch = next(data_iterator)
    timers("batch-generator").stop()

    forward_args: dict[str, Any] = {}
    loss_mask: torch.Tensor | None = None

    # PP rank 0 owns the embedding + vision fusion, so it needs the tokens and
    # images. Other stages run on the piped hidden states only.
    if is_first and "input_ids" in batch:
        tokens = batch["input_ids"].to("cuda")
        forward_args["input_ids"] = tokens
        forward_args["images"] = _images_to_cuda(batch.get("images") or [])
        forward_args["position_ids"] = None
        # None → the attention backend applies standard causal masking (BSHD).
        forward_args["attention_mask"] = None
        forward_args["packed_seq_params"] = None
    else:
        forward_args["attention_mask"] = None
        forward_args["packed_seq_params"] = None

    if is_last and "labels" in batch:
        forward_args["labels"] = batch["labels"].to("cuda")
        loss_mask = batch["loss_mask"].to("cuda")
        forward_args["loss_mask"] = loss_mask

    check_for_nan_in_loss = cfg.rerun_state_machine.check_for_nan_in_loss
    check_for_spiky_loss = cfg.rerun_state_machine.check_for_spiky_loss
    with straggler_timer:
        if return_schedule_plan:
            assert model.config.overlap_moe_expert_parallel_comm, (
                "overlap_moe_expert_parallel_comm must be enabled to return the schedule plan"
            )
            schedule_plan = model.build_schedule_plan(
                forward_args.get("input_ids"),
                None,
                forward_args.get("attention_mask"),
                labels=forward_args.get("labels"),
                loss_mask=forward_args.get("loss_mask"),
            )
            loss_function = _create_loss_function(loss_mask, check_for_nan_in_loss, check_for_spiky_loss)
            return schedule_plan, loss_function
        output_tensor = model(**forward_args)

    loss_function = _create_loss_function(loss_mask, check_for_nan_in_loss, check_for_spiky_loss)
    return output_tensor, loss_function


__all__ = ["forward_step"]
