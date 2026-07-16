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

from types import SimpleNamespace

import pytest
import torch

from megatron.bridge.models.stepfun.modelling_step37.image_insert_embedding import ImageForInsert
from megatron.bridge.models.stepfun.step37_mock_step import _accumulate_flops_metadata


pytestmark = pytest.mark.unit


def test_accumulate_flops_metadata_tracks_text_and_pre_downsample_patches():
    """Each mock microbatch contributes padded text lengths and raw ViT patches."""
    state = SimpleNamespace(
        cfg=SimpleNamespace(
            model=SimpleNamespace(vision_config=SimpleNamespace(patch_size=14)),
        )
    )
    tokens = torch.zeros((2, 16), dtype=torch.long)
    images = [ImageForInsert(insert_start_token=1, images=torch.zeros((4, 3, 28, 42)))]

    _accumulate_flops_metadata(state, tokens, images)
    _accumulate_flops_metadata(state, tokens, images)

    assert state._flops_seqlen_sum == 2 * (2 * 16)
    assert state._flops_seqlen_sq_sum == 2 * (2 * 16**2)
    assert state._flops_vision_patches == 2 * (4 * 2 * 3)


def test_accumulate_flops_metadata_handles_text_only_batch():
    """A text-only mock batch still initializes the vision accumulator to zero."""
    state = SimpleNamespace(
        cfg=SimpleNamespace(
            model=SimpleNamespace(vision_config=SimpleNamespace(patch_size=14)),
        )
    )

    _accumulate_flops_metadata(state, torch.zeros((1, 8), dtype=torch.long), [])

    assert state._flops_seqlen_sum == 8
    assert state._flops_seqlen_sq_sum == 64
    assert state._flops_vision_patches == 0
