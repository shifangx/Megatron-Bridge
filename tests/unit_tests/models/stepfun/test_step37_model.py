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

"""Unit tests for the Step3.7 pipeline-stage forward contract."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from megatron.bridge.models.stepfun.modelling_step37.model import Step37Model


_MODEL_MODULE = "megatron.bridge.models.stepfun.modelling_step37.model"


def test_forward_accepts_missing_input_ids_on_non_first_pipeline_stage():
    """Intermediate PP stages consume the activation installed by set_input_tensor."""
    expected_output = object()
    language_model = Mock(return_value=expected_output)
    model = SimpleNamespace(pre_process=False, language_model=language_model)

    with patch(f"{_MODEL_MODULE}.nvtx_range_push"), patch(f"{_MODEL_MODULE}.nvtx_range_pop"):
        output = Step37Model.forward(model)

    assert output is expected_output
    assert language_model.call_args.kwargs["input_ids"] is None
    assert language_model.call_args.kwargs["decoder_input"] is None


def test_forward_requires_input_ids_on_first_pipeline_stage():
    """The embedding-owning PP stage must still receive token ids."""
    model = SimpleNamespace(pre_process=True)

    with pytest.raises(ValueError, match="input_ids is required on the first pipeline stage"):
        Step37Model.forward(model)
