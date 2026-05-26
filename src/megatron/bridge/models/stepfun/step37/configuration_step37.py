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

"""HuggingFace ``PretrainedConfig`` for Step-3.7 (a.k.a. step3p7 / step37).

Step-3.7 is a vision-language model that wraps the Step-3.5-Flash language
backbone (see :mod:`megatron.bridge.models.stepfun.configuration_step35`) with
a Perception-Encoder (PE-G/14) vision tower and a single linear projector that
maps vision-encoder outputs into the LLM embedding space.

This config object is intentionally minimal: it carries only the fields the
Megatron-Bridge ``Step37Bridge`` reads to populate ``Step37ModelProvider``. The
canonical source-of-truth is the HF checkpoint under
``stepfun-ai/step3p7`` — values here are defaults for from-scratch runs.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from transformers import AutoConfig, PretrainedConfig

from megatron.bridge.models.stepfun.configuration_step35 import Step35Config


class Step37VisionConfig(PretrainedConfig):
    """Perception-Encoder (PE-G/14) vision tower config for Step-3.7."""

    model_type = "perception_encoder"

    def __init__(
        self,
        image_size: int = 728,
        patch_size: int = 14,
        hidden_size: int = 1536,
        num_hidden_layers: int = 47,
        num_attention_heads: int = 16,
        intermediate_size: int = 8960,
        output_dim: int = 6144,
        in_channels: int = 3,
        use_rope2d: bool = True,
        rope_theta: float = 1e4,
        hidden_act: str = "quickgelu",
        use_cls_token: bool = False,
        patch_embed_bias: bool = False,
        layer_norm_eps: float = 1e-5,
        attention_dropout: float = 0.0,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.output_dim = output_dim
        self.in_channels = in_channels
        self.use_rope2d = use_rope2d
        self.rope_theta = rope_theta
        self.hidden_act = hidden_act
        self.use_cls_token = use_cls_token
        self.patch_embed_bias = patch_embed_bias
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout


class Step37Config(PretrainedConfig):
    """HuggingFace config for Step-3.7 vision-language model.

    Composed of ``text_config`` (a :class:`Step35Config`) and ``vision_config``
    (a :class:`Step37VisionConfig`). Special token ids and image token counts
    sit at the top level — they describe how vision features are spliced into
    the LLM token sequence.
    """

    model_type = "step3p7"
    is_composition = True

    def __init__(
        self,
        text_config: Optional[Dict[str, Any]] = None,
        vision_config: Optional[Dict[str, Any]] = None,
        # Special tokens
        image_token_id: int = 128001,
        im_start_token_id: int = 128000,
        im_end_token_id: int = 128002,
        patch_start_token_id: int = 128003,
        patch_end_token_id: int = 128005,
        # Image token layout
        image_token_count: int = 169,
        patch_token_count: int = 81,
        # Projector
        projector_input_dim: int = 6144,
        projector_bias: bool = False,
        understand_projector_stride: int = 2,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        if text_config is None:
            self.text_config = Step35Config()
        elif isinstance(text_config, dict):
            self.text_config = Step35Config(**text_config)
        else:
            self.text_config = text_config

        if vision_config is None:
            self.vision_config = Step37VisionConfig()
        elif isinstance(vision_config, dict):
            self.vision_config = Step37VisionConfig(**vision_config)
        else:
            self.vision_config = vision_config

        self.image_token_id = image_token_id
        self.im_start_token_id = im_start_token_id
        self.im_end_token_id = im_end_token_id
        self.patch_start_token_id = patch_start_token_id
        self.patch_end_token_id = patch_end_token_id
        self.image_token_count = image_token_count
        self.patch_token_count = patch_token_count
        self.projector_input_dim = projector_input_dim
        self.projector_bias = projector_bias
        self.understand_projector_stride = understand_projector_stride


AutoConfig.register("step3p7", Step37Config, exist_ok=True)
