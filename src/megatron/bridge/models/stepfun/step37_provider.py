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

"""Megatron-Bridge model provider for Step-3.7 (step3p7 / step37).

The LLM tower is identical to Step-3.5-Flash so we inherit from
:class:`Step35ModelProvider` and add vision-config fields plus an overridden
``provide`` that wraps the produced GPT model with the vision tower, projector,
and image-token splicing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from megatron.bridge.models.stepfun.step35_provider import Step35ModelProvider
from megatron.bridge.models.stepfun.step37.projector import Step37Projector
from megatron.bridge.models.stepfun.step37.step37_model import Step37Model
from megatron.bridge.models.stepfun.step37.vision_model import Step37VisionTransformer


@dataclass
class Step37VisionProviderConfig:
    """Vision-tower hyperparameters (PE-G/14).

    Mirrors :class:`Step37VisionConfig` but stored as plain dataclass fields so
    Megatron-Bridge CLI overrides (``model.vision_config.hidden_size=...``) can
    reach them.
    """

    hidden_size: int = 1536
    num_hidden_layers: int = 47
    num_attention_heads: int = 16
    intermediate_size: int = 8960
    output_dim: int = 6144
    image_size: int = 728
    patch_size: int = 14
    in_channels: int = 3
    use_rope2d: bool = True
    rope_theta: float = 1e4
    hidden_act: str = "quickgelu"
    use_cls_token: bool = False
    patch_embed_bias: bool = False
    layer_norm_eps: float = 1e-5
    attention_dropout: float = 0.0


@dataclass
class Step37ModelProvider(Step35ModelProvider):
    """Provider for Step-3.7 vision-language model.

    Inherits every Step-3.5-Flash field (``layer_types``,
    ``sliding_attention_setting``, ``rotary_base_per_layer``, etc.) and adds
    vision-tower + projector + image-token-id fields.
    """

    vision_config: Step37VisionProviderConfig = field(default_factory=Step37VisionProviderConfig)
    projector_input_dim: int = 6144
    projector_bias: bool = False
    understand_projector_stride: int = 1
    image_token_id: int = 128001
    im_start_token_id: int = 128000
    image_token_count: int = 169

    freeze_language_model: bool = False
    freeze_vision_model: bool = True
    freeze_vision_projection: bool = False

    # Overridden so step37 recipes can pass ``vision_config={...}`` from the CLI
    # via a plain dict; the dataclass coercion happens in ``__post_init__``.
    def __post_init__(self) -> None:  # type: ignore[override]
        if isinstance(self.vision_config, dict):
            self.vision_config = Step37VisionProviderConfig(**self.vision_config)
        super_post_init = getattr(super(), "__post_init__", None)
        if callable(super_post_init):
            super_post_init()

    def provide(self, pre_process=None, post_process=None, vp_stage=None) -> Step37Model:
        """Build the Step37 multimodal model.

        Steps:
        1. Delegate to ``Step35ModelProvider.provide`` to construct the LLM
           backbone (GPTModel with hybrid full/sliding attention + MoE).
        2. Build the PE-G/14 vision tower and the linear projector.
        3. Wrap everything in :class:`Step37Model`.
        4. Apply freeze flags.
        """
        language_model = super().provide(
            pre_process=pre_process, post_process=post_process, vp_stage=vp_stage
        )

        # Only the pipeline stage that hosts the embedding gets a vision tower;
        # other stages only forward intermediate activations.
        is_first_pp_stage = pre_process is None or bool(pre_process)
        vision_tower = Step37VisionTransformer(self.vision_config) if is_first_pp_stage else None
        projector = (
            Step37Projector(
                input_dim=self.projector_input_dim,
                output_dim=self.hidden_size,
                bias=self.projector_bias,
                understand_projector_stride=self.understand_projector_stride,
            )
            if is_first_pp_stage
            else None
        )

        model = Step37Model(
            language_model=language_model,
            vision_model=vision_tower,
            projector=projector,
            config=self,
            image_token_id=self.image_token_id,
            im_start_token_id=self.im_start_token_id,
            image_token_count=self.image_token_count,
            pre_process=is_first_pp_stage,
            post_process=post_process if post_process is not None else True,
            add_encoder=is_first_pp_stage,
            add_decoder=True,
        )

        if self.freeze_language_model or self.freeze_vision_model or self.freeze_vision_projection:
            model.freeze(
                freeze_language_model=self.freeze_language_model,
                freeze_vision_model=self.freeze_vision_model,
                freeze_vision_projection=self.freeze_vision_projection,
            )
        return model
