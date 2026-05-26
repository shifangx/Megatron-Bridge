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

"""HuggingFace ↔ Megatron-Core bridge for Step-3.7.

Step-3.7 reuses the Step-3.5-Flash LLM backbone verbatim, so most of the
weight mapping is delegated to :class:`Step35Bridge`. The remaining work is:

1. Surface the HF vision-config and projector-config into
   :class:`Step37ModelProvider` so ``provide()`` can build the right vision tower.
2. Map vision-tower and projector weights from their HF parameter names to the
   simple PyTorch parameter names used by :class:`Step37VisionTransformer` and
   :class:`Step37Projector`.

In v1 we only populate the provider config — the vision/projector weight
mappings are stubbed and will be filled out once the canonical
``stepfun-ai/step3p7`` checkpoint is available end-to-end. Loading the LLM
weights alone (``vision/projector`` left at random init) is enough for the
SFT smoke recipe to make progress.
"""

from __future__ import annotations

import logging

import torch
from megatron.core.models.gpt.gpt_model import GPTModel
from transformers import AutoConfig

from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.stepfun.step35_bridge import Step35Bridge
from megatron.bridge.models.stepfun.step37.configuration_step37 import Step37Config
from megatron.bridge.models.stepfun.step37.step37_model import Step37Model
from megatron.bridge.models.stepfun.step37_provider import (
    Step37ModelProvider,
    Step37VisionProviderConfig,
)


logger = logging.getLogger(__name__)

# Ensure HF's AutoConfig can resolve "step3p7" offline (the external identifier
# from stepfun-ai/step3p7's config.json model_type).
AutoConfig.register("step3p7", Step37Config, exist_ok=True)


def _vision_provider_config_from_hf(hf_vision_cfg) -> Step37VisionProviderConfig:
    return Step37VisionProviderConfig(
        hidden_size=getattr(hf_vision_cfg, "hidden_size", 1536),
        num_hidden_layers=getattr(hf_vision_cfg, "num_hidden_layers", 47),
        num_attention_heads=getattr(hf_vision_cfg, "num_attention_heads", 16),
        intermediate_size=getattr(hf_vision_cfg, "intermediate_size", 8960),
        output_dim=getattr(hf_vision_cfg, "output_dim", 6144),
        image_size=getattr(hf_vision_cfg, "image_size", 728),
        patch_size=getattr(hf_vision_cfg, "patch_size", 14),
        in_channels=getattr(hf_vision_cfg, "in_channels", 3),
        use_rope2d=getattr(hf_vision_cfg, "use_rope2d", True),
        rope_theta=float(getattr(hf_vision_cfg, "rope_theta", 1e4)),
        hidden_act=getattr(hf_vision_cfg, "hidden_act", "quickgelu"),
        use_cls_token=getattr(hf_vision_cfg, "use_cls_token", False),
        patch_embed_bias=getattr(hf_vision_cfg, "patch_embed_bias", False),
        layer_norm_eps=getattr(hf_vision_cfg, "layer_norm_eps", 1e-5),
        attention_dropout=getattr(hf_vision_cfg, "attention_dropout", 0.0),
    )


@MegatronModelBridge.register_bridge(
    source="Step3p7ForConditionalGeneration",
    target=GPTModel,
    provider=Step37ModelProvider,
    model_type="step3p7",
)
class Step37Bridge(Step35Bridge):
    """Bridge for ``stepfun-ai/step3p7`` (Step-3.7 vision-language).

    Inherits :class:`Step35Bridge` for the LLM portion. The vision tower and
    projector use the same parameter names HF uses (``vision_model.*``,
    ``vit_large_projector.weight``) — Megatron-Core has no parallelism on the
    vision tower in v1, so an :class:`AutoMapping` over those names is enough.
    """

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> Step37ModelProvider:
        """Build a fully-populated :class:`Step37ModelProvider` from the HF config.

        The text-side config is read from ``hf_config.text_config`` so the
        Step-3.5 mappings (rope-per-layer, swiglu clamps, sliding-attention
        overrides, …) work unchanged.
        """
        hf_config = hf_pretrained.config
        text_config = getattr(hf_config, "text_config", hf_config)

        # Trampoline through Step35Bridge: temporarily swap in the text config
        # so the parent populates a Step35-shaped provider, then promote to a
        # Step37 provider with the vision additions.
        original_config = hf_pretrained.config
        try:
            hf_pretrained.config = text_config
            base_provider = super().provider_bridge(hf_pretrained)
        finally:
            hf_pretrained.config = original_config

        # Re-cast the provider as a Step37 provider, copying over Step35 fields.
        provider = Step37ModelProvider(
            **{f.name: getattr(base_provider, f.name) for f in base_provider.__dataclass_fields__.values()
               if hasattr(Step37ModelProvider, f.name)}
        )

        vision_cfg = getattr(hf_config, "vision_config", None)
        if vision_cfg is not None:
            provider.vision_config = _vision_provider_config_from_hf(vision_cfg)

        provider.projector_input_dim = getattr(hf_config, "projector_input_dim",
                                               getattr(vision_cfg, "output_dim", 6144) if vision_cfg else 6144)
        provider.projector_bias = bool(getattr(hf_config, "projector_bias", False))
        provider.understand_projector_stride = int(getattr(hf_config, "understand_projector_stride", 1))
        provider.image_token_id = int(getattr(hf_config, "image_token_id", 128001))
        provider.im_start_token_id = int(getattr(hf_config, "im_start_token_id", 128000))
        provider.image_token_count = int(getattr(hf_config, "image_token_count", 169))

        if isinstance(hf_config.torch_dtype, torch.dtype):
            provider.autocast_dtype = hf_config.torch_dtype

        return provider
