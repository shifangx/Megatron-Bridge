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

"""Step-3.7 vision-language Megatron model.

The model is a thin wrapper around three components:

1. :class:`Step37VisionTransformer` — Perception-Encoder (PE-G/14) vision tower.
2. :class:`Step37Projector` — single linear map to LLM hidden dim.
3. ``language_model`` — a Megatron-Core ``GPTModel`` produced by
   :class:`Step35ModelProvider` (i.e. the Step-3.5-Flash backbone).

Multimodal fusion mirrors the SteptronOss ``ImageInsertEmbedding`` logic: image
features are spliced into the token-embedding tensor at consecutive positions
starting at each ``<im_start>`` token id (``128000``). 169 patch slots per
image (= ``image_token_count``).

The model exposes the same forward kwargs the qwen3-vl step function passes
through, so the step function in :mod:`step37_step` can be a near-clone of the
qwen3-vl one.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from megatron.core.transformer import MegatronModule

from megatron.bridge.models.stepfun.step37.projector import Step37Projector
from megatron.bridge.models.stepfun.step37.vision_model import Step37VisionTransformer


class Step37Model(MegatronModule):
    """Step-3.7 multimodal model."""

    def __init__(
        self,
        language_model: nn.Module,
        vision_model: Step37VisionTransformer,
        projector: Step37Projector,
        config,
        image_token_id: int = 128001,
        im_start_token_id: int = 128000,
        image_token_count: int = 169,
        pre_process: bool = True,
        post_process: bool = True,
        add_encoder: bool = True,
        add_decoder: bool = True,
    ) -> None:
        super().__init__(config=config)
        self.language_model = language_model
        self.vision_model = vision_model if add_encoder else None
        self.vision_projection = projector if add_encoder else None
        self.image_token_id = image_token_id
        self.im_start_token_id = im_start_token_id
        self.image_token_count = image_token_count
        self.pre_process = pre_process
        self.post_process = post_process
        self.add_encoder = add_encoder
        self.add_decoder = add_decoder
        self.share_embeddings_and_output_weights = getattr(
            language_model, "share_embeddings_and_output_weights", False
        )

    # ------------------------------------------------------------------ utils
    def set_input_tensor(self, input_tensor):
        """Forward MCore PP plumbing to the language model."""
        if hasattr(self.language_model, "set_input_tensor"):
            self.language_model.set_input_tensor(input_tensor)

    def freeze(
        self,
        freeze_language_model: bool = False,
        freeze_vision_model: bool = True,
        freeze_vision_projection: bool = False,
    ) -> None:
        for p in self.language_model.parameters():
            p.requires_grad = not freeze_language_model
        if self.vision_model is not None:
            for p in self.vision_model.parameters():
                p.requires_grad = not freeze_vision_model
        if self.vision_projection is not None:
            for p in self.vision_projection.parameters():
                p.requires_grad = not freeze_vision_projection

    # -------------------------------------------------------------- fusion op
    def _encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run vision tower + projector. Returns ``(N_images, num_image_tokens, hidden)``."""
        if pixel_values is None or self.vision_model is None:
            return None
        if pixel_values.dim() == 5:
            # (B, N_per_sample, 3, H, W) → flatten to (B*N, 3, H, W)
            B, N, C, H, W = pixel_values.shape
            pixel_values = pixel_values.reshape(B * N, C, H, W)
        vision_features = self.vision_model(pixel_values)
        return self.vision_projection(vision_features)

    @staticmethod
    def _splice_image_features(
        input_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        image_features: torch.Tensor,
        image_token_id: int,
    ) -> torch.Tensor:
        """Replace embeddings at ``input_ids == image_token_id`` positions.

        Mirrors the SteptronOss ``ImageInsertEmbedding`` flow: every image
        contributes ``num_image_tokens`` consecutive replacements, in batch
        order. ``input_embeds`` and ``input_ids`` are aligned ``(B, S, ...)``.
        """
        if image_features is None or image_features.numel() == 0:
            return input_embeds
        mask = input_ids == image_token_id  # (B, S)
        num_image_slots = int(mask.sum().item())
        flat_features = image_features.reshape(-1, image_features.shape[-1])
        if flat_features.shape[0] < num_image_slots:
            raise ValueError(
                f"Not enough image features: have {flat_features.shape[0]}, "
                f"need {num_image_slots} ({image_token_id} occurrences in input_ids)"
            )
        flat_features = flat_features[:num_image_slots].to(input_embeds.dtype)
        out = input_embeds.clone()
        out[mask] = flat_features
        return out

    # ------------------------------------------------------------------ fwd
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
        packed_seq_params=None,
        **kwargs,
    ):
        # Vision-only encoder PP stages don't have the language model and
        # simply emit the encoded image features as the inter-stage tensor.
        decoder_input_override = decoder_input

        if self.pre_process and pixel_values is not None and self.vision_model is not None:
            image_features = self._encode_images(pixel_values)
            embed = self.language_model.embedding(input_ids=input_ids, position_ids=position_ids)
            # MCore GPTModel embedding returns (S, B, H); convert to (B, S, H) for splicing.
            if embed.dim() == 3 and embed.shape[0] == input_ids.shape[1]:
                embed_bsh = embed.transpose(0, 1).contiguous()
            else:
                embed_bsh = embed
            embed_bsh = self._splice_image_features(
                embed_bsh, input_ids, image_features, self.image_token_id
            )
            decoder_input_override = embed_bsh.transpose(0, 1).contiguous()

        return self.language_model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            decoder_input=decoder_input_override,
            labels=labels,
            loss_mask=loss_mask,
            packed_seq_params=packed_seq_params,
        )

    # ------------------------------------------------------------- state dict
    def sharded_state_dict(self, prefix: str = "", sharded_offsets=(), metadata=None):
        sd = {}
        if hasattr(self.language_model, "sharded_state_dict"):
            sd.update(
                self.language_model.sharded_state_dict(
                    prefix=f"{prefix}language_model.", sharded_offsets=sharded_offsets, metadata=metadata
                )
            )
        # Vision tower & projector are plain ``nn.Module``s — fall back to ``state_dict``.
        if self.vision_model is not None:
            for k, v in self.vision_model.state_dict(prefix=f"{prefix}vision_model.").items():
                sd[k] = v
        if self.vision_projection is not None:
            for k, v in self.vision_projection.state_dict(prefix=f"{prefix}vision_projection.").items():
                sd[k] = v
        return sd
