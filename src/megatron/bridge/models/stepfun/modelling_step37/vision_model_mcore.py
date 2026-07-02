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

"""Megatron-Core / TransformerEngine assembled Step3.7 vision tower.

This is the MCore counterpart to the pure-PyTorch :class:`Step37VisionModel`
(``vision_model.py``). It mirrors the Qwen3-VL pattern
(``modelling_qwen3_vl/vision_model.py``): the transformer stack is a real MCore
:class:`TransformerBlock` built from the TE ViT layer spec, which lets the PE-G/14
trunk benefit from TE fused kernels / CUDA graphs, while the non-transformer
pieces (conv patch-embed, absolute pos-emb, ``ln_pre``/``ln_post`` and the two
conv downsamplers) are kept **byte-identical** to the native tower so their HF
weights load unchanged.

Two Step3.7-specific deviations from a stock ViT are handled explicitly:

* **2D RoPE** — applied manually inside :class:`Step37VisionSelfAttention`
  (MCore's built-in rope is disabled via ``position_embedding_type="none"``),
  so the rotation is numerically identical to the native tower.
* **LayerScale** (``ls_1``/``ls_2`` γ) — *not* a module here. During HF→Megatron
  conversion the γ vectors are folded into ``linear_proj`` / ``linear_fc2``
  output rows (see ``step37_bridge``), keeping the standard TE layer
  (CUDA-graph friendly) intact.

Module layout (for weight mapping):
    vision_model.conv1 / ln_pre / positional_embedding / ln_post /
        vit_downsampler{1,2}          — same names/types as the native tower
    vision_model.decoder.layers.*     — MCore TransformerBlock (replaces the
        native ``transformer.resblocks.*``)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from megatron.core.models.common.vision_module.vision_module import VisionModule
from megatron.core.models.vision.vit_layer_specs import get_vit_layer_with_transformer_engine_spec
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig

from megatron.bridge.models.stepfun.modelling_step37.utils import EncoderRope2D
from megatron.bridge.models.stepfun.modelling_step37.vision_attention import Step37VisionSelfAttention


def get_step37_vision_layer_spec():
    """TE ViT layer spec with the Step3.7 2D-rope self-attention swapped in."""
    spec = get_vit_layer_with_transformer_engine_spec()
    spec.submodules.self_attention.module = Step37VisionSelfAttention
    return spec


class Step37VisionModelMcore(VisionModule):
    """PE-G/14 vision tower whose transformer stack is an MCore ``TransformerBlock``.

    Args:
        transformer_config: Megatron ``TransformerConfig`` for the vision tower
            (build via ``build_step37_vision_transformer_config``).
        vision_config: the HF ``StepRoboticsVisionEncoderConfig`` (for the conv /
            pos-emb / downsampler geometry that lives outside the transformer).
    """

    def __init__(
        self,
        transformer_config: TransformerConfig,
        vision_config,
    ) -> None:
        super().__init__(config=transformer_config)
        self.config = transformer_config
        vc = vision_config

        self.hidden_size = vc.width
        self.num_heads = vc.heads
        self.patch_size = vc.patch_size
        self.image_size = vc.image_size
        self.use_cls_token = getattr(vc, "use_cls_token", False)
        self.use_rope2d = getattr(vc, "use_rope2d", True)
        self.use_abs_posemb = getattr(vc, "use_abs_posemb", True)
        self.layer_norm_eps = vc.layer_norm_eps
        self.use_ln_pre = getattr(vc, "use_ln_pre", False)
        self.use_ln_post = getattr(vc, "use_ln_post", True)

        # ── Non-transformer pieces: identical names/types to the native tower ──
        self.conv1 = nn.Conv2d(
            in_channels=vc.num_channels,
            out_channels=self.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )
        self.ln_pre = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps) if self.use_ln_pre else nn.Identity()
        self.ln_post = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps) if self.use_ln_post else nn.Identity()

        grid_size = self.image_size // self.patch_size
        self.base_grid = (grid_size, grid_size)

        if self.use_cls_token:
            self.class_embedding = nn.Parameter(torch.randn(self.hidden_size) * (self.hidden_size**-0.5))
        else:
            self.class_embedding = None

        if self.use_abs_posemb:
            self.posemb_grid_size = self.image_size // self.patch_size
            self.positional_embedding = nn.Parameter(
                (self.hidden_size**-0.5)
                * torch.randn(int(self.use_cls_token) + self.posemb_grid_size**2, self.hidden_size)
            )

        # ── 2D rope: reused for freqs, applied inside the attention ──────────
        head_dim = self.hidden_size // self.num_heads
        self.rope = None
        if self.use_rope2d:
            self.rope = EncoderRope2D(
                dim=head_dim,
                max_grid_height=self.base_grid[0],
                max_grid_width=self.base_grid[1],
                use_cls_token=self.use_cls_token,
                theta=getattr(vc, "rope_theta", 10000),
                max_freq=getattr(vc, "rope_max_freq", 10),
                num_freqs=getattr(vc, "rope_num_freqs", 1),
                theta_rescale_factor=getattr(vc, "rope_theta_rescale_factor", 1.0),
            )

        # ── MCore transformer stack (replaces native transformer.resblocks) ──
        self.decoder = TransformerBlock(
            config=transformer_config,
            spec=get_step37_vision_layer_spec(),
            post_layer_norm=False,
            pre_process=True,
            post_process=True,
        )

        # ── Conv downsamplers (identical names/types to the native tower) ────
        self.vit_downsampler1 = nn.Conv2d(self.hidden_size, self.hidden_size * 2, kernel_size=3, stride=2, padding=1)
        self.vit_downsampler2 = nn.Conv2d(self.hidden_size * 2, self.hidden_size * 4, kernel_size=3, stride=2, padding=1)

    def set_input_tensor(self, input_tensor) -> None:
        """Vision tower always runs pre_process=post_process=True (no PP)."""
        raise NotImplementedError("Step37VisionModelMcore does not support pipeline parallelism")

    def sample_abs_posemb(self, grid_h: int, grid_w: int) -> torch.Tensor:
        if self.posemb_grid_size == grid_h and self.posemb_grid_size == grid_w:
            return self.positional_embedding[None, ...]

        pos_embed = self.positional_embedding
        if self.use_cls_token:
            cls_token_embed, pos_embed = pos_embed[:1], pos_embed[1:]
        pos_embed = (
            pos_embed.reshape(1, self.posemb_grid_size, self.posemb_grid_size, -1).permute(0, 3, 1, 2).contiguous()
        )
        pos_embed = F.interpolate(pos_embed, size=(grid_h, grid_w), mode="bilinear", align_corners=False)
        pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(-1, self.hidden_size)
        if self.use_cls_token:
            pos_embed = torch.cat([cls_token_embed, pos_embed], dim=0)
        return pos_embed[None, ...]

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run PE-G/14 (MCore transformer) + both downsamplers.

        Args:
            pixel_values: ``[B, C, H, W]`` with ``H = W = image_size``.

        Returns:
            ``[B, P', output_dim]`` image features (``P'=169`` for 728² inputs).
        """
        bsz, _, height, width = pixel_values.shape
        grid_h, grid_w = height // self.patch_size, width // self.patch_size

        hidden_state = self.conv1(pixel_values)  # [B, D, Gh, Gw]
        hidden_state = hidden_state.flatten(2).transpose(1, 2)  # [B, Gh*Gw, D]

        if self.use_cls_token:
            cls_token = self.class_embedding.view(1, 1, -1).expand(bsz, -1, -1)
            hidden_state = torch.cat([cls_token, hidden_state], dim=1)

        if self.use_abs_posemb:
            hidden_state = hidden_state + self.sample_abs_posemb(grid_h, grid_w)
        hidden_state = self.ln_pre(hidden_state)

        # 2D rope freqs for this grid (applied inside the attention).
        rotary_pos_emb = None
        if self.rope is not None:
            rotary_pos_emb = self.rope.get_freqs((grid_h, grid_w), hidden_state.device)

        # MCore transformer runs sequence-first: [S, B, D].
        hidden_state = hidden_state.transpose(0, 1).contiguous()
        hidden_state = self.decoder(
            hidden_states=hidden_state,
            attention_mask=None,
            rotary_pos_emb=rotary_pos_emb,
        )
        hidden_state = hidden_state.transpose(0, 1).contiguous()  # back to [B, S, D]

        if self.use_ln_post:
            hidden_state = self.ln_post(hidden_state)

        if self.use_cls_token:
            hidden_state = hidden_state[:, 1:, :]

        # Spatial reshape + 2× stride-2 downsampler → [B, P', D*4].
        B, P = hidden_state.shape[:2]
        HW = int(P**0.5)
        image_features = hidden_state.permute(0, 2, 1).view(B, -1, HW, HW)
        image_features = self.vit_downsampler1(image_features)
        image_features = self.vit_downsampler2(image_features)

        B, C, HW, _ = image_features.shape
        image_features = image_features.view(B, -1, HW * HW).permute(0, 2, 1)
        return image_features


__all__ = ["Step37VisionModelMcore", "get_step37_vision_layer_spec"]
