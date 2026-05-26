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

"""Perception-Encoder (PE-G/14) vision tower for Step-3.7.

This is a single-mesh, PyTorch-native re-implementation of the PE-G/14 vision
encoder used by SteptronOss' Step-3.7 model. It is intentionally simple — it
is meant to run alongside the LLM on the same parallelism group during SFT.
A future revision will move it to its own encoder mesh (mirroring
``MeshConnector`` in SteptronOss) and lift it onto Megatron-Core's
:class:`VisionTransformer`.

The tower produces, for each input image of shape ``(3, image_size, image_size)``,
a tensor of shape ``(num_patches, output_dim)`` where
``num_patches = (image_size // patch_size)**2`` (no CLS token, no spatial merging
inside the tower — the downstream projector + image-token layout handles the
169-token-per-image budget).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class PatchEmbed2D(nn.Module):
    """2D conv patch embedding without CLS token.

    Input:  ``(B, C, H, W)`` with ``H == W == image_size``.
    Output: ``(B, num_patches, hidden_size)``.
    """

    def __init__(self, in_channels: int, hidden_size: int, patch_size: int, bias: bool = False):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size, bias=bias)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        x = self.proj(pixel_values)  # (B, hidden, H/p, W/p)
        x = x.flatten(2).transpose(1, 2).contiguous()  # (B, num_patches, hidden)
        return x


def _build_axial_freqs(num_patches_per_side: int, head_dim: int, theta: float, device, dtype) -> torch.Tensor:
    """Return cos/sin tables for 2D-RoPE along one axis.

    Output shape: ``(num_patches_per_side, head_dim // 2)``.
    Half the head dimension is reserved for the row axis, half for the column
    axis; the caller concatenates the two halves.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    pos = torch.arange(num_patches_per_side, device=device, dtype=torch.float32)
    angles = torch.einsum("i,j->ij", pos, freqs)  # (P, head_dim/2)
    return angles.to(dtype=dtype)


def apply_rope2d(q: torch.Tensor, k: torch.Tensor, num_patches_per_side: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply 2D rotary position embeddings on (q, k).

    q, k have shape ``(B, H, N, D)`` where ``N == num_patches_per_side**2``.
    The first ``D/2`` channels encode row position, the next ``D/2`` encode
    column position. This is the convention used by the PE-G encoder in
    SteptronOss when ``use_rope2d=True``.
    """
    B, H, N, D = q.shape
    P = num_patches_per_side
    assert N == P * P, f"expected N=={P*P}, got {N}"
    assert D % 4 == 0, "head_dim must be divisible by 4 for 2D RoPE"

    half = D // 2
    angles_row = _build_axial_freqs(P, half, theta, q.device, q.dtype)  # (P, half/2)
    angles_col = _build_axial_freqs(P, half, theta, q.device, q.dtype)  # (P, half/2)

    rows = torch.arange(P, device=q.device).repeat_interleave(P)  # (N,)
    cols = torch.arange(P, device=q.device).repeat(P)  # (N,)
    row_angles = angles_row[rows]  # (N, half/2)
    col_angles = angles_col[cols]  # (N, half/2)

    row_cos = row_angles.cos().repeat_interleave(2, dim=-1)
    row_sin = row_angles.sin().repeat_interleave(2, dim=-1)
    col_cos = col_angles.cos().repeat_interleave(2, dim=-1)
    col_sin = col_angles.sin().repeat_interleave(2, dim=-1)

    def _rotate(t):
        return torch.stack((-t[..., 1::2], t[..., 0::2]), dim=-1).flatten(-2)

    def _apply(x):
        x_row, x_col = x[..., :half], x[..., half:]
        x_row = (x_row * row_cos) + (_rotate(x_row) * row_sin)
        x_col = (x_col * col_cos) + (_rotate(x_col) * col_sin)
        return torch.cat([x_row, x_col], dim=-1)

    return _apply(q), _apply(k)


class Step37VisionAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=True)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, num_patches_per_side: int, rope_theta: float, use_rope2d: bool) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv.unbind(0)  # each (B, H, N, D)
        if use_rope2d:
            q, k = apply_rope2d(q, k, num_patches_per_side, rope_theta)
        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)
        attn = attn.transpose(1, 2).reshape(B, N, C).contiguous()
        return self.proj(attn)


class Step37VisionMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.act = QuickGELU()
        self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Step37VisionBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        layer_norm_eps: float = 1e-5,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.attn = Step37VisionAttention(hidden_size, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.mlp = Step37VisionMLP(hidden_size, intermediate_size)

    def forward(self, x: torch.Tensor, num_patches_per_side: int, rope_theta: float, use_rope2d: bool) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), num_patches_per_side, rope_theta, use_rope2d)
        x = x + self.mlp(self.norm2(x))
        return x


class Step37VisionTransformer(nn.Module):
    """Perception-Encoder (PE-G/14) vision tower.

    Args mirror :class:`Step37VisionConfig`. Output is the last-layer hidden state
    projected to ``output_dim`` (typically 6144), shape
    ``(B, num_patches, output_dim)``.
    """

    def __init__(self, vision_config):
        super().__init__()
        self.config = vision_config
        self.image_size = vision_config.image_size
        self.patch_size = vision_config.patch_size
        self.num_patches_per_side = self.image_size // self.patch_size
        self.num_patches = self.num_patches_per_side ** 2
        self.use_rope2d = vision_config.use_rope2d
        self.rope_theta = vision_config.rope_theta

        self.patch_embed = PatchEmbed2D(
            in_channels=vision_config.in_channels,
            hidden_size=vision_config.hidden_size,
            patch_size=vision_config.patch_size,
            bias=vision_config.patch_embed_bias,
        )
        self.blocks = nn.ModuleList(
            [
                Step37VisionBlock(
                    hidden_size=vision_config.hidden_size,
                    intermediate_size=vision_config.intermediate_size,
                    num_heads=vision_config.num_attention_heads,
                    layer_norm_eps=vision_config.layer_norm_eps,
                    dropout=vision_config.attention_dropout,
                )
                for _ in range(vision_config.num_hidden_layers)
            ]
        )
        self.norm = nn.LayerNorm(vision_config.hidden_size, eps=vision_config.layer_norm_eps)
        self.head = nn.Linear(vision_config.hidden_size, vision_config.output_dim, bias=False)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode a batch of square images.

        Args:
            pixel_values: ``(B, 3, image_size, image_size)``.

        Returns:
            ``(B, num_patches, output_dim)`` patch features.
        """
        if pixel_values.dim() != 4:
            raise ValueError(f"expected pixel_values with 4 dims, got shape {tuple(pixel_values.shape)}")
        x = self.patch_embed(pixel_values)
        for block in self.blocks:
            x = block(x, self.num_patches_per_side, self.rope_theta, self.use_rope2d)
        x = self.norm(x)
        x = self.head(x)
        return x
