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

"""MCore ``SelfAttention`` subclass for the Step3.7 vision tower.

The PE-G/14 encoder uses a **2D rotary** positional embedding whose frequency
layout differs from the text-side (1D) rope MCore applies internally. This
subclass therefore disables MCore's built-in rope path (the vision
``TransformerConfig`` sets ``position_embedding_type="none"``) and instead
applies the *exact* native rotation (:func:`apply_rotary_emb` from
``modelling_step37.utils``) to the query/key tensors, so the MCore-assembled
vision tower is numerically equivalent to :class:`Step37VisionModel`.

The precomputed 2D freqs (``[1, 1, S, head_dim]``) are threaded in through the
standard MCore ``rotary_pos_emb`` argument by :class:`Step37VisionModelMcore`.
"""

from __future__ import annotations

from typing import Optional

import torch
from megatron.core.transformer.attention import SelfAttention

from megatron.bridge.models.stepfun.modelling_step37.utils import apply_rotary_emb


class Step37VisionSelfAttention(SelfAttention):
    """Self-attention for the Step3.7 vision encoder (2D rope, full attention)."""

    def _apply_vision_rope(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        """Apply the native 2D rope to an MCore ``[S, B, nh, hd]`` tensor.

        ``apply_rotary_emb`` expects the sequence on ``dim=-2`` with the freqs
        broadcasting as ``[1, 1, S, hd]``; MCore hands us ``[S, B, nh, hd]``,
        so we move to ``[B, nh, S, hd]``, rotate, and move back.
        """
        x_bnsd = x.permute(1, 2, 0, 3)
        x_bnsd = apply_rotary_emb(freqs, x_bnsd)
        return x_bnsd.permute(2, 0, 1, 3).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        key_value_states: Optional[torch.Tensor] = None,
        inference_context=None,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        rotary_pos_cos: Optional[torch.Tensor] = None,
        rotary_pos_sin: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        packed_seq_params=None,
        sequence_len_offset=None,
        *,
        inference_params=None,
        rotary_pos_cos_sin=None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Vision self-attention forward.

        ``rotary_pos_emb`` carries the precomputed 2D rope freqs (not the usual
        MCore rope tensor); it is applied manually to Q/K here. Inference /
        KV-cache paths are intentionally unsupported for the vision encoder.
        """
        assert inference_context is None and inference_params is None, (
            "Step37VisionSelfAttention does not support inference contexts"
        )

        # Q, K, V projection ([S, B, nh, hd]).
        query, key, value = self.get_query_key_value_tensors(hidden_states, key_value_states)

        # 2D rope (native-equivalent), applied only when freqs are provided.
        if rotary_pos_emb is not None:
            query = self._apply_vision_rope(query, rotary_pos_emb)
            key = self._apply_vision_rope(key, rotary_pos_emb)

        # Full (non-causal) attention within each image; batch dim isolates images.
        core_attn_out = self.core_attention(
            query,
            key,
            value,
            attention_mask,
            attn_mask_type=self.attn_mask_type,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
        )

        if packed_seq_params is not None and getattr(packed_seq_params, "qkv_format", None) == "thd":
            core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)

        output, bias = self.linear_proj(core_attn_out)
        return output, bias


__all__ = ["Step37VisionSelfAttention"]
