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

"""Vision→LLM projector used by Step-3.7.

The SteptronOss step3p7 model maps vision-encoder outputs into the LLM
embedding space with a single ``Linear(vision_out_dim, llm_hidden, bias=False)``
initialised with Kaiming-normal (fan-in). Optional ``stride`` lets us drop
every Kth patch token to land on the 169-token-per-image budget after the
vision tower emits a denser grid.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Step37Projector(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        bias: bool = False,
        understand_projector_stride: int = 1,
    ):
        super().__init__()
        self.understand_projector_stride = max(1, int(understand_projector_stride))
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)
        nn.init.kaiming_normal_(self.linear.weight, mode="fan_in", nonlinearity="linear")
        if bias:
            nn.init.zeros_(self.linear.bias)

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """Project vision features.

        Args:
            vision_features: ``(B, N, input_dim)`` raw vision-encoder output.

        Returns:
            ``(B, N // stride**2, output_dim)`` projected features.
        """
        if self.understand_projector_stride > 1:
            B, N, D = vision_features.shape
            s = self.understand_projector_stride
            side = int(N ** 0.5)
            if side * side != N:
                raise ValueError(f"projector stride>1 requires square token grid, got N={N}")
            if side % s != 0:
                raise ValueError(f"projector stride {s} does not divide grid side {side}")
            x = vision_features.reshape(B, side, side, D)
            x = x[:, ::s, ::s, :].contiguous()
            x = x.reshape(B, (side // s) * (side // s), D)
            return self.linear(x)
        return self.linear(vision_features)
