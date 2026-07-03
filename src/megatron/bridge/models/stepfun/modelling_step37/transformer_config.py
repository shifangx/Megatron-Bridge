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

"""Step3.7 transformer config and vision-config helper.

Mirrors ``qwen_vl/modelling_qwen3_vl/transformer_config.py``: the text-side
config is the standard Megatron ``TransformerConfig`` already used by Step-3.5,
extended with vision-tower fields. The HF ``StepRoboticsVisionEncoderConfig``
is passed straight through to the Megatron vision module — no separate
Megatron-side ``TransformerConfig`` is constructed for the vision tower, since
the PE-G/14 trunk does not use any Megatron tensor-parallel primitives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch.nn.functional as F
from megatron.core.transformer.transformer_config import TransformerConfig

from megatron.bridge.utils.cuda_graph import clear_cuda_graph_modules, set_cuda_graph_modules


@dataclass
class Step37TransformerConfig(TransformerConfig):
    """Step3.7 transformer config.

    Extends the Step-3.5 text-decoder ``TransformerConfig`` with the multimodal
    fields that ``Step37Model`` reads at construction time. All Step-3.5
    per-layer fields (``layer_types``, ``rotary_percents``,
    ``rotary_base_per_layer``, ``swiglu_limits``, ``swiglu_limits_shared``,
    ``attention_other_setting``, ``sliding_attention_setting``,
    ``head_wise_attn_gate``) are inherited from the Step-3.5 model provider —
    this class only adds the vision-side fields.
    """

    vision_config: Any = None
    image_token_id: int = 128001
    understand_projector_stride: int = 2
    projector_bias: bool = False
    language_max_sequence_length: int = 262144
    # Which vision-tower implementation to build:
    #   "native" — the HF-aligned pure-PyTorch PE-G/14 (:class:`Step37VisionModel`)
    #   "mcore"  — the Megatron-Core / TransformerEngine assembled tower
    #              (:class:`Step37VisionModelMcore`)
    vision_model_impl: str = "native"


def get_vision_model_config(vision_cfg: Any) -> Any:
    """Return the HF vision config unchanged.

    ``Step37VisionModel`` consumes the HF ``StepRoboticsVisionEncoderConfig``
    directly (it never uses Megatron tensor-parallel primitives), so this
    function is just a structural mirror of
    ``qwen_vl/modelling_qwen3_vl/transformer_config.get_vision_model_config``
    for parity with the Qwen3-VL package shape. It is intentionally a no-op.
    """
    return vision_cfg


_HIDDEN_ACT_TO_FN = {
    "gelu": F.gelu,
    "gelu_new": lambda x: F.gelu(x, approximate="tanh"),
    "gelu_pytorch_tanh": lambda x: F.gelu(x, approximate="tanh"),
    "quick_gelu": lambda x: x * F.sigmoid(1.702 * x),
    "relu": F.relu,
    "silu": F.silu,
}


def build_step37_vision_transformer_config(vision_cfg: Any, base_config: TransformerConfig) -> TransformerConfig:
    """Build a Megatron ``TransformerConfig`` for the MCore Step3.7 vision tower.

    Maps the HF ``StepRoboticsVisionEncoderConfig`` (PE-G/14) fields onto a
    dense, TP=1 ``TransformerConfig`` suitable for
    :class:`Step37VisionModelMcore`. Compute/dtype fields are copied from
    ``base_config`` (the text-side Step3.7 config) so the vision tower runs in
    the same precision as the decoder.

    Notes:
        * No ``position_embedding_type`` is set — it is not a core
          ``TransformerConfig`` field, and MCore's built-in rope is left unused.
          The 2D rope is applied manually inside
          :class:`Step37VisionSelfAttention`, not by MCore.
        * ``gated_linear_unit=False`` and biases enabled to match the PE MLP
          (``c_fc``/``c_proj``) and fused ``in_proj``/``out_proj`` layout.
    """
    width = int(vision_cfg.width)
    heads = int(vision_cfg.heads)
    mlp_ratio = float(getattr(vision_cfg, "mlp_ratio", 8960 / 1536))
    hidden_act = getattr(vision_cfg, "hidden_act", "gelu")
    activation_func = _HIDDEN_ACT_TO_FN.get(hidden_act, F.gelu)

    config = TransformerConfig(
        num_layers=int(vision_cfg.layers),
        hidden_size=width,
        num_attention_heads=heads,
        num_query_groups=heads,
        kv_channels=width // heads,
        ffn_hidden_size=int(width * mlp_ratio),
        normalization="LayerNorm",
        layernorm_epsilon=float(vision_cfg.layer_norm_eps),
        gated_linear_unit=False,
        activation_func=activation_func,
        add_bias_linear=True,
        add_qkv_bias=True,
        apply_rope_fusion=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        attention_softmax_in_fp32=True,
        bias_activation_fusion=False,
        # Vision tower runs on a replicated (TP=1) mesh.
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=1,
        sequence_parallel=False,
        # Inherit precision from the text-side config.
        params_dtype=base_config.params_dtype,
        bf16=base_config.bf16,
        fp16=base_config.fp16,
        autocast_dtype=getattr(base_config, "autocast_dtype", None),
        pipeline_dtype=getattr(base_config, "pipeline_dtype", None),
    )

    # Vision encoder CUDA graph settings. Mirror qwen_vl's get_vision_model_config:
    # the provider (``base_config``) carries vision-specific CUDA graph knobs
    # (``vision_cuda_graph_impl`` / ``vision_cuda_graph_scope`` /
    # ``max_vision_cuda_graph_seq_length``); translate them onto the vision
    # ``TransformerConfig`` so the MCore ``TransformerBlock`` captures TE CUDA
    # graphs. Absent / "none" leaves the tower running eagerly (backward compat).
    vision_cuda_graph_impl = getattr(base_config, "vision_cuda_graph_impl", "none")
    if vision_cuda_graph_impl and vision_cuda_graph_impl != "none":
        config.cuda_graph_impl = vision_cuda_graph_impl
        vision_scope = getattr(base_config, "vision_cuda_graph_scope", None)
        if vision_scope:
            set_cuda_graph_modules(config, vision_scope)
        else:
            clear_cuda_graph_modules(config)
        # Share the decoder's warmup-step count so both graphs warm up together.
        config.cuda_graph_warmup_steps = getattr(
            base_config, "cuda_graph_warmup_steps", config.cuda_graph_warmup_steps
        )
    else:
        config.cuda_graph_impl = "none"
        clear_cuda_graph_modules(config)

    max_vision_seq = getattr(base_config, "max_vision_cuda_graph_seq_length", None)
    if max_vision_seq is not None:
        config.max_vision_cuda_graph_seq_length = max_vision_seq

    return config
