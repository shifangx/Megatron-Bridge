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

"""Step-3.7 vision-language recipes.

Two factory functions:

* :func:`step37_smoke_sft_config` — single-GPU functional smoke (small LLM,
  vision tower trimmed, vision frozen). Mirrors
  ``Scripts-MBridge/7.4.step37_num_layers_6.sh``.
* :func:`step37_321b_a38b_sft_config` — full Step-3.7 SFT (288-expert MoE,
  45 LLM layers). Recommended parallelism: TP=1 PP=8 CP=8 EP=8 SP=on. Mirrors
  ``Scripts-MBridge/9.3.step3p7_sft_resume.sh``.

Both pull the Energon-based VLM data pipeline that qwen3-vl uses, with a
step3p7-flavoured tokenizer/processor path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from megatron.bridge.recipes.common import _sft_common_vlm
from megatron.bridge.recipes.utils.optimizer_utils import distributed_fused_adam_with_cosine_annealing
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.flex_dispatcher_backend import apply_flex_dispatcher_backend

from megatron.bridge.models.stepfun.step37_provider import (
    Step37ModelProvider,
    Step37VisionProviderConfig,
)


_HF_STEP3P7_PATH_DEFAULT = "stepfun-ai/step3p7"


def _build_step37_provider(
    *,
    num_layers: int = 45,
    hidden_size: int = 4096,
    num_attention_heads: int = 64,
    num_query_groups: int = 8,
    ffn_hidden_size: int = 11264,
    vocab_size: int = 128896,
    seq_length: int = 4096,
    num_moe_experts: Optional[int] = 288,
    moe_router_topk: int = 8,
    moe_shared_expert_intermediate_size: Optional[int] = 1280,
    moe_ffn_hidden_size: Optional[int] = 1280,
    vision_config: Optional[Step37VisionProviderConfig] = None,
    freeze_language_model: bool = False,
    freeze_vision_model: bool = True,
    freeze_vision_projection: bool = False,
) -> Step37ModelProvider:
    """Build a :class:`Step37ModelProvider` from scratch (no HF roundtrip).

    The smoke recipe uses this to side-step the HF download. Field values
    follow the architectural map distilled from the SteptronOss config.
    """
    provider_kwargs = dict(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_query_groups=num_query_groups,
        ffn_hidden_size=ffn_hidden_size,
        seq_length=seq_length,
        vocab_size=vocab_size,
        normalization="RMSNorm",
        gated_linear_unit=True,
        add_bias_linear=False,
        add_qkv_bias=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        kv_channels=hidden_size // num_attention_heads,
        position_embedding_type="rope",
        rotary_base=5_000_000.0,
        rotary_percent=1.0,
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
    )
    if num_moe_experts and num_moe_experts > 1:
        provider_kwargs.update(
            num_moe_experts=num_moe_experts,
            moe_router_topk=moe_router_topk,
            moe_router_load_balancing_type="aux_loss",
            moe_aux_loss_coeff=0.0,
            moe_router_pre_softmax=False,
            moe_grouped_gemm=True,
            moe_token_dispatcher_type="alltoall",
            moe_permute_fusion=True,
            moe_shared_expert_intermediate_size=moe_shared_expert_intermediate_size,
            moe_ffn_hidden_size=moe_ffn_hidden_size or moe_shared_expert_intermediate_size,
        )

    provider = Step37ModelProvider(**provider_kwargs)

    if vision_config is not None:
        provider.vision_config = vision_config
    provider.freeze_language_model = freeze_language_model
    provider.freeze_vision_model = freeze_vision_model
    provider.freeze_vision_projection = freeze_vision_projection

    # When MoE / hybrid-attention is on, the language tower needs Step35DecoderLayer.
    # Import lazily to avoid a hard dep when only the smoke (dense) recipe is loaded.
    if num_moe_experts and num_moe_experts > 1:
        from megatron.bridge.models.stepfun.step35_bridge import _build_step35_layer_spec

        provider.transformer_layer_spec = _build_step35_layer_spec

    return provider


def step37_smoke_sft_config(
    seq_length: int = 4096,
    train_iters: int = 10,
    global_batch_size: int = 4,
    micro_batch_size: int = 1,
) -> ConfigContainer:
    """Step-3.7 SFT smoke recipe: 6-layer LLM, MoE off, vision frozen.

    Recommended launch: single node, TP=1 PP=1.
    """
    cfg = _sft_common_vlm()

    vision_config = Step37VisionProviderConfig(
        num_hidden_layers=4,  # heavily trimmed for smoke
        hidden_size=512,
        num_attention_heads=8,
        intermediate_size=2048,
        output_dim=1024,
        image_size=224,
        patch_size=14,
    )

    cfg.model = _build_step37_provider(
        num_layers=6,
        hidden_size=1024,
        num_attention_heads=16,
        num_query_groups=4,
        ffn_hidden_size=2816,
        vocab_size=128896,
        seq_length=seq_length,
        num_moe_experts=None,
        vision_config=vision_config,
        freeze_language_model=False,
        freeze_vision_model=True,
        freeze_vision_projection=False,
    )
    cfg.model.projector_input_dim = vision_config.output_dim

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.attention_backend = "auto"
    cfg.model.cross_entropy_loss_fusion = True

    cfg.train.train_iters = train_iters
    cfg.train.global_batch_size = global_batch_size
    cfg.train.micro_batch_size = micro_batch_size
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100

    cfg.validation.eval_interval = 1000
    cfg.validation.eval_iters = 0

    opt_cfg, sched_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=2, lr_decay_iters=train_iters, max_lr=1e-4, min_lr=1e-5,
    )
    cfg.optimizer = opt_cfg
    cfg.scheduler = sched_cfg

    cfg.dataset.seq_length = seq_length
    cfg.dataset.hf_processor_path = _HF_STEP3P7_PATH_DEFAULT

    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.average_in_collective = True
    cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"

    cfg.mixed_precision = "bf16_mixed"
    return cfg


def step37_321b_a38b_sft_config(
    hf_path: str = _HF_STEP3P7_PATH_DEFAULT,
    seq_length: int = 4096,
    train_iters: int = 10,
    global_batch_size: int = 16,
    micro_batch_size: int = 1,
) -> ConfigContainer:
    """Full-size Step-3.7 SFT recipe.

    Recommended parallelism: TP=1 PP=8 CP=8 EP=8, SP=on. Mirrors the
    SteptronOss ``9.3.step3p7_sft_resume.sh`` layout.
    """
    cfg = _sft_common_vlm()

    cfg.model = _build_step37_provider(
        num_layers=45,
        hidden_size=4096,
        num_attention_heads=64,
        num_query_groups=8,
        ffn_hidden_size=11264,
        vocab_size=128896,
        seq_length=seq_length,
        num_moe_experts=288,
        moe_router_topk=8,
        moe_shared_expert_intermediate_size=1280,
        moe_ffn_hidden_size=1280,
        vision_config=Step37VisionProviderConfig(),
        freeze_language_model=False,
        freeze_vision_model=True,
        freeze_vision_projection=False,
    )

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 8
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 8
    cfg.model.expert_model_parallel_size = 8
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = True

    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.attention_backend = "auto"
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "te"
    cfg.model.moe_token_dispatcher_type = "alltoall"
    cfg.model.moe_flex_dispatcher_backend = "deepep"
    apply_flex_dispatcher_backend(cfg.model, cfg.model.moe_flex_dispatcher_backend)

    cfg.model.recompute_granularity = "full"
    cfg.model.recompute_method = "uniform"
    cfg.model.recompute_num_layers = 1

    cfg.train.train_iters = train_iters
    cfg.train.global_batch_size = global_batch_size
    cfg.train.micro_batch_size = micro_batch_size
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100

    cfg.validation.eval_interval = 1000
    cfg.validation.eval_iters = 0

    opt_cfg, sched_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=10, lr_decay_iters=train_iters, max_lr=5e-5, min_lr=5e-6,
    )
    cfg.optimizer = opt_cfg
    cfg.scheduler = sched_cfg
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32

    cfg.dataset.seq_length = seq_length
    cfg.dataset.hf_processor_path = hf_path

    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True

    cfg.mixed_precision = "bf16_mixed"
    return cfg


__all__ = [
    "step37_smoke_sft_config",
    "step37_321b_a38b_sft_config",
]
