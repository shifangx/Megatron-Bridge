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
"""GB200 performance recipes for Step-3.5-Flash (196B total / A11B active).

Two measured scales are exported per precision:

* 64 GPUs  — PP=8, EP=8,  GBS=1024
* 256 GPUs — PP=8, EP=32, GBS=8192

plus a 256-GPU MXFP8 ``large_scale`` proxy at GBS=512.
"""

from megatron.bridge.perf_recipes.environment import COMMON_PERF_ENV_VARS
from megatron.bridge.perf_recipes.stepfun.common import (
    CommOverlapConfig,
    ConfigContainer,
    _benchmark_common,
    _enable_full_iteration_mxfp8,
    _perf_precision,
    _step35_common,
    step35_196b_a11b_pretrain_config,
)


# Process settings shared by the TransformerEngine-CUDA-graph recipes below.
# EP-dependent entries are filled in per recipe.
_TE_GRAPH_ENV_VARS: dict[str, str | int | float | bool] = {
    **COMMON_PERF_ENV_VARS,
    # CUDA stream scheduling for this model and parallel layout.
    "CUDA_DEVICE_MAX_CONNECTIONS": 32,
    # CUDA graph and allocator behavior for this recipe.
    "NCCL_GRAPH_REGISTER": 0,
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
    # NCCL user-buffer and launch settings.
    "NCCL_NVLS_ENABLE": 0,
    # HybridEP topology for the target system.
    "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
    "NVLINK_DOMAIN_SIZE": 72,
    "USE_MNNVL": 1,
    # Transformer Engine overlap settings for this model.
    "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
    "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
}

# Process settings shared by the full-iteration-CUDA-graph MXFP8 recipes below.
_FULL_ITER_ENV_VARS: dict[str, str | int | float | bool] = {
    **COMMON_PERF_ENV_VARS,
    # CUDA stream scheduling for this model and parallel layout.
    "CUDA_DEVICE_MAX_CONNECTIONS": 32,
    # CUDA graph and allocator behavior for this recipe.
    "NCCL_GRAPH_REGISTER": 0,
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,graph_capture_record_stream_reuse:True",
    "TORCH_NCCL_AVOID_RECORD_STREAMS": 0,
    # NCCL user-buffer and launch settings.
    "NCCL_NVLS_ENABLE": 0,
    # HybridEP topology for the target system.
    "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
    "NVLINK_DOMAIN_SIZE": 72,
    "USE_MNNVL": 1,
    # Transformer Engine overlap settings for this model.
    "CUDNNFE_CLUSTER_OVERLAP_MARGIN": 8,
    "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
    "NVTE_CUTEDSL_FUSED_GROUPED_MLP": 1,
    "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
}


def _te_graph_config(precision: str, *, expert_model_parallel_size: int, global_batch_size: int) -> ConfigContainer:
    """Build a Step-3.5 GB200 recipe that captures attention and MoE routing in TE CUDA graphs."""
    cfg = step35_196b_a11b_pretrain_config()
    cfg.mixed_precision = _perf_precision(precision)
    _step35_common(cfg)

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 8
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = expert_model_parallel_size
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = global_batch_size
    cfg.train.micro_batch_size = 1

    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=True)

    _benchmark_common(cfg)
    # Keep process settings next to the recipe so users can see the exact benchmark environment.
    cfg.env_vars = {
        **_TE_GRAPH_ENV_VARS,
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": expert_model_parallel_size,
    }
    return cfg


def _full_iter_mxfp8_config(*, expert_model_parallel_size: int, global_batch_size: int) -> ConfigContainer:
    """Build a Step-3.5 GB200 MXFP8 recipe that captures the whole iteration in one CUDA graph."""
    cfg = step35_196b_a11b_pretrain_config()
    cfg.mixed_precision = _perf_precision("fp8_mx")
    _step35_common(cfg)

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 8
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = expert_model_parallel_size
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = global_batch_size
    cfg.train.micro_batch_size = 1

    _benchmark_common(cfg)
    _enable_full_iteration_mxfp8(cfg)
    # Keep process settings next to the recipe so users can see the exact benchmark environment.
    cfg.env_vars = {
        **_FULL_ITER_ENV_VARS,
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": expert_model_parallel_size,
    }
    return cfg


def step35_196b_a11b_pretrain_64gpu_gb200_bf16_config() -> ConfigContainer:
    """Step-3.5-Flash 196B-A11B pretrain: 64× GB200, BF16, PP=8 EP=8."""
    return _te_graph_config("bf16", expert_model_parallel_size=8, global_batch_size=1024)


def step35_196b_a11b_pretrain_64gpu_gb200_fp8cs_config() -> ConfigContainer:
    """Step-3.5-Flash 196B-A11B pretrain: 64× GB200, FP8 current-scaling, PP=8 EP=8."""
    return _te_graph_config("fp8_cs", expert_model_parallel_size=8, global_batch_size=1024)


def step35_196b_a11b_pretrain_64gpu_gb200_fp8mx_config() -> ConfigContainer:
    """Step-3.5-Flash 196B-A11B pretrain: 64× GB200, MXFP8, PP=8 EP=8."""
    return _te_graph_config("fp8_mx", expert_model_parallel_size=8, global_batch_size=1024)


def step35_196b_a11b_pretrain_64gpu_gb200_nvfp4_config() -> ConfigContainer:
    """Step-3.5-Flash 196B-A11B pretrain: 64× GB200, NVFP4, PP=8 EP=8."""
    return _te_graph_config("nvfp4", expert_model_parallel_size=8, global_batch_size=1024)


def step35_196b_a11b_pretrain_256gpu_gb200_bf16_config() -> ConfigContainer:
    """Step-3.5-Flash 196B-A11B pretrain: 256× GB200, BF16, PP=8 EP=32."""
    return _te_graph_config("bf16", expert_model_parallel_size=32, global_batch_size=8192)


def step35_196b_a11b_pretrain_256gpu_gb200_fp8cs_config() -> ConfigContainer:
    """Step-3.5-Flash 196B-A11B pretrain: 256× GB200, FP8 current-scaling, PP=8 EP=32."""
    return _te_graph_config("fp8_cs", expert_model_parallel_size=32, global_batch_size=8192)


def step35_196b_a11b_pretrain_256gpu_gb200_fp8mx_config() -> ConfigContainer:
    """Step-3.5-Flash 196B-A11B pretrain: 256× GB200, MXFP8, PP=8 EP=32."""
    return _full_iter_mxfp8_config(expert_model_parallel_size=32, global_batch_size=8192)


def step35_196b_a11b_pretrain_256gpu_gb200_nvfp4_config() -> ConfigContainer:
    """Step-3.5-Flash 196B-A11B pretrain: 256× GB200, NVFP4, PP=8 EP=32."""
    return _te_graph_config("nvfp4", expert_model_parallel_size=32, global_batch_size=8192)


def step35_196b_a11b_pretrain_256gpu_gb200_fp8mx_large_scale_config() -> ConfigContainer:
    """Step-3.5-Flash 196B-A11B pretrain: 256× GB200, MXFP8, PP=8 EP=32, GBS=512 proxy."""
    return _full_iter_mxfp8_config(expert_model_parallel_size=32, global_batch_size=512)
