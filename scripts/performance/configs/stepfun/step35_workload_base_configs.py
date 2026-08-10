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

"""Parallelism presets for Step-3.5-Flash performance configs.

Mirrors ``configs/qwen/qwen3_workload_base_configs.py`` but targets
``stepfun-ai/Step-3.5-Flash`` (~196B total / ~11B active MoE, 45 decoder
layers: 3 dense + 42 MoE, 288 routed + 1 shared expert, top-8, GQA hybrid
full/sliding attention).

Config naming convention (read by ``get_workload_base_config``):
    {MODEL}_{SIZE}_{TASK}_CONFIG_{GPU}_{PRECISION}_{VERSION}
e.g. STEP35_196B_A11B_PRETRAIN_CONFIG_GB200_FP8_MX_V2

V1: Blackwell num_gpus=64  (PP=8, EP=8),   GBS=1024
V2: Blackwell num_gpus=256 (PP=8, EP=32),  GBS=8192
Large-scale proxy: GBS=512

IMPORTANT — 45-layer constraints (enforced in step35_llm_pretrain.py):
  * PP=8 does not divide 45, so by default the builder sets
    num_layers_in_first_pipeline_stage=6 / num_layers_in_last_pipeline_stage=3
    (6 + 6*6 + 3 = 45), matching the validated Step-3.7 sbatch layout.
  * virtual_pipeline_model_parallel_size stays None by default (45 is not
    divisible by PP*VPP; enabling VPP requires an explicit pp_layout). The
    GB200 FP8_MX V2 preset is the exception: it sets VPP=3 with an explicit
    12-stage pp_layout, and the builder then clears the fixed first/last split
    in favor of that layout.

These presets mirror the Qwen3-235B tuning as a *starting point*; the exact
per-GPU/precision knobs should be perf-validated for Step-3.5.

Use --config_variant to select a variant.
Use --list_config_variants to see available variants interactively.
"""

from dataclasses import replace

from utils.utils import WorkloadBaseConfig


BASE_STEP35_196B_A11B_CONFIG = WorkloadBaseConfig(
    expert_tensor_parallel_size=1,
)


# =============================================================================
# Step-3.5 196B A11B presets - V1 (Blackwell num_gpus=64, GBS=1024)
# =============================================================================

STEP35_196B_A11B_PRETRAIN_CONFIG_GB300_BF16_V1 = replace(
    BASE_STEP35_196B_A11B_CONFIG,
    num_gpus=64,
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=8,
    expert_model_parallel_size=8,
    global_batch_size=1024,
    micro_batch_size=1,
    moe_flex_dispatcher_backend="hybridep",
    cuda_graph_impl="transformer_engine",
    cuda_graph_scope=["moe_router", "moe_preprocess"],
)


STEP35_196B_A11B_PRETRAIN_CONFIG_GB300_FP8_CS_V1 = STEP35_196B_A11B_PRETRAIN_CONFIG_GB300_BF16_V1
STEP35_196B_A11B_PRETRAIN_CONFIG_GB300_FP8_MX_V1 = STEP35_196B_A11B_PRETRAIN_CONFIG_GB300_FP8_CS_V1
STEP35_196B_A11B_PRETRAIN_CONFIG_GB300_NVFP4_V1 = STEP35_196B_A11B_PRETRAIN_CONFIG_GB300_FP8_CS_V1


STEP35_196B_A11B_PRETRAIN_CONFIG_GB200_BF16_V1 = replace(
    BASE_STEP35_196B_A11B_CONFIG,
    num_gpus=64,
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=8,
    expert_model_parallel_size=8,
    global_batch_size=1024,
    micro_batch_size=1,
    moe_flex_dispatcher_backend="hybridep",
    cuda_graph_impl="transformer_engine",
    cuda_graph_scope=["attn", "moe_router", "moe_preprocess"],
)


STEP35_196B_A11B_PRETRAIN_CONFIG_GB200_FP8_CS_V1 = STEP35_196B_A11B_PRETRAIN_CONFIG_GB200_BF16_V1
STEP35_196B_A11B_PRETRAIN_CONFIG_GB200_FP8_MX_V1 = STEP35_196B_A11B_PRETRAIN_CONFIG_GB200_FP8_CS_V1
STEP35_196B_A11B_PRETRAIN_CONFIG_GB200_NVFP4_V1 = STEP35_196B_A11B_PRETRAIN_CONFIG_GB200_FP8_CS_V1


STEP35_196B_A11B_PRETRAIN_CONFIG_B300_BF16_V1 = replace(
    BASE_STEP35_196B_A11B_CONFIG,
    num_gpus=64,
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=8,
    expert_model_parallel_size=8,
    global_batch_size=1024,
    micro_batch_size=1,
    moe_a2a_overlap=False,
    moe_flex_dispatcher_backend="hybridep",
)


STEP35_196B_A11B_PRETRAIN_CONFIG_B300_FP8_CS_V1 = STEP35_196B_A11B_PRETRAIN_CONFIG_B300_BF16_V1
STEP35_196B_A11B_PRETRAIN_CONFIG_B300_FP8_MX_V1 = STEP35_196B_A11B_PRETRAIN_CONFIG_B300_FP8_CS_V1
STEP35_196B_A11B_PRETRAIN_CONFIG_B300_NVFP4_V1 = STEP35_196B_A11B_PRETRAIN_CONFIG_B300_FP8_CS_V1


STEP35_196B_A11B_PRETRAIN_CONFIG_B200_BF16_V1 = replace(
    BASE_STEP35_196B_A11B_CONFIG,
    num_gpus=64,
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=8,
    expert_model_parallel_size=8,
    global_batch_size=1024,
    micro_batch_size=1,
    moe_a2a_overlap=False,
    moe_flex_dispatcher_backend="hybridep",
)


STEP35_196B_A11B_PRETRAIN_CONFIG_B200_FP8_CS_V1 = STEP35_196B_A11B_PRETRAIN_CONFIG_B200_BF16_V1
STEP35_196B_A11B_PRETRAIN_CONFIG_B200_FP8_MX_V1 = STEP35_196B_A11B_PRETRAIN_CONFIG_B200_FP8_CS_V1
STEP35_196B_A11B_PRETRAIN_CONFIG_B200_NVFP4_V1 = STEP35_196B_A11B_PRETRAIN_CONFIG_B200_FP8_CS_V1


STEP35_196B_A11B_PRETRAIN_CONFIG_H100_BF16_V1 = replace(
    BASE_STEP35_196B_A11B_CONFIG,
    num_gpus=256,
    tensor_model_parallel_size=2,
    pipeline_model_parallel_size=8,
    expert_model_parallel_size=16,
    global_batch_size=2048,
    micro_batch_size=1,
    moe_a2a_overlap=True,
    moe_flex_dispatcher_backend="hybridep",
)


STEP35_196B_A11B_PRETRAIN_CONFIG_H100_FP8_CS_V1 = STEP35_196B_A11B_PRETRAIN_CONFIG_H100_BF16_V1


# =============================================================================
# Step-3.5 196B A11B presets - V2 (Blackwell num_gpus=256, EP=32, GBS=8192)
# =============================================================================

STEP35_196B_A11B_PRETRAIN_CONFIG_GB300_BF16_V2 = replace(
    STEP35_196B_A11B_PRETRAIN_CONFIG_GB300_BF16_V1,
    num_gpus=256,
    expert_model_parallel_size=32,
    global_batch_size=8192,
    cuda_graph_scope=["attn", "moe_router", "moe_preprocess"],
)


STEP35_196B_A11B_PRETRAIN_CONFIG_GB300_FP8_CS_V2 = STEP35_196B_A11B_PRETRAIN_CONFIG_GB300_BF16_V2
STEP35_196B_A11B_PRETRAIN_CONFIG_GB300_FP8_MX_V2 = replace(
    STEP35_196B_A11B_PRETRAIN_CONFIG_GB300_FP8_CS_V2,
    moe_a2a_overlap=True,
    cuda_graph_impl="full_iteration",
    cuda_graph_scope=[],
    cutedsl_fused_grouped_mlp=True,
    fp8_dot_product_attention=True,
)
STEP35_196B_A11B_PRETRAIN_CONFIG_GB300_NVFP4_V2 = STEP35_196B_A11B_PRETRAIN_CONFIG_GB300_FP8_CS_V2


STEP35_196B_A11B_PRETRAIN_CONFIG_GB200_BF16_V2 = replace(
    STEP35_196B_A11B_PRETRAIN_CONFIG_GB200_BF16_V1,
    num_gpus=256,
    expert_model_parallel_size=32,
    global_batch_size=8192,
)


STEP35_196B_A11B_PRETRAIN_CONFIG_GB200_FP8_CS_V2 = STEP35_196B_A11B_PRETRAIN_CONFIG_GB200_BF16_V2
STEP35_196B_A11B_PRETRAIN_CONFIG_GB200_FP8_MX_V2 = replace(
    STEP35_196B_A11B_PRETRAIN_CONFIG_GB200_FP8_CS_V2,
    moe_a2a_overlap=True,
    cuda_graph_impl="full_iteration",
    cuda_graph_scope=[],
    cutedsl_fused_grouped_mlp=True,
    fp8_dot_product_attention=True,
    mtp_num_layers=1,
    # Nsys- and sweep-validated 256-GPU topology: PP4/VPP3/EP32. Decoder
    # counts per physical PP rank are [12, 13, 12, 8], with the extra PP1
    # decoder in its first virtual chunk and dense MTP isolated in the final
    # chunk. This replaces the fixed first/last split in the recipe builder.
    pipeline_model_parallel_size=4,
    expert_model_parallel_size=32,
    virtual_pipeline_model_parallel_size=3,
    pp_layout="Et*3|t*4|(t*4|)*9t*2mL",
)
STEP35_196B_A11B_PRETRAIN_CONFIG_GB200_NVFP4_V2 = STEP35_196B_A11B_PRETRAIN_CONFIG_GB200_FP8_CS_V2


STEP35_196B_A11B_PRETRAIN_CONFIG_B300_BF16_V2 = replace(
    STEP35_196B_A11B_PRETRAIN_CONFIG_B300_BF16_V1,
    num_gpus=256,
    expert_model_parallel_size=32,
    global_batch_size=8192,
)


STEP35_196B_A11B_PRETRAIN_CONFIG_B300_FP8_CS_V2 = STEP35_196B_A11B_PRETRAIN_CONFIG_B300_BF16_V2
STEP35_196B_A11B_PRETRAIN_CONFIG_B300_FP8_MX_V2 = STEP35_196B_A11B_PRETRAIN_CONFIG_B300_FP8_CS_V2
STEP35_196B_A11B_PRETRAIN_CONFIG_B300_NVFP4_V2 = STEP35_196B_A11B_PRETRAIN_CONFIG_B300_FP8_CS_V2


STEP35_196B_A11B_PRETRAIN_CONFIG_B200_BF16_V2 = replace(
    STEP35_196B_A11B_PRETRAIN_CONFIG_B200_BF16_V1,
    num_gpus=256,
    expert_model_parallel_size=32,
    global_batch_size=8192,
)


STEP35_196B_A11B_PRETRAIN_CONFIG_B200_FP8_CS_V2 = STEP35_196B_A11B_PRETRAIN_CONFIG_B200_BF16_V2
STEP35_196B_A11B_PRETRAIN_CONFIG_B200_FP8_MX_V2 = STEP35_196B_A11B_PRETRAIN_CONFIG_B200_FP8_CS_V2
STEP35_196B_A11B_PRETRAIN_CONFIG_B200_NVFP4_V2 = STEP35_196B_A11B_PRETRAIN_CONFIG_B200_FP8_CS_V2


STEP35_196B_A11B_PRETRAIN_CONFIG_H100_BF16_V2 = replace(
    STEP35_196B_A11B_PRETRAIN_CONFIG_H100_BF16_V1,
    expert_model_parallel_size=32,
    global_batch_size=8192,
)


STEP35_196B_A11B_PRETRAIN_CONFIG_H100_FP8_CS_V2 = STEP35_196B_A11B_PRETRAIN_CONFIG_H100_BF16_V2


# =============================================================================
# Step-3.5 196B A11B presets - Large Scale Proxy (GBS=512)
# =============================================================================

STEP35_196B_A11B_PRETRAIN_CONFIG_GB300_FP8_MX_LARGE_SCALE = replace(
    STEP35_196B_A11B_PRETRAIN_CONFIG_GB300_FP8_MX_V2,
    global_batch_size=512,
)


STEP35_196B_A11B_PRETRAIN_CONFIG_GB200_FP8_MX_LARGE_SCALE = replace(
    STEP35_196B_A11B_PRETRAIN_CONFIG_GB200_FP8_MX_V2,
    global_batch_size=512,
)


STEP35_196B_A11B_PRETRAIN_CONFIG_B300_FP8_MX_LARGE_SCALE = replace(
    STEP35_196B_A11B_PRETRAIN_CONFIG_B300_FP8_MX_V2,
    global_batch_size=512,
)


STEP35_196B_A11B_PRETRAIN_CONFIG_B200_FP8_MX_LARGE_SCALE = replace(
    STEP35_196B_A11B_PRETRAIN_CONFIG_B200_FP8_MX_V2,
    global_batch_size=512,
)


STEP35_196B_A11B_PRETRAIN_CONFIG_H100_FP8_CS_LARGE_SCALE = replace(
    STEP35_196B_A11B_PRETRAIN_CONFIG_H100_FP8_CS_V2,
    global_batch_size=512,
)


__all__ = [
    # V1
    "STEP35_196B_A11B_PRETRAIN_CONFIG_GB300_BF16_V1",
    "STEP35_196B_A11B_PRETRAIN_CONFIG_GB300_FP8_CS_V1",
    "STEP35_196B_A11B_PRETRAIN_CONFIG_GB300_FP8_MX_V1",
    "STEP35_196B_A11B_PRETRAIN_CONFIG_GB300_NVFP4_V1",
    "STEP35_196B_A11B_PRETRAIN_CONFIG_GB200_BF16_V1",
    "STEP35_196B_A11B_PRETRAIN_CONFIG_GB200_FP8_CS_V1",
    "STEP35_196B_A11B_PRETRAIN_CONFIG_GB200_FP8_MX_V1",
    "STEP35_196B_A11B_PRETRAIN_CONFIG_GB200_NVFP4_V1",
    "STEP35_196B_A11B_PRETRAIN_CONFIG_B300_BF16_V1",
    "STEP35_196B_A11B_PRETRAIN_CONFIG_B300_FP8_CS_V1",
    "STEP35_196B_A11B_PRETRAIN_CONFIG_B300_FP8_MX_V1",
    "STEP35_196B_A11B_PRETRAIN_CONFIG_B300_NVFP4_V1",
    "STEP35_196B_A11B_PRETRAIN_CONFIG_B200_BF16_V1",
    "STEP35_196B_A11B_PRETRAIN_CONFIG_B200_FP8_CS_V1",
    "STEP35_196B_A11B_PRETRAIN_CONFIG_B200_FP8_MX_V1",
    "STEP35_196B_A11B_PRETRAIN_CONFIG_B200_NVFP4_V1",
    "STEP35_196B_A11B_PRETRAIN_CONFIG_H100_BF16_V1",
    "STEP35_196B_A11B_PRETRAIN_CONFIG_H100_FP8_CS_V1",
    # V2
    "STEP35_196B_A11B_PRETRAIN_CONFIG_GB300_BF16_V2",
    "STEP35_196B_A11B_PRETRAIN_CONFIG_GB300_FP8_CS_V2",
    "STEP35_196B_A11B_PRETRAIN_CONFIG_GB300_FP8_MX_V2",
    "STEP35_196B_A11B_PRETRAIN_CONFIG_GB300_NVFP4_V2",
    "STEP35_196B_A11B_PRETRAIN_CONFIG_GB200_BF16_V2",
    "STEP35_196B_A11B_PRETRAIN_CONFIG_GB200_FP8_CS_V2",
    "STEP35_196B_A11B_PRETRAIN_CONFIG_GB200_FP8_MX_V2",
    "STEP35_196B_A11B_PRETRAIN_CONFIG_GB200_NVFP4_V2",
    "STEP35_196B_A11B_PRETRAIN_CONFIG_B300_BF16_V2",
    "STEP35_196B_A11B_PRETRAIN_CONFIG_B300_FP8_CS_V2",
    "STEP35_196B_A11B_PRETRAIN_CONFIG_B300_FP8_MX_V2",
    "STEP35_196B_A11B_PRETRAIN_CONFIG_B300_NVFP4_V2",
    "STEP35_196B_A11B_PRETRAIN_CONFIG_B200_BF16_V2",
    "STEP35_196B_A11B_PRETRAIN_CONFIG_B200_FP8_CS_V2",
    "STEP35_196B_A11B_PRETRAIN_CONFIG_B200_FP8_MX_V2",
    "STEP35_196B_A11B_PRETRAIN_CONFIG_B200_NVFP4_V2",
    "STEP35_196B_A11B_PRETRAIN_CONFIG_H100_BF16_V2",
    "STEP35_196B_A11B_PRETRAIN_CONFIG_H100_FP8_CS_V2",
    # Large Scale Proxy
    "STEP35_196B_A11B_PRETRAIN_CONFIG_GB300_FP8_MX_LARGE_SCALE",
    "STEP35_196B_A11B_PRETRAIN_CONFIG_GB200_FP8_MX_LARGE_SCALE",
    "STEP35_196B_A11B_PRETRAIN_CONFIG_B300_FP8_MX_LARGE_SCALE",
    "STEP35_196B_A11B_PRETRAIN_CONFIG_B200_FP8_MX_LARGE_SCALE",
    "STEP35_196B_A11B_PRETRAIN_CONFIG_H100_FP8_CS_LARGE_SCALE",
]
