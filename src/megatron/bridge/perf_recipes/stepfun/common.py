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
# ruff: noqa: F401
"""Common helpers for stepfun performance recipes."""

from megatron.bridge.perf_recipes._common import (
    _benchmark_common,
    _enable_overlap_param_gather_with_optimizer_step,
    _perf_precision,
)
from megatron.bridge.recipes.stepfun.step35 import step35_196b_a11b_pretrain_config
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import ConfigContainer


# Step-3.5-Flash has 45 decoder layers (3 dense + 42 MoE). PP=8 does not divide
# 45, so a fixed uneven split is required (6 + 6*6 + 3 = 45). This matches the
# validated Step-3.7 sbatch layout. VPP stays None everywhere: 45 is not
# divisible by PP*VPP, and enabling VPP would require an explicit pp_layout.
_NUM_LAYERS_IN_FIRST_PP_STAGE = 6
_NUM_LAYERS_IN_LAST_PP_STAGE = 3


def _step35_common(cfg: ConfigContainer) -> None:
    """Apply the model-level settings shared by every Step-3.5 perf recipe.

    Call this before ``_benchmark_common`` so that benchmark-mode defaults keep
    the final word on training length, logging, and CUDA-graph RNG wiring.
    """
    cfg.model.seq_length = 4096
    cfg.dataset.seq_length = 4096

    # MoE kernels: grouped GEMM + permute fusion are Step-3.5 safe. Router
    # fusion is left at the recipe default (Step-3.5 uses a sigmoid router with
    # bias + fp32 gate), so it is not force-enabled here.
    cfg.model.moe_permute_fusion = True
    cfg.model.moe_grouped_gemm = True

    # Selective recompute is enough for the MoE activations at seq_length=4096;
    # override the library recipe's conservative "full" recompute for throughput.
    cfg.model.recompute_granularity = "selective"
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.recompute_modules = ["core_attn"]

    # Dropless MoE requires forced load balancing for the flex dispatcher.
    cfg.model.moe_router_force_load_balancing = True

    # 45-layer uneven pipeline split (PP=8).
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.num_layers_in_first_pipeline_stage = _NUM_LAYERS_IN_FIRST_PP_STAGE
    cfg.model.num_layers_in_last_pipeline_stage = _NUM_LAYERS_IN_LAST_PP_STAGE

    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_shared_expert_overlap = False


def _enable_full_iteration_mxfp8(cfg: ConfigContainer) -> None:
    """Switch a Step-3.5 recipe to full-iteration CUDA graph capture.

    Dropless MoE produces variable-shaped per-expert tensors that CUDA graphs
    cannot capture, so pad to a fixed capacity and use paged stashing to recover
    the memory the padding costs.
    """
    cfg.model.cuda_graph_impl = "full_iteration"
    cfg.model.cuda_graph_scope = []
    cfg.rng.te_rng_tracker = True
    cfg.model.use_te_rng_tracker = True

    cfg.model.offload_modules = []
    cfg.model.moe_pad_experts_for_cuda_graph_inference = True
    cfg.model.moe_paged_stash = True
    cfg.model.moe_expert_rank_capacity_factor = 1.5
    cfg.model.moe_paged_stash_buffer_size_factor_cuda = 1.2
    cfg.model.moe_paged_stash_buffer_size_factor_cpu = 1.0

    cfg.model.high_priority_a2a_comm_stream = True
    cfg.model.use_transformer_engine_op_fuser = True
    cfg.model.moe_mlp_glu_interleave_size = 32
    cfg.model.moe_hybridep_num_sms_preprocessing = 32

    cfg.mixed_precision.fp8_dot_product_attention = True
    cfg.comm_overlap = CommOverlapConfig(
        tp_comm_overlap=True,
        overlap_moe_expert_parallel_comm=True,
        delay_wgrad_compute=True,
    )
