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

"""Performance-optimized Step-3.5-Flash (196B / A11B) pretrain configs.

Mirrors ``configs/qwen/qwen3_llm_pretrain.py``. ``run_script.py`` resolves
``step35_196b_a11b_pretrain_config_<gpu>`` via ``get_perf_optimized_recipe``
(``configs.stepfun``) and calls it with ``precision`` / ``mock`` /
``config_variant``. Each builder:

  1. fetches the ``STEP35_196B_A11B_PRETRAIN_CONFIG_<GPU>_<DTYPE>_<VER>``
     workload preset (parallelism / batch / MoE-CG knobs),
  2. loads the library recipe ``step35_196b_a11b_pretrain_config`` (the real
     Step-3.5 model/optimizer/data config),
  3. applies precision + common perf settings + the workload preset.

Launch example:
    --model_family_name stepfun --model_recipe_name step35_196b_a11b \\
    --gpu gb200 --compute_dtype fp8_mx --config_variant v2
"""

import logging

from utils.overrides import set_workload_base_configs
from utils.precision import get_precision_config
from utils.utils import get_workload_base_config

from megatron.bridge.recipes.stepfun.step35 import (
    step35_196b_a11b_pretrain_config as pretrain_config,
)
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.utils.cuda_graph import is_full_iteration_cuda_graph


logger = logging.getLogger(__name__)

# Step-3.5-Flash has 45 decoder layers. PP=8 does not divide 45, so a fixed
# uneven split is required (6 + 6*6 + 3 = 45). This matches the validated
# Step-3.7 sbatch (num_layers_in_first_pipeline_stage=6 /
# num_layers_in_last_pipeline_stage=3). All presets below use PP=8. Presets that
# instead carry an explicit VPP pp_layout (e.g. GB200 FP8_MX V2) override this
# fixed split in _build (the two are mutually exclusive).
_NUM_LAYERS_IN_FIRST_PP_STAGE = 6
_NUM_LAYERS_IN_LAST_PP_STAGE = 3


def set_step35_common_configs(cfg: ConfigContainer) -> None:
    """Set common performance configurations for all Step-3.5 configs."""
    cfg.model.seq_length = 4096
    cfg.dataset.sequence_length = 4096

    # MoE kernels: grouped GEMM + permute fusion are Step-3.5 safe; router
    # fusion is left at the recipe default (Step-3.5 uses a sigmoid router with
    # bias + fp32 gate, so it is not force-enabled here).
    cfg.model.moe_permute_fusion = True
    cfg.model.moe_grouped_gemm = True

    # Selective recompute is enough for the MoE activations at seq_length=4096;
    # override the recipe's conservative "full" recompute for throughput.
    cfg.model.recompute_granularity = "selective"
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.recompute_modules = ["core_attn"]

    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False

    # Dropless MoE requires forced load balancing for the flex dispatcher.
    cfg.model.moe_router_force_load_balancing = True

    # 45-layer uneven pipeline split (PP=8). VPP stays None (see module docstring).
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.num_layers_in_first_pipeline_stage = _NUM_LAYERS_IN_FIRST_PP_STAGE
    cfg.model.num_layers_in_last_pipeline_stage = _NUM_LAYERS_IN_LAST_PP_STAGE


def set_full_iter_cg_configs(cfg: ConfigContainer) -> None:
    """Apply MoE defaults required by full-iteration CUDA graph capture.

    Dropless MoE produces variable-shaped per-expert tensors that CUDA graphs
    cannot capture; pad to a fixed capacity and use paged stashing to recover
    memory. Gate callers on ``is_full_iteration_cuda_graph(cfg.model)``.
    """
    cfg.model.offload_modules = []
    cfg.model.moe_pad_experts_for_cuda_graph_inference = True
    cfg.model.moe_paged_stash = True
    cfg.model.moe_expert_rank_capacity_factor = 1.5
    cfg.model.moe_paged_stash_buffer_size_factor_cuda = 1.2
    cfg.model.moe_paged_stash_buffer_size_factor_cpu = 1.0


def _build(gpu: str, precision: str, config_variant: str, *, tp_comm_overlap: bool) -> ConfigContainer:
    """Shared builder body for every Step-3.5 per-GPU config."""
    base_cfg = get_workload_base_config(
        model_family_name="stepfun",
        model_recipe_name="step35_196b_a11b",
        gpu=gpu,
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )
    precision_config = get_precision_config(precision)

    cfg = pretrain_config()
    cfg.mixed_precision = precision_config
    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=tp_comm_overlap)
    cfg.model.moe_flex_dispatcher_backend = base_cfg.moe_flex_dispatcher_backend
    cfg.model.moe_token_dispatcher_type = "flex"

    set_step35_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)
    # A preset may carry an explicit VPP pp_layout string (e.g. GB200 FP8_MX V2).
    # That string fully specifies the per-virtual-stage layer split, so it replaces
    # the fixed uneven first/last split from set_step35_common_configs -- the two are
    # mutually exclusive. VPP itself is applied by set_workload_base_configs.
    if base_cfg.pp_layout:
        cfg.model.pipeline_model_parallel_layout = base_cfg.pp_layout
        cfg.model.num_layers_in_first_pipeline_stage = None
        cfg.model.num_layers_in_last_pipeline_stage = None
    if precision == "fp8_mx" and is_full_iteration_cuda_graph(cfg.model):
        set_full_iter_cg_configs(cfg)

    return cfg


def step35_196b_a11b_pretrain_config_gb300(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """GB300, baseline config."""
    return _build("gb300", precision, config_variant, tp_comm_overlap=True)


def step35_196b_a11b_pretrain_config_gb200(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """GB200, baseline config."""
    return _build("gb200", precision, config_variant, tp_comm_overlap=True)


def step35_196b_a11b_pretrain_config_b300(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """B300, baseline config."""
    return _build("b300", precision, config_variant, tp_comm_overlap=True)


def step35_196b_a11b_pretrain_config_b200(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """B200, baseline config."""
    return _build("b200", precision, config_variant, tp_comm_overlap=True)


def step35_196b_a11b_pretrain_config_h100(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """H100, baseline config."""
    return _build("h100", precision, config_variant, tp_comm_overlap=False)
