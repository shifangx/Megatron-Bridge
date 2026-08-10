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
# validated Step-3.7 sbatch layout. VPP stays None on this path: 45 is not
# divisible by PP*VPP, so any VPP needs the explicit layout below instead.
_NUM_LAYERS_IN_FIRST_PP_STAGE = 6
_NUM_LAYERS_IN_LAST_PP_STAGE = 3

# Explicit PP=4 / VPP=3 layout used by the full-iteration MXFP8 path, which needs
# a non-None VPP for EP A2A overlap. Expands to 12 stages (VPP = 12 // PP = 3):
#   Ettt | tttt x10 | ttL   ->  3 + 40 + 2 = 45 decoder layers, no MTP layer.
# EP A2A overlap does not support Step-3.5's dense MTP layer, so MTP is disabled
# on this path (see ``_enable_full_iteration_mxfp8``) and the layout carries no
# ``m`` entry.
_STEP35_PP4_VPP3_LAYOUT = "Et*3|(t*4|)*10t*2L"
_STEP35_PP4_VPP3_PP_SIZE = 4
_STEP35_PP4_VPP3_VPP_SIZE = 3

# Explicit PP=4 (no VPP) layout used by the per-op TE-graph recipes, which do not
# enable EP A2A overlap and so do not need the non-None VPP the full-iteration
# MXFP8 path forces. Four stages, one per PP rank:
#   Et*12 | t*12 | t*12 | t*9L   ->  12 + 12 + 12 + 9 = 45 decoder layers, no MTP.
_STEP35_PP4_LAYOUT = "Et*12|t*12|t*12|t*9L"
_STEP35_PP4_PP_SIZE = 4

# Step-3.5 per-layer overrides that ``Step35TransformerLayer`` indexes by the
# *global* 0-indexed layer id. MTP layers sit right after the decoder layers
# (indices ``num_layers .. num_layers + mtp_num_layers - 1``), so each of these
# lists is sized ``num_layers + mtp_num_layers`` when it comes from the HF
# config and must be resized in lockstep with ``mtp_num_layers``.
_STEP35_PER_LAYER_FIELDS = (
    "rotary_base_per_layer",
    "rotary_percents",
    "layer_types",
    "swiglu_limits",
    "swiglu_limits_shared",
)


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


def _use_pp4_vpp3_layout(cfg: ConfigContainer) -> None:
    """Replace the default uneven PP=8 split with the explicit PP=4 / VPP=3 layout.

    EP A2A overlap requires a non-None ``virtual_pipeline_model_parallel_size``
    at PP>1, which the 45-layer uneven split cannot provide. Megatron-Core also
    rejects an explicit layout combined with any of the coarse split knobs, so
    ``num_layers_in_{first,last}_pipeline_stage`` are cleared here.
    """
    cfg.model.pipeline_model_parallel_size = _STEP35_PP4_VPP3_PP_SIZE
    cfg.model.virtual_pipeline_model_parallel_size = _STEP35_PP4_VPP3_VPP_SIZE
    cfg.model.pipeline_model_parallel_layout = _STEP35_PP4_VPP3_LAYOUT
    cfg.model.num_layers_in_first_pipeline_stage = None
    cfg.model.num_layers_in_last_pipeline_stage = None


def _use_pp4_layout(cfg: ConfigContainer) -> None:
    """Replace the default uneven PP=8 split with the explicit PP=4 (no VPP) layout.

    Unlike ``_use_pp4_vpp3_layout`` this keeps
    ``virtual_pipeline_model_parallel_size`` at None: the per-op TE-graph recipes
    do not enable EP A2A overlap, so they do not need a non-None VPP. Megatron-Core
    rejects an explicit layout combined with the coarse split knobs, so
    ``num_layers_in_{first,last}_pipeline_stage`` are cleared here.
    """
    cfg.model.pipeline_model_parallel_size = _STEP35_PP4_PP_SIZE
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.pipeline_model_parallel_layout = _STEP35_PP4_LAYOUT
    cfg.model.num_layers_in_first_pipeline_stage = None
    cfg.model.num_layers_in_last_pipeline_stage = None


def _set_mtp_num_layers(cfg: ConfigContainer, mtp_num_layers: int | None) -> None:
    """Set ``mtp_num_layers`` and shrink the per-layer override lists to match.

    ``mtp_num_layers`` may be ``None`` to disable MTP entirely; it is treated as
    depth 0 for the list-trimming math but written back verbatim so the model
    reports MTP as off.

    ``Step35Bridge.provider_bridge`` fills ``rotary_base_per_layer`` (and the
    other lists in ``_STEP35_PER_LAYER_FIELDS``) straight from the HF config, so
    they are sized for the checkpoint's stock MTP depth. Megatron-Core asserts
    ``len(rotary_base_per_layer) == num_layers + mtp_num_layers``, so lowering
    ``mtp_num_layers`` without trimming these lists fails config validation.

    Only the trailing MTP entries are dropped; the ``num_layers`` decoder
    entries and the first ``mtp_num_layers`` MTP entries keep their global layer
    indices, which is what ``Step35TransformerLayer`` looks them up by.
    """
    old_mtp_num_layers = cfg.model.mtp_num_layers or 0
    new_mtp_num_layers = mtp_num_layers or 0
    if new_mtp_num_layers > old_mtp_num_layers:
        raise ValueError(
            f"Cannot raise mtp_num_layers from {old_mtp_num_layers} to {mtp_num_layers}: "
            "the per-layer override lists come from the HF config and have no entries to extend with."
        )

    num_layers = cfg.model.num_layers
    old_length = num_layers + old_mtp_num_layers
    new_length = num_layers + new_mtp_num_layers

    for field_name in _STEP35_PER_LAYER_FIELDS:
        values = getattr(cfg.model, field_name, None)
        # Leave anything that is not a full-depth per-layer list alone: unset
        # fields, and lists the provider sized by some other rule.
        if values is None or len(values) != old_length:
            continue
        setattr(cfg.model, field_name, list(values[:new_length]))

    cfg.model.mtp_num_layers = mtp_num_layers


def _enable_full_iteration_mxfp8(cfg: ConfigContainer) -> None:
    """Switch a Step-3.5 recipe to full-iteration CUDA graph capture.

    Dropless MoE produces variable-shaped per-expert tensors that CUDA graphs
    cannot capture, so pad to a fixed capacity and use paged stashing to recover
    the memory the padding costs.

    MTP is disabled here. ``CommOverlapConfig.setup`` asserts
    ``mtp_num_layers in (None, 0, 1)`` when EP A2A overlap is on, and while that
    admits a single MTP layer, Step-3.5's one MTP layer is a *dense* layer, which
    EP A2A overlap does not support. Setting ``mtp_num_layers`` to ``None`` sheds
    that layer and works around the known limitation. This is a deliberate
    deviation from the model's reference shape, applied only to the
    full-iteration MXFP8 recipes; the TE-graph recipes keep all 3 MTP layers.
    """
    _set_mtp_num_layers(cfg, None)
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
