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

"""Intermediate-activation dump helper for Step-3.7 mbridge inference.

Byte-for-byte compatible with the vLLM reference dumper
(``vllm/model_executor/models/step3_vl.py::_dump_tensor``) so mbridge and
vLLM dumps can be compared tag-for-tag:

* Same env switches: ``STEP3P7_DUMP_VIT`` / ``STEP3P7_DUMP_LLM`` /
  ``STEP3P7_DUMP_DIR`` / ``STEP3P7_DUMP_LLM_MAX_CALLS`` / ``STEP3P7_DUMP_TRIGGER``.
* Same filename convention: ``{tag}_call{idx}.pt`` holding ``tensor.detach().cpu()``.
* Same TP-rank-0-only gating and arm-file (trigger) gating.

Vision tags (``STEP3P7_DUMP_VIT``): ``vit_features``, ``vit_downsampled``,
``vit_projector_out``. LLM tags (``STEP3P7_DUMP_LLM``): ``llm_layer{idx}``.
"""

from __future__ import annotations

import os

import torch

# Per-tag call counter (reset when the arm-file token changes) and the last
# seen arm-file signature — mirrors the vLLM globals of the same purpose.
_DUMP_STEP: dict[str, int] = {}
_ARM_SIG: dict[str, str] = {"v": ""}


def _is_tp_rank0() -> bool:
    """Dump only on tensor-parallel rank 0 (matches vLLM)."""
    try:
        from megatron.core import parallel_state

        if parallel_state.is_initialized():
            return parallel_state.get_tensor_model_parallel_rank() == 0
    except Exception:
        pass
    return True


def dump_tensor(
    tag: str,
    tensor: torch.Tensor,
    *,
    enable_env: str = "STEP3P7_DUMP_VIT",
    max_calls: int | None = None,
) -> None:
    """Save ``tensor`` under ``{STEP3P7_DUMP_DIR}/{tag}_call{idx}.pt``.

    No-op unless ``enable_env`` is ``"1"``. When ``STEP3P7_DUMP_TRIGGER`` is
    set, only dumps while that file exists; a changed token resets the
    per-tag counters (so each armed query captures a fresh ``call0``).
    """
    if os.environ.get(enable_env, "0") != "1":
        return
    if not _is_tp_rank0():
        return

    trigger = os.environ.get("STEP3P7_DUMP_TRIGGER")
    if trigger:
        try:
            with open(trigger) as _f:
                sig = _f.read()
        except OSError:
            return
        if _ARM_SIG["v"] != sig:
            _ARM_SIG["v"] = sig
            _DUMP_STEP.clear()

    idx = _DUMP_STEP.get(tag, 0)
    if max_calls is not None and idx >= max_calls:
        return
    _DUMP_STEP[tag] = idx + 1

    t = tensor.detach()
    try:
        print(
            f"[DUMP] {tag} call#{idx} shape={tuple(t.shape)} "
            f"dtype={t.dtype} device={t.device} "
            f"mean={t.float().mean().item():.6f} std={t.float().std().item():.6f}",
            flush=True,
        )
    except Exception:
        pass
    dump_dir = os.environ.get("STEP3P7_DUMP_DIR", "/tmp")
    os.makedirs(dump_dir, exist_ok=True)
    torch.save(t.cpu(), os.path.join(dump_dir, f"{tag}_call{idx}.pt"))


def _max_llm_calls() -> int | None:
    raw = os.environ.get("STEP3P7_DUMP_LLM_MAX_CALLS", "1")
    return int(raw) if raw else None


__all__ = ["dump_tensor", "_max_llm_calls"]
