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

"""Self-contained mock dataset provider for Step3.7 SFT — **BSHD** layout.

This mirrors the role of :class:`MockVLMConversationProvider` (used by the
Qwen3-VL recipes) but targets Step3.7's native model contract:
``Step37Model.forward`` consumes ``list[ImageForInsert]`` directly and can
run in the standard **unpacked BSHD** attention layout (``[B, S]`` tokens,
``packed_seq_params=None``) — as opposed to the packed *THD* pipeline used by
:class:`Step37Flickr8kSFTDataProvider`.

Unlike the Flickr8k provider, nothing here is downloaded: every sample is
synthesized on the fly. Each ``__getitem__`` builds one fixed-length token row
with ``num_images`` image spans (``<im_start>`` + ``<im_patch>`` × 169 +
``<im_end>``) followed by random prompt / response tokens, plus a random image
pixel tensor ``[num_images, 3, image_size, image_size]``. The ``collate_fn``
stacks ``micro_batch_size`` rows into a ``[B, S]`` batch and merges the per-row
pixels into a single :class:`ImageForInsert` in row-major order (matching the
``<im_start>`` scan order in :meth:`ImageInsertEmbedding.insert_features`).

Pair this provider with ``--step_func step37_mock_step`` (the BSHD forward
step), which passes the batch straight to ``Step37Model.forward`` with
``packed_seq_params=None``.

Special-token ids and ``vocab_size`` are resolved from the tokenizer at
``tokenizer_path`` when left at their ``-1`` sentinel; pass them explicitly
(and leave ``tokenizer_path=None``) to build a fully offline mock — e.g. in a
unit test with no HF snapshot available.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Tuple

import numpy
import torch
from megatron.core.pipeline_parallel.utils import is_pp_first_stage, is_pp_last_stage
from megatron.core.process_groups_config import ProcessGroupCollection

from megatron.bridge.data.vlm_datasets.step37_flickr8k.template import (
    IMAGE_END_TOKEN,
    IMAGE_START_TOKEN,
    IMAGE_TOKEN,
    IMAGE_TOKEN_COUNT,
)
from megatron.bridge.models.stepfun.modelling_step37.image_insert_embedding import ImageForInsert
from megatron.bridge.training.config import DatasetBuildContext, DatasetProvider


class _Step37MockDataset(torch.utils.data.Dataset):
    """Map-style dataset that synthesizes Step3.7 BSHD SFT rows on the fly.

    ``__getitem__`` returns one sample dict whose keys depend on which
    **pipeline stage** owns the current rank (see ``load_inputs`` /
    ``load_labels``). At most::

        {
            "input_ids": LongTensor[S],   # tokens[:-1]  (first PP stage)
            "labels":    LongTensor[S],   # tokens[1:]   (last PP stage)
            "loss_mask": FloatTensor[S],  # 1.0 on the response span only (last PP stage)
            "image":     FloatTensor[num_images, 3, H, W],  # (first PP stage)
        }

    Under pipeline parallelism only the first stage runs the embedding + vision
    fusion (so it needs ``input_ids`` + ``image``) and only the last stage
    computes the loss (so it needs ``labels`` + ``loss_mask``). Middle stages
    operate on piped hidden states alone and therefore synthesize **nothing** —
    this in particular skips the expensive ``[num_images, 3, H, W]`` random
    image tensor on every rank that would only throw it away. See
    ``step37_mock_step.forward_step`` for the consuming side.

    ``collate_fn`` mirrors this: it only stacks the keys that are present, so a
    middle-stage batch is an empty payload (with an empty ``images`` list) that
    still advances the sampler in lockstep with the other stages.
    """

    def __init__(
        self,
        *,
        length: int,
        seq_length: int,
        num_images: int,
        image_size: int,
        image_token_count: int,
        img_start_token_id: int,
        img_patch_token_id: int,
        img_end_token_id: int,
        pad_token_id: int,
        vocab_size: int,
        prompt_length: int,
        response_length: int,
        random_seed: int,
        load_inputs: bool = True,
        load_labels: bool = True,
    ) -> None:
        super().__init__()
        self._length = int(max(0, length))
        # PP-stage gating: first stage needs input_ids + image; last stage needs
        # labels + loss_mask; middle stages need neither.
        self.load_inputs = bool(load_inputs)
        self.load_labels = bool(load_labels)
        self.seq_length = int(seq_length)
        self.num_images = int(num_images)
        self.image_size = int(image_size)
        self.image_token_count = int(image_token_count)
        self.img_start_token_id = int(img_start_token_id)
        self.img_patch_token_id = int(img_patch_token_id)
        self.img_end_token_id = int(img_end_token_id)
        self.pad_token_id = int(pad_token_id)
        self.vocab_size = int(vocab_size)
        self.prompt_length = int(prompt_length)
        self.response_length = int(response_length)
        self.random_seed = int(random_seed)

        # One image span is ``<im_start>`` + N ``<im_patch>`` + ``<im_end>``.
        image_block_len = self.num_images * (self.image_token_count + 2)
        content_len = image_block_len + self.prompt_length + self.response_length
        # ``+ 1`` because the row is shifted by one (input=[:-1], labels=[1:]).
        if content_len > self.seq_length + 1:
            raise ValueError(
                f"Mock content length {content_len} exceeds seq_length+1={self.seq_length + 1}. "
                "Lower num_images / prompt_length / response_length or raise seq_length."
            )
        self._image_block_len = image_block_len

    def __len__(self) -> int:
        return self._length

    def _text_tokens(self, rng: numpy.random.Generator, n: int) -> list[int]:
        """Random text token ids in ``[1, vocab_size)`` avoiding special ids."""
        if n <= 0:
            return []
        specials = {
            self.img_start_token_id,
            self.img_patch_token_id,
            self.img_end_token_id,
            self.pad_token_id,
        }
        tokens = rng.integers(low=1, high=max(2, self.vocab_size), size=n).tolist()
        return [t if t not in specials else self.pad_token_id for t in tokens]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self._length == 0:
            raise IndexError("Empty dataset")
        # Deterministic per-index streams keep every DP rank / worker in sync.
        rng = numpy.random.default_rng(seed=self.random_seed + int(idx))

        sample: dict[str, Any] = {}

        # The token stream is only synthesized when this PP stage consumes it —
        # the first stage reads ``input_ids`` and the last stage reads
        # ``labels`` / ``loss_mask`` (both are shifts of the same stream, so the
        # RNG draws stay identical and first/last stay aligned). Middle stages
        # skip it entirely. The image draw comes *after* the text draws, so a
        # stage that skips the image never perturbs the token stream.
        if self.load_inputs or self.load_labels:
            full_len = self.seq_length + 1
            tokens: list[int] = []
            loss: list[float] = []

            # 1) Image spans (never contribute to the loss).
            for _ in range(self.num_images):
                tokens.append(self.img_start_token_id)
                tokens.extend([self.img_patch_token_id] * self.image_token_count)
                tokens.append(self.img_end_token_id)
            loss.extend([0.0] * self._image_block_len)

            # 2) Prompt text (masked out of the loss).
            prompt = self._text_tokens(rng, self.prompt_length)
            tokens.extend(prompt)
            loss.extend([0.0] * len(prompt))

            # 3) Response text (the only supervised span).
            response = self._text_tokens(rng, self.response_length)
            tokens.extend(response)
            loss.extend([1.0] * len(response))

            # 4) Right-pad to the shifted length.
            pad = full_len - len(tokens)
            if pad > 0:
                tokens.extend([self.pad_token_id] * pad)
                loss.extend([0.0] * pad)

            full_tokens = torch.tensor(tokens[:full_len], dtype=torch.long)
            full_loss = torch.tensor(loss[:full_len], dtype=torch.float32)

            if self.load_inputs:
                sample["input_ids"] = full_tokens[:-1].contiguous()
            if self.load_labels:
                sample["labels"] = full_tokens[1:].contiguous()
                sample["loss_mask"] = full_loss[1:].contiguous()

        # Vision pixels are consumed only by the first PP stage (embedding +
        # vision fusion). Never materialize the heavy tensor on stages that
        # would discard it.
        if self.load_inputs and self.num_images > 0:
            sample["image"] = torch.from_numpy(
                rng.standard_normal(
                    size=(self.num_images, 3, self.image_size, self.image_size),
                    dtype=numpy.float32,
                )
            )

        return sample

    def collate_fn(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Stack ``[B, S]`` tensors and merge per-row pixels into one insert.

        Only the keys the current PP stage synthesized are stacked, so a
        middle-stage batch carries just an empty ``images`` list. The consuming
        ``forward_step`` guards on ``"input_ids" in batch`` / ``"labels" in
        batch``, so omitting a key on the stages that do not need it is safe.
        """
        out: dict[str, Any] = {}

        if batch and all("input_ids" in b for b in batch):
            out["input_ids"] = torch.stack([b["input_ids"] for b in batch], dim=0)
        if batch and all("labels" in b for b in batch):
            out["labels"] = torch.stack([b["labels"] for b in batch], dim=0)
            out["loss_mask"] = torch.stack([b["loss_mask"] for b in batch], dim=0)

        images: list[ImageForInsert] = []
        pixels = [b["image"] for b in batch if b.get("image") is not None]
        if pixels:
            # Row-major concat matches the ``<im_start>`` scan order (batch 0
            # first) in ImageInsertEmbedding.insert_features.
            images.append(
                ImageForInsert(
                    insert_start_token=self.img_start_token_id,
                    images=torch.cat(pixels, dim=0),
                )
            )
        out["images"] = images

        return out


@dataclass(kw_only=True)
class Step37MockSFTDataProvider(DatasetProvider):
    """Mock (synthetic) Step3.7 SFT dataset provider in BSHD layout.

    Set ``cfg.dataset = Step37MockSFTDataProvider(...)`` on a Step3.7 SFT
    recipe and run with ``--step_func step37_mock_step`` to exercise the
    unpacked BSHD training path without any dataset download. See the module
    docstring for the batch schema.
    """

    # ── Tokenizer / vocab ─────────────────────────────────────────────────
    tokenizer_path: Optional[str] = None
    """Local HF snapshot / model id used only to resolve ``vocab_size`` and the
    special token ids when they are left at ``-1``. ``None`` requires those
    fields to be provided explicitly (fully offline mock)."""
    vocab_size: int = -1
    """Random text-token upper bound. Resolved from the tokenizer if ``< 0``."""

    # ── Sample synthesis ──────────────────────────────────────────────────
    num_samples: int = 1000
    num_images: int = 1
    image_size: int = 728
    """Square image edge fed to the vision tower. ``728`` yields exactly 169
    features per image (13×13 after the two stride-2 downsamplers), matching
    ``image_token_count``."""
    image_token_count: int = IMAGE_TOKEN_COUNT
    prompt_length: int = 32
    response_length: int = 64
    random_seed: int = 1234

    # ── Special-token ids (resolved from the tokenizer when ``-1``) ───────
    img_start_token_id: int = -1
    img_patch_token_id: int = -1
    img_end_token_id: int = -1
    pad_token_id: int = 0
    image_start_token: str = IMAGE_START_TOKEN
    image_token: str = IMAGE_TOKEN
    image_end_token: str = IMAGE_END_TOKEN

    # ── mbridge framework defaults ────────────────────────────────────────
    seq_length: int = 2048
    dataloader_type: Optional[Literal["single", "cyclic", "external"]] = "single"
    skip_getting_attention_mask_from_dataset: bool = True
    global_data_keys: list = field(default_factory=list)
    """No cross-PP broadcast keys: BSHD attention needs neither ``cu_seqlens``
    nor ``position_id`` (causal masking is applied by the attention backend)."""

    def __post_init__(self):  # type: ignore[override]
        super_post = getattr(super(), "__post_init__", None)
        if super_post is not None:
            super_post()

    def _resolve_from_tokenizer(self) -> None:
        """Fill ``vocab_size`` / special ids from the tokenizer if unset."""
        needs_ids = min(self.img_start_token_id, self.img_patch_token_id, self.img_end_token_id) < 0
        if self.vocab_size >= 0 and not needs_ids:
            return
        if not self.tokenizer_path:
            raise ValueError(
                "Step37MockSFTDataProvider needs either tokenizer_path set or "
                "vocab_size + img_{start,patch,end}_token_id provided explicitly."
            )
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(self.tokenizer_path, trust_remote_code=False)
        if self.vocab_size < 0:
            self.vocab_size = int(tok.vocab_size)
        if self.img_start_token_id < 0:
            self.img_start_token_id = int(tok.convert_tokens_to_ids(self.image_start_token))
        if self.img_patch_token_id < 0:
            self.img_patch_token_id = int(tok.convert_tokens_to_ids(self.image_token))
        if self.img_end_token_id < 0:
            self.img_end_token_id = int(tok.convert_tokens_to_ids(self.image_end_token))

    @staticmethod
    def _resolve_pp_stage(pg_collection: Optional[ProcessGroupCollection]) -> Tuple[bool, bool]:
        """Return ``(is_first_pp_stage, is_last_pp_stage)`` for this rank.

        Falls back to ``(True, True)`` when no pipeline process group is
        available (e.g. unit tests or single-stage runs), which reproduces the
        original "synthesize everything" behavior.
        """
        pp = getattr(pg_collection, "pp", None) if pg_collection is not None else None
        if pp is None:
            return True, True
        return is_pp_first_stage(pp), is_pp_last_stage(pp)

    def _make_dataset(
        self, size: int, *, load_inputs: bool, load_labels: bool
    ) -> Optional[_Step37MockDataset]:
        if not size or size <= 0:
            return None
        return _Step37MockDataset(
            length=size,
            seq_length=self.seq_length,
            num_images=self.num_images,
            image_size=self.image_size,
            image_token_count=self.image_token_count,
            img_start_token_id=self.img_start_token_id,
            img_patch_token_id=self.img_patch_token_id,
            img_end_token_id=self.img_end_token_id,
            pad_token_id=self.pad_token_id,
            vocab_size=self.vocab_size,
            prompt_length=self.prompt_length,
            response_length=self.response_length,
            random_seed=self.random_seed,
            load_inputs=load_inputs,
            load_labels=load_labels,
        )

    def build_datasets(
        self, context: DatasetBuildContext
    ) -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:
        """Build synthetic train / valid / test datasets in BSHD layout.

        Each rank only synthesizes the slice its pipeline stage consumes: the
        first PP stage gets ``input_ids`` + ``image``, the last PP stage gets
        ``labels`` + ``loss_mask``, and middle stages get an empty payload.
        """
        self._resolve_from_tokenizer()
        is_first, is_last = self._resolve_pp_stage(context.pg_collection)
        make = lambda size: self._make_dataset(size, load_inputs=is_first, load_labels=is_last)
        train_ds = make(context.train_samples)
        valid_ds = make(context.valid_samples)
        test_ds = make(context.test_samples)
        return train_ds, valid_ds, test_ds


__all__ = ["Step37MockSFTDataProvider"]
