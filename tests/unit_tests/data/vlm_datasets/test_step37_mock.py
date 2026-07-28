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

import pytest
import torch

from megatron.bridge.data.vlm_datasets.step37_mock import Step37MockSFTDataProvider, _Step37MockDataset
from megatron.bridge.models.stepfun.modelling_step37.image_insert_embedding import ImageForInsert
from megatron.bridge.training.config import DatasetBuildContext


pytestmark = pytest.mark.unit

# Offline knobs: explicit ids + vocab so no tokenizer / download is needed.
_SEQ_LEN = 512
_NUM_IMAGES = 2
_IMAGE_SIZE = 28
_IMAGE_TOKEN_COUNT = 4
_IMG_START, _IMG_PATCH, _IMG_END = 50, 51, 52


def _make_provider(**overrides) -> Step37MockSFTDataProvider:
    kwargs = dict(
        tokenizer_path=None,
        vocab_size=100,
        seq_length=_SEQ_LEN,
        num_images=_NUM_IMAGES,
        image_size=_IMAGE_SIZE,
        image_token_count=_IMAGE_TOKEN_COUNT,
        img_start_token_id=_IMG_START,
        img_patch_token_id=_IMG_PATCH,
        img_end_token_id=_IMG_END,
        prompt_length=8,
        response_length=8,
    )
    kwargs.update(overrides)
    return Step37MockSFTDataProvider(**kwargs)


def test_build_datasets_splits():
    provider = _make_provider()
    ctx = DatasetBuildContext(train_samples=10, valid_samples=4, test_samples=0)
    train_ds, valid_ds, test_ds = provider.build_datasets(ctx)
    assert len(train_ds) == 10
    assert len(valid_ds) == 4
    assert test_ds is None


def test_sample_is_bshd_shaped():
    provider = _make_provider()
    ctx = DatasetBuildContext(train_samples=4, valid_samples=0, test_samples=0)
    train_ds, _, _ = provider.build_datasets(ctx)

    sample = train_ds[0]
    assert sample["input_ids"].shape == (_SEQ_LEN,)
    assert sample["labels"].shape == (_SEQ_LEN,)
    assert sample["loss_mask"].shape == (_SEQ_LEN,)
    assert sample["image"].shape == (_NUM_IMAGES, 3, _IMAGE_SIZE, _IMAGE_SIZE)

    # labels are the row shifted by one → labels[:-1] == input_ids[1:].
    assert torch.equal(sample["labels"][:-1], sample["input_ids"][1:])
    # Exactly one <im_start> per image span.
    assert int((sample["input_ids"] == _IMG_START).sum()) == _NUM_IMAGES
    # Only the response span is supervised.
    assert sample["loss_mask"].sum() > 0

    # Determinism: same index → identical sample.
    assert torch.equal(train_ds[0]["input_ids"], train_ds[0]["input_ids"])


def test_collate_merges_images_bshd():
    provider = _make_provider()
    ctx = DatasetBuildContext(train_samples=4, valid_samples=0, test_samples=0)
    train_ds, _, _ = provider.build_datasets(ctx)

    batch = train_ds.collate_fn([train_ds[0], train_ds[1]])

    assert batch["input_ids"].shape == (2, _SEQ_LEN)
    assert batch["labels"].shape == (2, _SEQ_LEN)
    assert batch["loss_mask"].shape == (2, _SEQ_LEN)

    # BSHD → no packed-sequence artifacts.
    assert "cu_seqlens" not in batch
    assert "position_ids" not in batch

    # One merged ImageForInsert holding all rows' pixels in batch-major order.
    images = batch["images"]
    assert len(images) == 1
    assert isinstance(images[0], ImageForInsert)
    assert images[0].insert_start_token == _IMG_START
    assert images[0].images.shape == (2 * _NUM_IMAGES, 3, _IMAGE_SIZE, _IMAGE_SIZE)

    # Total <im_start> markers == total inserted images across the batch.
    assert int((batch["input_ids"] == _IMG_START).sum()) == 2 * _NUM_IMAGES


def test_content_too_long_raises():
    with pytest.raises(ValueError):
        provider = _make_provider(seq_length=8, response_length=64)
        provider.build_datasets(DatasetBuildContext(train_samples=1, valid_samples=0, test_samples=0))


# ── PP-stage-aware synthesis ──────────────────────────────────────────────
#
# Under pipeline parallelism each stage only synthesizes the slice it consumes:
# the first stage gets ``input_ids`` + ``image``, the last stage gets ``labels``
# + ``loss_mask``, and middle stages get an empty payload. These tests drive the
# dataset directly with the ``load_inputs`` / ``load_labels`` flags that
# ``Step37MockSFTDataProvider._resolve_pp_stage`` derives from the PP group, so
# no distributed init is required.


def _make_stage_dataset(load_inputs: bool, load_labels: bool, *, length: int = 4) -> _Step37MockDataset:
    return _Step37MockDataset(
        length=length,
        seq_length=_SEQ_LEN,
        num_images=_NUM_IMAGES,
        image_size=_IMAGE_SIZE,
        image_token_count=_IMAGE_TOKEN_COUNT,
        img_start_token_id=_IMG_START,
        img_patch_token_id=_IMG_PATCH,
        img_end_token_id=_IMG_END,
        pad_token_id=0,
        vocab_size=100,
        prompt_length=8,
        response_length=8,
        random_seed=1234,
        load_inputs=load_inputs,
        load_labels=load_labels,
    )


def test_first_stage_only_inputs_and_image():
    ds = _make_stage_dataset(load_inputs=True, load_labels=False)
    sample = ds[0]
    assert set(sample.keys()) == {"input_ids", "image"}
    assert sample["input_ids"].shape == (_SEQ_LEN,)
    assert sample["image"].shape == (_NUM_IMAGES, 3, _IMAGE_SIZE, _IMAGE_SIZE)


def test_last_stage_only_labels_and_loss_mask():
    ds = _make_stage_dataset(load_inputs=False, load_labels=True)
    sample = ds[0]
    assert set(sample.keys()) == {"labels", "loss_mask"}
    assert sample["labels"].shape == (_SEQ_LEN,)
    assert sample["loss_mask"].shape == (_SEQ_LEN,)
    # No heavy image tensor materialized on the last stage.
    assert "image" not in sample


def test_middle_stage_empty_payload():
    ds = _make_stage_dataset(load_inputs=False, load_labels=False)
    sample = ds[0]
    assert sample == {}
    # Middle-stage collate carries only an empty images list (no tensors).
    batch = ds.collate_fn([ds[0], ds[1]])
    assert batch == {"images": []}


def test_first_and_last_stage_stay_aligned():
    # The token stream is drawn identically regardless of which slice a stage
    # keeps, so the first stage's input_ids and the last stage's labels are the
    # same shifted row — exactly as if a single stage had produced both.
    first = _make_stage_dataset(load_inputs=True, load_labels=False)
    last = _make_stage_dataset(load_inputs=False, load_labels=True)
    full = _make_stage_dataset(load_inputs=True, load_labels=True)

    assert torch.equal(first[2]["input_ids"], full[2]["input_ids"])
    assert torch.equal(last[2]["labels"], full[2]["labels"])
    assert torch.equal(last[2]["loss_mask"], full[2]["loss_mask"])
    # labels are the row shifted by one relative to the first stage's inputs.
    assert torch.equal(last[2]["labels"][:-1], first[2]["input_ids"][1:])


def test_first_stage_collate_omits_labels():
    ds = _make_stage_dataset(load_inputs=True, load_labels=False)
    batch = ds.collate_fn([ds[0], ds[1]])
    assert batch["input_ids"].shape == (2, _SEQ_LEN)
    assert "labels" not in batch
    assert "loss_mask" not in batch
    assert len(batch["images"]) == 1
    assert batch["images"][0].images.shape == (2 * _NUM_IMAGES, 3, _IMAGE_SIZE, _IMAGE_SIZE)
