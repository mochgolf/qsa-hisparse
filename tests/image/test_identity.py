"""``identity_for(req)``: records, page digests, per-attempt cache, bypass set."""

import dataclasses
import hashlib
from types import SimpleNamespace

import pytest
import torch

from image.fixtures import image_req, item
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import Modality, MultimodalInputFormat
from sglang_qsa_hisparse.hisparse.image_identity import ImageRecord
from sglang_qsa_hisparse.hisparse.image_request import BYPASS, identity_for


def test_text_request_has_no_identity_and_is_left_untouched():
    req = SimpleNamespace(rid="text", multimodal_inputs=None)
    assert identity_for(req) is None
    assert not hasattr(req, "qsa_image_identity")


def test_records_follow_offsets_and_keep_item_order_grid_and_full_span():
    # Items listed out of prompt order: the ordinal is the item index.
    req = image_req(
        "two", [item("key-y", 100, 120, (1, 8, 10)), item("key-x", 10, 74)], 200
    )
    identity = identity_for(req)
    assert identity.records == (
        ImageRecord("key-x", 1, 10, 74, (1, 4, 8)),
        ImageRecord("key-y", 0, 100, 120, (1, 8, 10)),
    )
    assert all(type(v) is int for r in identity.records for v in r.grid)
    # Independent page-cumulative digest over int64 positions, complete pages.
    positions = req.multimodal_inputs.mrope_positions
    previous, expected = b"", []
    for page in range(200 // 64):
        block = positions[:, page * 64 : (page + 1) * 64].to(torch.int64)
        previous = hashlib.sha256(previous + block.numpy().tobytes()).digest()
        expected.append(previous)
    assert identity.page_digests == tuple(expected)
    # The straddling image keeps its full span; later images are not in the key.
    assert identity.key_at(64) == (identity.records[:1], expected[0])
    assert identity.key_at(128) == (identity.records, expected[1])


def test_identity_is_computed_once_per_attempt_and_mrope_length():
    req = image_req("cached", [item("key-x", 10, 74)], 130)
    first = identity_for(req)
    assert identity_for(req) is first
    req.cache_request_handle = dataclasses.replace(
        req.cache_request_handle, attempt_id=1
    )
    second = identity_for(req)
    assert second is not first and second == first
    # A retracted request extends its positions over its output tokens.
    mm = req.multimodal_inputs
    mm.mrope_positions = torch.cat([mm.mrope_positions, torch.full((3, 70), 130)], 1)
    extended = identity_for(req)
    assert extended.records == first.records
    assert extended.page_digests[:2] == first.page_digests
    assert len(extended.page_digests) == 3


def _bypass_cases():
    def video(req):
        req.multimodal_inputs.mm_items[0].modality = Modality.VIDEO

    def audio(req):
        req.multimodal_inputs.mm_items[0].modality = Modality.AUDIO

    def embeddings(req):
        req.multimodal_inputs.mm_items[0].precomputed_embeddings = torch.zeros(9, 4)

    def embedding_format(req):
        req.multimodal_inputs.mm_items[0].format = (
            MultimodalInputFormat.PRECOMPUTED_EMBEDDING
        )

    def multi_span(req):
        req.multimodal_inputs.mm_items[0].offsets = [(10, 20), (30, 40)]

    def no_offsets(req):
        req.multimodal_inputs.mm_items[0].offsets = None

    def no_artifact_key(req):
        req.multimodal_inputs.mm_items[1] = item(None, 100, 120)

    def no_mrope(req):
        req.multimodal_inputs.mrope_positions = None

    def no_items(req):
        req.multimodal_inputs.mm_items = []

    return [video, audio, embeddings, embedding_format, multi_span, no_offsets,
            no_artifact_key, no_mrope, no_items]  # fmt: skip


@pytest.mark.parametrize("change", _bypass_cases(), ids=lambda f: f.__name__)
def test_unsupported_inputs_bypass(change):
    req = image_req("bypass", [item("key-x", 10, 74), item("key-y", 100, 120)], 200)
    change(req)
    assert identity_for(req) is BYPASS


def test_skipped_mm_hashing_bypasses():
    req = image_req("skip-hash", [item("key-x", 10, 74)], 128)
    with envs.SGLANG_MM_SKIP_COMPUTE_HASH.override(True):
        assert identity_for(req) is BYPASS
