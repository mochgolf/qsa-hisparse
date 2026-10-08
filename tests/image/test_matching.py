"""Image prefix matching (Track I acceptance 1, 2, 4, 7), counterexamples first.

Every request below has equal prefix tokens to the reference ``A`` unless
stated: all images share one pad value, which forces the 30-bit pad
collision, so a miss can only come from the image key. ``A`` holds image
``X`` at ``[10, 74)`` (straddling 64) and ``Y`` at ``[100, 120)``; its
checkpoints are at 64, 128 and 192.
"""

import hashlib
import unittest
from array import array
from unittest.mock import patch

import torch

from image.fixtures import PAD, image_req, item, mm_inputs, positions, tokens
from prefix import test_prefix_cache as prefix_tests
from sglang.srt.multimodal.cache.identity import build_artifact_key
from sglang_qsa_hisparse.hisparse.image_request import BYPASS, identity_for
from sglang_qsa_hisparse.hisparse.prefix import (
    HostPrefixCache,
    PrefixSegment,
    PrefixSnapshot,
    image_key,
    key_digest,
    token_bytes,
)
from sglang_qsa_hisparse.hisparse.prefix_cache import QSAHostPrefixCache

NS = ("model", "position=0")
FINGERPRINT = "sha256:" + "ab" * 32


def key(content, **preprocess_kwargs):
    digest = "sha256:" + hashlib.sha256(content).hexdigest()
    return build_artifact_key(
        digest,
        modality="image",
        processor_fingerprint=FINGERPRINT,
        preprocess_kwargs=preprocess_kwargs,
    )


X, Y, Z = key(b"x"), key(b"y"), key(b"z")


def reference_items(x=X, y=Y, x_grid=(1, 16, 16), y_grid=(1, 8, 10)):
    return [item(x, 10, 74, x_grid), item(y, 100, 120, y_grid)]


def checkpoint(ids, key_value):
    length = len(ids)
    return PrefixSnapshot(
        NS,
        token_bytes(ids),
        (
            PrefixSegment(
                0,
                length,
                torch.zeros((1, 2, length, 1, 256), dtype=torch.uint8),
                torch.zeros((1, length // 4, 1, 16), dtype=torch.bfloat16),
            ),
        ),
        (torch.zeros((4, 1, 16), dtype=torch.bfloat16),),
        torch.zeros((4, 3), dtype=torch.int64),
        ([torch.zeros((2, 1, 3))], torch.zeros((2, 1, 3))),
        key_value,
    )


def cache_of(ids, identity, lengths=(64, 128, 192)):
    cache = HostPrefixCache(100_000_000)
    for length in lengths:
        snapshot = checkpoint(ids[:length], image_key(identity, length))
        cache.publish(cache.reserve(1_000_000), snapshot, completed=True)
    return cache


def hit(cache, ids, identity):
    reader = cache.acquire(NS, token_bytes(ids), identity=identity)
    if reader is None:
        return 0
    length = reader.snapshot.length
    reader.close()
    return length


def reference():
    a = image_req("a", reference_items(), 200)
    ids = tokens(200, a.multimodal_inputs.mm_items)
    return ids, cache_of(ids, identity_for(a))


def length_hit(items, *, ids=None, length=200, mrope=None):
    a_ids, cache = reference()
    req = image_req("r", items, length)
    if mrope is not None:
        req.multimodal_inputs.mrope_positions = mrope
    return hit(cache, a_ids if ids is None else ids, identity_for(req))


# Acceptance 1: equal tokens are not enough; every image intersecting [0, L)
# must match by artifact key, order, full offsets and grid.


def test_forced_pad_collision_with_different_artifact_key_misses():
    assert length_hit(reference_items()) == 192  # Control: same identity.
    # Y differs (fully inside [64, 128)); tokens are equal by the forced pad.
    assert length_hit(reference_items(y=Z)) == 64


def test_same_content_with_different_preprocessing_misses():
    detail = key(b"y", detail="high")
    assert detail != Y  # Same content digest, different preprocessing.
    assert length_hit(reference_items(y=detail)) == 64


def test_swapped_image_order_misses():
    swapped = reference_items()[::-1]  # Same spans and keys, items reordered.
    assert length_hit(swapped) == 0
    # Same items, contents swapped between the two spans.
    crossed = [item(Y, 10, 74, (1, 16, 16)), item(X, 100, 120, (1, 8, 10))]
    assert length_hit(crossed) == 0


def test_different_grid_misses():
    assert length_hit(reference_items(x_grid=(1, 8, 32))) == 0  # Same span length.
    assert length_hit(reference_items(y_grid=(1, 4, 20))) == 64


def test_straddling_image_that_differs_misses_at_every_later_boundary():
    # X spans [10, 74): the checkpoint at 64 holds part of it, keyed by all of it.
    assert length_hit(reference_items(x=Z)) == 0
    assert length_hit([item(X, 10, 80, (1, 16, 16)), item(Y, 100, 120)]) == 0


def test_different_image_after_the_boundary_still_hits():
    items = reference_items(y=Z)
    ids = tokens(200, items)
    ids[100:120] = [PAD + 1] * 20  # A real Z: its own pad value too.
    assert length_hit(items, ids=ids) == 64
    # An extra image that starts at the boundary is not part of the key.
    extra = reference_items() + [item(Z, 128, 140)]
    assert length_hit(extra, ids=tokens(200, extra)) == 128


# Acceptance 2: prefix M-RoPE positions by page-cumulative digest; no
# whole-prompt digest, so requests that differ only after L share.


def test_mrope_digest_difference_misses():
    early, late = positions(200, shift_from=32), positions(200, shift_from=150)
    assert length_hit(reference_items(), mrope=early) == 0
    assert length_hit(reference_items(), mrope=late) == 128


def test_different_suffix_and_length_share_the_prefix():
    ids, cache = reference()
    longer = image_req("longer", reference_items() + [item(Z, 210, 290)], 300)
    suffix = ids[:140] + list(range(5000, 5160))
    assert hit(cache, suffix, identity_for(longer)) == 128


def test_identity_beyond_mrope_coverage_never_matches():
    ids, cache = reference()
    short = image_req("short", reference_items(), 150)  # Two complete pages.
    assert hit(cache, ids, identity_for(short)) == 128


# Acceptance 7: text requests behave as before and never meet image checkpoints.


def test_text_and_image_checkpoints_are_disjoint():
    ids, cache = reference()
    assert hit(cache, ids, None) == 0  # Text request with A's exact tokens.
    text = HostPrefixCache(100_000_000)
    text.publish(text.reserve(1_000_000), checkpoint(ids[:128], None), completed=True)
    assert hit(text, ids, None) == 128
    a = image_req("a", reference_items(), 200)
    assert hit(text, ids, identity_for(a)) == 0


def test_signature_carries_a_digest_of_the_key():
    identity = identity_for(image_req("a", reference_items(), 200))
    ids = tokens(200, reference_items())
    snapshot = checkpoint(ids[:128], identity.key_at(128))
    sha = hashlib.sha256(token_bytes(ids[:128])).hexdigest()
    assert snapshot.signature == (NS, 128, sha, key_digest(identity.key_at(128)))
    other = identity_for(image_req("b", reference_items(y=Z), 200))
    assert key_digest(other.key_at(128)) != key_digest(identity.key_at(128))
    assert key_digest(other.key_at(64)) == key_digest(identity.key_at(64))
    assert checkpoint(ids[:128], None).signature == (NS, 128, sha, None)


def namespace_with_images(cache, req):
    """Stand-in for I-B's ``_namespace``: supported image requests pass."""
    if identity_for(req) is BYPASS:
        return None
    return (
        cache.runtime.prefix_namespace,
        cache.host.epoch,
        req.extra_key,
        req.cache_salt,
        req.lora_id,
    )


class TestImageCheckpoints(unittest.TestCase):
    """The fork prefix suite's fake runtime with real pools and allocators."""

    req = prefix_tests.TestRuntimeHostPrefixes.req
    match = prefix_tests.TestRuntimeHostPrefixes.match

    def setUp(self):
        prefix_tests.TestRuntimeHostPrefixes.setUp(self)
        self.patches.enter_context(
            patch.object(QSAHostPrefixCache, "_namespace", namespace_with_images)
        )
        self.votes = []
        self.patches.enter_context(
            patch.object(
                QSAHostPrefixCache, "_converge", lambda _, vote: self.votes.append(vote)
            )
        )

    def image(self, name, items, length=200):
        req = self.req(name, tokens(length, items))
        req.multimodal_inputs = mm_inputs(items, length)
        return req

    def rows(self, req, length):
        self.a.req_pool.alloc([req])
        pages = self.a.runner.token_to_kv_pool_allocator.alloc(length)
        req.prefix_indices = pages
        self.a.req_table[req.kv.req_pool_idx, :length] = pages.int()
        req.kv.kv_allocated_len = req.kv.kv_committed_len = length

    def prefill(self, req, length):
        """Own a row, pages and a lease as a cold prefill of ``length`` does."""
        self.rows(req, length)
        return self.a._acquire_request(req.kv.req_pool_idx, req.rid)

    def release(self, req):
        lease = self.a.release(req.kv.req_pool_idx, req.rid)
        self.a.runner.token_to_kv_pool_allocator.free(req.prefix_indices)
        self.a.req_pool.free_mamba_cache(req)
        self.a.req_pool.free(req)
        req.kv.mark_kv_released()
        self.a.after_release(lease)

    def capture(self, req, *lengths):
        state = self.prefill(req, max(lengths))
        for length in lengths:
            state.seq_len = length
            self.cache._capture(req)
        self.release(req)

    def entries(self, length):
        return [
            s for s, _ in self.a.prefix_cache.entries.values() if s.length == length
        ]

    def test_capture_keys_checkpoints_and_tp_signatures(self):
        a = self.image("a", reference_items())
        self.capture(a, 64, 128)
        identity = identity_for(a)
        (first,), (second,) = self.entries(64), self.entries(128)
        self.assertEqual(first.image_key, identity.key_at(64))
        self.assertEqual(second.image_key, identity.key_at(128))
        # The ancestor lookup found the request's own basis under its key.
        self.assertIs(second.segments[0], first.segments[0])
        ids = a.full_untruncated_fill_ids
        self.assertEqual(
            self.votes,
            [
                (
                    first.namespace,
                    length,
                    hashlib.sha256(token_bytes(ids[:length])).hexdigest(),
                    key_digest(identity.key_at(length)),
                    True,
                )
                for length in (64, 128)
            ],
        )
        self.assertEqual(first.signature[-1], key_digest(identity.key_at(64)))

    def test_ancestor_lookup_requires_the_capturing_requests_key(self):
        self.capture(self.image("a", reference_items()), 64)
        # Same tokens (forced pad collision), different straddling image: an
        # identity-blind lookup would find A's checkpoint and skip this capture.
        self.capture(self.image("b", reference_items(x=Z)), 64)
        self.assertEqual(self.votes[-1][-1], True)
        keys = {s.image_key[0][0].artifact_key for s in self.entries(64)}
        self.assertEqual(keys, {X, Z})
        # The same identity is still recognized as already captured.
        self.capture(self.image("a-again", reference_items()), 64)
        self.assertEqual(self.votes[-1][-1], False)
        self.assertEqual(len(self.entries(64)), 2)

    def test_match_prefix_requires_the_key_at_the_candidate_length(self):
        self.capture(self.image("a", reference_items()), 64, 128)
        self.capture(self.image("b", reference_items(x=Z)), 64)
        cases = (
            ("same", reference_items(), 128),
            ("other-straddling", reference_items(x=Z), 64),
            ("third-straddling", reference_items(x=key(b"w")), 0),
            ("after-boundary", reference_items(y=Z), 64),
        )
        for name, items, expected in cases:
            with self.subTest(name):
                req = self.image(name, items)
                self.assertEqual(self.match(req), expected)
                reader = self.cache.matches.get(req.cache_request_handle)
                if expected:
                    snapshot = reader.snapshot
                    key_value = identity_for(req).key_at(expected)
                    self.assertEqual(snapshot.image_key, key_value)
                    self.assertEqual(self.votes[-1][-1], snapshot.signature)
                    self.cache.release_aborted_request(req.cache_request_handle)
        text = self.req("text", tokens(200, reference_items()))
        self.assertEqual(self.match(text), 0)

    def state_views(self, req, lease):
        """Every request-scoped tensor a checkpoint covers, as writable views."""
        row, slot = req.kv.req_pool_idx, req.kv.mamba_pool_idx
        physical = self.a.slots.staging_slice(lease, 0, 128)
        logical = req.prefix_indices[::4].long() // 4
        mamba = self.a.req_pool.mamba_pool
        short, ngram = mamba._slot_siblings
        ring = slice(row * 4, row * 4 + 4)
        return (
            [(self.a.full.k_buffer[li], physical) for li in range(2)]
            + [(self.a.full.v_buffer[li], physical) for li in range(2)]
            + [(t, logical) for t in self.a.pool.qsa_compressed_k_buffer_pool]
            + [(t, ring) for t in self.a.pool.qsa_key_state_buffer_pool]
            + [(self.a.pool.qsa_rope_position_buffer, ring)]
            + [(mamba.mamba_cache.conv[0], (slice(None), slot))]
            + [(mamba.mamba_cache.temporal, (slice(None), slot))]
            + [(short.conv_state, (slice(None), slot)), (ngram.context, slot)]
        )

    def test_image_checkpoint_restores_exact_state_into_a_reused_slot(self):
        # Acceptance 4 on CPU: raw K/V, compressed index, pending ring, rope
        # rows and every recurrent/PLE state round-trip for an image request.
        a = self.image("a", reference_items())
        state = self.prefill(a, 128)
        state.seq_len = 128
        views, expected = self.state_views(a, state.lease), []
        for n, (tensor, where) in enumerate(views):
            shape = tensor[where].shape
            value = torch.arange(shape.numel()).reshape(shape).add(17 * n) % 251
            tensor[where] = value.to(tensor.dtype)
            expected.append(tensor[where].clone())
        self.cache._capture(a)
        self.release(a)
        for tensor, _ in views:
            tensor.fill_(99)  # Poison every backing between owners.
        warm = self.image("warm", reference_items())
        warm.full_untruncated_fill_ids[130:] = array("q", range(5000, 5070))
        self.assertEqual(self.match(warm), 128)
        self.rows(warm, 128)
        self.assertEqual(warm.kv.req_pool_idx, state.lease.req_pool_idx)
        reader = self.cache.matches[warm.cache_request_handle]
        self.a.restore_prefix(warm, reader.snapshot)
        lease = self.a.requests[warm.kv.req_pool_idx].lease
        self.assertGreater(lease.generation, state.lease.generation)
        for (tensor, where), value in zip(self.state_views(warm, lease), expected):
            self.assertTrue(torch.equal(tensor[where], value))
        self.cache.release_aborted_request(warm.cache_request_handle)
