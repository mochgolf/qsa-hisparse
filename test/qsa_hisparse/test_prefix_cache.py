"""Host ownership and exact state tests; no CUDA/DMA or live TP claims.

Reference bytes are constructed from position/layer arithmetic independently
of QSA packing and indexing helpers. Actual paged/row/Mamba allocators execute.
"""

import unittest
from array import array
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import replace
from threading import Barrier
from threading import Event as ThreadEvent
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import Req, ReqKvInfo
from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sglang.srt.mem_cache.allocation import alloc_for_extend
from sglang.srt.mem_cache.allocator.mamba import MambaSlotAllocator
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    CacheRequestHandle,
    CacheRequestOutcome,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.chunk_cache import ChunkCache
from sglang.srt.mem_cache.memory_pool import (
    HybridReqToTokenPool,
    MambaPool,
    ReqToTokenPool,
)
from sglang.srt.mem_cache.ple_state_pool import NGramPool, ShortConvPool
from sglang.srt.mem_cache.qsa_hisparse.prefix import (
    HostPrefixCache,
    PrefixSegment,
    PrefixSnapshot,
    token_bytes,
)
from sglang.srt.mem_cache.qsa_hisparse.prefix_cache import QSAHostPrefixCache
from sglang.srt.mem_cache.qsa_hisparse.runtime import QSAHiSparseRuntime
from sglang.srt.mem_cache.qsa_hisparse.slots import QSAHiSparseSlots
from sglang.srt.mem_cache.radix_cache import RadixKey


class Event:
    def record(self, stream=None):
        self.recorded = True

    def synchronize(self):
        assert self.recorded

    def query(self):
        return self.recorded


def snapshot(length=64, namespace=("model", "position=0"), segments=None):
    if segments is None:
        raw = (
            torch.arange(2 * length * 256)
            .remainder(251)
            .to(torch.uint8)
            .reshape(1, 2, length, 1, 256)
        )
        index = (
            torch.arange(length // 4 * 16)
            .reshape(1, length // 4, 1, 16)
            .to(torch.bfloat16)
        )
        segments = (PrefixSegment(0, length, raw, index),)
    return PrefixSnapshot(
        namespace,
        token_bytes(range(length)),
        segments,
        (torch.full((4, 1, 16), 7, dtype=torch.bfloat16),),
        torch.arange(12).reshape(4, 3),
        (
            [torch.full((2, 1, 3), 13, dtype=torch.bfloat16)],
            torch.full((2, 1, 3), 19, dtype=torch.float32),
        ),
    )


def footprint(s):
    # Independent sum for a newly constructed, unshared fixture.
    return (
        len(s.tokens)
        + sum(x.raw.nbytes + x.index.nbytes for x in s.segments)
        + sum(t.nbytes for t in s.pending)
        + s.rope.nbytes
        + sum(t.nbytes for t in s.mamba[0])
        + s.mamba[1].nbytes
    )


class TestHostPrefixOwnership(unittest.TestCase):
    def test_pool_flush_preserves_new_leases_and_rejects_stale_generations(self):
        for prefix_enabled in (False, True):
            with self.subTest(prefix_enabled=prefix_enabled):
                pool = ReqToTokenPool(1, 64, "cpu", False)
                slots = QSAHiSparseSlots(64, 64, 1)
                host = HostPrefixCache(100_000) if prefix_enabled else None
                previous = None
                for cycle in range(4):
                    index = pool.alloc_rows(1)[0]
                    lease = slots.acquire(
                        index, int(pool.req_generation[index]), f"request-{cycle}"
                    )
                    if previous is not None:
                        with self.assertRaisesRegex(RuntimeError, "stale QSA lease"):
                            slots.require(previous)
                    waits = []

                    def drain(kind):
                        self.assertEqual(slots.active[index], lease)
                        self.assertEqual(pool.available_size(), 0)
                        waits.append(kind)

                    terminal = Mock()
                    terminal.synchronize.side_effect = lambda: drain("model")
                    copy = Mock()
                    copy.synchronize.side_effect = lambda: drain("copy")
                    slots.drain(lease, terminal, [copy])
                    self.assertEqual(waits, ["model", "copy"])
                    slots.logical_flushed(lease, SimpleNamespace(free_group=None))
                    slots.commit_release(lease)
                    pool.free_rows([index])
                    if host is not None:
                        host.reset()
                    pool.clear()
                    self.assertEqual(pool.available_size(), 1)
                    with self.assertRaisesRegex(RuntimeError, "stale QSA request"):
                        slots.acquire(index, lease.generation, lease.rid)
                    previous = lease

    def publish(self, cache, s, size=None):
        reservation = cache.reserve(footprint(s) if size is None else size)
        self.assertIsNotNone(reservation)
        cache.publish(reservation, s, completed=True)

    def test_exact_tokens_namespace_and_checkpoint_limits(self):
        cache = HostPrefixCache(1_000_000)
        self.publish(cache, snapshot(64))
        self.publish(cache, snapshot(128))
        for n, expected in ((63, 0), (64, 64), (127, 64), (128, 128), (129, 128)):
            reader = cache.acquire(("model", "position=0"), token_bytes(range(140)), n)
            self.assertEqual(0 if reader is None else reader.snapshot.length, expected)
            if reader is not None:
                reader.close()
        for ns in (
            ("another-model", "position=0"),
            ("model", "position=1"),
            ("model", "adapter=1"),
        ):
            self.assertIsNone(cache.acquire(ns, token_bytes(range(140))))
        branch = list(range(140))
        branch[64] = 9000
        reader = cache.acquire(("model", "position=0"), token_bytes(branch))
        self.assertEqual(reader.snapshot.length, 64)
        reader.close()
        branch[0] = 9001
        self.assertIsNone(cache.acquire(("model", "position=0"), token_bytes(branch)))

    def test_partial_pages_and_incomplete_writes_never_publish(self):
        cache = HostPrefixCache(1_000_000)
        for s, error in (
            (snapshot(63), "page64"),
            (replace(snapshot(), segments=()), "missing prefix"),
        ):
            r = cache.reserve(footprint(s))
            with self.assertRaisesRegex(ValueError, error):
                cache.publish(r, s, completed=True)
            self.assertEqual(cache.stats()["prefix_cache_pending_bytes"], 0)
        r = cache.reserve(footprint(snapshot()))
        with self.assertRaisesRegex(RuntimeError, "incomplete writes"):
            cache.publish(r, snapshot(), completed=False)
        self.assertEqual(cache.used_bytes, 0)
        self.assertEqual(len(cache.entries), 0)

    def test_reader_protects_eviction_and_reset_then_releases(self):
        s = snapshot()
        cache = HostPrefixCache(footprint(s))
        self.publish(cache, s)
        reader = cache.acquire(s.namespace, s.tokens)
        self.assertIsNone(cache.reserve(footprint(s)))
        cache.reset()
        self.assertIsNone(cache.acquire(s.namespace, s.tokens))
        self.assertEqual(cache.used_bytes, footprint(s))
        self.assertEqual(reader.snapshot.signature, s.signature)
        reader.close()
        reader.close()  # Repeated cancellation cannot release a newer reader.
        self.assertEqual(cache.used_bytes, 0)
        self.publish(cache, snapshot())
        self.assertEqual(cache.stats()["prefix_cache_readers"], 0)

    def test_pending_budget_and_flushed_publication(self):
        s = snapshot()
        cache = HostPrefixCache(footprint(s))
        r = cache.reserve(footprint(s))
        self.assertIsNone(cache.reserve(1))
        cache.reset()
        with self.assertRaisesRegex(RuntimeError, "flushed epoch"):
            cache.publish(r, s, completed=True)
        self.assertEqual(cache.stats()["prefix_cache_pending_bytes"], 0)
        self.assertEqual(cache.used_bytes, 0)
        r = cache.reserve(1)
        with self.assertRaisesRegex(RuntimeError, "exceeded"):
            cache.publish(r, s, completed=True)
        self.assertEqual(cache.used_bytes, 0)

    def test_pending_publishers_reserve_entry_slots(self):
        cache = HostPrefixCache(1_000_000, max_entries=1)
        reservation = cache.reserve(100_000)
        self.assertIsNone(cache.reserve(100_000))
        reservation.close()
        self.assertEqual(cache.stats()["prefix_cache_pending_entries"], 0)
        self.publish(cache, snapshot())
        self.publish(cache, snapshot(namespace=("replacement",)))
        self.assertEqual(len(cache.entries), 1)

    def test_shared_segments_count_once_across_eviction(self):
        first = snapshot()
        extra = snapshot().segments[0]
        extra = replace(extra, start=64, stop=128)
        second = snapshot(128, segments=(first.segments[0], extra))
        cache = HostPrefixCache(1_000_000, max_entries=2)
        self.publish(cache, first)
        self.publish(cache, second)
        shared = first.segments[0].raw.nbytes + first.segments[0].index.nbytes
        self.assertEqual(
            cache.used_bytes, footprint(first) + footprint(second) - shared
        )
        reader = cache.acquire(second.namespace, second.tokens)
        self.publish(cache, snapshot(64, namespace=("branch",)))
        self.assertEqual(len(cache.entries), 2)
        self.assertEqual(
            reader.snapshot.segments[0].raw.data_ptr(), first.segments[0].raw.data_ptr()
        )
        reader.close()
        cache.reset()
        self.assertEqual(cache.used_bytes, 0)

    def test_eight_concurrent_readers_keep_reset_storage_bounded(self):
        s = snapshot()
        cache = HostPrefixCache(footprint(s))
        self.publish(cache, s)
        rendezvous, finish = Barrier(9), ThreadEvent()

        def read():
            reader = cache.acquire(s.namespace, s.tokens)
            rendezvous.wait(timeout=10)
            finish.wait(timeout=10)
            length = reader.snapshot.length
            reader.close()
            return length

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(read) for _ in range(8)]
            rendezvous.wait(timeout=10)
            try:
                self.assertEqual(cache.stats()["prefix_cache_readers"], 8)
                cache.reset()
                self.assertEqual(cache.used_bytes, footprint(s))
                self.assertIsNone(cache.reserve(footprint(s)))
            finally:
                finish.set()
            self.assertEqual(
                [future.result(timeout=10) for future in futures], [64] * 8
            )
        self.assertEqual(cache.used_bytes, 0)


class TestRuntimeHostPrefixes(unittest.TestCase):
    def setUp(self):
        from sglang.srt.runtime_context import get_context

        self.patches = ExitStack()
        self.addCleanup(self.patches.close)
        self.patches.enter_context(get_context().override_server_args())
        self.patches.enter_context(patch.object(torch.cuda, "Event", Event))
        self.patches.enter_context(
            patch.object(torch.cuda, "current_stream", return_value=Mock())
        )
        self.patches.enter_context(
            patch("sglang.srt.mem_cache.memory_pool.current_platform.synchronize")
        )
        self.patches.enter_context(
            patch(
                "sglang.srt.mem_cache.allocation.attention_backends",
                return_value=("torch_native", "torch_native"),
            )
        )
        a = self.a = QSAHiSparseRuntime.__new__(QSAHiSparseRuntime)
        a.mode, a.device, a.strict, a.path = "p2-offload", "cpu", True, None
        a.capacity, a.max_requests, a.layer_ids = 256, 2, [3, 7]
        a.slots = QSAHiSparseSlots(a.capacity, 64, 2)
        a.full = SimpleNamespace(
            k_buffer=[
                torch.zeros((a.slots.raw_pool_size + 64, 1, 256), dtype=torch.uint8)
                for _ in a.layer_ids
            ],
            v_buffer=[
                torch.zeros((a.slots.raw_pool_size + 64, 1, 256), dtype=torch.uint8)
                for _ in a.layer_ids
            ],
        )
        a.pool = SimpleNamespace(
            dtype=torch.uint8,
            index_state_dtype=torch.bfloat16,
            qsa_key_state_buffer_pool=[
                torch.zeros((12, 1, 16), dtype=torch.bfloat16) for _ in a.layer_ids
            ],
            qsa_rope_position_buffer=torch.zeros((12, 3), dtype=torch.int64),
            qsa_compressed_k_buffer_pool=[
                torch.zeros((144, 1, 16), dtype=torch.bfloat16) for _ in a.layer_ids
            ],
        )
        a.pool.qsa_hisparse = a
        rp = a.req_pool = HybridReqToTokenPool.__new__(HybridReqToTokenPool)
        ReqToTokenPool.__init__(rp, 2, a.capacity, "cpu", False)
        rp.enable_mamba_extra_buffer = False
        rp.req_index_to_mamba_index_mapping = torch.zeros(3, dtype=torch.int32)
        rp.mamba_allocator = MambaSlotAllocator(2, "cpu")
        mamba = rp.mamba_pool = MambaPool.__new__(MambaPool)
        mamba.replayssm_write_pos = None
        mamba.mamba_cache = MambaPool.State(
            conv=[torch.zeros((3, 3, 2, 3), dtype=torch.bfloat16)],
            temporal=torch.zeros((3, 3, 2, 2), dtype=torch.float32),
        )
        short = ShortConvPool.__new__(ShortConvPool)
        short.conv_state = torch.zeros((2, 3, 2, 3), dtype=torch.bfloat16)
        short.layer_map = {3: 0, 7: 1}
        ngram = NGramPool.__new__(NGramPool)
        ngram.context = torch.zeros((3, 3), dtype=torch.int64)
        mamba._slot_siblings = (short, ngram)
        a.req_table = rp.req_to_token
        a.runner = SimpleNamespace(
            token_to_kv_pool_allocator=PagedTokenToKVPoolAllocator(
                512, 64, torch.uint8, "cpu", a.pool, False
            )
        )
        a.copy_stream, a.producer_stream = Mock(), Mock()
        a.host_slabs = torch.empty((2, 2, 64, 2048), dtype=torch.uint8)
        a._allocate_decode_workspace()
        a.requests, a.pending_releases, a.batch_requests = {}, [], []
        a.record = Mock()
        a.prefix_cache = HostPrefixCache(4_000_000)
        a.prefix_namespace = ("fixture-model", "normal-text-position=0", "TP2")
        params = CacheInitParams(True, rp, a.runner.token_to_kv_pool_allocator, 64)
        self.cache = QSAHostPrefixCache(ChunkCache(params), a, None)

    def test_finished_request_releases_owned_pages(self):
        req = self.req("finished", range(64))
        self.a.req_pool.alloc([req])
        allocator = self.a.runner.token_to_kv_pool_allocator
        indices = allocator.alloc(64)
        self.a.req_table[req.kv.req_pool_idx, :64] = indices.int()
        req.kv.kv_allocated_len = req.kv.kv_committed_len = 64
        available_before = allocator.available_size()

        self.cache.cache_finished_req(req, is_insert=False, owned_kv_len=64)

        self.assertEqual(allocator.available_size(), available_before + 64)

    def req(self, name, tokens):
        return SimpleNamespace(
            rid=name,
            kv=ReqKvInfo(),
            cache_request_handle=CacheRequestHandle(name, 1),
            full_untruncated_fill_ids=array("q", tokens),
            prefix_indices=torch.empty(0, dtype=torch.int64),
            extra_key=None,
            cache_salt=None,
            lora_id=None,
            last_node=None,
            extend_range=SimpleNamespace(start=0, end=len(tokens), length=len(tokens)),
        )

    def source(self, length=128):
        req = self.req("source", range(length))
        self.a.req_pool.alloc([req])
        req.prefix_indices = self.a.runner.token_to_kv_pool_allocator.alloc(length)
        # Reverse full page order: compressed keys follow logical physical pages,
        # while raw staging follows token positions. The oracle does neither gather.
        req.prefix_indices = req.prefix_indices.reshape(-1, 64).flip(0).flatten()
        self.a.req_table[req.kv.req_pool_idx, :length] = req.prefix_indices.int()
        req.kv.kv_allocated_len = req.kv.kv_committed_len = length
        state = self.a._acquire_request(req.kv.req_pool_idx, req.rid)
        state.seq_len = length
        self.expected_raw, self.expected_index = [], []
        for li in range(2):
            k = (
                (torch.arange(length * 256).reshape(length, 1, 256) + li * 31)
                .remainder(256)
                .to(torch.uint8)
            )
            v = (
                (torch.arange(length * 256).reshape(length, 1, 256) * 3 + li * 19)
                .remainder(256)
                .to(torch.uint8)
            )
            self.a.full.k_buffer[li][64 : 64 + length] = k
            self.a.full.v_buffer[li][64 : 64 + length] = v
            index = (
                torch.arange(length // 4 * 16).reshape(length // 4, 1, 16) + li * 61
            ).to(torch.bfloat16)
            self.a.pool.qsa_compressed_k_buffer_pool[li][
                req.prefix_indices[::4] // 4
            ] = index
            self.expected_raw.append((k, v))
            self.expected_index.append(index)
        mid = req.kv.mamba_pool_idx
        self.a.req_pool.mamba_pool.mamba_cache.conv[0][:, mid] = torch.arange(
            18
        ).reshape(3, 2, 3)
        self.a.req_pool.mamba_pool.mamba_cache.temporal[:, mid] = (
            torch.arange(12).reshape(3, 2, 2) / 8
        )
        short, ngram = self.a.req_pool.mamba_pool._slot_siblings
        short.conv_state[:, mid] = torch.arange(12).reshape(2, 2, 3) + 50
        ngram.context[mid] = torch.tensor([91, 92, 93])
        for li, target in enumerate(self.a.pool.qsa_key_state_buffer_pool):
            target[req.kv.req_pool_idx * 4 : (req.kv.req_pool_idx + 1) * 4] = li + 200
        self.a.pool.qsa_rope_position_buffer[
            req.kv.req_pool_idx * 4 : (req.kv.req_pool_idx + 1) * 4
        ] = torch.arange(12).reshape(4, 3) + 1000
        self.cache._capture(req)
        lease = self.a.release(req.kv.req_pool_idx, req.rid)
        self.a.runner.token_to_kv_pool_allocator.free(req.prefix_indices)
        self.a.req_pool.free_mamba_cache(req)
        self.a.req_pool.free(req)
        req.kv.mark_kv_released()
        self.a.after_release(lease)
        return req, state.lease

    def match(self, req, limit=None):
        result = self.cache.match_prefix(
            MatchPrefixParams(
                RadixKey(
                    req.full_untruncated_fill_ids,
                    limit=len(req.full_untruncated_fill_ids) - 1
                    if limit is None
                    else limit,
                ),
                req=req,
            )
        )
        req.prefix_indices = result.device_indices
        req.kv.cache_protected_len = result.cache_protected_len or 0
        return len(req.prefix_indices)

    def batch(self, req, prefix):
        req.extend_range = SimpleNamespace(
            start=prefix,
            end=len(req.full_untruncated_fill_ids),
            length=len(req.full_untruncated_fill_ids) - prefix,
        )
        return SimpleNamespace(
            tree_cache=self.cache,
            req_to_token_pool=self.a.req_pool,
            reqs=[req],
            device="cpu",
            prefix_lens=[prefix],
            extend_lens=[req.extend_range.length],
            extend_num_tokens=req.extend_range.length,
            seq_lens_cpu=torch.tensor([req.extend_range.end]),
            seq_lens=torch.tensor([req.extend_range.end]),
            maybe_evict_swa=Mock(),
            is_dllm=lambda: False,
        )

    def admission_req(self, name, tokens, *, ignore_eos=False, salt=None):
        req = self.req(name, tokens)
        req.origin_input_ids = req.full_untruncated_fill_ids
        req.sampling_params = SimpleNamespace(ignore_eos=ignore_eos, max_new_tokens=8)
        req.output_ids = []
        req.retracted_stain = False
        req.storage_hit_length = 0
        req.host_hit_length = req.host_loaded_length = req.swa_host_hit_length = 0
        req.storage_hit_start = None
        req.host_hit_is_storage = False
        req.cache_salt = salt
        req.needs_host_load_back = lambda: False
        req.materialized_host_hit_len = lambda: 0
        req.fulfilled_storage_hit_len = lambda prefix: 0

        def set_range(start, end):
            req.extend_range = SimpleNamespace(start=start, end=end, length=end - start)

        req.set_extend_range = set_range
        return req

    def adder(self, chunk=4096):
        return PrefillAdder(
            64,
            self.cache,
            self.a.runner.token_to_kv_pool_allocator,
            None,
            1.0,
            4096,
            chunk,
        )

    def test_final_checkpoint_page_boundaries_for_normal_and_ignore_eos_controls(self):
        for ignore_eos in (False, True):
            for length, first_end in (
                (63, 63),
                (64, 64),
                (65, 64),
                (127, 64),
                (128, 128),
                (129, 128),
                (201, 192),
            ):
                ranges, charges = [], []
                for salt in (None, "independent-cold-control"):
                    req = self.admission_req(
                        "boundary", range(length), ignore_eos=ignore_eos, salt=salt
                    )
                    self.match(req)
                    adder = self.adder()
                    adder.add_one_req(req, False, None)
                    self.assertEqual(adder.can_run_list, [req])
                    self.assertEqual(req.extend_range.end, first_end)
                    self.assertEqual(adder.new_chunked_req is req, first_end < length)
                    ranges.append((req.extend_range.start, req.extend_range.end))
                    charges.append(
                        (
                            adder.memory_budget.total_offset,
                            adder.memory_budget.current_offset,
                        )
                    )
                self.assertEqual(ranges[0], ranges[1])
                self.assertEqual(charges[0], charges[1])

    def test_checkpoint_limits_around_real_prefill_chunk_boundaries(self):
        for length, boundary in (
            (4095, 4032),
            (4096, None),
            (4097, 4096),
            (4098, 4096),
            (8192, None),
            (8193, 8192),
        ):
            req = self.req("large-boundary", range(length))
            self.assertEqual(self.cache.prefill_checkpoint_limit(req), boundary)
            if boundary is not None:
                req.prefix_indices = torch.empty(boundary, dtype=torch.int64)
                self.assertIsNone(self.cache.prefill_checkpoint_limit(req))
        req = self.req("unsupported-boundary", range(4097))
        req.multimodal_inputs = object()
        self.assertIsNone(self.cache.prefill_checkpoint_limit(req))

    def test_chunk_continuation_captures_aligned_boundary_before_logits_tail(self):
        req = self.admission_req("continued", range(201), ignore_eos=True)
        req.prefix_indices = torch.arange(64)
        adder = self.adder()
        self.assertIs(adder.add_chunked_req(req), req)
        self.assertEqual((req.extend_range.start, req.extend_range.end), (64, 192))
        req.prefix_indices = torch.arange(192)
        adder = self.adder()
        self.assertIsNone(adder.add_chunked_req(req))
        self.assertEqual((req.extend_range.start, req.extend_range.end), (192, 201))

    def test_ordinary_short_prompt_publishes_and_reuses_actual_checkpoint(self):
        req = self.admission_req("short-cold", range(73))
        self.match(req)
        adder = self.adder()
        adder.add_one_req(req, False, None)
        self.assertEqual((req.extend_range.start, req.extend_range.end), (0, 64))
        batch = self.batch(req, 0)
        batch.seq_lens = batch.seq_lens_cpu = torch.tensor([64])
        batch.extend_lens = [64]
        batch.extend_num_tokens = 64
        req.extend_range = SimpleNamespace(start=0, end=64, length=64)
        alloc_for_extend(batch)
        state = self.a._acquire_request(req.kv.req_pool_idx, req.rid)
        state.seq_len = 64
        self.cache.cache_unfinished_req(req, chunked=True)
        warm = self.req("short-warm", range(73))
        self.assertEqual(self.match(warm), 64)
        self.assertEqual(self.cache.pending_prefix_tokens(warm), 64)

    def test_logprob_clipped_cold_capture_uses_its_own_prefix_bytes(self):
        self.source(64)
        old = self.a.prefix_cache.acquire(
            self.cache._namespace(self.req("key", range(128))), token_bytes(range(128))
        )
        old_raw = old.snapshot.segments[0].raw.data_ptr()
        old.close()
        req = self.req("logprob-cold", range(128))
        self.assertEqual(self.match(req, limit=0), 0)
        alloc_for_extend(self.batch(req, 0))
        state = self.a._acquire_request(req.kv.req_pool_idx, req.rid)
        state.seq_len = 128
        for li in range(2):
            self.a.full.k_buffer[li][64:192].fill_(211 + li)
            self.a.full.v_buffer[li][64:192].fill_(219 + li)
            physical = self.a.req_table[req.kv.req_pool_idx, :128:4].long() // 4
            self.a.pool.qsa_compressed_k_buffer_pool[li][physical] = 377 + li
        self.cache._capture(req)
        reader = self.a.prefix_cache.acquire(
            self.cache._namespace(req), token_bytes(range(128))
        )
        try:
            checkpoint = reader.snapshot
            self.assertEqual(checkpoint.length, 128)
            self.assertEqual(len(checkpoint.segments), 1)
            self.assertNotEqual(checkpoint.segments[0].raw.data_ptr(), old_raw)
            for li in range(2):
                self.assertTrue(
                    bool((checkpoint.segments[0].raw[li, 0] == 211 + li).all())
                )
                self.assertTrue(
                    bool((checkpoint.segments[0].raw[li, 1] == 219 + li).all())
                )
                self.assertTrue(
                    bool((checkpoint.segments[0].index[li] == 377 + li).all())
                )
        finally:
            reader.close()

    def test_active_request_shares_only_its_restored_and_published_segments(self):
        self.source(64)
        req = self.req("incremental", range(193))
        self.assertEqual(self.match(req), 64)
        original = self.cache.matches[req.cache_request_handle].snapshot.segments[0]
        alloc_for_extend(self.batch(req, 64))
        state = self.a.requests[req.kv.req_pool_idx]
        self.assertIsInstance(state.prefix_basis_entry_id, int)
        state.seq_len = 128
        self.a.full.k_buffer[0][128:192].fill_(211)
        self.cache._capture(req)
        published = self.a.prefix_cache.acquire(
            self.cache._namespace(req), token_bytes(range(128))
        )
        first_two = published.snapshot.segments
        self.assertEqual(len(first_two), 2)
        self.assertEqual(first_two[0].raw.data_ptr(), original.raw.data_ptr())
        self.assertEqual(first_two[0].index.data_ptr(), original.index.data_ptr())
        self.assertEqual(state.prefix_basis_entry_id, published.entry_id)
        published.close()
        state.seq_len = 192
        self.a.full.k_buffer[0][192:256].fill_(223)
        self.cache._capture(req)
        reader = self.a.prefix_cache.acquire(
            self.cache._namespace(req), token_bytes(range(192))
        )
        try:
            self.assertEqual(len(reader.snapshot.segments), 3)
            for old, new in zip(first_two, reader.snapshot.segments):
                self.assertEqual(old.raw.data_ptr(), new.raw.data_ptr())
                self.assertEqual(old.index.data_ptr(), new.index.data_ptr())
            self.assertTrue(bool((reader.snapshot.segments[1].raw[0, 0] == 211).all()))
            self.assertTrue(bool((reader.snapshot.segments[2].raw[0, 0] == 223).all()))
        finally:
            reader.close()

    def test_evicted_active_basis_falls_back_to_copying_all_current_bytes(self):
        self.source(64)
        req = self.req("evicted-basis", range(129))
        self.assertEqual(self.match(req), 64)
        alloc_for_extend(self.batch(req, 64))
        state = self.a.requests[req.kv.req_pool_idx]
        self.assertIsInstance(state.prefix_basis_entry_id, int)
        reservation = self.a.prefix_cache.reserve(self.a.prefix_cache.budget_bytes)
        self.assertIsNotNone(reservation)
        reservation.close()
        self.assertEqual(self.a.prefix_cache.used_bytes, 0)
        state.seq_len = 128
        self.a.full.k_buffer[0][128:192].fill_(211)
        self.cache._capture(req)
        reader = self.a.prefix_cache.acquire(
            self.cache._namespace(req), token_bytes(range(128))
        )
        try:
            self.assertEqual(len(reader.snapshot.segments), 1)
            raw = reader.snapshot.segments[0].raw
            self.assertTrue(torch.equal(raw[0, 0, :64], self.expected_raw[0][0]))
            self.assertTrue(bool((raw[0, 0, 64:] == 211).all()))
        finally:
            reader.close()

    def test_restore_recycled_slot_exact_raw_index_pending_and_ple(self):
        _, old_lease = self.source()
        req = self.req("branch", list(range(128)) + [9000])
        self.assertEqual(self.match(req), 128)
        # Poison every request-scoped backing between owners, including raw FP8 NaNs.
        for t in self.a.full.k_buffer + self.a.full.v_buffer:
            t.fill_(255)
        for t in self.a.req_pool.mamba_pool.mamba_cache.conv:
            t.fill_(-1)
        self.a.req_pool.mamba_pool.mamba_cache.temporal.fill_(-2)
        short, ngram = self.a.req_pool.mamba_pool._slot_siblings
        short.conv_state.fill_(-3)
        ngram.context.fill_(-4)
        out, _, _ = alloc_for_extend(self.batch(req, 128))
        new_lease = self.a.requests[req.kv.req_pool_idx].lease
        self.assertEqual(new_lease.req_pool_idx, old_lease.req_pool_idx)
        self.assertGreater(new_lease.generation, old_lease.generation)
        with self.assertRaisesRegex(RuntimeError, "stale QSA lease"):
            self.a.slots.require(old_lease)
        self.assertTrue(bool((req.prefix_indices >= 64).all()))
        self.assertEqual(req.kv.cache_protected_len, 0)
        self.assertEqual(out.numel(), 1)
        for li, (k, v) in enumerate(self.expected_raw):
            self.assertTrue(torch.equal(self.a.full.k_buffer[li][64:192], k))
            self.assertTrue(torch.equal(self.a.full.v_buffer[li][64:192], v))
            slots = self.a.req_table[req.kv.req_pool_idx, :128:4].long() // 4
            self.assertTrue(
                torch.equal(
                    self.a.pool.qsa_compressed_k_buffer_pool[li][slots].view(
                        torch.uint8
                    ),
                    self.expected_index[li].view(torch.uint8),
                )
            )
            self.assertTrue(
                torch.equal(
                    self.a.pool.qsa_key_state_buffer_pool[li][
                        req.kv.req_pool_idx * 4 : (req.kv.req_pool_idx + 1) * 4
                    ],
                    torch.full((4, 1, 16), li + 200, dtype=torch.bfloat16),
                )
            )
        mid = req.kv.mamba_pool_idx
        self.assertTrue(
            torch.equal(
                short.conv_state[:, mid],
                (torch.arange(12).reshape(2, 2, 3) + 50).to(torch.bfloat16),
            )
        )
        self.assertTrue(torch.equal(ngram.context[mid], torch.tensor([91, 92, 93])))
        self.assertTrue(
            torch.equal(
                self.a.req_pool.mamba_pool.mamba_cache.temporal[:, mid],
                torch.arange(12).reshape(3, 2, 2) / 8,
            )
        )
        self.assertFalse(req.kv.mamba_needs_clear)
        self.assertEqual(req.host_loaded_length, 128)
        self.assertEqual(self.a.prefix_cache.reused_tokens, 128)
        self.assertEqual(self.a.prefix_cache.stats()["prefix_cache_readers"], 0)
        # Mutating a continuation cannot mutate the published recurrent or KV state.
        short.conv_state[:, mid].fill_(999)
        self.a.full.k_buffer[0][64:192].fill_(17)
        reader = self.a.prefix_cache.acquire(
            self.cache._namespace(req), token_bytes(range(128))
        )
        self.assertTrue(
            torch.equal(reader.snapshot.segments[0].raw[0, 0], self.expected_raw[0][0])
        )
        self.assertTrue(
            torch.equal(
                reader.snapshot.mamba[2][0][:, 0],
                (torch.arange(12).reshape(2, 2, 3) + 50).to(torch.bfloat16),
            )
        )
        reader.close()

    def test_first_restore_failure_releases_pages_rows_and_snapshot_pin(self):
        self.source()
        req = self.req("failure", list(range(128)) + [9000])
        self.match(req)
        logical_before = self.a.runner.token_to_kv_pool_allocator.available_size()
        with patch.object(
            self.a, "_restore_mamba", side_effect=RuntimeError("restore copy failed")
        ):
            with self.assertRaisesRegex(RuntimeError, "restore copy failed"):
                alloc_for_extend(self.batch(req, 128))
        self.assertEqual(
            self.a.runner.token_to_kv_pool_allocator.available_size(), logical_before
        )
        self.assertEqual(self.a.req_pool.available_size(), 2)
        self.assertEqual(self.a.req_pool.mamba_allocator.available_size(), 2)
        self.assertEqual(len(self.a.slots.active), 0)
        self.assertIsNone(self.a.slots.prefill_owner)
        self.assertEqual(self.a.prefix_cache.stats()["prefix_cache_readers"], 0)
        self.assertFalse(req.kv.holds_kv)
        self.assertEqual(self.a.prefix_cache.reused_tokens, 0)

    def test_queued_abort_flush_and_unsupported_identity_misses(self):
        self.source(64)
        for name in (
            "multimodal_inputs",
            "positional_embed_overrides",
            "input_embeds",
            "position_ids",
        ):
            req = self.req("unsupported-" + name, range(65))
            setattr(req, name, object())
            self.assertEqual(self.match(req), 0)
        req = self.req("salt", range(65))
        req.cache_salt = "different"
        self.assertEqual(self.match(req), 0)
        req = self.req("queued", range(65))
        self.assertEqual(self.match(req), 64)
        self.cache.release_aborted_request(req.cache_request_handle)
        self.cache.release_aborted_request(req.cache_request_handle)
        self.assertEqual(self.a.prefix_cache.stats()["prefix_cache_readers"], 0)
        self.assertEqual(self.match(req), 64)
        self.cache.reset()
        self.assertEqual(self.match(req), 0)
        self.assertEqual(self.a.prefix_cache.used_bytes, 0)
        self.assertEqual(self.a.req_pool.available_size(), 2)

    def test_later_chunk_failure_preserves_previous_lease_and_frees_new_pages(self):
        self.source(64)
        req = self.req("later", range(129))
        self.match(req)
        batch = self.batch(req, 64)
        batch.seq_lens = batch.seq_lens_cpu = torch.tensor([128])
        batch.extend_lens = [64]
        batch.extend_num_tokens = 64
        req.extend_range = SimpleNamespace(start=64, end=128, length=64)
        alloc_for_extend(batch)
        self.a.requests[req.kv.req_pool_idx].seq_len = 128
        previous_lease = self.a.requests[req.kv.req_pool_idx].lease
        previous_indices = self.a.req_table[req.kv.req_pool_idx, :128].clone()
        req.prefix_indices = previous_indices.long()
        before = self.a.runner.token_to_kv_pool_allocator.available_size()
        self.a.req_pool.alloc_aux_to_lengths = Mock(
            side_effect=RuntimeError("later aux failed")
        )
        with self.assertRaisesRegex(RuntimeError, "later aux failed"):
            alloc_for_extend(self.batch(req, 128))
        self.assertEqual(
            self.a.runner.token_to_kv_pool_allocator.available_size(), before
        )
        self.assertEqual(self.a.req_pool.available_size(), 1)
        self.assertEqual(self.a.req_pool.mamba_allocator.available_size(), 1)
        self.assertEqual(req.kv.kv_allocated_len, 128)
        self.assertEqual(req.kv.kv_committed_len, 128)
        self.a.slots.require(previous_lease, "prefill")
        self.assertTrue(
            torch.equal(self.a.req_table[req.kv.req_pool_idx, :128], previous_indices)
        )
        self.assertEqual(self.a.prefix_cache.stats()["prefix_cache_readers"], 0)

    def test_pending_host_hit_ignore_eos_can_continue_in_chunks(self):
        self.source(64)
        req = self.req("long-suffix", range(201))
        self.match(req)
        req.sampling_params = SimpleNamespace(ignore_eos=True, max_new_tokens=8)
        req.output_ids = []
        req.retracted_stain = False
        req.storage_hit_length = 0
        req.host_hit_length = req.host_loaded_length = req.swa_host_hit_length = 0
        req.storage_hit_start = None
        req.host_hit_is_storage = False
        req.needs_host_load_back = lambda: False
        req.materialized_host_hit_len = lambda: 0
        req.fulfilled_storage_hit_len = lambda prefix: 0

        def set_range(start, end):
            req.extend_range = SimpleNamespace(start=start, end=end, length=end - start)

        req.set_extend_range = set_range
        adder = PrefillAdder(
            64,
            self.cache,
            self.a.runner.token_to_kv_pool_allocator,
            None,
            1.0,
            4096,
            64,
        )
        adder.add_one_req(req, False, None)
        self.assertEqual(adder.can_run_list, [req])
        self.assertIs(adder.new_chunked_req, req)
        self.assertEqual((req.extend_range.start, req.extend_range.end), (64, 128))

    def test_weight_load_attempt_invalidates_even_without_regular_flush(self):
        from sglang.srt.managers.scheduler_components.weight_updater import (
            SchedulerWeightUpdaterManager,
        )

        self.source(64)
        manager = SchedulerWeightUpdaterManager(
            tp_worker=None,
            draft_worker=None,
            tp_cpu_group=None,
            memory_saver_adapter=None,
            flush_cache=Mock(),
            is_fully_idle=lambda: True,
            scheduler=SimpleNamespace(tree_cache=self.cache),
        )
        old = self.a.prefix_cache.epoch
        with manager._observe_weight_load("fixture"):
            self.assertEqual(self.a.prefix_cache.epoch, old + 1)
        manager.flush_cache.assert_not_called()
        req = self.req("new-model", range(65))
        self.assertEqual(self.match(req), 0)

    def test_tp_disagreement_rejects_match_before_marker_or_admission(self):
        self.source(64)
        req = self.req("tp-disagree", range(65))
        self.cache.tp_group = object()

        def disagree(votes, signature, group):
            votes[:] = [signature, (signature[0], None)]

        with (
            patch.object(torch.distributed, "get_world_size", return_value=2),
            patch.object(torch.distributed, "all_gather_object", side_effect=disagree),
        ):
            with self.assertRaisesRegex(RuntimeError, "TP ranks disagree"):
                self.match(req)
        self.assertEqual(req.prefix_indices.numel(), 0)
        self.assertEqual(self.a.prefix_cache.stats()["prefix_cache_readers"], 0)

    def test_real_request_prefix_limit_preserves_requested_prompt_logprobs(self):
        self.source(64)
        self.source(128)
        for start, expected in (
            (0, 0),
            (63, 0),
            (64, 64),
            (127, 64),
            (128, 128),
            (-1, 128),
        ):
            req = Req.__new__(Req)
            req.__dict__.update(self.req("logprobs-" + str(start), range(129)).__dict__)
            req.origin_input_ids = array("q", range(129))
            req.output_ids = array("q")
            req.dllm_config = None
            req.return_logprob = True
            req.logprob_start_len = start
            req.session = req.positional_embed_overrides = req.multimodal_inputs = None
            req.is_retracted = False
            req.init_next_round_input(self.cache)
            self.assertEqual(len(req.prefix_indices), expected)
            self.cache.release_aborted_request(req.cache_request_handle)

    def test_old_abort_handle_cannot_release_new_attempt_reader(self):
        self.source(64)
        old = self.req("same-rid", range(65))
        old.cache_request_handle = CacheRequestHandle("same-rid", 1)
        self.match(old)
        self.cache.release_aborted_request(old.cache_request_handle)
        new = self.req("same-rid", range(65))
        new.cache_request_handle = CacheRequestHandle("same-rid", 2)
        self.match(new)
        self.cache.release_aborted_request(old.cache_request_handle)
        self.assertEqual(self.a.prefix_cache.stats()["prefix_cache_readers"], 1)
        self.cache.release_aborted_request(new.cache_request_handle)
        self.assertEqual(self.a.prefix_cache.stats()["prefix_cache_readers"], 0)

    def test_real_request_force_miss_drops_old_match_without_acquiring_snapshot(self):
        self.source(64)
        req = Req.__new__(Req)
        req.__dict__.update(self.req("forced-cold", range(65)).__dict__)
        req.origin_input_ids, req.output_ids = array("q", range(65)), array("q")
        req.dllm_config = None
        req.return_logprob, req.logprob_start_len = False, -1
        req.session = req.positional_embed_overrides = req.multimodal_inputs = None
        req.is_retracted = False
        req.init_next_round_input(self.cache)
        self.assertEqual(len(req.prefix_indices), 64)
        self.assertEqual(self.a.prefix_cache.stats()["prefix_cache_readers"], 1)
        with (
            patch.object(envs.SGLANG_RADIX_FORCE_MISS, "get", return_value=True),
            patch.object(
                self.a.prefix_cache, "acquire", wraps=self.a.prefix_cache.acquire
            ) as acquire,
            patch.object(
                self.cache, "_converge", wraps=self.cache._converge
            ) as converge,
        ):
            req.init_next_round_input(self.cache)
            self.assertEqual(len(req.prefix_indices), 0)
            self.assertEqual(self.a.prefix_cache.stats()["prefix_cache_readers"], 0)
            acquire.assert_not_called()
            self.assertEqual(
                converge.call_args.args[0],
                (req.cache_request_handle, self.cache._namespace(req), None),
            )
            self.assertEqual(self.cache.prefill_checkpoint_limit(req), 64)
        # The knob suppresses reads while leaving normal cold publication on.
        self.a.prefix_cache.reset()
        batch = self.batch(req, 0)
        req.extend_range = SimpleNamespace(start=0, end=64, length=64)
        batch.extend_lens, batch.extend_num_tokens = [64], 64
        batch.seq_lens = batch.seq_lens_cpu = torch.tensor([64])
        with patch.object(envs.SGLANG_RADIX_FORCE_MISS, "get", return_value=True):
            alloc_for_extend(batch)
            state = self.a._acquire_request(req.kv.req_pool_idx, req.rid)
            state.seq_len = 64
            self.cache.cache_unfinished_req(req, chunked=True)
        self.assertEqual(len(self.a.prefix_cache.entries), 1)
        probe = self.req("probe", range(65))
        self.assertEqual(self.match(probe), 64)
        self.cache.release_aborted_request(probe.cache_request_handle)

    def test_producer_failure_prevents_publication_and_releases_reservation(self):
        # A failed producer must not publish even if a different current stream
        # reports completion. Keep the request owner for normal abort/drain.
        self.source(64)
        req = self.req("producer-failure", range(128))
        self.a.req_pool.alloc([req])
        self.a.req_table[req.kv.req_pool_idx, :128] = (
            self.a.runner.token_to_kv_pool_allocator.alloc(128).int()
        )
        req.kv.kv_allocated_len = req.kv.kv_committed_len = 128
        state = self.a._acquire_request(req.kv.req_pool_idx, req.rid)
        state.seq_len = 128
        self.a.producer_stream.synchronize.side_effect = RuntimeError(
            "producer still failed"
        )
        with self.assertRaisesRegex(RuntimeError, "producer still failed"):
            self.cache._capture(req)
        self.assertEqual(self.a.prefix_cache.stats()["prefix_cache_pending_bytes"], 0)
        self.assertEqual(self.a.prefix_cache.stats()["prefix_cache_pending_entries"], 0)
        self.assertEqual(self.a.prefix_cache.stats()["prefix_cache_readers"], 0)
        self.assertEqual(len(self.a.prefix_cache.entries), 1)
        self.a.slots.require(state.lease, "prefill")

    def test_failed_restore_drain_retains_ownership_until_explicit_cleanup(self):
        self.source(64)
        req = self.req("failed-dma", range(65))
        self.match(req)
        before = self.a.runner.token_to_kv_pool_allocator.available_size()
        stream = Mock()
        stream.synchronize.side_effect = RuntimeError("DMA drain unavailable")

        def failed_submit(req, snapshot):
            self.a._acquire_request(req.kv.req_pool_idx, req.rid)
            raise RuntimeError("restore submission failed")

        with (
            patch.object(torch.cuda, "current_stream", return_value=stream),
            patch.object(self.a, "restore_prefix", side_effect=failed_submit),
        ):
            with self.assertRaisesRegex(
                RuntimeError, "restore submission failed"
            ) as caught:
                alloc_for_extend(self.batch(req, 64))
        self.assertTrue(
            any("retained ownership" in note for note in caught.exception.__notes__)
        )
        self.assertEqual(self.a.prefix_cache.stats()["prefix_cache_readers"], 1)
        self.assertEqual(self.a.req_pool.available_size(), 1)
        self.assertLess(
            self.a.runner.token_to_kv_pool_allocator.available_size(), before
        )
        self.assertEqual(len(self.a.slots.active), 1)
        self.assertIn(req.cache_request_handle, self.cache.restoring)
        record = self.cache.restoring[req.cache_request_handle]
        for _ in range(2):
            self.cache.finish(req.cache_request_handle, CacheRequestOutcome.ABORT)
            self.assertIs(self.cache.restoring[req.cache_request_handle], record)
            self.assertIsNotNone(record["reader"].snapshot)
            self.assertEqual(self.a.prefix_cache.stats()["prefix_cache_readers"], 1)
        with self.assertRaisesRegex(RuntimeError, "undrained prefix restore"):
            self.match(req)
        with self.assertRaisesRegex(RuntimeError, "undrained prefix restore"):
            self.cache.prepare_prefix_for_extend([req])
        with self.assertRaisesRegex(RuntimeError, "must drain through rollback"):
            self.cache.before_release(req, is_insert=False)
        self.assertIs(self.cache.restoring[req.cache_request_handle], record)
        self.assertIsNotNone(record["reader"].snapshot)
        stream.synchronize.side_effect = None
        self.cache.rollback_prefix_for_extend([req])
        self.assertEqual(self.a.prefix_cache.stats()["prefix_cache_readers"], 0)
        self.assertEqual(
            self.a.runner.token_to_kv_pool_allocator.available_size(), before
        )
        self.assertEqual(self.a.req_pool.available_size(), 2)
        self.cache.finish(req.cache_request_handle, CacheRequestOutcome.ABORT)
        self.assertEqual(self.a.prefix_cache.stats()["prefix_cache_readers"], 0)
        self.assertNotIn(req.cache_request_handle, self.cache.restoring)

    def test_old_active_request_cannot_publish_after_model_invalidation(self):
        self.source(64)
        req = self.req("old-model", range(65))
        self.match(req)
        alloc_for_extend(self.batch(req, 64))
        self.a.requests[req.kv.req_pool_idx].seq_len = 64
        self.cache.invalidate_model()
        self.cache._capture(req)
        self.assertEqual(self.a.prefix_cache.used_bytes, 0)
        self.assertEqual(len(self.a.prefix_cache.entries), 0)

    def test_rollback_retry_after_drain_or_commit_ledger_failure_never_double_frees(
        self,
    ):
        for failure in ("release_drained", "before_commit", "logical_release_complete"):
            with self.subTest(failure=failure):
                self.source(64)
                req = self.req("rollback-" + failure, range(65))
                self.match(req)
                allocator = self.a.runner.token_to_kv_pool_allocator
                before = allocator.available_size()
                self.a.req_pool.alloc([req])
                self.cache.prepare_prefix_for_extend([req])
                suffix = allocator.alloc(64)[:1]
                req.extend_range = SimpleNamespace(start=64, end=65, length=1)
                self.cache.note_extend_allocation([req], suffix)
                req.kv.kv_allocated_len = req.kv.kv_committed_len = 65
                state = self.a._acquire_request(req.kv.req_pool_idx, req.rid)
                lease = state.lease
                with ExitStack() as injected:
                    if failure == "before_commit":
                        injected.enter_context(
                            patch.object(
                                self.a,
                                "after_release",
                                side_effect=OSError("commit unavailable"),
                            )
                        )
                    else:

                        def failed_ledger(event, *args, **kwargs):
                            if event == failure:
                                raise OSError("ledger unavailable")

                        injected.enter_context(
                            patch.object(self.a, "record", side_effect=failed_ledger)
                        )
                    with self.assertRaises(OSError):
                        self.cache.rollback_prefix_for_extend([req])
                if failure == "logical_release_complete":
                    self.assertNotIn(req.cache_request_handle, self.cache.restoring)
                    self.assertEqual(
                        self.a.prefix_cache.stats()["prefix_cache_readers"], 0
                    )
                    self.assertNotIn(lease.req_pool_idx, self.a.slots.active)
                else:
                    record = self.cache.restoring[req.cache_request_handle]
                    self.assertIsNotNone(record["reader"].snapshot)
                    self.assertEqual(record["req_idx"], lease.req_pool_idx)
                    self.assertEqual(record["lease"], lease)
                    if failure == "before_commit":
                        self.assertIsNone(record["prefix"])
                        self.assertIsNone(record["suffix"])
                        self.assertIsNone(req.kv.req_pool_idx)
                self.cache.rollback_prefix_for_extend([req])
                self.cache.rollback_prefix_for_extend([req])
                self.assertEqual(allocator.available_size(), before)
                self.assertEqual(self.a.req_pool.available_size(), 2)
                self.assertEqual(self.a.req_pool.mamba_allocator.available_size(), 2)
                self.assertEqual(len(self.a.slots.active), 0)
                self.assertEqual(self.a.prefix_cache.stats()["prefix_cache_readers"], 0)
                self.assertNotIn(req.cache_request_handle, self.cache.restoring)

    def test_missing_recurrent_or_pending_state_fails_closed(self):
        self.source(64)
        req = self.req("missing", range(65))
        self.match(req)
        reader = self.cache.matches[req.cache_request_handle]
        reader.snapshot = replace(reader.snapshot, pending=())
        with self.assertRaisesRegex(ValueError, "missing pending"):
            alloc_for_extend(self.batch(req, 64))
        self.assertEqual(self.a.req_pool.available_size(), 2)
        self.assertEqual(self.a.prefix_cache.stats()["prefix_cache_readers"], 0)

    def test_host_prefix_charge_changes_admission_and_rejection_refund(self):
        req = self.req("budget", range(129))
        adder = PrefillAdder.__new__(PrefillAdder)
        adder.tree_cache = SimpleNamespace(pending_prefix_tokens=lambda r: 128)
        adder.memory_budget = SimpleNamespace(total_offset=7, current_offset=3)
        adder.rem_chunk_tokens = None
        adder.can_run_list = []

        def reject(*args):
            self.assertEqual(adder.memory_budget.total_offset, 135)
            self.assertEqual(adder.memory_budget.current_offset, 131)
            return AddReqResult.NO_TOKEN

        adder._add_one_req = reject
        self.assertEqual(adder.add_one_req(req, False, None), AddReqResult.NO_TOKEN)
        self.assertEqual(adder.memory_budget.total_offset, 7)
        self.assertEqual(adder.memory_budget.current_offset, 3)

        def admit(*args):
            adder.can_run_list.append(req)
            return AddReqResult.CONTINUE

        adder._add_one_req = admit
        adder.add_one_req(req, False, None)
        self.assertEqual(adder.memory_budget.total_offset, 135)


if __name__ == "__main__":
    unittest.main()
