"""Release ordering and physical-slot ownership through W2's hooks.

Every path runs the patched SGLang entry point (B04, B05, M01, M03, M02, M04)
against the real paged allocator, request-row pool, chunk cache, QSA slots and
the runtime's release methods. Independent spies append to one ledger, and
each expected order is written out from the lifecycle contract: capture,
runtime release (drain), logical free, row free, mark released, after_release,
free-group end, after_logical_flush, physical slot reuse.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.mem_cache import allocation, common
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle, MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.chunk_cache import ChunkCache
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang_qsa_hisparse.hisparse.prefix_cache import QSAHostPrefixCache
from sglang_qsa_hisparse.hisparse.runtime import QSAHiSparseRuntime
from sglang_qsa_hisparse.hisparse.slots import QSAHiSparseSlots

pytestmark = pytest.mark.usefixtures("w2_patches", "published_context")

PAGE = 4
TOKENS = 8

# Every lease release continues so after the logical free (deferred in a group).
RELEASE_TAIL = [
    "req_to_token_pool.free",
    "mark_kv_released",
    "runtime.after_release",
]
DEFERRED_FLUSH = [
    "free_group_end",
    "logical_pages_freed",
    "runtime.after_logical_flush",
    "runtime.after_release",
    "physical_slot_released",
]


class Event:
    """CPU stand-in for torch.cuda.Event: recorded work is complete."""

    def record(self, stream=None):
        pass

    def synchronize(self):
        pass

    def query(self):
        return True


class LedgerKv(ReqKvInfo):
    __slots__ = ("ledger",)

    def mark_kv_released(self):
        self.ledger.append("mark_kv_released")
        ReqKvInfo.mark_kv_released(self)


def spy(ledger, label, fn):
    def recorded(*args, **kwargs):
        ledger.append(label)
        return fn(*args, **kwargs)

    return recorded


class World:
    def __init__(self, cache_type=ChunkCache, rows=1):
        self.ledger = ledger = []
        kvcache = SimpleNamespace()
        allocator = PagedTokenToKVPoolAllocator(64, PAGE, torch.uint8, "cpu", kvcache, False)
        pool = ReqToTokenPool(
            size=rows, max_context_len=64, device="cpu", enable_memory_saver=False
        )
        rt = QSAHiSparseRuntime.__new__(QSAHiSparseRuntime)
        rt.slots = QSAHiSparseSlots(staging_tokens=64, page_size=PAGE, max_requests=1)
        rt.req_pool, rt.device, rt.producer_stream = pool, "cpu", object()
        rt.runner = SimpleNamespace(token_to_kv_pool_allocator=allocator)
        rt.requests, rt.pending_releases, rt.batch_requests = {}, [], []
        rt.record = lambda event, lease=None, **extra: None
        rt.prefix_namespace = ("fixture",)
        rt.prefix_cache = SimpleNamespace(epoch=0, reused_tokens=0)
        kvcache.qsa_hisparse = rt
        params = CacheInitParams(
            disable=True,
            req_to_token_pool=pool,
            token_to_kv_pool_allocator=allocator,
            page_size=PAGE,
        )
        cache = ChunkCache(params)
        if cache_type is QSAHostPrefixCache:
            cache = QSAHostPrefixCache(cache, rt, None)
        else:
            cache.before_release = lambda req, is_insert: ledger.append(
                ("before_release", is_insert)
            )
        cache.cache_finished_req = spy(ledger, "cache_finished_req", cache.cache_finished_req)
        pool.alloc = spy(ledger, "req_to_token_pool.alloc", pool.alloc)
        pool.free = spy(ledger, "req_to_token_pool.free", pool.free)
        allocator._release_page_ids = spy(
            ledger, "logical_pages_freed", allocator._release_page_ids
        )
        allocator.free_group_end = spy(ledger, "free_group_end", allocator.free_group_end)
        rt.release = spy(ledger, "runtime.release", rt.release)
        rt.after_release = spy(ledger, "runtime.after_release", rt.after_release)
        rt.after_logical_flush = spy(
            ledger, "runtime.after_logical_flush", rt.after_logical_flush
        )
        rt.slots.commit_release = spy(
            ledger, "physical_slot_released", rt.slots.commit_release
        )
        self.allocator, self.pool, self.runtime, self.cache = allocator, pool, rt, cache

    def req(self, rid="req"):
        kv = LedgerKv()
        kv.ledger = self.ledger
        return SimpleNamespace(
            rid=rid,
            kv=kv,
            cache_request_handle=CacheRequestHandle(rid, 1),
            full_untruncated_fill_ids=list(range(TOKENS + PAGE)),
            prefix_indices=torch.empty(0, dtype=torch.int64),
            effective_kv_committed_len=lambda: kv.kv_committed_len,
            multimodal_inputs=None,
            return_routed_experts=False,
            finished=lambda: True,
            finished_reason=None,
            time_stats=Mock(),
        )

    def leased_req(self, *, decoding=True):
        """A request holding a row, TOKENS logical tokens and a physical lease."""
        req = self.req()
        self.pool.alloc([req])
        row = req.kv.req_pool_idx
        self.pool.req_to_token[row, :TOKENS] = self.allocator.alloc(TOKENS).int()
        req.kv.kv_committed_len = req.kv.kv_allocated_len = TOKENS
        self.lease = self.acquire(row, req.rid)
        if decoding:
            self.runtime.slots.begin_handoff(self.lease)
            self.runtime.slots.finish_handoff(self.lease, Event())
            self.runtime.slots.admit_decode(self.lease)
        self.ledger.clear()
        return req

    def acquire(self, row, rid):
        rt = self.runtime
        lease = rt.slots.acquire(row, int(self.pool.req_generation[row]), rid)
        rt.requests[row] = SimpleNamespace(lease=lease, states=[], handoff_event=None, host=None)
        return lease

    def processor(self):
        ledger = self.ledger
        return SchedulerBatchResultProcessor(
            is_generation=True,
            disaggregation_mode=None,
            enable_overlap=False,
            enable_overlap_mlx=False,
            model_config=SimpleNamespace(think_end_ids=None),
            token_to_kv_pool_allocator=self.allocator,
            tree_cache=self.cache,
            hisparse_coordinator=SimpleNamespace(
                request_finished=lambda req: ledger.append("request_finished")
            ),
            req_to_token_pool=self.pool,
            decode_offload_manager=None,
            metrics_collector=None,
            metrics_reporter=Mock(),
            draft_worker=None,
            model_worker=SimpleNamespace(
                prepare_for_kv_cache_release=lambda req: ledger.append(
                    "prepare_for_kv_cache_release"
                )
            ),
            logprob_result_processor=None,
            output_streamer=Mock(),
            beam_coordinator=Mock(),
            abort_request=lambda *args, **kwargs: None,
        )

    def slot_reusable(self):
        return self.lease.slot in self.runtime.slots.free_slots

    def events(self):
        # A logical free can release several page runs back to back.
        return [e for i, e in enumerate(self.ledger) if i == 0 or e != self.ledger[i - 1]]


@pytest.fixture(autouse=True)
def cpu_events(monkeypatch):
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: Event())


def test_finish_releases_the_lease_after_the_free_group_flush():
    world = World()
    req = world.leased_req()
    batch = SimpleNamespace(mamba_track_mask_cpu=None)

    world.allocator.free_group_begin()  # As process_batch_result_decode does.
    world.processor()._handle_finish_state_updated_req(req, batch, None, 0, None)
    assert not world.slot_reusable()
    world.allocator.free_group_end()

    assert world.events() == [
        "request_finished",
        "prepare_for_kv_cache_release",
        ("before_release", True),
        "runtime.release",
        "cache_finished_req",
        *RELEASE_TAIL,
        *DEFERRED_FLUSH,
    ]
    assert world.slot_reusable()


def test_sampling_mask_abort_skips_capture_and_defers_the_slot():
    world = World()
    req = world.leased_req()

    world.allocator.free_group_begin()
    world.processor()._handle_sampling_mask_abort(req)
    world.allocator.free_group_end()

    assert world.events() == [
        "request_finished",
        "prepare_for_kv_cache_release",
        ("before_release", False),
        "runtime.release",
        "cache_finished_req",
        *RELEASE_TAIL,
        *DEFERRED_FLUSH,
    ]


def test_chunked_abort_outside_a_free_group_frees_pages_before_the_slot():
    # process_pending_chunked_abort releases a prefill-phase lease directly.
    world = World()
    req = world.leased_req(decoding=False)

    common.release_kv_cache(req, world.cache, is_insert=False)

    assert world.events() == [
        ("before_release", False),
        "runtime.release",
        "cache_finished_req",
        "logical_pages_freed",
        *RELEASE_TAIL,
        "physical_slot_released",
    ]


def test_failed_restore_rolls_back_before_the_error_propagates():
    world = World(QSAHostPrefixCache)
    cache, rt, ledger = world.cache, world.runtime, world.ledger
    req = world.req("restore")
    snapshot = SimpleNamespace(
        length=TOKENS,
        signature=("restore", TOKENS),
        namespace=(rt.prefix_namespace, 0, None, None, None),
    )
    reader = SimpleNamespace(snapshot=snapshot, entry_id=1, close=Mock())
    rt.prefix_cache.acquire = lambda namespace, tokens, count: reader
    for name in (
        "prepare_prefix_for_extend",
        "note_extend_allocation",
        "restore_prefix_for_extend",
        "rollback_prefix_for_extend",
    ):
        setattr(cache, name, spy(ledger, name, getattr(cache, name)))

    def failing_restore(req, snapshot):
        ledger.append("runtime.restore_prefix")
        world.lease = world.acquire(req.kv.req_pool_idx, req.rid)
        raise RuntimeError("restore copy failed")

    rt.restore_prefix = failing_restore
    key = RadixKey(req.full_untruncated_fill_ids, limit=len(req.full_untruncated_fill_ids) - 1)
    req.prefix_indices = cache.match_prefix(MatchPrefixParams(key, req=req)).device_indices
    assert len(req.prefix_indices) == TOKENS
    seq_len = len(req.full_untruncated_fill_ids)
    req.extend_range = SimpleNamespace(start=TOKENS, end=seq_len, length=seq_len - TOKENS)
    batch = SimpleNamespace(
        tree_cache=cache,
        req_to_token_pool=world.pool,
        reqs=[req],
        device="cpu",
        prefix_lens=[TOKENS],
        extend_lens=[seq_len - TOKENS],
        extend_num_tokens=seq_len - TOKENS,
        seq_lens=torch.tensor([seq_len]),
        seq_lens_cpu=torch.tensor([seq_len]),
        maybe_evict_swa=lambda: None,
        is_dllm=lambda: False,
    )
    free_tokens = world.allocator.available_size()

    with pytest.raises(RuntimeError, match="restore copy failed"):
        allocation.alloc_for_extend(batch)

    assert world.events() == [
        "req_to_token_pool.alloc",
        "prepare_prefix_for_extend",
        "note_extend_allocation",
        "restore_prefix_for_extend",
        "runtime.restore_prefix",
        "rollback_prefix_for_extend",
        "runtime.release",
        "logical_pages_freed",
        *RELEASE_TAIL,
        "physical_slot_released",
    ]
    assert world.allocator.available_size() == free_tokens
    reader.close.assert_called_once_with()


def test_physical_slot_is_never_reusable_before_the_logical_flush():
    world = World(rows=2)
    req = world.leased_req()
    world.pool.alloc([fresh := world.req("fresh")])  # Another row, no lease yet.
    rt = world.runtime

    free_tokens = world.allocator.available_size()
    world.allocator.free_group_begin()
    common.release_kv_cache(req, world.cache)
    # Released and drained, but its logical pages are still held by the open group.
    assert world.allocator.available_size() == free_tokens
    assert not world.slot_reusable()
    with pytest.raises(RuntimeError, match="capacity exhausted"):
        world.acquire(fresh.kv.req_pool_idx, fresh.rid)
    assert rt.pending_releases == [world.lease]

    world.allocator.free_group_end()
    assert world.allocator.available_size() == free_tokens + TOKENS
    assert world.events()[-len(DEFERRED_FLUSH) :] == DEFERRED_FLUSH
    assert world.slot_reusable() and not rt.pending_releases
    world.acquire(fresh.kv.req_pool_idx, fresh.rid)


def test_single_request_runtime_releases_its_pending_lease_after_the_flush():
    world = World()
    ledger = world.ledger
    adapter = SimpleNamespace(
        pending_release="lease",
        after_release=lambda lease: ledger.append(("after_release", lease)),
    )
    world.allocator.get_kvcache().qsa_hisparse = adapter
    world.allocator.free_group_begin()
    world.allocator.free(world.allocator.alloc(TOKENS))
    world.allocator.free_group_end()

    assert world.events() == ["free_group_end", "logical_pages_freed", ("after_release", "lease")]


@pytest.mark.parametrize("target", [True, False])
def test_flush_keeps_row_generations_monotonic_for_the_target_model(target_model, target):
    world = World()
    target_model(target)
    req = world.leased_req()
    common.release_kv_cache(req, world.cache)
    released = world.lease

    world.pool.clear()  # Scheduler.flush_cache, after the release committed.
    world.pool.alloc([fresh := world.req("fresh")])
    row = fresh.kv.req_pool_idx

    if not target:
        # Pinned behavior: the flush restarts generations, so the row's new
        # owner carries the released request's generation.
        assert int(world.pool.req_generation[row]) == released.generation
        return
    assert int(world.pool.req_generation[row]) == released.generation + 1
    world.acquire(row, fresh.rid)
    with pytest.raises(RuntimeError, match="stale QSA lease callback"):
        world.runtime.slots.require(released)
