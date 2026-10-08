"""Scheduler gates under W2's hooks: host-prefix wrap (S01) and coordinator
adoption (S02), multimodal decode batches (S03), the staging-to-decode
transition (S04), lease admission (S05), idle leak checks (S07), idleness (S08), abort of staging requests (S09),
weight-load invalidation (B06) and the chunk checkpoint cap (P01). Expected
values follow production's code at 897286b12a (the fork's ee8fe158d6 for
these gates)."""

from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.schedule_batch import FINISH_ABORT, NextBatchPlan, ScheduleBatch
from sglang.srt.managers.schedule_policy import PrefillAdder
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.weight_updater import (
    SchedulerWeightUpdaterManager,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.chunk_cache import ChunkCache
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang_qsa_hisparse.hisparse.prefix_cache import QSAHostPrefixCache
from sglang_qsa_hisparse.hisparse.slots import QSAHiSparseSlots

pytestmark = pytest.mark.usefixtures("w2_patches")


def scheduler(**fields):
    s = Scheduler.__new__(Scheduler)
    s.__dict__.update(fields)
    return s


class Event:
    def synchronize(self):
        pass

    def query(self):
        return True


# S01, S02 -------------------------------------------------------------------


def coordinator_scheduler(
    *,
    adapter=True,
    prefix_cache="host",
    cache_type=ChunkCache,
    hierarchical=False,
    mamba_extra_buffer=False,
    runner_coordinator=True,
):
    runtime = SimpleNamespace(uses_qsa_hisparse_leases=True, prefix_cache=prefix_cache)
    kvcache = SimpleNamespace(qsa_hisparse=runtime) if adapter else SimpleNamespace()
    allocator = SimpleNamespace(get_kvcache=lambda: kvcache, device="cpu")
    pool = SimpleNamespace(enable_mamba_extra_buffer=mamba_extra_buffer)
    params = CacheInitParams(
        disable=True, req_to_token_pool=pool, token_to_kv_pool_allocator=allocator, page_size=64
    )
    return scheduler(
        enable_hisparse=False,
        enable_hierarchical_cache=hierarchical,
        token_to_kv_pool_allocator=allocator,
        req_to_token_pool=pool,
        tree_cache=cache_type(params),
        tp_worker=SimpleNamespace(
            model_runner=SimpleNamespace(
                hisparse_coordinator=Mock() if runner_coordinator else None
            )
        ),
        forward_stream="forward-stream",
        tp_cpu_group="tp-group",
    )


def test_host_prefix_cache_wraps_the_chunk_cache_after_adopting_the_coordinator():
    s = coordinator_scheduler()
    chunk_cache = s.tree_cache
    coordinator = s.tp_worker.model_runner.hisparse_coordinator

    s.init_hisparse_coordinator()

    assert s.hisparse_coordinator is coordinator
    coordinator.set_decode_producer_stream.assert_called_once_with("forward-stream")
    assert type(s.tree_cache) is QSAHostPrefixCache
    assert s.tree_cache.req_to_token_pool is chunk_cache.req_to_token_pool
    assert (s.tree_cache.host, s.tree_cache.tp_group) == ("host", "tp-group")


@pytest.mark.parametrize(
    "setting, message",
    [
        ({"cache_type": type("RadixLike", (ChunkCache,), {})}, "private chunk-cache"),
        ({"hierarchical": True}, "private chunk-cache"),
        ({"mamba_extra_buffer": True}, "one mutable Mamba slot"),
    ],
)
def test_host_prefix_cache_rejects_unsupported_schedulers(setting, message):
    s = coordinator_scheduler(**setting)
    with pytest.raises(ValueError, match=message):
        s.init_hisparse_coordinator()


@pytest.mark.parametrize(
    "setting, adopted",
    [
        ({"prefix_cache": None}, True),  # Leases without host prefixes.
        ({"prefix_cache": None, "runner_coordinator": False}, False),  # None guard.
        ({"adapter": False}, False),  # Upstream without enable_hisparse.
    ],
)
def test_without_host_prefixes_the_tree_cache_is_unchanged(setting, adopted):
    s = coordinator_scheduler(**setting)
    chunk_cache = s.tree_cache
    s.init_hisparse_coordinator()
    assert s.tree_cache is chunk_cache
    assert (s.hisparse_coordinator is not None) == adopted


# S03 ------------------------------------------------------------------------


@pytest.mark.parametrize("target", [True, False])
def test_rebuilt_decode_batch_carries_multimodal_inputs_for_the_target_model(
    target_model, target
):
    target_model(target)
    reqs = [
        SimpleNamespace(
            kv=SimpleNamespace(req_pool_idx=row),
            origin_input_ids=[1, 2],
            output_ids=[3],
            multimodal_inputs=mm,
        )
        for row, mm in ((1, "image-inputs"), (2, None))
    ]
    batch = SimpleNamespace(return_logprob=False, multimodal_inputs=None)
    s = scheduler(
        device="cpu",
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
        tree_cache=None,
        model_config=SimpleNamespace(vocab_size=8),
        enable_overlap=False,
        spec_algorithm=None,
        future_map=Mock(),
    )
    with (
        patch.object(ScheduleBatch, "init_new", return_value=batch),
        patch.object(SamplingBatchInfo, "from_schedule_batch"),
    ):
        assert s._build_hisparse_decode_batch(reqs) is batch
    assert batch.multimodal_inputs == (["image-inputs", None] if target else None)


# S04 ------------------------------------------------------------------------


def test_a_qsa_coordinator_owns_the_prefill_to_decode_transition():
    # Upstream gates this on enable_hisparse, which QSA leaves unset.
    coordinator = Mock()
    coordinator.collect_ready_reqs.return_value = []
    running = MagicMock(is_prefill_only=False, batch_is_full=True)
    running.is_empty.return_value = True
    last_batch = MagicMock()
    last_batch.forward_mode.is_extend.return_value = True
    passthrough = MagicMock(side_effect=lambda batch, **_: batch)
    s = scheduler(
        scheduler_stage_metrics=None,
        dllm_config=None,
        enable_hisparse=False,
        hisparse_coordinator=coordinator,
        enable_fpm=False,
        enable_hierarchical_cache=True,
        enable_hicache_storage=False,
        tree_cache=SimpleNamespace(check_hicache_events=Mock()),
        chunked_req=None,
        _pending_chunked_abort_req=None,
        require_mlp_sync=False,
        prefill_decode_interval=0,
        _prefill_decode_interval_remaining=0,
        get_new_batch_prefill=Mock(
            return_value=NextBatchPlan(batch_to_run=None, running_batch=running)
        ),
        dp_attn_adapter=SimpleNamespace(
            maybe_prepare_mlp_sync_batch=passthrough,
            maybe_convert_decode_to_extend=passthrough,
        ),
        ngram_embedding_manager=SimpleNamespace(prepare_for_forward=passthrough),
    )

    s.get_next_batch_to_run(running_batch=running, last_batch=last_batch)

    coordinator.collect_ready_reqs.assert_called_once_with()
    assert running.batch_is_full is False
    last_batch.filter_batch.assert_not_called()


# S05 ------------------------------------------------------------------------


def test_lease_admission_is_one_request_per_free_slot_and_none_during_prefill():
    slots = QSAHiSparseSlots(staging_tokens=64, page_size=4, max_requests=2)
    coordinator = SimpleNamespace(
        uses_qsa_hisparse_leases=True, adapter=SimpleNamespace(slots=slots)
    )
    s = scheduler(
        hisparse_coordinator=coordinator,
        req_to_token_pool=SimpleNamespace(available_size=lambda: 6),
        running_batch=None,
        beam_coordinator=SimpleNamespace(pending_member_rows=lambda batch: 0),
    )
    parallel = SimpleNamespace(pp_max_micro_batch_size=8)
    with patch("sglang.srt.managers.scheduler.get_parallel", return_value=parallel):
        assert s.get_num_allocatable_reqs(0) == 1
        assert s.get_num_allocatable_reqs(0, 1) == 1
        with pytest.raises(ValueError, match="beam/prefix ownership"):
            s.get_num_allocatable_reqs(0, 2)
        for row in (1, 2):
            lease = slots.acquire(row, 1, f"r{row}")
            assert s.get_num_allocatable_reqs(0) == 0  # Prefill owns staging.
            slots.begin_handoff(lease)
            slots.finish_handoff(lease, Event())
        assert s.get_num_allocatable_reqs(0) == 0  # No free slot.
        coordinator.uses_qsa_hisparse_leases = False
        assert s.get_num_allocatable_reqs(0) == 6


# S07 ------------------------------------------------------------------------


@pytest.mark.parametrize("coordinator", [None, "qsa-coordinator"])
def test_idle_pool_leak_checks_are_skipped_while_a_coordinator_exists(coordinator):
    s = scheduler(
        hisparse_coordinator=coordinator,
        disaggregation_mode=DisaggregationMode.NULL,
        enable_unified_memory=False,
        **{
            name: MagicMock()
            for name in (
                "scheduler_stage_metrics",
                "invariant_checker",
                "pool_stats_observer",
                "token_to_kv_pool_allocator",
                "metrics_reporter",
                "kv_events_publisher",
                "new_token_ratio_tracker",
                "load_publisher",
                "load_inquirer",
                "maybe_send_health_check_signal",
                "publish_load_snapshot",
                "maybe_sleep_on_idle",
            )
        },
    )
    s.is_fully_idle = lambda: True
    s.invariant_checker._check_all_pools.return_value = (False, [])
    s.token_to_kv_pool_allocator.verify_byte_accounting.return_value = []

    s.on_idle()

    assert s.invariant_checker._check_all_pools.called == (coordinator is None)
    assert s.invariant_checker._check_req_pool.called == (coordinator is None)
    s.invariant_checker._check_tree_cache.assert_called_once_with()
    s.maybe_sleep_on_idle.assert_called_once_with()


# S08 ------------------------------------------------------------------------


def test_staging_requests_keep_the_scheduler_busy_except_for_health_checks():
    coordinator = SimpleNamespace(has_ongoing_staging=lambda: True)
    s = scheduler(
        running_batch=SimpleNamespace(is_empty=lambda: True),
        chunked_req=None,
        dllm_manager=SimpleNamespace(any_staging_reqs=lambda: False),
        last_batch=None,
        enable_overlap=False,
        waiting_queue=[],
        _engine_paused=False,
        disaggregation_mode=DisaggregationMode.NULL,
        grammar_manager=SimpleNamespace(grammar_queue=[]),
        enable_hisparse=False,
        enable_hierarchical_cache=False,
        enable_lmcache=False,
        hisparse_coordinator=coordinator,
    )
    parallel = SimpleNamespace(pp_size=1)
    with patch("sglang.srt.managers.scheduler.get_parallel", return_value=parallel):
        assert not s.is_fully_idle()
        assert s.is_fully_idle(for_health_check=True)
        assert not s.is_fully_idle(ignore_waiting=True)
        coordinator.has_ongoing_staging = lambda: False
        assert s.is_fully_idle()


# S09 ------------------------------------------------------------------------


class StagedReq:
    rid = "staged"
    to_finish = None

    def finished(self):
        return False


@pytest.mark.parametrize("leases", [True, False])
def test_abort_reaches_requests_waiting_in_qsa_staging(leases):
    staged = StagedReq()
    s = scheduler(
        chunked_req=None,
        mm_receiver=None,
        waiting_queue=[],
        dllm_config=None,
        grammar_manager=Mock(),
        disaggregation_mode=DisaggregationMode.NULL,
        running_batch=SimpleNamespace(reqs=[]),
        last_batch=None,
        enable_continuous_input_polling=False,
        hisparse_coordinator=SimpleNamespace(
            uses_qsa_hisparse_leases=leases,
            ack_staging_queue=[SimpleNamespace(req=staged)],
        ),
    )
    parallel = SimpleNamespace(pp_size=1)
    with patch("sglang.srt.managers.scheduler.get_parallel", return_value=parallel):
        s.abort_request(
            SimpleNamespace(
                rid="staged", abort_all=False, abort_message=None, finished_reason=None
            )
        )
    assert isinstance(staged.to_finish, FINISH_ABORT) == leases


# B06 ------------------------------------------------------------------------


def test_any_weight_load_attempt_invalidates_host_prefixes_first():
    events = []
    cache = SimpleNamespace(invalidate_model=lambda: events.append("invalidate"))

    def failed_load(model_path, load_format, recapture_cuda_graph=False):
        events.append("load")
        return False, "load failed"

    runner = SimpleNamespace(
        weight_updater=SimpleNamespace(update_weights_from_disk=failed_load)
    )
    # The manager reads its TP group from the published parallel bundle.
    parallel = SimpleNamespace(tp_group=SimpleNamespace(cpu_group=None))
    with patch(
        "sglang.srt.managers.scheduler_components.weight_updater.get_parallel",
        return_value=parallel,
    ):
        manager = SchedulerWeightUpdaterManager(
            tp_worker=SimpleNamespace(weight_update_runners=lambda: [("target", runner)]),
            draft_worker=None,
            memory_saver_adapter=None,
            flush_cache=Mock(),
            is_fully_idle=lambda: True,
            scheduler=SimpleNamespace(tree_cache=cache),
        )
    output = manager.update_weights_from_disk(
        SimpleNamespace(
            model_path="m", load_format=None, recapture_cuda_graph=False, flush_cache=False
        )
    )
    assert not output.success
    assert events == ["invalidate", "load"]
    manager.flush_cache.assert_not_called()


# P01 ------------------------------------------------------------------------


def chunked_continuation(checkpoint_limit, chunked_req_limit=None):
    """add_chunked_req for a 201-token prompt with 64 tokens already cached."""
    allocator = PagedTokenToKVPoolAllocator(4096, 64, torch.uint8, "cpu", SimpleNamespace(), False)
    pool = ReqToTokenPool(size=1, max_context_len=4096, device="cpu", enable_memory_saver=False)
    cache = ChunkCache(
        CacheInitParams(
            disable=True, req_to_token_pool=pool, token_to_kv_pool_allocator=allocator, page_size=64
        )
    )
    cache.prefill_checkpoint_limit = lambda req: checkpoint_limit
    adder = PrefillAdder(64, cache, allocator, None, 1.0, 4096, 4096)
    adder.chunked_req_limit = chunked_req_limit  # As the scheduler sets it.
    req = SimpleNamespace(
        full_untruncated_fill_ids=list(range(201)),
        prefix_indices=torch.arange(64),
        sampling_params=SimpleNamespace(max_new_tokens=8),
        output_ids=[],
        retracted_stain=False,
        kv=SimpleNamespace(mamba_pool_idx=None),
    )
    req.set_extend_range = lambda start, end: setattr(
        req, "extend_range", SimpleNamespace(start=start, end=end, length=end - start)
    )
    return adder.add_chunked_req(req), req


@pytest.mark.parametrize(
    "checkpoint_limit, chunked_req_limit, end, unfinished",
    [
        (None, None, 201, False),  # No checkpoint: the whole remainder.
        (128, None, 192, True),  # Stops at the last full page (fork 960-964).
        (128, 64, 128, True),  # The tighter upstream limit still applies.
        (128, 200, 192, True),
    ],
)
def test_a_chunk_continuation_stops_at_the_host_prefix_checkpoint(
    checkpoint_limit, chunked_req_limit, end, unfinished
):
    result, req = chunked_continuation(checkpoint_limit, chunked_req_limit)
    assert (req.extend_range.start, req.extend_range.end) == (64, end)
    assert (result is req) == unfinished
