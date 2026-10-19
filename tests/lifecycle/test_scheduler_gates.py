"""Scheduler gates under W2's hooks: coordinator adoption (S02), multimodal
decode batches (S03), the staging-to-decode transition (S04), lease admission
(S05), the QSA prefill adder (S06, with P01's chunk checkpoint cap), idle leak
checks (S07), idleness (S08), abort of staging requests (S09) and weight-load
invalidation (B06). Expected values follow production's code at 897286b12a
(the fork's ee8fe158d6 for these gates). S01's host-prefix cache selection
and rejections moved into M05's factory hook at v0.5.22
(test_tree_cache_selection.py)."""

from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.schedule_batch import (
    FINISH_ABORT,
    NextBatchPlan,
    Req,
    ScheduleBatch,
)
from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.weight_updater import (
    SchedulerWeightUpdaterManager,
)
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang_qsa_hisparse.hisparse.prefix_cache import QSAHostPrefixCache
from sglang_qsa_hisparse.hisparse.slots import QSAHiSparseSlots
from sglang_qsa_hisparse.patches.hisparse.scheduler import QSAPrefillAdder

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


# S02 ------------------------------------------------------------------------


def coordinator_scheduler(*, adapter=True, prefix_cache="host", runner_coordinator=True):
    runtime = SimpleNamespace(uses_qsa_hisparse_leases=True, prefix_cache=prefix_cache)
    kvcache = SimpleNamespace(qsa_hisparse=runtime) if adapter else SimpleNamespace()
    allocator = SimpleNamespace(get_kvcache=lambda: kvcache, device="cpu")
    return scheduler(
        enable_hisparse=False,
        enable_hierarchical_cache=False,
        token_to_kv_pool_allocator=allocator,
        tree_cache=object(),
        tp_worker=SimpleNamespace(
            model_runner=SimpleNamespace(
                hisparse_coordinator=Mock() if runner_coordinator else None
            )
        ),
        forward_stream="forward-stream",
        tp_cpu_group="tp-group",
    )


def test_a_lease_scheduler_adopts_the_runner_coordinator_and_keeps_its_cache():
    s = coordinator_scheduler()
    tree_cache = s.tree_cache  # M05 already built the host-prefix cache.
    coordinator = s.tp_worker.model_runner.hisparse_coordinator

    s.init_hisparse_coordinator()

    assert s.hisparse_coordinator is coordinator
    coordinator.set_decode_producer_stream.assert_called_once_with("forward-stream")
    assert s.tree_cache is tree_cache


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
    tree_cache = s.tree_cache
    s.init_hisparse_coordinator()
    assert s.tree_cache is tree_cache
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
    batch = SimpleNamespace(
        return_logprob=False,
        multimodal_inputs=None,
        spec_algorithm=SimpleNamespace(is_none=lambda: True),  # Read at v0.5.22.
    )
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
                "dp_attn_adapter",  # v0.5.22: drop_sync_wait_carry() first.
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


# B03 and the coordinator's speculative-decoding call ------------------------


def test_prefill_results_stage_qsa_leases_without_notifying_finished_ones(
    published_context,
):
    # Production's gate admits an unfinished prefill into QSA staging; v0.5.22's
    # new request_finished call for requests finishing in a prefill batch
    # (#37771) stays on enable_hisparse, which QSA leaves unset (production
    # never notified QSA leases there).
    from lifecycle.test_hisparse_batch_result_processor_hidden_states import (
        _PrefillReq,
    )
    from sglang.srt.managers.scheduler_components.batch_result_processor import (
        SchedulerBatchResultProcessor,
    )
    from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode

    finished, staged = (
        _PrefillReq(rid=rid, inflight_middle_chunks=0, return_hidden_states=False)
        for rid in ("finished", "staged")
    )
    finished.finished = lambda: bool(finished.output_ids)
    for req in (finished, staged):
        req.kv = SimpleNamespace(kv_committed_len=0)
    batch = SimpleNamespace(
        reqs=[finished, staged],
        decoding_reqs=None,
        return_logprob=False,
        return_hidden_states=False,
        return_hidden_states_mode=CaptureHiddenMode.NULL,
        spec_info=None,
        spec_algorithm=SimpleNamespace(is_none=lambda: True),
        prefill_stats=None,
        dp_cooperation_info=None,
    )
    result = SimpleNamespace(
        copy_done=None,
        auxiliary_host_output=None,
        routed_experts_output=None,
        indexer_topk_output=None,
        logits_output=SimpleNamespace(
            hidden_states=None, customized_info=None, sampling_mask_output=None
        ),
        next_token_ids=torch.tensor([5, 6]),
        extend_input_len_per_req=None,
        extend_logprob_start_len_per_req=None,
        grammar_advanced=False,
        can_run_cuda_graph=False,
        skipped_output_comm=False,
    )
    coordinator = Mock()
    processor = SchedulerBatchResultProcessor(
        is_generation=True,
        disaggregation_mode=None,
        enable_overlap=False,
        enable_overlap_mlx=False,
        model_config=SimpleNamespace(think_end_ids=None),
        token_to_kv_pool_allocator=Mock(),
        tree_cache="tree-cache",
        hisparse_coordinator=coordinator,
        req_to_token_pool=None,
        decode_offload_manager=None,
        metrics_collector=None,
        metrics_reporter=Mock(),
        draft_worker=None,
        model_worker=Mock(),
        logprob_result_processor=None,
        output_streamer=Mock(),
        beam_coordinator=Mock(),
        abort_request=lambda *args, **kwargs: None,
    )
    patches = "sglang_qsa_hisparse.patches.hisparse.scheduler"
    with (
        patch(f"{patches}.release_kv_cache") as release,
        patch(f"{patches}.checkpoint_kv_cache") as checkpoint,
        patch.object(SchedulerBatchResultProcessor, "_maybe_collect_routed_experts"),
        patch.object(SchedulerBatchResultProcessor, "_maybe_collect_indexer_topk"),
    ):
        processor.process_batch_result_prefill(batch, result)

    release.assert_called_once_with(finished, "tree-cache", checkpoint=True)
    checkpoint.assert_called_once_with(staged, "tree-cache")
    coordinator.admit_request_into_staging.assert_called_once_with(staged)
    coordinator.request_finished.assert_not_called()


def test_the_qsa_coordinator_takes_prefill_spec_scratch_as_a_no_op():
    # v0.5.22's alloc_for_extend calls it on every prefill batch's coordinator.
    from sglang_qsa_hisparse.hisparse.coordinator import QSAHiSparseCoordinator

    coordinator = QSAHiSparseCoordinator.__new__(QSAHiSparseCoordinator)
    coordinator.adapter = Mock()
    assert coordinator.allocate_spec_scratch([1, 2]) is None
    assert coordinator.adapter.mock_calls == []


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


# S06 and P01 ------------------------------------------------------------------


class AdderReq(SimpleNamespace):
    extend_len = Req.extend_len


def disabled_cache(cache_class=UnifiedRadixCache):
    """--disable-radix-cache's tree (QSAHostPrefixCache: no snapshots needed)."""
    runtime = SimpleNamespace(prefix_cache=SimpleNamespace(epoch=0), prefix_namespace=())
    allocator = PagedTokenToKVPoolAllocator(
        4096, 64, torch.uint8, "cpu", SimpleNamespace(qsa_hisparse=runtime), False
    )
    pool = ReqToTokenPool(size=1, max_context_len=4096, device="cpu", enable_memory_saver=False)
    return cache_class(
        CacheInitParams(
            disable=True,
            req_to_token_pool=pool,
            token_to_kv_pool_allocator=allocator,
            page_size=64,
            tree_components=(ComponentType.FULL,),
        )
    )


@pytest.mark.parametrize("leases", [True, False])
def test_qsa_leases_admit_one_prefill_request_per_pass(published_context, leases):
    s = scheduler(
        hisparse_coordinator=SimpleNamespace(uses_qsa_hisparse_leases=leases)
    )
    adder_class = s.get_prefill_adder_class()
    assert adder_class is (QSAPrefillAdder if leases else PrefillAdder)
    if not leases:
        return
    cache = disabled_cache()
    # As _get_new_batch_prefill_raw builds it, with the scheduler's setting.
    adder = adder_class(
        64, cache, cache.token_to_kv_pool_allocator, None, 1.0, 4096, 4096,
        prefill_max_requests=None,
    )
    assert adder.prefill_max_requests == 1
    adder.can_run_list.append("chunked continuation")
    assert adder.add_one_req(object(), False, None) == AddReqResult.OTHER


def chunked_continuation(checkpoint_limit, chunked_req_limit=None):
    """add_chunked_req for a 201-token prompt with 64 tokens already cached."""
    cache = disabled_cache(QSAHostPrefixCache)
    cache.prefill_checkpoint_limit = lambda req: checkpoint_limit
    adder = QSAPrefillAdder(64, cache, cache.token_to_kv_pool_allocator, None, 1.0, 4096, 4096)
    adder.chunked_req_limit = chunked_req_limit  # As the scheduler sets it.
    req = AdderReq(
        full_untruncated_fill_ids=list(range(201)),
        prefix_len=64,
        sampling_params=SimpleNamespace(max_new_tokens=8),
        output_ids=[],
        retracted_stain=False,
        kv=SimpleNamespace(mamba_pool_idx=None, req_pool_idx=1),
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
    published_context, checkpoint_limit, chunked_req_limit, end, unfinished
):
    result, req = chunked_continuation(checkpoint_limit, chunked_req_limit)
    assert (req.prefix_len, req.extend_end) == (64, end)
    assert (result is req) == unfinished
