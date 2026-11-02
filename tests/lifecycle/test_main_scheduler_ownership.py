"""Real main scheduler collaborators preserve QSA's staging ownership."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import NextBatchPlan
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.invariant_checker import (
    SchedulerInvariantChecker,
)
from sglang.srt.managers.scheduler_components.running_reqs import RunningReqs
from sglang_qsa_hisparse.hisparse.coordinator import QSAHiSparseCoordinator

pytestmark = pytest.mark.usefixtures("w2_patches", "published_context")


class Request:
    rid = "staged"

    def finished(self):
        return False


class Batch:
    def __init__(self, reqs=(), *, extend=False):
        self.reqs = list(reqs)
        self.chunked_req = None
        self.is_prefill_only = False
        self.batch_is_full = False
        self.forward_mode = SimpleNamespace(is_extend=lambda: extend)

    def is_empty(self):
        return not self.reqs

    def batch_size(self):
        return len(self.reqs)

    def filter_batch(self, **kwargs):
        pass

    def merge_batch(self, other):
        self.reqs.extend(other.reqs)


def make_scheduler():
    coordinator = QSAHiSparseCoordinator.__new__(QSAHiSparseCoordinator)
    coordinator.ack_staging_queue = []
    coordinator.tp_group = None
    coordinator.initial_batch_target = 0
    coordinator.wait_initial_pair = False
    coordinator.adapter = SimpleNamespace(slots=Mock(), _request=Mock(), record=Mock())
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.__dict__.update(
        waiting_queue=[],
        grammar_manager=SimpleNamespace(grammar_queue=[]),
        chunked_req=None,
        running_reqs=RunningReqs(),
        disaggregation_mode=DisaggregationMode.NULL,
        disagg_prefill_bootstrap_queue=None,
        disagg_prefill_transfer_queue=None,
        disagg_decode_prealloc_queue=None,
        disagg_decode_transfer_queue=None,
        hisparse_coordinator=coordinator,
        deferred_responses=None,
        uocr_runtime=None,
        dllm_config=None,
        enable_hisparse=False,
    )
    return scheduler, coordinator


def test_qsa_prefill_joins_running_only_after_host_ready():
    scheduler, coordinator = make_scheduler()
    req = Request()
    prefill, running = Batch([req], extend=True), Batch()
    scheduler._join_launched_reqs(prefill)
    # QSA leaves enable_hisparse unset. Main must still exclude this prefill
    # from running; its result hands it to the coordinator's host staging.
    assert list(scheduler.running_reqs) == []
    event = Mock()
    event.query.return_value = False
    coordinator.ack_staging_queue = [
        SimpleNamespace(
            req=req,
            lease=SimpleNamespace(req_pool_idx=1, generation=1, rid=req.rid, slot=0),
            event=event,
        )
    ]
    checker = SchedulerInvariantChecker(
        disaggregation_mode=DisaggregationMode.NULL,
        page_size=64,
        tree_cache=None,
        token_to_kv_pool_allocator=None,
        req_to_token_pool=None,
        pool_stats_observer=None,
        get_last_batch=lambda: prefill,
        get_running_batch=lambda: running,
        scheduler_stage_metrics=None,
        get_request_lists=scheduler.request_lists,
        get_running_reqs=lambda: scheduler.running_reqs,
        prefill_launch_starts_running=scheduler._prefill_launch_starts_running,
    )
    scheduler.__dict__.update(
        scheduler_stage_metrics=None,
        invariant_checker=checker,
        _pending_chunked_abort_req=None,
        enable_hierarchical_cache=False,
        enable_hicache_storage=False,
        enable_unified_cache_external_linker=False,
        require_mlp_sync=False,
        enable_lmcache=False,
        _should_defer_prefill=lambda batch: False,
        _arm_prefill_decode_interval=lambda batch: None,
        get_new_batch_prefill=lambda batch: NextBatchPlan(
            batch_to_run=None, running_batch=batch
        ),
        update_running_batch=lambda batch: batch,
        _build_hisparse_decode_batch=lambda reqs: Batch(reqs),
        dp_attn_adapter=SimpleNamespace(
            maybe_prepare_mlp_sync_batch=lambda batch, **kwargs: batch,
            maybe_convert_decode_to_extend=lambda batch: batch,
        ),
        ngram_embedding_manager=SimpleNamespace(
            prepare_for_forward=lambda batch, **kwargs: batch
        ),
    )

    def vote(votes, signature, group):
        votes[:] = [signature]

    with (
        patch.object(envs.SGLANG_ENABLE_RUNNING_REQS_CHECK, "get", return_value=True),
        patch("torch.distributed.all_reduce"),
        patch("torch.distributed.get_world_size", return_value=1),
        patch("torch.distributed.all_gather_object", side_effect=vote),
    ):
        checker.maybe_check_request_lists()
        parked = scheduler.get_next_batch_to_run(running, prefill)
        assert parked.batch_to_run is None
        assert list(scheduler.running_reqs) == []
        assert scheduler.request_lists()["hisparse_staging"] == [req]
        event.query.return_value = True
        ready = scheduler.get_next_batch_to_run(running, prefill)
        assert ready.running_batch.reqs == [req]
        assert list(scheduler.running_reqs) == [req]
        assert scheduler.request_lists()["hisparse_staging"] == []
        checker.maybe_check_request_lists()


@pytest.mark.parametrize(
    "coordinator_kind,enable_hisparse,role,expected",
    [
        ("qsa", False, DisaggregationMode.NULL, False),
        ("other", False, DisaggregationMode.NULL, True),
        ("none", True, DisaggregationMode.NULL, False),
        ("none", False, DisaggregationMode.PREFILL, False),
    ],
)
def test_staging_gate_keeps_native_and_other_coordinator_results(
    coordinator_kind, enable_hisparse, role, expected
):
    scheduler, _ = make_scheduler()
    if coordinator_kind == "other":
        scheduler.hisparse_coordinator = SimpleNamespace()
    elif coordinator_kind == "none":
        scheduler.hisparse_coordinator = None
    scheduler.enable_hisparse = enable_hisparse
    scheduler.disaggregation_mode = role
    assert scheduler._prefill_launch_starts_running() is expected
