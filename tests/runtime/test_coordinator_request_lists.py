"""Main's scheduler observers must see requests parked in QSA host staging."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.invariant_checker import (
    SchedulerInvariantChecker,
)
from sglang_qsa_hisparse.hisparse.coordinator import QSAHiSparseCoordinator


class Request:
    def __init__(self, rid):
        self.rid = rid
        self.kv = SimpleNamespace(holds_kv=True)


def test_scheduler_and_row_observer_include_real_qsa_staging(monkeypatch):
    monkeypatch.delenv("SGLANG_QSA_P2_VALIDATE_INITIAL_PAIR", raising=False)
    monkeypatch.setenv("SGLANG_ENABLE_RUNNING_REQS_CHECK", "1")
    coordinator = QSAHiSparseCoordinator(
        SimpleNamespace(max_requests=2, mode="p2-offload", real=torch.ones(1)),
        None,
    )
    first, second = Request("first"), Request("second")
    coordinator.ack_staging_queue = [
        SimpleNamespace(req=first), SimpleNamespace(req=second)
    ]
    queue = coordinator.ack_staging_queue
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.waiting_queue = []
    scheduler.grammar_manager = SimpleNamespace(grammar_queue=[])
    scheduler.chunked_req = None
    scheduler.running_reqs = []
    scheduler.disagg_prefill_bootstrap_queue = None
    scheduler.disagg_prefill_transfer_queue = None
    scheduler.dllm_config = None
    scheduler.disagg_decode_prealloc_queue = None
    scheduler.disagg_decode_transfer_queue = None
    scheduler.hisparse_coordinator = coordinator
    scheduler.uocr_runtime = None
    scheduler.deferred_responses = None

    lists = scheduler.request_lists()
    assert lists == {
        "waiting": [], "grammar": [], "chunked": [], "running": [],
        "hisparse_staging": [first, second],
    }
    assert coordinator.ack_staging_queue is queue
    lists["hisparse_staging"].clear()
    assert coordinator.request_lists() == {"hisparse_staging": [first, second]}

    checker = SchedulerInvariantChecker.__new__(SchedulerInvariantChecker)
    checker.get_request_lists = scheduler.request_lists
    checker.get_last_batch = lambda: None
    checker.get_running_batch = lambda: None
    checker.tree_cache = SimpleNamespace(session_records=lambda: {})
    assert checker._requests_owning_rows() == [first, second]
    checker.maybe_check_request_lists()

    # The observer follows the queue after staging admission without retaining
    # an old snapshot or assigning a second owner to a request.
    coordinator.ack_staging_queue.pop(0)
    scheduler.running_reqs.append(first)
    assert checker._requests_owning_rows() == [first, second]
    checker.maybe_check_request_lists()
    scheduler.running_reqs.append(second)
    with pytest.raises(RuntimeError, match="second is in both running and hisparse_staging"):
        checker.maybe_check_request_lists()
