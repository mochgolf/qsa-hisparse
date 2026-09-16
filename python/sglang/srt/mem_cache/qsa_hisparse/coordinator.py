"""Scheduler staging and TP readiness for request-scoped QSA leases."""

import os
from types import SimpleNamespace

import torch


class QSAHiSparseCoordinator:
    """QSA implementation of the existing staging/decode scheduler hooks."""

    uses_qsa_hisparse_leases = True

    def __init__(self, adapter, tp_group):
        self.adapter, self.tp_group = adapter, tp_group
        self.ack_staging_queue = []
        initial_batch = os.environ.get("SGLANG_QSA_P2_VALIDATE_INITIAL_PAIR", "0")
        if initial_batch not in ("0", "1", "2", "4", "8"):
            raise ValueError("QSA P2 initial-batch validation must be 0, 1, 2, 4 or 8")
        # Keep 1 as the accepted B2 spelling. Larger values let bounded tests
        # park sequential near-context prefills until one real decode batch exists.
        self.initial_batch_target = 2 if initial_batch == "1" else int(initial_batch)
        if self.initial_batch_target > adapter.max_requests:
            raise ValueError("QSA P2 initial-batch target exceeds lease capacity")
        self.wait_initial_pair = self.initial_batch_target != 0
        self.num_real_reqs = (
            adapter.real
            if adapter.mode == "p2-offload"
            else torch.ones(1, dtype=torch.int32, device=adapter.device)
        )

    def set_decode_producer_stream(self, stream):
        if stream is None:
            raise ValueError(
                "QSA coordinator requires the real forward producer stream"
            )
        self.adapter.producer_stream = stream

    def admit_request_into_staging(self, req):
        if getattr(req, "beam_group", None) is not None:
            raise ValueError("QSA P2 does not support beam/prefix ownership sharing")
        if self.adapter.producer_stream is None:
            raise RuntimeError("QSA handoff has no prefill producer stream")
        producer = torch.cuda.Event()
        producer.record(self.adapter.producer_stream)
        torch.cuda.current_stream(self.adapter.device).wait_event(producer)
        lease, event = self.adapter.handoff(req)
        req.hisparse_staging = True
        self.ack_staging_queue.append(
            SimpleNamespace(req=req, lease=lease, event=event)
        )
        self.adapter.record("ready_parked", lease)

    def has_ongoing_staging(self):
        return bool(self.ack_staging_queue)

    def collect_ready_reqs(self):
        # An empty local queue must still rendezvous: the peer may have an item.
        total = torch.tensor(len(self.ack_staging_queue), dtype=torch.int64)
        torch.distributed.all_reduce(total, group=self.tp_group)
        if int(total) == 0:
            return []
        signatures = [
            (x.lease.req_pool_idx, x.lease.generation, x.lease.rid, x.lease.slot)
            for x in self.ack_staging_queue
        ]
        count = 0
        for item in self.ack_staging_queue:
            self.adapter.slots.require(item.lease, "host_ready")
            self.adapter._request(item.lease.req_pool_idx, item.lease.rid)
            if not item.event.query():
                break
            count += 1
        # Admission is rare; compare identities as well as the ready prefix count.
        world = torch.distributed.get_world_size(self.tp_group)
        votes = [None] * world
        torch.distributed.all_gather_object(
            votes, (signatures, count, self.initial_batch_target), group=self.tp_group
        )
        if any(
            ids != signatures or target != self.initial_batch_target
            for ids, _, target in votes
        ):
            raise RuntimeError("QSA TP ranks disagree on staging lease identities")
        count = min(n for _, n, _ in votes)
        if self.wait_initial_pair:
            if count < self.initial_batch_target:
                return []
            self.adapter.record(
                "initial_pair_released",
                ordered_leases=signatures,
                target_count=self.initial_batch_target,
            )
            self.wait_initial_pair = False
            self.initial_batch_target = 0
        ready, self.ack_staging_queue = (
            self.ack_staging_queue[:count],
            self.ack_staging_queue[count:],
        )
        for item in ready:
            self.adapter.slots.admit_decode(item.lease)
            item.req.hisparse_staging = False
            self.adapter.record("ready_converged", item.lease)
        return [item.req for item in ready]

    def map_last_loc_to_buffer(
        self,
        seq_lens,
        out_cache_loc,
        req_pool_indices,
        seq_lens_cpu,
        req_pool_indices_cpu,
    ):
        # Keep out_cache_loc logical: index-K consumes it later in the model.
        for req_idx in req_pool_indices_cpu.tolist():
            state = self.adapter.requests[req_idx]
            self.adapter._request(req_idx, state.lease.rid)
            self.adapter.slots.require(state.lease, "decode")

    def wait_for_pending_backup(self):
        # Each request/layer waits its own writeback in the shared V3 algorithms.
        return

    def request_finished(self, req):
        # common.release_kv_cache owns the single drain -> logical flush -> release.
        self.adapter._request(req.kv.req_pool_idx, req.rid)

    def retract_req(self, req):
        raise RuntimeError("QSA P2 cannot retract an offloaded request into prefill")

    def destroy(self):
        if self.adapter.producer_stream is not None:
            self.adapter.producer_stream.synchronize()
        self.adapter.copy_stream.synchronize()
