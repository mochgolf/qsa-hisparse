"""QSA's single physical prefill lane and request-scoped decode leases.

Logical token pages stay in the ordinary paged allocator until request release.
These leases own physical backing only; a batch row is never a lease identity.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class QSAHiSparseLease:
    req_pool_idx: int
    generation: int
    rid: str
    slot: int


class QSAHiSparseSlots:
    def __init__(self, staging_tokens: int, page_size: int, max_requests: int):
        if (page_size < 4 or page_size % 4 or staging_tokens <= 0
                or staging_tokens % page_size or not 1 <= max_requests <= 8):
            raise ValueError("QSA P2 needs page-aligned C4 staging and at most eight leases")
        self.staging_tokens = staging_tokens
        self.page_size = page_size
        self.max_requests = max_requests
        self.active = {}
        self.phases = {}
        self.last_generation = {}
        self.free_slots = list(range(max_requests))
        self.prefill_owner = None

    @property
    def raw_pool_size(self):
        # MHA adds page_size padding. Staging occupies [page_size, size+page_size).
        # Each lease has a padding row followed by C4 members 1..4.
        return self.staging_tokens + 5 * self.max_requests

    def acquire(self, req_pool_idx: int, generation: int, rid: str):
        if req_pool_idx <= 0 or generation <= 0 or not rid:
            raise ValueError("QSA lease requires a real request slot, generation and ID")
        if req_pool_idx in self.active:
            raise RuntimeError("request slot already owns a QSA lease")
        if generation <= self.last_generation.get(req_pool_idx, 0):
            raise RuntimeError("stale QSA request generation")
        if self.prefill_owner is not None:
            raise RuntimeError("only one QSA prefill may own physical staging")
        if not self.free_slots:
            raise RuntimeError("QSA host/hot lease capacity exhausted")
        lease = QSAHiSparseLease(req_pool_idx, generation, rid, self.free_slots.pop(0))
        self.active[req_pool_idx] = lease
        self.last_generation[req_pool_idx] = generation
        self.phases[req_pool_idx] = "prefill"
        self.prefill_owner = lease
        return lease

    def require(self, lease, *phases):
        if self.active.get(lease.req_pool_idx) != lease:
            raise RuntimeError("stale QSA lease callback")
        if phases and self.phases[lease.req_pool_idx] not in phases:
            raise RuntimeError("QSA lease phase does not permit this operation")

    def staging_slice(self, lease, start: int, stop: int):
        self.require(lease, "prefill", "copying")
        if self.prefill_owner != lease:
            raise RuntimeError("raw staging belongs to another request")
        if not 0 <= start <= stop <= self.staging_tokens:
            raise ValueError("raw staging position exceeds one-context backing")
        # ponytail: fixed position addressing requires one prefill and no prefix sharing.
        return slice(self.page_size + start, self.page_size + stop)

    def ring_slice(self, lease):
        self.require(lease, "prefill", "copying", "host_ready", "decode")
        start = self.page_size + self.staging_tokens + 5 * lease.slot
        return slice(start, start + 5)

    def ring_write_location(self, lease, seq_len: int):
        self.require(lease, "decode")
        if not 1 <= seq_len <= self.staging_tokens:
            raise ValueError("QSA decode length exceeds request capacity")
        return self.ring_slice(lease).start + 1 + (seq_len - 1) % 4

    def begin_handoff(self, lease):
        self.require(lease, "prefill")
        if self.prefill_owner != lease:
            raise RuntimeError("handoff lost its staging owner")
        self.phases[lease.req_pool_idx] = "copying"

    def finish_handoff(self, lease, completion_event):
        self.require(lease, "copying")
        if completion_event is None or not completion_event.query():
            raise RuntimeError("cannot reuse staging before all layer copies complete")
        if self.prefill_owner != lease:
            raise RuntimeError("handoff lost its staging owner")
        self.phases[lease.req_pool_idx] = "host_ready"
        self.prefill_owner = None

    def admit_decode(self, lease):
        # The coordinator calls this only after TP-wide ready convergence.
        self.require(lease, "host_ready")
        self.phases[lease.req_pool_idx] = "decode"

    def drain(self, lease, terminal_event, copy_events):
        self.require(lease, "prefill", "copying", "host_ready", "decode", "failed")
        copy_events = tuple(copy_events)
        if terminal_event is None or any(event is None for event in copy_events):
            raise ValueError("release requires explicit terminal and copy events")
        self.phases[lease.req_pool_idx] = "releasing"
        try:
            terminal_event.synchronize()
            for event in copy_events:
                event.synchronize()
        except Exception:
            # Retain every lease on failure; the caller must not free its backing.
            self.phases[lease.req_pool_idx] = "failed"
            raise
        if self.prefill_owner == lease:
            self.prefill_owner = None
        self.phases[lease.req_pool_idx] = "drained"

    def logical_flushed(self, lease, allocator):
        self.require(lease, "drained")
        if allocator.free_group is not None:
            raise RuntimeError("QSA logical free group has not flushed")
        self.phases[lease.req_pool_idx] = "logical_flushed"

    def commit_release(self, lease):
        self.require(lease, "logical_flushed")
        del self.active[lease.req_pool_idx]
        del self.phases[lease.req_pool_idx]
        self.free_slots.append(lease.slot)
        self.free_slots.sort()

    def snapshot(self):
        return {
            "staging_capacity_tokens": self.staging_tokens,
            "staging_available_tokens": self.staging_tokens if self.prefill_owner is None else 0,
            "staging_owner": None if self.prefill_owner is None else self.prefill_owner.req_pool_idx,
            "raw_pool_size_tokens": self.raw_pool_size,
            "ring_reserved_tokens": 5 * self.max_requests,
            "lease_capacity": self.max_requests,
            "lease_active": len(self.active),
            "lease_free": len(self.free_slots),
        }
