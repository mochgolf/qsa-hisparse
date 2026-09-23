"""Scheduler adapter for host prefixes without upstream radix page sharing."""

import hashlib

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.base_prefix_cache import MatchResult
from sglang.srt.mem_cache.chunk_cache import ChunkCache
from sglang.srt.mem_cache.qsa_hisparse.prefix import token_bytes


class QSAHostPrefixCache(ChunkCache):
    def __init__(self, chunk_cache, runtime, tp_group):
        self.__dict__.update(chunk_cache.__dict__)
        self.runtime, self.tp_group = runtime, tp_group
        self.host = runtime.prefix_cache
        self.matches = {}
        self.restoring = {}

    def _namespace(self, req):
        # Exact model/position identity for plain text; unsupported inputs miss.
        if any(
            getattr(req, name, None) is not None
            for name in (
                "multimodal_inputs",
                "positional_embed_overrides",
                "input_embeds",
                "position_ids",
                "mrope_positions",
                "session",
                "beam_group",
            )
        ) or getattr(req, "return_hidden_states", False):
            return None
        return (
            self.runtime.prefix_namespace,
            self.host.epoch,
            getattr(req, "extra_key", None),
            getattr(req, "cache_salt", None),
            getattr(req, "lora_id", None),
        )

    def _converge(self, signature):
        if self.tp_group is None:  # Scaled CPU reference tests.
            return
        votes = [None] * torch.distributed.get_world_size(self.tp_group)
        torch.distributed.all_gather_object(votes, signature, group=self.tp_group)
        if any(vote != signature for vote in votes):
            raise RuntimeError(
                "QSA TP ranks disagree on prefix checkpoint identity/readiness"
            )

    def match_prefix(self, params):
        req = params.req
        if req is None:
            return super().match_prefix(params)
        handle = req.cache_request_handle
        namespace = self._namespace(req)
        if handle in self.restoring:
            self._converge((handle, namespace, "undrained restore"))
            raise RuntimeError("QSA cannot rematch an undrained prefix restore")
        self.release_aborted_request(handle)
        if (
            namespace is None
            or getattr(req, "skip_radix_cache_insert", False)
            or params.key.is_bigram
            or envs.SGLANG_RADIX_FORCE_MISS.get()
        ):
            self._converge((handle, namespace, None))
            return super().match_prefix(params)
        reader = self.host.acquire(namespace, token_bytes(params.key), len(params.key))
        try:
            # A rank-local hit changes chunk shape and logical admission demand.
            # Converge before either rank can make those scheduler decisions.
            self._converge(
                (
                    handle,
                    namespace,
                    None if reader is None else reader.snapshot.signature,
                )
            )
        except BaseException:
            if reader is not None:
                reader.close()
            raise
        if reader is None:
            return super().match_prefix(params)
        self.matches[handle] = reader
        # This tensor communicates a length only. It never names physical pages;
        # prepare_prefix_for_extend MUST replace it before allocation/index use.
        return MatchResult(
            device_indices=torch.full((reader.snapshot.length,), -1, dtype=torch.int64),
            last_device_node=None,
            last_host_node=None,
            best_match_node=None,
            cache_protected_len=0,
        )

    def pending_prefix_tokens(self, req):
        reader = self.matches.get(req.cache_request_handle)
        return 0 if reader is None else reader.snapshot.length

    def prefill_checkpoint_limit(self, req):
        """Split an unaligned final prefill at its last reusable page.

        Salt does not change checkpoint scheduling. A salted miss computes the
        prefix before its unaligned suffix and supplies an uncached control.
        """
        if self._namespace(req) is None or getattr(
            req, "skip_radix_cache_insert", False
        ):
            return None
        length = len(req.full_untruncated_fill_ids)
        if length % self.page_size == 0:
            return None  # The existing complete-final-prefill capture is sufficient.
        boundary = length // self.page_size * self.page_size
        remaining = boundary - len(req.prefix_indices)
        return remaining if remaining > 0 else None

    def prepare_prefix_for_extend(self, reqs):
        for req in reqs:
            handle = req.cache_request_handle
            if handle in self.restoring:
                raise RuntimeError("QSA cannot overwrite an undrained prefix restore")
            reader = self.matches.get(handle)
            self.restoring[handle] = dict(
                req=req,
                reader=reader,
                prefix=None,
                suffix=None,
                fresh=req.kv.kv_allocated_len == 0,
                previous_allocated=req.kv.kv_allocated_len,
                previous_committed=req.kv.kv_committed_len,
            )
            signature = None if reader is None else reader.snapshot.signature
            self._converge((handle, signature))
            if reader is None:
                if bool((req.prefix_indices < 0).any()):
                    raise RuntimeError("QSA prefix marker lost its snapshot owner")
                continue
            count = reader.snapshot.length
            if len(req.prefix_indices) != count or not bool(
                (req.prefix_indices == -1).all()
            ):
                raise RuntimeError("QSA prefix marker was mutated before allocation")
            if not self.restoring[handle]["fresh"]:
                raise RuntimeError(
                    "QSA prefix restore must begin with a fresh request row"
                )
            prefix = self.token_to_kv_pool_allocator.alloc(count)
            if prefix is None:
                raise RuntimeError("QSA admitted prefix has insufficient logical pages")
            self.restoring[handle]["prefix"] = prefix
            req.prefix_indices = prefix.long()
            req.kv.cache_protected_len = 0

    def note_extend_allocation(self, reqs, out_cache_loc):
        offset = 0
        for req in reqs:
            count = req.extend_range.length
            record = self.restoring.get(req.cache_request_handle)
            if record is not None:
                record["suffix"] = out_cache_loc[offset : offset + count]
            offset += count

    def restore_prefix_for_extend(self, reqs):
        for req in reqs:
            handle = req.cache_request_handle
            record = self.restoring.get(handle)
            if record is None:
                continue
            reader = record["reader"]
            if reader is None:
                continue
            snapshot = reader.snapshot
            namespace = self._namespace(req)
            if snapshot.namespace != namespace:
                raise RuntimeError("QSA prefix identity changed before restoration")
            record["restore_stream"] = torch.cuda.current_stream(self.runtime.device)
            self.runtime.restore_prefix(req, snapshot)
            self.runtime.note_prefix_basis(req, snapshot, reader.entry_id)
            self.host.reused_tokens += snapshot.length
            req.host_hit_length = req.host_loaded_length = snapshot.length
            reader.close()
            self.matches.pop(handle, None)
        self.restoring.clear()

    def rollback_prefix_for_extend(self, reqs):
        """Undo a failed first allocation/restore without a historical lease."""
        for req in reqs:
            handle = req.cache_request_handle
            record = self.restoring.get(handle)
            if record is None:
                self.release_aborted_request(handle)
                continue
            req_idx = req.kv.req_pool_idx
            restore_stream = record.get("restore_stream")
            if restore_stream is not None:
                restore_stream.synchronize()
                record.pop("restore_stream", None)
            if not record["fresh"]:
                suffix = record["suffix"]
                if suffix is not None:
                    # An extension inside an already owned partial page must
                    # retain that page. Only positions above its end are new.
                    old = record["previous_allocated"]
                    skip = (-old) % self.page_size
                    if len(suffix) > skip:
                        self.token_to_kv_pool_allocator.free(suffix[skip:])
                    record["suffix"] = None
                req.kv.kv_allocated_len = record["previous_allocated"]
                req.kv.kv_committed_len = record["previous_committed"]
                self.restoring.pop(handle, None)
                self.release_aborted_request(handle)
                continue
            # A ledger write can fail after drain or commit. Keep the original
            # identity and record each completed cleanup stage before retrying.
            record.setdefault("req_idx", req_idx)
            if "lease" not in record:
                record["lease"] = self.runtime.slots.active.get(record["req_idx"])
            lease = record["lease"]
            if not record.get("drained", False):
                if (
                    lease is not None
                    and self.runtime.slots.active.get(lease.req_pool_idx) == lease
                    and self.runtime.slots.phases[lease.req_pool_idx]
                    in ("drained", "logical_flushed")
                ):
                    record["drained"] = True
                else:
                    lease = self.runtime.release(record["req_idx"], req.rid)
                    record["lease"] = lease
                    record["drained"] = True
            allocations = [
                record[k] for k in ("prefix", "suffix") if record[k] is not None
            ]
            if allocations:
                self.token_to_kv_pool_allocator.free(torch.cat(allocations))
                record["prefix"] = record["suffix"] = None
            if req.kv.mamba_pool_idx is not None:
                self.req_to_token_pool.free_mamba_cache(req)
            if req.kv.req_pool_idx is not None:
                self.req_to_token_pool.free(req)
            req.kv.mark_kv_released()
            req.kv.kv_allocated_len = req.kv.kv_committed_len = 0
            req.prefix_indices = torch.empty(0, dtype=torch.int64)
            completed = False
            try:
                self.runtime.after_release(lease)
                completed = True
            finally:
                # Once release committed, or durable ownership moved to the
                # runtime's logical-flush queue, an event-log failure must not
                # preserve a transaction that would replay already freed pages.
                if (
                    completed
                    or lease is None
                    or self.runtime.slots.active.get(lease.req_pool_idx) != lease
                    or lease in self.runtime.pending_releases
                ):
                    if record["reader"] is not None:
                        record["reader"].close()
                    self.matches.pop(handle, None)
                    self.restoring.pop(handle, None)

    def _capture(self, req):
        namespace = self._namespace(req)
        if namespace is None or getattr(req, "skip_radix_cache_insert", False):
            return
        state = self.runtime.requests.get(req.kv.req_pool_idx)
        if (
            state is None
            or self.runtime.slots.phases[state.lease.req_pool_idx] != "prefill"
        ):
            return
        length = state.seq_len
        if not length or length % 64:
            return
        tokens = token_bytes(req.full_untruncated_fill_ids[:length])
        captured = self.runtime.capture_prefix(req, namespace, tokens)
        reservation = None if captured is None else captured[0]
        try:
            signature = (
                namespace,
                length,
                hashlib.sha256(tokens).hexdigest(),
                captured is not None,
            )
            self._converge(
                signature
            )  # Capture above completed all layer and sibling copies.
            if captured is not None:
                entry_id = self.host.publish(reservation, captured[1], completed=True)
                self.runtime.note_prefix_basis(req, captured[1], entry_id)
                self.runtime.record(
                    "prefix_publish_complete",
                    state.lease,
                    checkpoint_tokens=length,
                    **self.host.stats(),
                )
        finally:
            if reservation is not None:
                reservation.close()

    def before_release(self, req, is_insert):
        if req.cache_request_handle in self.restoring:
            raise RuntimeError(
                "QSA prefix restore must drain through rollback before release"
            )
        if is_insert:
            self._capture(req)

    def cache_unfinished_req(self, req, chunked=False):
        self._capture(req)
        super().cache_unfinished_req(req, chunked=chunked)

    def cache_finished_req(self, req, is_insert=True, *, owned_kv_len):
        self.release_aborted_request(req.cache_request_handle)
        super().cache_finished_req(req, is_insert, owned_kv_len=owned_kv_len)

    def release_aborted_request(self, handle):
        # Failed restore submission can retain an undrained stream and its
        # snapshot. Only successful rollback may drop that ownership pin.
        if handle in self.restoring:
            return
        reader = self.matches.pop(handle, None)
        if reader is not None:
            reader.close()

    def reset(self):
        if self.restoring:
            raise RuntimeError("QSA prefix flush cannot race a restore")
        for handle in list(self.matches):
            self.release_aborted_request(handle)
        self.host.reset()

    def release_host_resources(self):
        self.reset()

    def invalidate_model(self):
        self.reset()
