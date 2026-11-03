"""Host prefixes for QSA leases on upstream's disabled UnifiedRadixCache.

M05 builds this cache under ``--disable-radix-cache`` with
``create_unified_radix_cache(ctx, cache_class=QSAHostPrefixCache)``. A
host hit is reported as ``host_hit_length`` and accepted in
``init_load_back`` (request-owned: ``load_back_is_cache_owned`` is False).
Allocation (M01's hook around ``alloc_for_extend``) records every request in
``prepare_prefix_for_extend``, takes fresh logical pages for a hit from
``prefix_device_indices`` and restores the snapshot in
``maybe_hand_to_session``, which upstream calls once per request after the
row is written; a failed allocation is undone by
``rollback_prefix_for_extend``. ``checkpoint`` captures before upstream's
disabled checkpoint, which frees a finished request's mamba slot.
"""

import hashlib

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.mem_cache.static_kv_storage import kv_storage_of
from sglang_qsa_hisparse.hisparse.image_identity import ImagePrefixIdentity
from sglang_qsa_hisparse.hisparse.image_request import identity_for
from sglang_qsa_hisparse.hisparse.prefix import image_key, key_digest, token_bytes


class QSAHostPrefixCache(UnifiedRadixCache):
    # UnifiedRadixCache.__init__ calls reset() before the host cache is bound.
    host = None

    def __init__(self, params):
        runtime = params.token_to_kv_pool_allocator.get_kvcache().qsa_hisparse
        if runtime.prefix_cache is None:
            raise ValueError("QSA host prefixes require the runtime's host prefix cache")
        self.runtime = runtime
        self.matches = {}
        self.restoring = {}
        super().__init__(params)  # self.tp_group = params.tp_cache_group
        self.host = runtime.prefix_cache

    def _namespace(self, req):
        # Exact model/position identity for plain text and supported images
        # (whose image identity is matched separately); unsupported inputs miss.
        image = identity_for(req)
        if image is not None and not isinstance(image, ImagePrefixIdentity):
            return None  # Track I bypass set.
        if any(
            getattr(req, name, None) is not None
            for name in (
                *(("multimodal_inputs",) if image is None else ()),
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
        reader = self.host.acquire(
            namespace,
            token_bytes(params.key),
            len(params.key),
            identity=identity_for(req),
        )
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
        # A host hit at the root: admission accepts it in init_load_back, and
        # the restored pages belong to the request (no tree node holds them).
        return super().match_prefix(params)._replace(
            host_hit_length=reader.snapshot.length
        )

    def init_load_back(self, params):
        reader = self.matches.get(params.req.cache_request_handle)
        if reader is None:
            return super().init_load_back(params)
        return reader.snapshot.length, params.req.last_node

    def load_back_is_cache_owned(self):
        return False

    def pending_prefix_tokens(self, req):
        reader = self.matches.get(req.cache_request_handle)
        return 0 if reader is None else reader.snapshot.length

    def prefill_checkpoint_limit(self, req):
        """Split an unaligned final prefill at its last reusable page.

        Salt does not change checkpoint scheduling. A salted miss computes the
        prefix before its unaligned suffix and supplies an uncached control.
        Supported image requests split the same way, also inside an image.
        """
        if self._namespace(req) is None or getattr(
            req, "skip_radix_cache_insert", False
        ):
            return None
        length = len(req.full_untruncated_fill_ids)
        if length % self.page_size == 0:
            return None  # The existing complete-final-prefill capture is sufficient.
        boundary = length // self.page_size * self.page_size
        # A pending host hit becomes the prefix when admission accepts it.
        prefix = max(req.prefix_len, self.pending_prefix_tokens(req))
        remaining = boundary - prefix
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
                fresh=req.kv.kv_allocated_len == 0,
                previous_allocated=req.kv.kv_allocated_len,
                previous_committed=req.kv.kv_committed_len,
            )
            signature = None if reader is None else reader.snapshot.signature
            self._converge((handle, signature))
            if reader is None:
                if not req.kv.holds_kv and req.prefix_len:
                    raise RuntimeError("QSA prefix lost its snapshot owner")
                continue
            if req.prefix_len != reader.snapshot.length:
                raise RuntimeError("QSA prefix was not admitted before allocation")
            if not self.restoring[handle]["fresh"]:
                raise RuntimeError(
                    "QSA prefix restore must begin with a fresh request row"
                )

    def prefix_device_indices(self, req):
        record = self.restoring.get(req.cache_request_handle)
        if record is None or record["reader"] is None:
            return super().prefix_device_indices(req)
        if record["prefix"] is None:
            prefix = self.token_to_kv_pool_allocator.alloc(req.prefix_len)
            if prefix is None:
                raise RuntimeError("QSA admitted prefix has insufficient logical pages")
            record["prefix"] = prefix
        return record["prefix"].long()

    def maybe_hand_to_session(self, req):
        super().maybe_hand_to_session(req)
        handle = req.cache_request_handle
        record = self.restoring.get(handle)
        if record is None:
            return
        reader = record["reader"]
        if reader is not None:
            snapshot = reader.snapshot
            namespace = self._namespace(req)
            if snapshot.namespace != namespace:
                raise RuntimeError("QSA prefix identity changed before restoration")
            record["restore_stream"] = torch.cuda.current_stream(self.runtime.device)
            self.runtime.restore_prefix(req, snapshot)
            self.runtime.note_prefix_basis(req, snapshot, reader.entry_id)
            self.host.reused_tokens += snapshot.length
            reader.close()
            self.matches.pop(handle, None)
        self.restoring.pop(handle)

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
            # Upstream writes the row and then sets kv_allocated_len to the new
            # length; before that, its own failure path frees out_cache_loc.
            if "suffix" not in record:
                start = record["previous_allocated"]
                if record["prefix"] is not None:
                    start = len(record["prefix"])
                end = req.kv.kv_allocated_len
                record["suffix"] = (
                    self.req_to_token_pool.req_to_token[req_idx, start:end].long()
                    if req_idx is not None and end > record["previous_allocated"]
                    else None
                )
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
                kv_storage_of(
                    self.req_to_token_pool, self.token_to_kv_pool_allocator
                ).release_state_slot(req)
            if req.kv.req_pool_idx is not None:
                self.req_to_token_pool.free(req)
            req.kv.mark_kv_released()
            req.kv.kv_allocated_len = req.kv.kv_committed_len = 0
            req.prefix_len = 0
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
                key_digest(image_key(identity_for(req), length)),
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

    def claim_kv_row(self, req):
        # First hook of every release path, before any row or lease teardown:
        # a pending host restore must drain through rollback, not be released.
        if req.cache_request_handle in self.restoring:
            raise RuntimeError(
                "QSA prefix restore must drain through rollback before release"
            )
        return super().claim_kv_row(req)

    def checkpoint(self, req, *, up_to, **kwargs):
        # The capture observes the live lease and mamba slot; the disabled
        # checkpoint frees a finished request's mamba slot.
        self._capture(req)
        return super().checkpoint(req, up_to=up_to, **kwargs)

    def on_release(self, req, *, checkpointed):
        self.release_aborted_request(req.cache_request_handle)
        return super().on_release(req, checkpointed=checkpointed)

    def release_aborted_request(self, handle):
        # Failed restore submission can retain an undrained stream and its
        # snapshot. Only successful rollback may drop that ownership pin.
        if handle not in self.restoring:
            reader = self.matches.pop(handle, None)
            if reader is not None:
                reader.close()
        super().release_aborted_request(handle)

    def _drop_host_prefixes(self):
        if self.restoring:
            raise RuntimeError("QSA prefix flush cannot race a restore")
        for handle in list(self.matches):
            self.release_aborted_request(handle)
        self.host.reset()

    def reset(self):
        if self.host is not None:
            self._drop_host_prefixes()
        super().reset()

    def release_host_resources(self):
        self._drop_host_prefixes()
        super().release_host_resources()

    def invalidate_model(self):
        self._drop_host_prefixes()
