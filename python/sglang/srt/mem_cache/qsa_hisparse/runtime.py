"""Multi-request QSA offload with one prefill arena and persistent decode leases.

The runtime owns allocation and lifecycle. Graph replay only routes existing
leases into fixed buffers; scheduler readiness lives in coordinator.py.
"""

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.qsa_hisparse.config import (
    prefix_cache_options,
    validate_configuration,
)
from sglang.srt.mem_cache.qsa_hisparse.layout import (
    pack_c4,
    stage_short_prefix,
    unpack_index,
)
from sglang.srt.mem_cache.qsa_hisparse.single_request import QSAHiSparseSingleRequest
from sglang.srt.mem_cache.qsa_hisparse.slots import QSAHiSparseSlots
from sglang.srt.utils.nvtx_utils import (
    NVTX_OPERATIONS_ENABLED,
    operations_nvtx_range,
    profile_method,
)


class _RequestCache(QSAHiSparseSingleRequest):
    """Reuse per-request writeback and compact-byte checks on private views.

    Construction, handoff, batch metadata and release belong to the runtime.
    The inherited single-owner lifecycle methods are never used here.
    """

    def __init__(self, adapter, lease):
        self.adapter, self.lease = adapter, lease
        self.pool, self.device = adapter.pool, adapter.device
        self.strict, self.capacity = adapter.strict, adapter.capacity
        self.generation = lease.generation
        self.copy_stream = adapter.copy_stream
        self.seq_len = self.decode_steps = 0
        self.offloaded = False
        self.states = []
        self.host = (
            None if adapter.host_slabs is None else adapter.host_slabs[lease.slot]
        )
        ring = adapter.slots.ring_slice(lease)
        self.full = SimpleNamespace(
            k_buffer=[k[ring] for k in adapter.full.k_buffer],
            v_buffer=[v[ring] for v in adapter.full.v_buffer],
        )
        if self.host is not None:
            self.compact_base = lease.slot * 2052
            self.compact = adapter.compact[
                :, self.compact_base : self.compact_base + 2052
            ]
            self.compact_table = adapter.compact_table[lease.slot : lease.slot + 1]
            self.compressed_len = adapter.compressed_lens[lease.slot : lease.slot + 1]
            self.compact.zero_()
            self.compact_table.zero_()
            for ring_k, ring_v in zip(self.full.k_buffer, self.full.v_buffer):
                ring_k.zero_()
                ring_v.zero_()
        self.handoff_event = None

    def make_state(self):
        self.adapter.slots.require(self.lease, "copying")
        self.adapter._request(self.lease.req_pool_idx, self.lease.rid)
        shared = self.adapter.layer_states[len(self.states)]
        slot = self.lease.slot
        state = {name: shared[name][slot : slot + 1] for name in ("tokens", "lru")}
        state["hot"] = shared["hot"].view(self.adapter.max_requests, 2112, 2048)[slot]
        state["hot"].zero_()
        state["tokens"].fill_(-1)
        state["lru"].copy_(self.adapter.initial_lru)
        state.update(done=None, generation=self.generation, writeback_bytes=0)
        return state

    def record(self, event, **extra):
        self.adapter.record(
            event,
            self.lease,
            seq_len=self.seq_len,
            decode_steps=self.decode_steps,
            **extra,
        )


class QSAHiSparseRuntime:
    """Bounded multi-request resident/offload runtime, including decode graphs."""

    uses_qsa_hisparse_leases = True

    def __init__(self, runner, mode):
        if mode not in ("p2-offload", "p2-resident"):
            raise ValueError("QSA P2 requires an explicit resident/offload arm")
        if os.environ.get("SGLANG_QSA_HISPARSE_V3_CAPTURE"):
            raise ValueError("QSA P2 does not support V3 capture")
        self.runner, self.mode = runner, mode
        self.pool = runner.token_to_kv_pool
        self.full = self.pool.full_kv_pool
        self.graph_enabled = (
            mode == "p2-offload"
            and runner.server_args.cuda_graph_backend_decode == "full"
        )
        self.observe = os.environ.get("SGLANG_QSA_HISPARSE_V3_OBSERVE", "strict")
        if self.observe not in ("strict", "light"):
            raise ValueError("QSA P2 requires strict/light observation")
        self.strict = self.observe == "strict"
        validate_configuration(
            runner.server_args,
            self.pool,
            p2=True,
            graph=self.graph_enabled,
            strict=self.strict,
        )
        self.max_requests = int(runner.server_args.max_running_requests)
        self.graph_batch_sizes = tuple(range(1, self.max_requests + 1))
        from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
        from sglang.srt.model_executor.cuda_graph_config import (
            check_cuda_graph_backend,
            cuda_graph_fully_disabled,
        )

        graph_ok = (
            (
                check_cuda_graph_backend("decode", "full")
                and check_cuda_graph_backend("prefill", "disabled")
            )
            if self.graph_enabled
            else cuda_graph_fully_disabled()
        )
        if self.graph_enabled:
            from sglang.srt.runtime_context import get_exec

            cfg = get_exec().graph.cuda_graph_config.decode
            graph_ok = (
                graph_ok
                and cfg.bs == list(self.graph_batch_sizes)
                and cfg.max_bs == self.max_requests
            )

        if (
            not str(self.pool.device).startswith("cuda")
            or not graph_ok
            or type(self.full) is not MHATokenToKVPool
            or type(runner.token_to_kv_pool_allocator)
            is not PagedTokenToKVPoolAllocator
            or getattr(runner, "_unified_memory_pool", None) is not None
            or getattr(self.pool, "qsa_hisparse", None) is not None
        ):
            raise ValueError(
                "QSA P2 needs its bounded backend and independent static pools"
            )
        if getattr(runner.server_args, "enable_mixed_chunk", False) or getattr(
            runner.server_args, "enable_priority_preemption", False
        ):
            raise ValueError("QSA P2 forbids mixed prefill/decode and preemption")
        self.capacity = 262144
        budget, entries = prefix_cache_options()
        self.prefix_cache = None
        if budget:
            if mode != "p2-offload" or any(
                getattr(runner.server_args, flag, False)
                for flag in (
                    "enable_lora",
                    "enable_linear_replayssm",
                    "enable_mamba_extra_buffer",
                )
            ):
                raise ValueError(
                    "QSA host prefixes require offload without LoRA/replay/extra Mamba buffers"
                )
            from sglang.srt.mem_cache.qsa_hisparse.prefix import HostPrefixCache

            self.prefix_cache = HostPrefixCache(budget, entries)
            # The cache is process local. Flush advances its epoch after weight updates.
            self.prefix_namespace = (
                str(getattr(runner.server_args, "model_path", "")),
                str(getattr(runner.server_args, "revision", "")),
                str(self.pool.dtype),
                runner.server_args.tp_size,
                json.dumps(
                    getattr(runner.model_config, "hf_config", {}).to_dict(),
                    sort_keys=True,
                )
                if hasattr(getattr(runner.model_config, "hf_config", None), "to_dict")
                else repr(getattr(runner.model_config, "hf_config", None)),
            )
        self.slots = QSAHiSparseSlots(self.capacity, 64, self.max_requests)
        self.req_pool = runner.req_to_token_pool
        if (
            self.prefix_cache is not None
            and getattr(self.req_pool, "mamba_ckpt_pool", None) is not None
        ):
            raise ValueError("QSA host prefixes require plain recurrent checkpoints")
        if self.prefix_cache is not None and any(
            getattr(self.req_pool.mamba_pool.mamba_cache, name, None) is not None
            for name in ("replayssm_d", "replayssm_k", "replayssm_g")
        ):
            raise ValueError("QSA host prefixes cannot checkpoint ReplaySSM scratch")
        self.req_table = self.req_pool.req_to_token
        if self.req_table.shape[1] < self.capacity:
            raise ValueError("QSA P2 request table is smaller than a context")
        expected_raw = (
            self.slots.raw_pool_size if mode == "p2-offload" else self.pool.size
        )
        if (
            self.full.size != expected_raw
            or self.full.page_size != 64
            or self.full.dtype != self.pool.dtype
            or self.full.device != self.pool.device
            or self.full.head_num != 1
            or self.full.head_dim != 256
            or self.full.layer_num != 12
            or self.full.kv_cache_layout.lower() != "nhd"
            or any(
                tuple(t.shape) != (expected_raw + 64, 1, 256)
                for t in self.full.k_buffer + self.full.v_buffer
            )
        ):
            raise ValueError(
                "QSA P2 raw backing geometry/capacity does not match the lease layout"
            )
        self.device, self.rank = self.pool.device, runner.ps.tp_rank
        self.layer_ids = list(self.pool.full_attention_layer_id_mapping)
        self.raw_ptrs = [t.data_ptr() for t in self.full.k_buffer + self.full.v_buffer]
        self.index_ptr = self.pool.qsa_compressed_flat.data_ptr()
        self.copy_stream = torch.cuda.Stream(device=self.device)
        self.producer_stream = None
        self.requests = {}
        self.pending_releases = []
        self.batch_requests = []
        self.offloaded = False
        self.forward_id = 0
        self.raw_write_locs = None
        self.graph_capture_size = None
        self.graph_batch = None
        self.graph_copy_pending = False
        self.graph_failed_leases = set()
        self.graph_copy_epoch = 0
        directory = os.environ.get("SGLANG_QSA_HISPARSE_V3_EVENTS")
        self.path = None
        if directory:
            Path(directory).mkdir(parents=True, exist_ok=True)
            self.path = Path(directory) / f"rank-{self.rank}.jsonl"
        start = time.monotonic()
        self.host_slabs = None
        host_alloc_wall_ms = 0.0
        self.workspace = []
        if mode == "p2-offload":
            self.host_slabs = torch.empty(
                (self.max_requests, 12, self.capacity // 4, 2048),
                dtype=torch.uint8,
                pin_memory=True,
            )
            host_alloc_wall_ms = (time.monotonic() - start) * 1000
            self._allocate_decode_workspace()
            if self.graph_enabled:
                self._allocate_graph_workspace()
        self.record(
            "init", allocation_phase="startup", host_alloc_wall_ms=host_alloc_wall_ms
        )

    def _allocate_decode_workspace(self):
        """Fixed backing; persistent rows are lease slots, scratch rows are batch order."""
        device, layers, blocks = self.device, len(self.layer_ids), self.capacity // 4
        self.indices = (
            unpack_index(device)[None]
            + torch.arange(self.max_requests, device=device)[:, None] * 4096
        ).flatten()
        self.gathered = torch.empty(
            (self.max_requests * 512, 2048), dtype=torch.uint8, device=device
        )
        self.unpacked = torch.empty(
            (self.max_requests, 2, 2048, 1, 256), dtype=torch.uint8, device=device
        )
        self.compact = torch.zeros(
            (2, self.max_requests * 2052, 1, 256), dtype=torch.uint8, device=device
        )
        self.compact_table = torch.zeros(
            (self.max_requests, self.capacity), dtype=torch.int32, device=device
        )
        self.real = torch.zeros(1, dtype=torch.int32, device=device)
        self.compressed_lens = torch.zeros(
            self.max_requests, dtype=torch.int32, device=device
        )
        self.batch_lens = torch.zeros(
            self.max_requests, dtype=torch.int32, device=device
        )
        self.batch_slots = torch.zeros(
            self.max_requests, dtype=torch.int32, device=device
        )
        self.batch_write_locs = torch.zeros(
            self.max_requests, dtype=torch.int64, device=device
        )
        self.blocks = torch.empty(
            (self.max_requests, 512), dtype=torch.int32, device=device
        )
        self.out = torch.empty(
            (self.max_requests, 512), dtype=torch.int32, device=device
        )
        self.gather_indices = torch.empty(
            self.max_requests * 512, dtype=torch.int64, device=device
        )
        self.miss_src = torch.empty(
            (self.max_requests, 512), dtype=torch.int64, device=device
        )
        self.miss_dst = torch.empty(
            (self.max_requests, 512), dtype=torch.int32, device=device
        )
        self.miss_count = torch.zeros(
            self.max_requests, dtype=torch.int32, device=device
        )
        self.initial_lru = torch.arange(2048, dtype=torch.int16, device=device)[None]
        self.workspace = [
            self.indices,
            self.gathered,
            self.unpacked,
            self.compact,
            self.compact_table,
            self.real,
            self.compressed_lens,
            self.batch_lens,
            self.batch_slots,
            self.batch_write_locs,
            self.blocks,
            self.out,
            self.gather_indices,
            self.miss_src,
            self.miss_dst,
            self.miss_count,
            self.initial_lru,
        ]
        self.layer_states = []
        for li in range(layers):
            shared = {
                "hot": torch.zeros(
                    (self.max_requests * 2112, 2048), dtype=torch.uint8, device=device
                ),
                "tokens": torch.full(
                    (self.max_requests, 2112), -1, dtype=torch.int32, device=device
                ),
                "lru": self.initial_lru.repeat(self.max_requests, 1),
                "device_locs": torch.arange(
                    self.max_requests * 2112, dtype=torch.int32, device=device
                ).view(self.max_requests, 2112),
                "host_locs": (
                    torch.arange(blocks, dtype=torch.int64, device=device)[None]
                    + (
                        torch.arange(self.max_requests, device=device)[:, None] * layers
                        + li
                    )
                    * blocks
                ),
            }
            self.layer_states.append(shared)
            self.workspace.extend(t for name, t in shared.items() if name != "hot")

    def _allocate_graph_workspace(self):
        self.graph_seq_lens = torch.zeros(
            self.max_requests, dtype=torch.int32, device=self.device
        )
        self.graph_row_ids = torch.arange(
            self.max_requests, dtype=torch.int32, device=self.device
        )[:, None]
        self.workspace.extend((self.graph_seq_lens, self.graph_row_ids))
        self.graph_producer = torch.cuda.Event()
        self.graph_copy_done = torch.cuda.Event()
        self.graph_audit = None
        if self.strict:
            layers = len(self.layer_ids)
            self.graph_audit = {
                "raw": torch.zeros(
                    (layers, self.max_requests, 2051),
                    dtype=torch.int32,
                    device=self.device,
                ),
                "compact": torch.zeros(
                    (layers, *self.compact.shape), dtype=torch.uint8, device=self.device
                ),
                "mapping": torch.zeros(
                    (layers, self.max_requests, 2051),
                    dtype=torch.int32,
                    device=self.device,
                ),
                "miss_count": torch.zeros(
                    (layers, self.max_requests), dtype=torch.int32, device=self.device
                ),
                "valid_counts": torch.zeros(
                    (layers, self.max_requests), dtype=torch.int32, device=self.device
                ),
                "resolver_out": torch.zeros(
                    (layers, self.max_requests, 512),
                    dtype=torch.int32,
                    device=self.device,
                ),
                "miss_src": torch.zeros(
                    (layers, self.max_requests, 512),
                    dtype=torch.int64,
                    device=self.device,
                ),
                "miss_dst": torch.zeros(
                    (layers, self.max_requests, 512),
                    dtype=torch.int32,
                    device=self.device,
                ),
            }
            self.workspace.extend(self.graph_audit.values())

    @contextmanager
    def graph_capture(self, count, *, native_key=None, native_backend=None):
        if (
            not self.graph_enabled
            or count not in self.graph_batch_sizes
            or self.requests
            or self.slots.active
            or self.pending_releases
            or self.graph_batch is not None
            or self.graph_capture_size is not None
        ):
            raise RuntimeError(
                "QSA graph capture needs an exact configured batch and no live leases"
            )
        from sglang.srt.model_executor.runner.flashinfer_autotune import (
            should_run_flashinfer_autotune,
        )

        if should_run_flashinfer_autotune(self.runner):
            raise RuntimeError(
                "QSA graph capture does not permit earlier autotune dummy forwards"
            )
        self.graph_warmups = 0
        info = {
            "graph_key": None if native_key is None else vars(native_key),
            "graph_backend": type(native_backend).__name__,
            "graph_pool": getattr(native_backend, "_pool", None),
        }
        capture_error = None
        try:
            self.graph_capture_size = count
            self.offloaded = True
            self.raw_write_locs = self.batch_write_locs[:count]
            self.row_slots = self.batch_slots[:count]
            self.real.zero_()
            self.batch_write_locs.zero_()  # Reserved raw row zero, never a lease ring.
            self.batch_slots.zero_()
            self.batch_lens.zero_()
            self.graph_seq_lens.zero_()
            self.miss_count.zero_()
            if self.path is not None:
                self.record(
                    "graph_capture_begin",
                    **info,
                    real_count=int(self.real[0]),
                    graph_buffers=self.graph_buffer_inventory(),
                )
            yield
            if self.path is not None:
                self.record(
                    "graph_capture_complete",
                    **info,
                    real_count=int(self.real[0]),
                    warmups=self.graph_warmups,
                    graph_buffers=self.graph_buffer_inventory(),
                    native_key_present=native_key
                    in getattr(native_backend, "_graphs", {}),
                )
        except BaseException as error:
            capture_error = error
            if self.path is not None:
                try:
                    self.record("graph_capture_failed", **info, error=repr(error))
                except BaseException as observation_error:
                    error.add_note(f"QSA capture failure ledger: {observation_error!r}")
            raise
        finally:
            self.graph_capture_size = None
            self.offloaded = False
            self.raw_write_locs = None
            try:
                self.real.zero_()
                if self.requests or self.slots.active or self.pending_releases:
                    raise RuntimeError("QSA graph capture acquired live ownership")
                if self.path is not None:
                    self.record(
                        "graph_capture_reset",
                        **info,
                        capture_mode=self.graph_capture_size,
                        real_count=int(self.real[0]),
                    )
            except BaseException as error:
                if capture_error is None:
                    raise
                capture_error.add_note(f"QSA capture cleanup failure: {error!r}")

    def after_graph_warmup(self):
        if (
            self.graph_capture_size not in self.graph_batch_sizes
            or self.requests
            or self.slots.active
        ):
            raise RuntimeError("QSA graph warmup changed ownership")
        self.graph_warmups += 1
        if self.path is not None:
            self.record(
                "graph_warmup_complete",
                capture_size=self.graph_capture_size,
                warmup=self.graph_warmups,
                real_count=int(self.real[0]),
            )

    def graph_buffer_inventory(self):
        tensors = {
            name: getattr(self, name)
            for name in (
                "host_slabs",
                "indices",
                "gathered",
                "unpacked",
                "compact",
                "compact_table",
                "real",
                "compressed_lens",
                "batch_lens",
                "batch_slots",
                "batch_write_locs",
                "blocks",
                "out",
                "gather_indices",
                "miss_src",
                "miss_dst",
                "miss_count",
                "graph_seq_lens",
                "graph_row_ids",
            )
        }
        for li, state in enumerate(self.layer_states):
            tensors.update(
                {f"layer{li}.{name}": value for name, value in state.items()}
            )
            tensors[f"layer{li}.raw_k"] = self.full.k_buffer[li]
            tensors[f"layer{li}.raw_v"] = self.full.v_buffer[li]
        tensors.update(
            {f"audit.{name}": value for name, value in (self.graph_audit or {}).items()}
        )
        return {
            name: {
                "ptr": value.data_ptr(),
                "bytes": value.nbytes,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
            for name, value in tensors.items()
        }

    def prepare_graph_replay(self, batch, count):
        if (
            not self.graph_enabled
            or count != batch.batch_size
            or count not in self.graph_batch_sizes
            or not batch.forward_mode.is_decode()
            or self.graph_batch is not None
            or self.graph_capture_size is not None
        ):
            raise RuntimeError("QSA graph replay requires exact unpadded decode rows")
        self.begin_batch(batch, graph=True)
        # One stream wait protects every newest row before a new replay can write.
        if self.graph_copy_pending:
            torch.cuda.current_stream(self.device).wait_event(self.graph_copy_done)
            self.record(
                "graph_copy_wait",
                copy_epoch=self.graph_copy_epoch,
                completion_observed=False,
            )
        self.graph_seq_lens[:count].copy_(batch.seq_lens_cpu[:count])

    @contextmanager
    def graph_replay_scope(self):
        try:
            yield
        except BaseException as error:
            self.fail_graph_replay(error)
            raise

    def _drain_graph_streams(self):
        errors = []
        producer = (
            self.producer_stream
            if self.producer_stream is not None
            else torch.cuda.current_stream(self.device)
        )
        for stream in (producer, self.copy_stream):
            try:
                stream.synchronize()
            except BaseException as error:
                errors.append(error)
        if errors:
            for error in errors[1:]:
                errors[0].add_note(f"Additional QSA graph drain failure: {error!r}")
            raise errors[0]

    def fail_graph_replay(self, cause=None):
        saved = self.graph_batch
        if saved is None:
            return
        leases = {
            lease
            for lease, _ in saved
            if self.slots.active.get(lease.req_pool_idx) == lease
        }
        self.graph_failed_leases.update(leases)
        for lease in leases:
            self.slots.phases[lease.req_pool_idx] = "failed"
        try:
            self._drain_graph_streams()
        except BaseException as error:
            if cause is None:
                raise
            cause.add_note(
                f"QSA graph leases remain blocked on stream drain: {error!r}"
            )
        else:
            self.graph_failed_leases.difference_update(leases)
        finally:
            self.graph_batch = None

    def finish_graph_replay(self, count, *, native_key=None, native_backend=None):
        saved = self.graph_batch
        if saved is None or len(saved) != count:
            raise RuntimeError("QSA graph replay has no matching prepared batch")
        try:
            if native_backend is not None and (
                native_key not in native_backend._graphs or native_key.size != count
            ):
                raise RuntimeError(
                    "QSA graph native replay key differs from prepared rows"
                )
            for lease, seq in saved:
                state = self._request(lease.req_pool_idx, lease.rid)
                self.slots.require(lease, "decode")
                if state.lease != lease or state.seq_len != seq:
                    raise RuntimeError(
                        "QSA graph writeback generation/sequence changed"
                    )
            producer = torch.cuda.current_stream(self.device)
            self.graph_producer.record(producer)
            closing = [
                (self.requests[lease.req_pool_idx], seq // 4 - 1)
                for lease, seq in saved
                if seq % 4 == 0
            ]
            if self.strict:
                self.graph_producer.synchronize()
                self._check_graph_close(closing, host=False)
            if closing:
                self.graph_copy_epoch += 1
                with torch.cuda.stream(self.copy_stream):
                    self.copy_stream.wait_event(self.graph_producer)
                    for state, block in closing:
                        for li, layer_state in enumerate(state.states):
                            state.host[li, block].copy_(
                                layer_state["hot"][2048], non_blocking=True
                            )
                    self.graph_copy_done.record(self.copy_stream)
                self.graph_copy_pending = True
                for state, _ in closing:
                    for layer_state in state.states:
                        layer_state.update(done=self.graph_copy_done, copy_begin=None)
                        layer_state["copy_epoch"] = self.graph_copy_epoch
                        layer_state["writeback_bytes"] += 2048
                self.record(
                    "graph_copy_submit",
                    copy_epoch=self.graph_copy_epoch,
                    copies=[
                        {
                            "req_pool_idx": s.lease.req_pool_idx,
                            "rid": s.lease.rid,
                            "generation": s.lease.generation,
                            "slot": s.lease.slot,
                            "seq_len": s.seq_len,
                            "block": block,
                            "layer_id": self.layer_ids[li],
                            "bytes": 2048,
                            "host_dst_ptr": s.host[li, block].data_ptr(),
                        }
                        for s, block in closing
                        for li in range(len(s.states))
                    ],
                )
            if self.strict:
                if closing:
                    self.graph_copy_done.synchronize()
                    self._check_graph_close(closing, host=True)
                    self.record("graph_copy_complete", copy_epoch=self.graph_copy_epoch)
                self._check_graph_selected(saved)
            self.record(
                "graph_replay",
                batch_size=count,
                graph_key=None if native_key is None else vars(native_key),
                graph_backend=type(native_backend).__name__,
                native_key_present=native_key in getattr(native_backend, "_graphs", {}),
                ordered_leases=[
                    [x.req_pool_idx, x.generation, x.rid, x.slot] for x, _ in saved
                ],
                close_rows=len(closing),
                writeback_bytes=len(closing) * len(self.layer_ids) * 2048,
            )
        except BaseException as error:
            # A failed copy/event submission may have no completion event to drain.
            # Retain leases and drain both streams before any scheduler can reuse them.
            self.fail_graph_replay(error)
            raise
        finally:
            self.graph_batch = None

    def _check_graph_close(self, closing, *, host):
        for state, block in closing:
            for li in range(len(state.states)):
                expected = torch.cat(
                    (
                        state.full.k_buffer[li][1:5].view(torch.uint8).flatten(),
                        state.full.v_buffer[li][1:5].view(torch.uint8).flatten(),
                    )
                )
                actual = (
                    state.host[li, block].to(expected.device)
                    if host
                    else state.states[li]["hot"][2048]
                )
                if not torch.equal(actual, expected):
                    raise AssertionError(
                        f"QSA graph closed C4 differs from raw K/V ring: host={host}"
                    )

    def _check_graph_selected(self, saved):
        for row, (lease, seq) in enumerate(saved):
            state = self.requests[lease.req_pool_idx]
            selected = min(seq // 4, 512) * 4
            valid = selected + seq % 4
            for li, lid in enumerate(self.layer_ids):
                raw = self.graph_audit["raw"][li, row : row + 1]
                compact = self.graph_audit["compact"][
                    li, :, lease.slot * 2052 : (lease.slot + 1) * 2052
                ]
                state.check_selected(
                    SimpleNamespace(layer_id=lid),
                    raw,
                    raw[:, :2048:4] // 4,
                    self.graph_audit["miss_count"][li, row : row + 1],
                    compact=compact,
                    mapping=self.graph_audit["mapping"][li, row, :valid],
                )
                if int(self.graph_audit["valid_counts"][li, row]) != valid:
                    raise AssertionError("QSA graph FA2 valid count differs")
                if state.decode_steps in (1, 2, 3, 384, 767) or seq % 4 == 0:
                    misses = int(self.graph_audit["miss_count"][li, row])
                    self.record(
                        "graph_selected_check",
                        lease,
                        layer_id=lid,
                        seq_len=seq,
                        resolver_out=self.graph_audit["resolver_out"][li, row]
                        .cpu()
                        .tolist(),
                        selected_blocks=(raw[0, :selected:4] // 4).cpu().tolist(),
                        miss_count=misses,
                        miss_src=self.graph_audit["miss_src"][li, row, :misses]
                        .cpu()
                        .tolist(),
                        miss_dst=self.graph_audit["miss_dst"][li, row, :misses]
                        .cpu()
                        .tolist(),
                        closed_block_selected=bool(
                            seq % 4 == 0
                            and torch.any(raw[0, :selected:4] // 4 == seq // 4 - 1)
                        ),
                    )

    def native_lease_snapshot(self, state, *, include_pages=False):
        lease = state.lease
        self.slots.require(lease)
        idx = lease.req_pool_idx
        if int(self.req_pool.req_generation[idx]) != lease.generation:
            raise RuntimeError("QSA P2 snapshot has a stale native request generation")
        extra_buffer = self.req_pool.enable_mamba_extra_buffer
        snapshot = {
            "logical_row_ptr": self.req_table[idx].data_ptr(),
            "mamba_pool_idx": int(self.req_pool.req_index_to_mamba_index_mapping[idx]),
            "mamba_extra_buffer_enabled": extra_buffer,
            "mamba_track_slots": (
                self.req_pool.req_index_to_mamba_ping_pong_track_buffer_mapping[
                    idx
                ].tolist()
                if extra_buffer
                else []
            ),
        }
        if include_pages:
            # Decode-boundary evidence only: read current native page identities,
            # not a saved pointer or a second allocator's shadow mapping.
            starts = self.req_table[idx, : state.seq_len : self.slots.page_size].cpu()
            if torch.any(starts % self.slots.page_size):
                raise RuntimeError("QSA P2 logical page start is not aligned")
            snapshot.update(
                logical_seq_len=state.seq_len,
                logical_page_ids=(starts // self.slots.page_size).tolist(),
            )
        return snapshot

    def record(self, event, lease=None, **extra):
        if self.path is None:
            return
        if self.observe == "light" and event in (
            "prefill_chunk",
            "decode_batch",
            "graph_replay",
            "graph_copy_wait",
            "graph_copy_submit",
        ):
            # Keep the ordered schedule without synchronizing GPU page inventories.
            row = {
                "event": event,
                "time_ns": time.time_ns(),
                "rank": self.rank,
                "mode": self.mode,
                "observe": self.observe,
                "forward_id": self.forward_id,
                **extra,
            }
            with self.path.open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            return
        reserved = 0 if self.host_slabs is None else self.host_slabs.nbytes
        active = sum(
            0 if s.host is None else s.host.nbytes for s in self.requests.values()
        )
        _, mamba_sizes, _ = self.req_pool.mamba_pool.get_contiguous_buf_infos()
        row = {
            "event": event,
            "time_ns": time.time_ns(),
            "rank": self.rank,
            "mode": self.mode,
            "observe": self.observe,
            "forward_id": self.forward_id,
            "req_pool_idx": None if lease is None else lease.req_pool_idx,
            "generation": None if lease is None else lease.generation,
            "rid": None if lease is None else lease.rid,
            "lease_slot": None if lease is None else lease.slot,
            "phase": None
            if lease is None
            else self.slots.phases.get(lease.req_pool_idx),
            **self.slots.snapshot(),
            "logical_capacity": self.pool.size,
            "logical_available": self.runner.token_to_kv_pool_allocator.available_size(),
            "raw_bytes": sum(t.nbytes for t in self.full.k_buffer + self.full.v_buffer),
            "raw_backing_size_tokens": self.full.size,
            "raw_ptrs": self.raw_ptrs,
            "index_ptr": self.index_ptr,
            "raw_storage_unchanged": self.raw_ptrs
            == [t.data_ptr() for t in self.full.k_buffer + self.full.v_buffer],
            "index_bytes": self.pool.qsa_compressed_flat.nbytes,
            "index_storage_unchanged": self.index_ptr
            == self.pool.qsa_compressed_flat.data_ptr(),
            "host_reserved_bytes": reserved,
            "host_bytes": active,
            "host_free_bytes": reserved - active,
            "pending_release_count": sum(
                self.slots.active.get(x.req_pool_idx) == x
                for x in self.pending_releases
            ),
            "hot_bytes": sum(
                t["hot"].nbytes for s in self.requests.values() for t in s.states
            ),
            "hot_reserved_bytes": sum(
                s["hot"].nbytes for s in getattr(self, "layer_states", [])
            ),
            "workspace_bytes": sum(t.nbytes for t in self.workspace),
            "mamba_bytes": sum(mamba_sizes),
            "mamba_available": self.req_pool.mamba_allocator.available_size(),
            "leases": [
                {
                    "req_pool_idx": s.lease.req_pool_idx,
                    "generation": s.lease.generation,
                    "rid": s.lease.rid,
                    "slot": s.lease.slot,
                    "phase": self.slots.phases[s.lease.req_pool_idx],
                    **self.native_lease_snapshot(
                        s,
                        include_pages=event
                        in ("handoff_complete", "prefill_chunk", "decode_batch"),
                    ),
                    "host_ptr": None if s.host is None else s.host.data_ptr(),
                    "hot_ptrs": [x["hot"].data_ptr() for x in s.states],
                    "ring_k_ptrs": [x.data_ptr() for x in s.full.k_buffer],
                    "ring_v_ptrs": [x.data_ptr() for x in s.full.v_buffer],
                }
                for s in self.requests.values()
            ],
            "cuda_allocated": torch.cuda.memory_allocated(self.device),
            "cuda_reserved": torch.cuda.memory_reserved(self.device),
            **extra,
        }
        with self.path.open("a") as stream:
            stream.write(json.dumps(row) + "\n")

    def _request(self, req_idx, rid):
        state = self.requests[req_idx]
        lease = state.lease
        self.slots.require(lease)
        if lease.rid != rid or lease.generation != int(
            self.req_pool.req_generation[req_idx]
        ):
            raise RuntimeError("QSA P2 request identity/generation changed")
        return state

    def _acquire_request(self, req_idx, rid):
        lease = self.slots.acquire(
            req_idx, int(self.req_pool.req_generation[req_idx]), rid
        )
        state = _RequestCache(self, lease)
        state.prefix_epoch = getattr(getattr(self, "prefix_cache", None), "epoch", None)
        state.prefix_basis_entry_id = None
        state.prefix_basis_length = 0
        self.requests[req_idx] = state
        self.record("begin_prefill", lease)
        return state

    def note_prefix_basis(self, req, snapshot, entry_id):
        # Remember identity only. Holding tensor references here would keep
        # evicted host storage alive outside the cache's byte accounting.
        state = self._request(req.kv.req_pool_idx, req.rid)
        state.prefix_basis_entry_id = entry_id
        state.prefix_basis_length = snapshot.length

    def prefix_checkpoint_bytes(self, delta, token_count):
        """Exact new tensor/key bytes reserved before a checkpoint is copied."""
        mamba = self.req_pool.mamba_pool
        size = sum(t[:, :1].nbytes for t in mamba.mamba_cache.conv)
        size += mamba.mamba_cache.temporal[:, :1].nbytes
        for sibling in mamba._slot_siblings:
            for _, tensor, _, _ in sibling.iter_transfer_state_entries():
                size += tensor[:1].nbytes
        size += sum(t[:4].nbytes for t in self.pool.qsa_key_state_buffer_pool)
        size += self.pool.qsa_rope_position_buffer[:4].nbytes
        size += delta * len(self.layer_ids) * 2 * 256
        size += sum(
            t[: delta // 4].nbytes for t in self.pool.qsa_compressed_k_buffer_pool
        )
        return size + token_count * 8

    @staticmethod
    def _capture_slot_tensor(source, slot, *, layered=True):
        shape = source[:, :1].shape if layered else source[:1].shape
        target = torch.empty(shape, dtype=source.dtype, device="cpu")
        if layered:
            for li in range(source.shape[0]):
                target[li, 0].copy_(source[li, slot])
        else:
            target[0].copy_(source[slot])
        return target

    @staticmethod
    def _restore_slot_tensor(target, data, slot, *, layered=True):
        if layered:
            for li in range(target.shape[0]):
                target[li, slot].copy_(data[li, 0])
        else:
            target[slot].copy_(data[0])

    def _mamba_sibling_tensors(self):
        from sglang.srt.mem_cache.ple_state_pool import NGramPool, ShortConvPool

        for sibling in self.req_pool.mamba_pool._slot_siblings:
            if isinstance(sibling, ShortConvPool):
                yield sibling.conv_state, True
            elif isinstance(sibling, NGramPool):
                yield sibling.context, False
            else:
                raise ValueError(
                    "QSA host prefixes do not support this Mamba slot sibling"
                )

    def _capture_mamba(self, slot):
        cache = self.req_pool.mamba_pool.mamba_cache
        conv = [self._capture_slot_tensor(t, slot) for t in cache.conv]
        temporal = self._capture_slot_tensor(cache.temporal, slot)
        siblings = [
            self._capture_slot_tensor(t, slot, layered=layered)
            for t, layered in self._mamba_sibling_tensors()
        ]
        return (conv, temporal, siblings) if siblings else (conv, temporal)

    def _restore_mamba(self, data, slot):
        cache = self.req_pool.mamba_pool.mamba_cache
        for target, source in zip(cache.conv, data[0]):
            self._restore_slot_tensor(target, source, slot)
        self._restore_slot_tensor(cache.temporal, data[1], slot)
        if self.req_pool.mamba_pool._slot_siblings:
            for (target, layered), source in zip(
                self._mamba_sibling_tensors(), data[2]
            ):
                self._restore_slot_tensor(target, source, slot, layered=layered)
        torch.cuda.current_stream(self.device).synchronize()

    @staticmethod
    def _check_slot_tensor(target, data, slot, *, layered=True):
        if layered:
            return all(
                torch.equal(
                    target[li, slot].cpu().contiguous().view(torch.uint8),
                    data[li, 0].contiguous().view(torch.uint8),
                )
                for li in range(target.shape[0])
            )
        return torch.equal(
            target[slot].cpu().contiguous().view(torch.uint8),
            data[0].contiguous().view(torch.uint8),
        )

    def _check_restored_mamba(self, data, slot):
        cache = self.req_pool.mamba_pool.mamba_cache
        good = all(
            self._check_slot_tensor(target, source, slot)
            for target, source in zip(cache.conv, data[0])
        ) and self._check_slot_tensor(cache.temporal, data[1], slot)
        if self.req_pool.mamba_pool._slot_siblings:
            good = good and all(
                self._check_slot_tensor(target, source, slot, layered=layered)
                for (target, layered), source in zip(
                    self._mamba_sibling_tensors(), data[2]
                )
            )
        return good

    def capture_prefix(self, req, namespace, tokens):
        """Capture only an actual complete forward boundary, before handoff."""
        from sglang.srt.mem_cache.qsa_hisparse.prefix import (
            PrefixSegment,
            PrefixSnapshot,
        )

        state = self._request(req.kv.req_pool_idx, req.rid)
        length = state.seq_len
        if (
            self.slots.phases[state.lease.req_pool_idx] != "prefill"
            or not length
            or length % 64
            or state.prefix_epoch != self.prefix_cache.epoch
        ):
            return None
        if len(tokens) != length * 8:
            raise RuntimeError("QSA prefix token/checkpoint boundary differs")
        previous = self.prefix_cache.acquire(namespace, tokens, count=False)
        reservation = None
        retained = False
        try:
            ancestor = None if previous is None else previous.snapshot
            if ancestor is not None and ancestor.length == length:
                return None
            if (
                previous is not None
                and previous.entry_id != state.prefix_basis_entry_id
            ):
                previous.close()
                previous = None
                if state.prefix_basis_length:
                    previous = self.prefix_cache.acquire(
                        namespace, tokens, state.prefix_basis_length, count=False
                    )
                    if (
                        previous is not None
                        and previous.entry_id != state.prefix_basis_entry_id
                    ):
                        previous.close()
                        previous = None
                ancestor = None if previous is None else previous.snapshot
            # Token equality alone cannot justify shared raw/index bytes: an
            # uncached forward can differ with its chunk partition. Share only
            # a checkpoint this active request restored or published itself.
            start = 0 if ancestor is None else ancestor.length
            reservation = self.prefix_cache.reserve(
                self.prefix_checkpoint_bytes(length - start, length)
            )
            if reservation is None:
                return None
            reservation.basis_reader = previous
            retained = True
            # Every copied tensor is immutable after publication. Avoid a growing
            # device-side stack; each transfer has at most one prefill chunk.
            segments = [] if ancestor is None else list(ancestor.segments)
            if self.producer_stream is None:
                raise RuntimeError(
                    "QSA prefix capture has no registered model producer"
                )
            self.producer_stream.synchronize()
            torch.cuda.current_stream(self.device).synchronize()
            for offset in range(start, length, 4096):
                stop = min(offset + 4096, length)
                raw = torch.empty(
                    (len(self.layer_ids), 2, stop - offset, 1, 256), dtype=torch.uint8
                )
                shape = self.pool.qsa_compressed_k_buffer_pool[0].shape[1:]
                index = torch.empty(
                    (len(self.layer_ids), (stop - offset) // 4, *shape),
                    dtype=self.pool.index_state_dtype,
                )
                physical = self.slots.staging_slice(state.lease, offset, stop)
                logical = self.req_table[req.kv.req_pool_idx, offset:stop:4].long() // 4
                for li in range(len(self.layer_ids)):
                    raw[li, 0].copy_(self.full.k_buffer[li][physical].view(torch.uint8))
                    raw[li, 1].copy_(self.full.v_buffer[li][physical].view(torch.uint8))
                    index[li].copy_(self.pool.qsa_compressed_k_buffer_pool[li][logical])
                segments.append(PrefixSegment(offset, stop, raw, index))
            ring = slice(req.kv.req_pool_idx * 4, (req.kv.req_pool_idx + 1) * 4)
            pending = tuple(
                t[ring].to("cpu", copy=True)
                for t in self.pool.qsa_key_state_buffer_pool
            )
            rope = self.pool.qsa_rope_position_buffer[ring].to("cpu", copy=True)
            mamba = self._capture_mamba(int(req.kv.mamba_pool_idx))
            snapshot = PrefixSnapshot(
                namespace, tokens, tuple(segments), pending, rope, mamba
            )
            return reservation, snapshot
        except BaseException:
            if reservation is not None:
                reservation.close()
            raise
        finally:
            if previous is not None and not retained:
                previous.close()

    def restore_prefix(self, req, snapshot):
        """Restore into a fresh private lease and freshly allocated index pages."""
        if req.kv.req_pool_idx in self.requests:
            raise RuntimeError("QSA prefix restore cannot mutate an existing lease")
        self.validate_prefix_checkpoint(snapshot)
        state = self._acquire_request(req.kv.req_pool_idx, req.rid)
        ring = slice(req.kv.req_pool_idx * 4, (req.kv.req_pool_idx + 1) * 4)
        for segment in snapshot.segments:
            physical = self.slots.staging_slice(
                state.lease, segment.start, segment.stop
            )
            logical = (
                self.req_table[
                    req.kv.req_pool_idx, segment.start : segment.stop : 4
                ].long()
                // 4
            )
            for li in range(len(self.layer_ids)):
                self.full.k_buffer[li][physical].view(torch.uint8).copy_(
                    segment.raw[li, 0]
                )
                self.full.v_buffer[li][physical].view(torch.uint8).copy_(
                    segment.raw[li, 1]
                )
                index = self.pool.qsa_compressed_k_buffer_pool[li]
                index[logical] = segment.index[li].to(index.device)
                if self.strict and (
                    not torch.equal(
                        self.full.k_buffer[li][physical].view(torch.uint8).cpu(),
                        segment.raw[li, 0],
                    )
                    or not torch.equal(
                        self.full.v_buffer[li][physical].view(torch.uint8).cpu(),
                        segment.raw[li, 1],
                    )
                    or not torch.equal(
                        index[logical].cpu().view(torch.uint8),
                        segment.index[li].view(torch.uint8),
                    )
                ):
                    raise AssertionError("QSA restored prefix bytes/index differ")
        for target, data in zip(self.pool.qsa_key_state_buffer_pool, snapshot.pending):
            target[ring].copy_(data)
        self.pool.qsa_rope_position_buffer[ring].copy_(snapshot.rope)
        self._restore_mamba(snapshot.mamba, int(req.kv.mamba_pool_idx))
        if self.strict:
            if (
                not self._check_restored_mamba(
                    snapshot.mamba, int(req.kv.mamba_pool_idx)
                )
                or any(
                    not torch.equal(
                        target[ring].cpu().view(torch.uint8), data.view(torch.uint8)
                    )
                    for target, data in zip(
                        self.pool.qsa_key_state_buffer_pool, snapshot.pending
                    )
                )
                or not torch.equal(
                    self.pool.qsa_rope_position_buffer[ring].cpu(), snapshot.rope
                )
            ):
                raise AssertionError("QSA restored recurrent/pending state differs")
        req.kv.mamba_needs_clear = False
        req.kv.mamba_cow_src_index = None
        state.seq_len = snapshot.length
        self.record(
            "prefix_restore_complete",
            state.lease,
            reused_tokens=snapshot.length,
            **self.prefix_cache.stats(),
        )

    def validate_prefix_checkpoint(self, snapshot):
        """A missing state component must never turn into approximate reuse."""
        from sglang.srt.mem_cache.ple_state_pool import NGramPool, ShortConvPool

        for segment in snapshot.segments:
            if (
                segment.raw.shape
                != (len(self.layer_ids), 2, segment.stop - segment.start, 1, 256)
                or segment.raw.dtype != torch.uint8
            ):
                raise ValueError("QSA prefix raw geometry differs")
            shape = self.pool.qsa_compressed_k_buffer_pool[0].shape[1:]
            if (
                segment.index.shape
                != (len(self.layer_ids), (segment.stop - segment.start) // 4, *shape)
                or segment.index.dtype != self.pool.index_state_dtype
            ):
                raise ValueError("QSA prefix compressed geometry differs")
        if len(snapshot.pending) != len(self.pool.qsa_key_state_buffer_pool):
            raise ValueError("QSA prefix checkpoint has missing pending layers")
        expected = [(t[:4].shape, t.dtype) for t in self.pool.qsa_key_state_buffer_pool]
        actual = [(t.shape, t.dtype) for t in snapshot.pending]
        if (
            actual != expected
            or snapshot.rope.shape != self.pool.qsa_rope_position_buffer[:4].shape
            or snapshot.rope.dtype != torch.int64
        ):
            raise ValueError("QSA prefix checkpoint pending/position geometry differs")
        mamba = self.req_pool.mamba_pool
        expected_arity = 3 if mamba._slot_siblings else 2
        if len(snapshot.mamba) != expected_arity:
            raise ValueError("QSA prefix checkpoint has missing recurrent siblings")
        conv, temporal = snapshot.mamba[:2]
        if (
            len(conv) != len(mamba.mamba_cache.conv)
            or [(t.shape, t.dtype) for t in conv]
            != [(t[:, :1].shape, t.dtype) for t in mamba.mamba_cache.conv]
            or (temporal.shape, temporal.dtype)
            != (
                mamba.mamba_cache.temporal[:, :1].shape,
                mamba.mamba_cache.temporal.dtype,
            )
        ):
            raise ValueError("QSA prefix recurrent geometry differs")
        if mamba._slot_siblings:
            if len(snapshot.mamba[2]) != len(mamba._slot_siblings):
                raise ValueError("QSA prefix checkpoint has missing PLE siblings")
            for sibling, data in zip(mamba._slot_siblings, snapshot.mamba[2]):
                if isinstance(sibling, ShortConvPool):
                    source = sibling.conv_state[:, :1]
                elif isinstance(sibling, NGramPool):
                    source = sibling.context[:1]
                else:
                    raise ValueError(
                        "QSA prefix checkpoint has an unsupported slot sibling"
                    )
                if (data.shape, data.dtype) != (source.shape, source.dtype):
                    raise ValueError("QSA prefix PLE geometry differs")

    def begin_batch(self, batch, *, graph=False):
        if batch.forward_mode.is_idle():
            self.batch_requests = []
            return
        decode = batch.forward_mode.is_decode()
        if not decode and not batch.forward_mode.is_extend():
            raise RuntimeError("QSA P2 supports plain extend/decode only")
        if (
            not 1 <= batch.batch_size <= self.max_requests
            or (not decode and batch.batch_size != 1)
            or batch.req_pool_indices_cpu is None
            or batch.seq_lens_cpu is None
            or not batch.rids
            or len(batch.rids) != batch.batch_size
        ):
            raise RuntimeError("QSA P2 requires real bounded rows and CPU metadata")
        req_indices = batch.req_pool_indices_cpu.tolist()
        seq_lens = batch.seq_lens_cpu.tolist()
        if (
            len(req_indices) != batch.batch_size
            or len(seq_lens) != batch.batch_size
            or len(set(req_indices)) != len(req_indices)
        ):
            raise RuntimeError("QSA P2 batch metadata aliases or pads requests")
        if self.strict and (
            batch.req_pool_indices.cpu().tolist() != req_indices
            or batch.seq_lens.cpu().tolist() != seq_lens
        ):
            raise RuntimeError("QSA P2 CPU/GPU row identities or lengths differ")
        states = []
        for req_idx, rid, seq in zip(req_indices, batch.rids, seq_lens):
            if not 1 <= seq <= self.capacity:
                raise ValueError("QSA P2 request exceeds context capacity")
            if req_idx not in self.requests:
                if decode:
                    raise RuntimeError("QSA P2 decode without an admitted lease")
                self._acquire_request(req_idx, rid)
            state = self._request(req_idx, rid)
            self.slots.require(state.lease, "decode" if decode else "prefill")
            if graph and (
                len(state.states) != len(self.layer_ids)
                or any(s["generation"] != state.generation for s in state.states)
                or state.graph_identity != self.native_lease_snapshot(state)
            ):
                raise RuntimeError("QSA graph native/layer ownership changed")
            states.append(state)
        # Validate the entire batch before advancing either request's decode state.
        if (
            len({s.lease.slot for s in states}) != len(states)
            or len(set(batch.rids)) != len(states)
            or not all(batch.rids)
        ):
            raise RuntimeError(
                "QSA P2 batch aliases physical slots or request identities"
            )
        if graph:
            self.graph_batch = tuple(
                (state.lease, seq) for state, seq in zip(states, seq_lens)
            )
        self.forward_id += 1
        self.batch_requests = states
        for state, seq in zip(states, seq_lens):
            state.seq_len = seq
            if decode:
                state.decode_steps += 1
                if state.host is not None:
                    state.compressed_len.fill_(seq // 4)
        self.offloaded = decode and self.mode == "p2-offload"
        if self.mode == "p2-offload":
            if decode:
                locations = [
                    self.slots.ring_write_location(s.lease, s.seq_len)
                    for s in self.batch_requests
                ]
                count = len(states)
                self.raw_write_locs = self.batch_write_locs[:count]
                self.row_slots = self.batch_slots[:count]
                self.raw_write_locs.copy_(torch.tensor(locations, dtype=torch.int64))
                self.row_slots.copy_(
                    torch.tensor([s.lease.slot for s in states], dtype=torch.int32)
                )
                self.batch_lens[:count].copy_(
                    torch.tensor([seq // 4 for seq in seq_lens], dtype=torch.int32)
                )
                self.real.fill_(count)
            else:
                state = self.batch_requests[0]
                extend = batch.extend_seq_lens_cpu[0]
                segment = self.slots.staging_slice(
                    state.lease, state.seq_len - extend, state.seq_len
                )
                self.raw_write_locs = torch.arange(
                    segment.start, segment.stop, dtype=torch.int64, device=self.device
                )
        if decode:
            self.record(
                "decode_batch",
                batch_size=len(self.batch_requests),
                rows=[
                    {
                        "req_pool_idx": s.lease.req_pool_idx,
                        "generation": s.lease.generation,
                        "rid": s.lease.rid,
                        "lease_slot": s.lease.slot,
                        "seq_len": s.seq_len,
                        "tail": s.seq_len % 4,
                        "compressed_len": s.seq_len // 4,
                        "closes_c4": s.seq_len % 4 == 0,
                        "ring_location": self.slots.ring_write_location(
                            s.lease, s.seq_len
                        ),
                    }
                    for s in self.batch_requests
                ],
            )
        else:
            state = self.batch_requests[0]
            self.record(
                "prefill_chunk",
                batch_size=1,
                rows=[
                    {
                        "req_pool_idx": state.lease.req_pool_idx,
                        "generation": state.lease.generation,
                        "rid": state.lease.rid,
                        "lease_slot": state.lease.slot,
                        "start": state.seq_len - batch.extend_seq_lens_cpu[0],
                        "end": state.seq_len,
                    }
                ],
            )

    def write_locations(self, logical_locs):
        if self.mode == "p2-resident":
            return logical_locs
        if (
            self.raw_write_locs is None
            or self.raw_write_locs.numel() != logical_locs.numel()
        ):
            raise RuntimeError("QSA P2 raw writer does not match current batch")
        return self.raw_write_locs

    def prefill_slots(self, req_idx, seq_len):
        if self.mode == "p2-resident":
            return self.req_table[req_idx, :seq_len].long()
        state = self.requests[req_idx]
        segment = self.slots.staging_slice(state.lease, 0, seq_len)
        return torch.arange(
            segment.start, segment.stop, dtype=torch.int64, device=self.device
        )

    def handoff(self, req):
        state = self._request(req.kv.req_pool_idx, req.rid)
        try:
            return self._handoff(req)
        except Exception:
            # Even a failed submission may have queued work before recording its event.
            try:
                self.copy_stream.synchronize()
            finally:
                self.slots.phases[state.lease.req_pool_idx] = "failed"
                self.record("handoff_failed", state.lease)
            raise

    def _handoff(self, req):
        state = self._request(req.kv.req_pool_idx, req.rid)
        lease, prompt_len = state.lease, state.seq_len
        before = self.runner.token_to_kv_pool_allocator.available_size()
        self.slots.begin_handoff(lease)
        self.record("handoff_begin", lease, prompt_len=prompt_len)
        start = time.monotonic()
        if self.mode == "p2-offload":
            blocks, tail = divmod(prompt_len, 4)
            for li, lid in enumerate(self.layer_ids):
                k, v = self.full.k_buffer[li], self.full.v_buffer[li]
                for offset in range(0, blocks * 4, 4096):
                    stop = min(offset + 4096, blocks * 4)
                    segment = self.slots.staging_slice(lease, offset, stop)
                    kb, vb = k[segment].view(torch.uint8), v[segment].view(torch.uint8)
                    packed = pack_c4(kb, vb)
                    producer, done = torch.cuda.Event(), torch.cuda.Event()
                    producer.record()
                    with torch.cuda.stream(self.copy_stream):
                        self.copy_stream.wait_event(producer)
                        dst = state.host[li, offset // 4 : stop // 4]
                        dst.copy_(packed, non_blocking=True)
                        done.record(self.copy_stream)
                    state.handoff_event = done
                    # Preserve the accepted bounded handoff's buffer lifetime.
                    done.synchronize()
                    if self.strict and (
                        not torch.equal(dst[:, :1024].reshape(-1, 1, 256), kb.cpu())
                        or not torch.equal(dst[:, 1024:].reshape(-1, 1, 256), vb.cpu())
                    ):
                        raise AssertionError("QSA P2 handoff K/V bytes differ")
                if tail:
                    segment = self.slots.staging_slice(
                        lease, prompt_len - tail, prompt_len
                    )
                    state.full.k_buffer[li][1 : 1 + tail].copy_(k[segment])
                    state.full.v_buffer[li][1 : 1 + tail].copy_(v[segment])
                layer_state = state.make_state()
                stage_short_prefix(
                    layer_state["hot"], layer_state["tokens"], state.host[li, :blocks]
                )
                if blocks:
                    layer_state["hot"][2048].copy_(
                        state.host[li, blocks - 1], non_blocking=True
                    )
                    layer_state["tokens"][0, 2048] = blocks - 1
                state.states.append(layer_state)
                self.record(
                    "handoff_layer",
                    lease,
                    layer=lid,
                    d2h_bytes=blocks * 2048,
                    bytes_checked=self.strict,
                )
            state.offloaded = True
        done = torch.cuda.Event()
        done.record()
        done.synchronize()
        state.handoff_event = done
        self.slots.finish_handoff(lease, done)
        if getattr(self, "graph_enabled", False):
            state.graph_identity = self.native_lease_snapshot(state)
        if before != self.runner.token_to_kv_pool_allocator.available_size():
            raise AssertionError("QSA P2 handoff changed logical index ownership")
        self.record(
            "handoff_complete",
            lease,
            prompt_len=prompt_len,
            wall_ms=(time.monotonic() - start) * 1000,
            host_slab_ptr=None if state.host is None else state.host.data_ptr(),
        )
        return lease, done

    def after_store(self, layer, *, graph=False):
        if graph:
            from sglang.srt.layers.attention.qsa.hisparse_graph import close_c4

            li = self.pool._transfer_full_attention_id(layer.layer_id)
            count = self.graph_capture_size or len(self.graph_batch or ())
            if not self.graph_enabled or count not in self.graph_batch_sizes:
                raise RuntimeError("QSA C4 close has no graph context")
            close_c4[(count,)](
                self.full.k_buffer[li].view(torch.uint8),
                self.full.v_buffer[li].view(torch.uint8),
                self.layer_states[li]["hot"],
                self.layer_states[li]["tokens"],
                self.batch_slots,
                self.graph_seq_lens,
                self.real,
                self.slots.page_size + self.slots.staging_tokens,
            )
            return
        if self.offloaded:
            for state in self.batch_requests:
                self.slots.require(state.lease, "decode")
                state.after_store(layer)

    @profile_method("qsa.selected", nvtx_enabled=NVTX_OPERATIONS_ENABLED)
    def selected(self, layer, raw_indices, *, graph=False):
        from sglang.kernels.ops.kvcache.hisparse import load_cache_to_device_buffer_mla

        count = (
            (self.graph_capture_size or len(self.graph_batch or ()))
            if graph
            else len(self.batch_requests)
        )
        if (
            not self.offloaded
            or not 1 <= count <= self.max_requests
            or raw_indices.shape != (count, 2051)
            or raw_indices.dtype != torch.int32
            or raw_indices.device != self.compact.device
        ):
            raise RuntimeError(
                "QSA P2 selection rows do not match current decode batch"
            )
        li = self.pool._transfer_full_attention_id(layer.layer_id)
        for state in () if graph else self.batch_requests:
            self._request(state.lease.req_pool_idx, state.lease.rid)
            self.slots.require(state.lease, "decode")
            if state.states[li]["generation"] != state.generation:
                raise RuntimeError("stale QSA batched selected state")
        for state in () if graph else self.batch_requests:
            done = state.states[li]["done"]
            if done is not None:
                torch.cuda.current_stream(self.device).wait_event(done)
        shared, blocks = self.layer_states[li], self.blocks[:count]
        with operations_nvtx_range("qsa.resolve_refetch"):
            torch.div(raw_indices[:, :2048:4], 4, rounding_mode="floor", out=blocks)
            load_cache_to_device_buffer_mla(
                blocks,
                shared["tokens"],
                shared["host_locs"],
                shared["device_locs"],
                self.host_slabs.view(-1, 2048),
                shared["hot"],
                self.out[:count],
                self.row_slots,
                self.batch_lens[:count],
                shared["lru"],
                2048,
                512,
                2048,
                64,
                1024,
                self.real,
                self.miss_src[:count],
                self.miss_dst[:count],
                self.miss_count[:count],
            )
        with operations_nvtx_range("qsa.hot_gather"):
            gather_indices = self.gather_indices[: count * 512]
            gather_indices.copy_(self.out[:count].view(-1))
            gather_indices.view(count, 512).masked_fill_(
                self.initial_lru[:, :512] >= self.batch_lens[:count, None], 0
            )
            if graph:
                # Preserve native invalid output; mask only inactive capture rows.
                gather_indices.view(count, 512).masked_fill_(
                    self.graph_row_ids[:count] >= self.real, 0
                )
            torch.index_select(
                shared["hot"], 0, gather_indices, out=self.gathered[: count * 512]
            )
        with operations_nvtx_range("qsa.indexed_unpack"):
            torch.index_select(
                self.gathered.view(-1, 256),
                0,
                self.indices[: count * 4096],
                out=self.unpacked[:count].view(-1, 256),
            )
        if graph:
            from sglang.srt.layers.attention.qsa.hisparse_graph import finish_compact

            finish_compact[(count, 513)](
                self.unpacked,
                self.full.k_buffer[li].view(torch.uint8),
                self.full.v_buffer[li].view(torch.uint8),
                self.compact,
                self.compact_table,
                raw_indices,
                self.batch_slots,
                self.graph_seq_lens,
                self.real,
                self.capacity,
                self.slots.page_size + self.slots.staging_tokens,
                self.max_requests * 2052,
            )
            if self.graph_audit is not None:
                self.graph_audit["raw"][li, :count].copy_(raw_indices)
                self.graph_audit["compact"][li].copy_(self.compact)
                mapping = self.compact_table[
                    self.row_slots[:, None].long(), raw_indices.clamp_min(0).long()
                ]
                self.graph_audit["mapping"][li, :count].copy_(mapping)
                self.graph_audit["miss_count"][li, :count].copy_(
                    self.miss_count[:count]
                )
                self.graph_audit["resolver_out"][li, :count].copy_(self.out[:count])
                self.graph_audit["miss_src"][li, :count].copy_(self.miss_src[:count])
                self.graph_audit["miss_dst"][li, :count].copy_(self.miss_dst[:count])
        else:
            for row, state in enumerate(self.batch_requests):
                state.finish_selected(
                    layer,
                    raw_indices[row : row + 1],
                    blocks[row : row + 1],
                    self.unpacked[row],
                    self.miss_count[row : row + 1],
                )
        return (
            self.compact[0].view(self.pool.dtype),
            self.compact[1].view(self.pool.dtype),
            self.compact_table,
            self.row_slots,
        )

    def capture_decode(self, *args, **kwargs):
        if (
            getattr(self, "graph_enabled", False)
            and self.graph_audit is not None
            and (self.graph_capture_size is not None or self.graph_batch is not None)
        ):
            li = self.pool._transfer_full_attention_id(args[0].layer_id)
            valid_counts = args[8]
            self.graph_audit["valid_counts"][li, : valid_counts.numel()].copy_(
                valid_counts
            )

    def release(self, req_idx, rid):
        if req_idx not in self.requests:
            if req_idx in self.slots.active:
                raise RuntimeError(
                    "QSA P2 cannot release a partially initialized lease"
                )
            return None  # Allocated, but no model forward and no physical lease yet.
        state = self._request(req_idx, rid)
        if state.lease in getattr(self, "graph_failed_leases", ()):
            # A failed completion record may leave DMA without a usable event.
            self._drain_graph_streams()
            self.graph_failed_leases.remove(state.lease)
        self.record("release_begin", state.lease)
        terminal = torch.cuda.Event()
        if self.producer_stream is None:
            raise RuntimeError("QSA P2 release has no registered producer stream")
        terminal.record(self.producer_stream)
        events = [s["done"] for s in state.states if s["done"] is not None]
        if state.handoff_event is not None:
            events.append(state.handoff_event)
        self.slots.drain(state.lease, terminal, events)
        self.record(
            "release_drained",
            state.lease,
            pending_events=0,
            graph_copy_epochs=sorted(
                {s["copy_epoch"] for s in state.states if "copy_epoch" in s}
            ),
        )
        return state.lease

    def after_release(self, lease):
        if lease is None:
            return
        self.slots.require(lease, "drained", "logical_flushed")
        allocator = self.runner.token_to_kv_pool_allocator
        if allocator.free_group is not None:
            self.slots.require(lease, "drained")
            if lease in self.pending_releases:
                raise RuntimeError("duplicate QSA P2 deferred release")
            self.pending_releases.append(lease)
            self.record("logical_release_pending", lease)
            return
        if self.slots.phases[lease.req_pool_idx] == "drained":
            self.slots.logical_flushed(lease, allocator)
        state = self.requests[lease.req_pool_idx]
        state.states.clear()
        state.host = None
        self.batch_requests = [s for s in self.batch_requests if s.lease != lease]
        self.slots.commit_release(lease)
        del self.requests[lease.req_pool_idx]
        self.record("logical_release_complete", lease)

    def after_logical_flush(self):
        while self.pending_releases:
            lease = self.pending_releases[0]
            try:
                self.after_release(lease)
            finally:
                # Keep unfinished leases reachable on failure. A ledger failure
                # after commit must not replay the already released generation.
                if lease.req_pool_idx not in self.slots.active:
                    self.pending_releases.pop(0)
