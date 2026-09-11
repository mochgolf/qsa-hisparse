"""Opt-in, single-request eager QSA prefill-to-host handoff.

Logical slots still own compressed index-K. Only raw attention backing moves.
This experiment deliberately does not implement concurrent prefill or graphs.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch

from sglang.srt.utils.nvtx_utils import (
    NVTX_OPERATIONS_ENABLED,
    operations_nvtx_range,
    profile_method,
)


def pack_c4(k, v):
    """Pack complete raw byte rows; also used by the CPU layout check."""
    if k.shape != v.shape or k.shape[-1] != 256 or k.numel() % 1024:
        raise ValueError("complete C4 K/V rows required")
    return torch.cat((k.reshape(-1, 1024), v.reshape(-1, 1024)), dim=1)


def unpack_index(device):
    rows = torch.arange(2048, device=device, dtype=torch.int64)
    k = (rows // 4) * 8 + rows % 4
    return torch.cat((k, k + 4))


def stage_short_prefix(hot, tokens, records):
    """HiSparse's <=HOT fast path assumes a fully resident ordered prefix."""
    count = records.shape[0]
    if count <= 2048:
        hot[:count].copy_(records, non_blocking=True)
        tokens[0, :count] = torch.arange(count, dtype=torch.int32, device=hot.device)


def validate_configuration(args, pool, *, p2=False, graph=False, strict=True):
    if graph and not p2:
        raise ValueError("only QSA P2 offload supports the bounded graph path")
    max_requests = getattr(args, "max_running_requests", None) if p2 else 1
    if p2 and max_requests not in (2, 4, 8):
        raise ValueError("QSA P2 requires max_running_requests in (2, 4, 8)")
    required = {
        "max_running_requests": max_requests,
        "tp_size": 2,
        "pp_size": 1,
        "disable_radix_cache": True,
        "disable_overlap_schedule": True,
        "cuda_graph_backend_decode": "full" if graph else "disabled",
        "cuda_graph_backend_prefill": "disabled",
        "context_length": 262144,
        "max_total_tokens": max_requests * 262144,
        "skip_server_warmup": True,
        "random_seed": 147342228,
        "speculative_algorithm": None,
        "disaggregation_mode": "null",
    }
    # The strict comparison oracle needs batch-invariant math. P2 light runtime
    # can use the user's native GEMM path; deterministic GEMM dominates B1 cost.
    if not p2 or strict:
        required["enable_deterministic_inference"] = True
    if graph:
        required.update(disable_cuda_graph_padding=True,
                        cuda_graph_bs_decode=list(range(1, max_requests + 1)),
                        cuda_graph_max_bs_decode=max_requests, enable_torch_compile=False)
    for name, value in required.items():
        if getattr(args, name, None) != value:
            raise ValueError(f"QSA V3 requires {name}={value!r}")
    chunks = (2048, 4096) if p2 else (2048,)
    if getattr(args, "chunked_prefill_size", None) not in chunks:
        raise ValueError(f"QSA requires chunked_prefill_size in {chunks}")
    if getattr(args, "enable_hisparse", False):
        raise ValueError("QSA V3 cannot use the MLA HiSparse coordinator")
    if any(getattr(args, x, False) for x in (
        "enable_dp_attention", "enable_two_batch_overlap", "enable_single_batch_overlap",
        "enable_streaming_session",
    )):
        raise ValueError("QSA V3 does not support DP/overlap")
    full = pool.full_kv_pool
    if (pool.size != max_requests * 262144 or pool.page_size != 64 or pool.qsa_compress_ratio != 4
            or pool.qsa_token_topk != 2048 or pool.full_layer_nums != 12
            or pool.head_num != 1 or pool.head_dim != 256
            or pool.dtype != torch.float8_e4m3fn
            or full.use_hnd or full.kv_cache_layout == "vectorized_5d"
            or full.post_capture_active or full.is_quantized_kv_cache):
        raise ValueError("QSA V3 requires plain NHD FP8 TP2 C4/page64 pool")


class QSAHiSparseV3:
    HOT, PAGE, ITEM, TOPK = 2048, 64, 2048, 512

    def write_locations(self, logical_locs):
        return self.ring_loc if self.offloaded else logical_locs

    def prefill_slots(self, req_idx, seq_len):
        return self.req_table[req_idx, :seq_len].long()

    def __init__(self, runner, mode):
        if mode not in ("resident", "offload"):
            raise ValueError("SGLANG_QSA_HISPARSE_V3 must be resident or offload")
        self.observe = os.environ.get("SGLANG_QSA_HISPARSE_V3_OBSERVE", "strict")
        if self.observe not in ("strict", "light"):
            raise ValueError("SGLANG_QSA_HISPARSE_V3_OBSERVE must be strict or light")
        self.strict = self.observe == "strict"
        capture = os.environ.get("SGLANG_QSA_HISPARSE_V3_CAPTURE")
        if capture and not self.strict:
            raise ValueError("QSA V3 light observation cannot enable tensor capture")
        self.pool = runner.token_to_kv_pool
        validate_configuration(runner.server_args, self.pool)
        from sglang.srt.model_executor.cuda_graph_config import cuda_graph_fully_disabled

        if not cuda_graph_fully_disabled():
            raise ValueError("QSA V3 requires both resolved graph phases disabled")
        if getattr(self.pool, "qsa_hisparse_v3", None) is not None:
            raise ValueError("QSA V3 requires one shared prefill/decode backend")
        self.runner, self.mode = runner, mode
        self.device = self.pool.device
        self.full = self.pool.full_kv_pool
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

        if type(self.full) is not MHATokenToKVPool or getattr(runner, "_unified_memory_pool", None) is not None:
            raise ValueError("QSA V3 requires independently owned plain MHA backing")
        from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator

        if type(runner.token_to_kv_pool_allocator) is not PagedTokenToKVPoolAllocator:
            raise ValueError("QSA V3 requires the ordinary paged logical allocator")
        self.req_table = runner.req_to_token_pool.req_to_token
        self.capacity = min(self.pool.size, self.req_table.shape[1])
        if self.capacity != 262144:
            raise ValueError("QSA V3 requires request-table capacity >=262144")
        self.layer_ids = list(self.pool.full_attention_layer_id_mapping)
        self.rank = runner.ps.tp_rank
        self.copy_stream = torch.cuda.Stream(device=self.device)
        self.owner = None
        self.owner_rid = None
        self.seen_rids = set()
        self.failed = False
        self.releasing = False
        self.pending_release = None
        self.capture_dir = Path(capture) if capture else None
        self.capture_layers = {}
        if self.capture_dir is not None:
            self.capture_dir.mkdir(parents=True, exist_ok=True)
        self.generation = 0
        self.offloaded = False
        self.states = []
        self.host = None
        self.host_slab = None
        self.decode_steps = 0
        self.seq_len = 0
        self.is_decode = False
        self.raw_shapes = (self.full.k_buffer[0].shape, self.full.v_buffer[0].shape)
        self.raw_released = False
        self.index_ptr = self.pool.qsa_compressed_flat.data_ptr()
        self.mamba_pool = runner.req_to_token_pool.mamba_pool
        self.mamba_ptrs, self.mamba_sizes, _ = self.mamba_pool.get_contiguous_buf_infos()
        self.path = None
        directory = os.environ.get("SGLANG_QSA_HISPARSE_V3_EVENTS")
        if directory:
            Path(directory).mkdir(parents=True, exist_ok=True)
            self.path = Path(directory) / f"rank-{self.rank}.jsonl"
        # ponytail: one persistent slab for the guarded B1 adapter; P2 needs per-request leases.
        host_alloc_start = time.monotonic()
        if self.mode == "offload":
            self.host_slab = torch.empty(
                (12, (self.capacity + 3) // 4, self.ITEM),
                dtype=torch.uint8, pin_memory=True,
            )
        self.record("init", allocation_phase="startup",
                    host_alloc_wall_ms=(time.monotonic() - host_alloc_start) * 1000
                    if self.mode == "offload" else 0.0)

    def record(self, event, **extra):
        if self.path is None or (not self.strict and event in (
            "decode_step", "prefill_step", "selected_check",
        )):
            return
        raw = sum(t.numel() * t.element_size() for t in self.full.k_buffer + self.full.v_buffer)
        host_reserved = 0 if self.host_slab is None else self.host_slab.nbytes
        host_active = 0 if self.host is None else self.host.nbytes
        row = {
            "event": event, "time_ns": time.time_ns(), "rank": self.rank,
            "mode": self.mode, "observe": self.observe, "generation": self.generation,
            "req_pool_idx": self.owner, "seq_len": self.seq_len,
            "rid": self.owner_rid,
            "decode_steps": self.decode_steps, "raw_bytes": raw,
            "index_bytes": self.pool.qsa_compressed_flat.numel() * 2,
            "index_storage_unchanged": self.index_ptr == self.pool.qsa_compressed_flat.data_ptr(),
            "host_bytes": host_active,
            "host_reserved_bytes": host_reserved,
            "host_free_bytes": host_reserved - host_active,
            "host_slab_ptr": None if self.host_slab is None else self.host_slab.data_ptr(),
            "hot_bytes": sum(s["hot"].numel() for s in self.states),
            "workspace_bytes": sum(
                t.nbytes for name in ("indices", "gathered", "unpacked", "compact", "compact_table", "zero_req", "real", "ring_loc", "compressed_len")
                if (t := getattr(self, name, None)) is not None
            ),
            "mamba_bytes": sum(self.mamba_sizes),
            "mamba_storage_unchanged": self.mamba_ptrs == self.mamba_pool.get_contiguous_buf_infos()[0],
            "mamba_available": self.runner.req_to_token_pool.mamba_allocator.available_size(),
            "logical_available": self.runner.token_to_kv_pool_allocator.available_size(),
            "cuda_allocated": torch.cuda.memory_allocated(self.device),
            "cuda_reserved": torch.cuda.memory_reserved(self.device),
            "cuda_max_allocated": torch.cuda.max_memory_allocated(self.device),
            "cuda_max_reserved": torch.cuda.max_memory_reserved(self.device),
            **extra,
        }
        with self.path.open("a") as stream:
            stream.write(json.dumps(row) + "\n")

    @profile_method("qsa.batch_metadata", nvtx_enabled=NVTX_OPERATIONS_ENABLED)
    def begin_batch(self, batch):
        if self.failed or self.releasing:
            raise RuntimeError("QSA V3 cannot serve a failed/releasing lease")
        if batch.forward_mode.is_idle():
            return
        if batch.batch_size != 1:
            raise RuntimeError("QSA V3 admits exactly one active request")
        req = int(batch.req_pool_indices[0].item())
        if not batch.rids or len(batch.rids) != 1:
            raise RuntimeError("QSA V3 requires a real request ID")
        rid = batch.rids[0]
        seq = int(batch.seq_lens[0].item())
        decode = batch.forward_mode.is_decode()
        if not decode and not batch.forward_mode.is_extend():
            raise RuntimeError("QSA V3 supports only plain extend/decode")
        if self.owner is None:
            if decode:
                raise RuntimeError("decode without a prefill lease")
            # ponytail: bounded experiment requires unique rids; no async lease registry.
            if rid in self.seen_rids:
                raise RuntimeError("QSA V3 requires unique request IDs across reuse")
            self.seen_rids.add(rid)
            self.owner, self.owner_rid = req, rid
            self.seq_len = seq
            self.generation += 1
            self.decode_steps = 0
            torch.cuda.reset_peak_memory_stats(self.device)
            if self.raw_released:
                self.full._create_buffers()
                for k, v in zip(self.full.k_buffer, self.full.v_buffer):
                    if (k.shape, v.shape) != self.raw_shapes:
                        raise AssertionError("restored raw buffer shape differs")
                if self.full.k_data_ptrs.cpu().tolist() != [k.data_ptr() for k in self.full.k_buffer]:
                    raise AssertionError("restored K pointer metadata differs")
                if self.full.v_data_ptrs.cpu().tolist() != [v.data_ptr() for v in self.full.v_buffer]:
                    raise AssertionError("restored V pointer metadata differs")
                self.raw_released = False
                self.record("restore_staging", shapes_and_pointers_checked=True)
            self.record("begin_prefill")
        if (req, rid) != (self.owner, self.owner_rid):
            raise RuntimeError("QSA V3 request owner changed before release")
        if self.offloaded and not decode:
            raise RuntimeError("QSA V3 cannot retract an offloaded request into prefill")
        self.seq_len, self.is_decode = seq, decode
        if decode:
            if seq - 1 < 2048:
                raise RuntimeError("QSA V3 requires at least 2048 prompt tokens")
            if self.mode == "offload" and not self.offloaded:
                try:
                    self.handoff(seq - 1)
                except Exception:
                    self.failed = True
                    self.copy_stream.synchronize()
                    self.record("handoff_failed", completed_layers=len(self.states))
                    raise
            elif self.mode == "resident" and self.decode_steps == 0:
                self.record("resident_boundary", prompt_len=seq - 1)
            self.decode_steps += 1
            if self.offloaded:
                self.ring_loc.fill_(1 + (seq - 1) % 4)
                self.compressed_len.fill_(seq // 4)
            self.record("decode_step")
        else:
            self.record("prefill_step")

    @profile_method("qsa.handoff", nvtx_enabled=NVTX_OPERATIONS_ENABLED)
    def handoff(self, prompt_len):
        if (self.mode != "offload" or self.owner is None or self.failed or self.releasing
                or self.host is not None or self.host_slab is None):
            raise RuntimeError("QSA V3 handoff requires an unclaimed host slab and active lease")
        start_wall = time.monotonic()
        allocated_before = torch.cuda.memory_allocated(self.device)
        logical_available = self.runner.token_to_kv_pool_allocator.available_size()
        # Keep append capacity for C4 blocks completed during decode.
        self.host = self.host_slab
        self.record("handoff_begin", prompt_len=prompt_len)
        blocks, tail = divmod(prompt_len, 4)
        with operations_nvtx_range("qsa.handoff.allocate_validate"):
            slots = self.req_table[self.owner, :prompt_len].long()
            slots_cpu = slots.cpu()
            complete = slots_cpu[:blocks * 4].reshape(-1, 4)
            if not torch.equal(complete, complete[:, :1] + torch.arange(4)):
                raise RuntimeError("C4 physical source members are not contiguous")
            if len(torch.unique(slots_cpu)) != prompt_len:
                raise RuntimeError("prefill physical slots alias")
        for li, lid in enumerate(self.layer_ids):
            copy_ms = 0.0
            k, v = self.full.k_buffer[li], self.full.v_buffer[li]
            # Copy bounded chunks; keep both original buffers alive through D2H.
            for offset in range(0, blocks * 4, 4096):
                stop = min(offset + 4096, blocks * 4)
                src = slots[offset:stop]
                with operations_nvtx_range("qsa.handoff.gather_pack"):
                    kb = k.view(torch.uint8).index_select(0, src)
                    vb = v.view(torch.uint8).index_select(0, src)
                    packed = pack_c4(kb, vb)
                producer, copy_begin, done = torch.cuda.Event(), torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                producer.record()
                dst = self.host[li, offset // 4:stop // 4]
                with operations_nvtx_range("qsa.handoff.d2h_submit"):
                    with torch.cuda.stream(self.copy_stream):
                        self.copy_stream.wait_event(producer)
                        copy_begin.record(self.copy_stream)
                        dst.copy_(packed, non_blocking=True)
                        done.record(self.copy_stream)
                with operations_nvtx_range("qsa.handoff.wait_copy"):
                    done.synchronize()
                copy_ms += copy_begin.elapsed_time(done)
                # Independent source slices, not unpacking the candidate pack.
                if self.strict:
                    if not torch.equal(dst[:, :1024].reshape(-1, 1, 256), kb.cpu()):
                        raise AssertionError("handoff K bytes differ")
                    if not torch.equal(dst[:, 1024:].reshape(-1, 1, 256), vb.cpu()):
                        raise AssertionError("handoff V bytes differ")
            # The existing store_cache writer skips reserved padding slot 0.
            ring_k = torch.zeros((5, 1, 256), dtype=k.dtype, device=self.device)
            ring_v = torch.zeros_like(ring_k)
            if tail:
                ring_k[1:1 + tail].copy_(k.view(torch.uint8).index_select(0, slots[-tail:]).view(k.dtype))
                ring_v[1:1 + tail].copy_(v.view(torch.uint8).index_select(0, slots[-tail:]).view(v.dtype))
            state = self.make_state()
            stage_short_prefix(state["hot"], state["tokens"], self.host[li, :blocks])
            state["hot"][self.HOT].copy_(self.host[li, blocks - 1], non_blocking=True)
            state["tokens"][0, self.HOT] = blocks - 1
            self.states.append(state)
            self.full.k_buffer[li], self.full.v_buffer[li] = ring_k, ring_v
            del k, v, kb, vb, packed
            self.record("handoff_layer", layer=lid, source_bytes=prompt_len * 512,
                        d2h_bytes=blocks * self.ITEM, d2h_event_ms=copy_ms, bytes_checked=self.strict)
        self.full._init_data_ptrs_and_strides()
        self.raw_released = True
        self.indices = unpack_index(self.device)
        self.gathered = torch.empty((512, 2048), dtype=torch.uint8, device=self.device)
        self.unpacked = torch.empty((2, 2048, 1, 256), dtype=torch.uint8, device=self.device)
        self.compact = torch.zeros((2, 2052, 1, 256), dtype=torch.uint8, device=self.device)
        self.compact_table = torch.zeros((1, self.capacity), dtype=torch.int32, device=self.device)
        self.zero_req = torch.zeros(1, dtype=torch.int32, device=self.device)
        self.real = torch.ones(1, dtype=torch.int32, device=self.device)
        self.ring_loc = torch.zeros(1, dtype=torch.int64, device=self.device)
        self.compressed_len = torch.zeros(1, dtype=torch.int32, device=self.device)
        self.offloaded = True
        if self.runner.token_to_kv_pool_allocator.available_size() != logical_available:
            raise AssertionError("handoff freed index-K logical ownership")
        self.record("handoff_complete", prompt_len=prompt_len,
                    wall_ms=(time.monotonic() - start_wall) * 1000,
                    allocated_before=allocated_before,
                    allocated_drop_bytes=allocated_before - torch.cuda.memory_allocated(self.device),
                    pending_tail=tail, logical_lease_preserved=True)

    @profile_method("qsa.state_allocate", nvtx_enabled=NVTX_OPERATIONS_ENABLED)
    def make_state(self):
        physical = self.HOT + self.PAGE
        return {
            "hot": torch.zeros((physical, self.ITEM), dtype=torch.uint8, device=self.device),
            "tokens": torch.full((1, physical), -1, dtype=torch.int32, device=self.device),
            "lru": torch.arange(self.HOT, dtype=torch.int16, device=self.device)[None],
            "host_locs": torch.arange((self.capacity + 3) // 4, dtype=torch.int64, device=self.device)[None],
            "device_locs": torch.arange(physical, dtype=torch.int32, device=self.device)[None],
            "out": torch.empty((1, self.TOPK), dtype=torch.int32, device=self.device),
            "miss_src": torch.empty((1, self.TOPK), dtype=torch.int64, device=self.device),
            "miss_dst": torch.empty((1, self.TOPK), dtype=torch.int32, device=self.device),
            "miss_count": torch.zeros(1, dtype=torch.int32, device=self.device),
            "done": None,
            "generation": self.generation,
            "writeback_bytes": 0,
        }

    @profile_method("qsa.writeback", nvtx_enabled=NVTX_OPERATIONS_ENABLED)
    def after_store(self, layer):
        if not self.offloaded or self.seq_len % 4:
            return
        li = self.pool._transfer_full_attention_id(layer.layer_id)
        state = self.states[li]
        if state["generation"] != self.generation:
            raise RuntimeError("stale QSA V3 writeback state")
        if state["done"] is not None:
            torch.cuda.current_stream(self.device).wait_event(state["done"])
        hot = state["hot"][self.HOT]
        hot[:1024].copy_(self.full.k_buffer[li][1:5].view(torch.uint8).reshape(-1))
        hot[1024:].copy_(self.full.v_buffer[li][1:5].view(torch.uint8).reshape(-1))
        block = self.seq_len // 4 - 1
        if block < self.HOT:
            state["hot"][block].copy_(hot)
            state["tokens"][0, block] = block
        state["tokens"][0, self.HOT] = block
        producer = torch.cuda.Event()
        copy_begin = torch.cuda.Event(enable_timing=True) if self.strict else None
        done = torch.cuda.Event(enable_timing=self.strict)
        producer.record()
        with torch.cuda.stream(self.copy_stream):
            self.copy_stream.wait_event(producer)
            if copy_begin is not None:
                copy_begin.record(self.copy_stream)
            self.host[li, block].copy_(hot, non_blocking=True)
            done.record(self.copy_stream)
        state["done"] = done
        state["copy_begin"] = copy_begin
        state["writeback_bytes"] += self.ITEM

    @profile_method("qsa.selected", nvtx_enabled=NVTX_OPERATIONS_ENABLED)
    def selected(self, layer, raw_indices):
        from sglang.kernels.ops.kvcache.hisparse import load_cache_to_device_buffer_mla

        if raw_indices.shape != (1, 2051):
            raise RuntimeError("QSA V3 requires the fixed width 2051 raw selection")
        li = self.pool._transfer_full_attention_id(layer.layer_id)
        state = self.states[li]
        if state["generation"] != self.generation:
            raise RuntimeError("stale QSA V3 selected state")
        if state["done"] is not None:
            torch.cuda.current_stream(self.device).wait_event(state["done"])
        with operations_nvtx_range("qsa.resolve_refetch"):
            blocks = (raw_indices[:, :2048:4] // 4).to(torch.int32).contiguous()
            load_cache_to_device_buffer_mla(
                blocks, state["tokens"], state["host_locs"], state["device_locs"],
                self.host[li], state["hot"], state["out"], self.zero_req.long(),
                self.compressed_len, state["lru"], 2048, 512, 2048, 64, 1024,
                self.real, state["miss_src"], state["miss_dst"], state["miss_count"],
            )
        with operations_nvtx_range("qsa.hot_gather"):
            torch.index_select(state["hot"], 0, state["out"].reshape(-1).long(), out=self.gathered)
        with operations_nvtx_range("qsa.indexed_unpack"):
            torch.index_select(self.gathered.view(-1, 256), 0, self.indices,
                               out=self.unpacked.view(-1, 256))
        self.finish_selected(layer, raw_indices, blocks, self.unpacked, state["miss_count"])
        return self.compact[0].view(self.pool.dtype), self.compact[1].view(self.pool.dtype), self.compact_table, self.zero_req

    def finish_selected(self, layer, raw_indices, blocks, unpacked, miss_count):
        """Shared compact/tail mapping and independent host-byte oracle."""
        li = self.pool._transfer_full_attention_id(layer.layer_id)
        state = self.states[li]
        with operations_nvtx_range("qsa.compact_mapping"):
            self.compact[:, 1:2049].copy_(unpacked)
            tail = self.seq_len % 4
            if tail:
                self.compact[0, 2049:2049 + tail].copy_(self.full.k_buffer[li][1:1 + tail].view(torch.uint8))
                self.compact[1, 2049:2049 + tail].copy_(self.full.v_buffer[li][1:1 + tail].view(torch.uint8))
            valid = raw_indices[0, :2048 + tail].long()
            base = getattr(self, "compact_base", 0)
            self.compact_table[0, valid] = torch.arange(1 + base, 2049 + tail + base, dtype=torch.int32, device=self.device)
        self.check_selected(layer, raw_indices, blocks, miss_count)

    def check_selected(self, layer, raw_indices, blocks, miss_count, *, compact=None, mapping=None):
        """Independent host-byte oracle, also callable on post-replay snapshots."""
        check = self.strict and (self.decode_steps in (1, 2, 3, 384, 767) or self.seq_len % 4 == 0)
        if check:
            li = self.pool._transfer_full_attention_id(layer.layer_id)
            state = self.states[li]
            tail, base = self.seq_len % 4, getattr(self, "compact_base", 0)
            valid = raw_indices[0, :2048 + tail].long()
            compact = self.compact if compact is None else compact
            mapping = self.compact_table[0, valid] if mapping is None else mapping
            ids = blocks[0].cpu().long()
            if len(torch.unique(ids)) != 512 or int(ids.min()) < 0 or int(ids.max()) >= self.seq_len // 4:
                raise AssertionError("invalid selected C4 set")
            if not torch.equal(raw_indices[0, :2048].cpu().reshape(-1, 4), ids[:, None] * 4 + torch.arange(4)):
                raise AssertionError("invalid C4 expansion")
            # Sync only the copy completion needed by this byte check, never the device.
            if state["done"] is not None:
                state["done"].synchronize()
            expected = self.host[li].index_select(0, ids)
            ek, ev = expected[:, :1024].reshape(-1, 1, 256), expected[:, 1024:].reshape(-1, 1, 256)
            if not torch.equal(compact[0, 1:2049].cpu(), ek) or not torch.equal(compact[1, 1:2049].cpu(), ev):
                raise AssertionError("selected unpack K/V bytes differ")
            if not torch.equal(raw_indices[0, 2048:].cpu(), torch.cat((
                torch.arange(self.seq_len - tail, self.seq_len, dtype=torch.int32),
                torch.full((3 - tail,), -1, dtype=torch.int32),
            ))):
                raise AssertionError("pending tail indices/mask differ")
            for plane, source in ((0, self.full.k_buffer[li]), (1, self.full.v_buffer[li])):
                if not torch.equal(compact[plane, 2049:2049 + tail].cpu(), source[1:1 + tail].view(torch.uint8).cpu()):
                    raise AssertionError("pending tail bytes differ")
            if not torch.equal(mapping.cpu(), torch.arange(1 + base, 2049 + tail + base, dtype=torch.int32)):
                raise AssertionError("compact physical mapping differs")
            self.record("selected_check", layer=layer.layer_id, bytes_checked=2048 * 512,
                        tail=tail, tail_bytes_checked=tail * 512, mapping_checked=True,
                        page_boundary=self.seq_len % 64 == 0,
                        latest_writeback_event_ms=(state["copy_begin"].elapsed_time(state["done"])
                                                  if state.get("copy_begin") is not None else None),
                        writeback_bytes=state["writeback_bytes"], miss_count=int(miss_count[0]))

    def release(self, req_idx, rid):
        if self.owner is None:
            # An admitted but never-forwarded request owns no adapter storage.
            if rid in self.seen_rids:
                raise RuntimeError("duplicate QSA V3 release")
            return None
        if (int(req_idx), rid) != (self.owner, self.owner_rid) or self.releasing:
            raise RuntimeError("stale QSA V3 request release")
        self.releasing = True
        lease = (self.owner, self.owner_rid, self.generation)
        self.record("release_begin", rid=rid)
        start = time.monotonic()
        terminal = torch.cuda.Event()
        terminal.record()
        terminal.synchronize()
        terminal_ms = (time.monotonic() - start) * 1000
        start = time.monotonic()
        self.copy_stream.synchronize()
        drain_ms = (time.monotonic() - start) * 1000
        self.states.clear()
        self.host = None
        if self.offloaded:
            for name in ("indices", "gathered", "unpacked", "compact", "compact_table", "zero_req", "real", "ring_loc", "compressed_len"):
                setattr(self, name, None)
        self.offloaded = False
        self.record("release", rid=rid, pending_events=0, terminal_wall_ms=terminal_ms, copy_drain_wall_ms=drain_ms)
        return lease

    def after_release(self, lease):
        if lease is None:
            return
        if not self.releasing or lease != (self.owner, self.owner_rid, self.generation):
            raise RuntimeError("stale QSA V3 release commit")
        if self.runner.token_to_kv_pool_allocator.free_group is not None:
            self.pending_release = lease
            self.record("logical_release_pending")
            return
        self.record("logical_release_complete")
        self.pending_release = None
        self.owner = self.owner_rid = None
        self.releasing = False

    def capture_decode(self, layer, q, packed_k, packed_v, raw_indices, output, k_scale, v_scale,
                       valid_counts, cu_seqlens_q, cu_seqlens_k):
        """Bounded, opt-in evidence for execution-01's exact output mismatch."""
        if self.capture_dir is None or self.decode_steps > 127:
            return
        row = {"q": q.detach().cpu(), "output": output.detach().cpu(),
               "indices": raw_indices.detach().cpu(), "k_scale": k_scale, "v_scale": v_scale,
               "valid_counts": valid_counts.cpu(), "cu_q": cu_seqlens_q.cpu(), "cu_k": cu_seqlens_k.cpu(),
               "q_stride": q.stride(), "k_shape": packed_k.shape, "k_stride": packed_k.stride(),
               "v_shape": packed_v.shape, "v_stride": packed_v.stride()}
        if self.decode_steps in (1, 2, 93, 94):
            valid = int(row["cu_k"][-1])
            row["k"] = packed_k[:valid].detach().cpu()
            row["v"] = packed_v[:valid].detach().cpu()
        self.capture_layers[layer.layer_id] = row
        if layer.layer_id == self.layer_ids[-1]:
            torch.save(
                {"rid": self.owner_rid, "generation": self.generation,
                 "rank": self.rank, "step": self.decode_steps, "seq_len": self.seq_len,
                 "layers": self.capture_layers},
                self.capture_dir / f"gen-{self.generation}-step-{self.decode_steps:04d}-rank-{self.rank}.pt",
            )
            self.capture_layers = {}
