#!/usr/bin/env python3
"""Independent QSA/HiSparse cache-I/O microbenchmark.

The benchmark deliberately exercises the upstream HiSparse GPU resolve kernel
through ``sglang.kernels.ops.kvcache.hisparse``.  The CPU side only parses the
rank-0 trace, builds synthetic payloads, and performs post-fetch validation;
it never supplies a miss/victim list to the measured C path.

The payload is a byte-level FP8 stand-in.  Native K/V are represented as
``[token, 1, 256]`` uint8 arrays.  The prototype host layout packs four native
tokens into one 2048-byte C4 record ``[K4(1024), V4(1024)]``.  K/V scales are
separate non-unit layer-level metadata, copied/read alongside every arm and
checked independently; they are not incorrectly counted as per-block IO.

No model output or end-to-end QSA semantics are claimed here.  The real
38K trace supplies selected C4 IDs and layer/position cadence.  ``--span
261144`` relocates that same selection pattern near a 261K logical context to
exercise wide host addressing; it is explicitly synthetic until a real 261K
rank-0 trace is supplied.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import multiprocessing as mp
import os
import queue as queue_module
import shutil
import statistics
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SGLANG_ROOT = Path(
    os.environ.get("QSA_HISPARSE_SOURCE", ROOT / "sources" / "sglang-hisparse")
)
DEFAULT_TRACE = Path(os.environ.get("QSA_HISPARSE_TRACE", "<TRACE_JSONL>"))
LAYER_IDS = (3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47)
HOT_BLOCKS = 2048
TOP_K = 512
C4 = 4
HEAD_DIM = 256
FP8_BYTES = 1
RAW_KV_BYTES = C4 * HEAD_DIM * FP8_BYTES * 2
RAW_K_OR_V_BYTES = C4 * HEAD_DIM * FP8_BYTES
ITEM_BYTES = RAW_K_OR_V_BYTES * 2
PAGE_SIZE = 64
BLOCK_SIZE = 1024
BARRIER_TIMEOUT_S = 300


def _percentiles(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"p50_ms": None, "p95_ms": None, "p99_ms": None, "max_ms": None}
    ordered = sorted(float(x) for x in values)

    def nearest_rank(frac: float) -> float:
        return ordered[min(len(ordered) - 1, max(0, math.ceil(frac * len(ordered)) - 1))]

    return {
        "p50_ms": nearest_rank(0.50),
        "p95_ms": nearest_rank(0.95),
        "p99_ms": nearest_rank(0.99),
        "max_ms": ordered[-1],
        "mean_ms": statistics.fmean(ordered),
        "n": len(ordered),
    }


def _toy_lru(capacity: int, current: list[int], selected: Sequence[int]) -> tuple[list[int], list[int], list[int]]:
    """Small CPU diagnostic only; never feeds a measured GPU call."""
    cache = list(current)
    hits = [x for x in selected if x in cache]
    misses = [x for x in selected if x not in cache]
    # Match the intended contract: stale entries first, then newly fetched
    # blocks, then current hits.  Upstream's parallel tie order may differ.
    stale = [x for x in cache if x not in hits]
    evicted = stale[: len(misses)]
    survivors = stale[len(misses) :]
    cache = (survivors + misses + hits)[-capacity:]
    return cache, hits, evicted


def run_toy_check() -> dict[str, Any]:
    cache, hits, evicted = _toy_lru(4, [0, 1, 2, 3], [1, 3, 4, 5])
    assert hits == [1, 3]
    assert evicted == [0, 2]
    assert cache == [4, 5, 1, 3]
    cache, hits, evicted = _toy_lru(4, cache, [1, 3, 6, 7])
    assert hits == [1, 3]
    assert evicted == [4, 5]
    assert cache == [6, 7, 1, 3]
    complete_lru = list(range(4))
    assert sorted(complete_lru) == list(range(4))
    try:
        invalid_lru = [0, 1, 2, -1]
        if sorted(invalid_lru) != list(range(4)):
            raise ValueError("LRU must be a complete physical-slot permutation")
    except ValueError:
        pass
    else:
        raise AssertionError("CPU self-check failed to reject an LRU containing -1/missing slot")
    return {"status": "passed", "checks": ["hit", "miss", "evict", "MRU-order", "complete-LRU"]}


def _read_trace(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line_no, line in enumerate(f, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_no}: {exc}") from exc
            required = {
                "request_id",
                "rank",
                "layer_id",
                "position",
                "compressed_length",
                "block_indices",
            }
            missing = required.difference(row)
            if missing:
                raise ValueError(f"{path}:{line_no} missing {sorted(missing)}")
            if int(row["rank"]) != 0:
                continue
            blocks = [int(x) for x in row["block_indices"]]
            compressed_length = int(row["compressed_length"])
            if len(blocks) != TOP_K or len(set(blocks)) != TOP_K:
                raise ValueError(
                    f"{path}:{line_no} expected {TOP_K} unique C4 IDs, got {len(blocks)}"
                )
            if compressed_length <= 0 or min(blocks) < 0 or max(blocks) >= compressed_length:
                raise ValueError(f"{path}:{line_no} has invalid C4 IDs for compressed_length={compressed_length}")
            rows.append(
                {
                    "request_id": str(row["request_id"]),
                    "rank": 0,
                    "layer_id": int(row["layer_id"]),
                    "position": int(row["position"]),
                    "compressed_length": compressed_length,
                    "block_indices": blocks,
                    "forward_mode": row.get("forward_mode", "decode"),
                    "cuda_graph": bool(row.get("cuda_graph", True)),
                }
            )
    if not rows:
        raise ValueError(f"trace has no rank-0 rows: {path}")
    layers = tuple(sorted({row["layer_id"] for row in rows}))
    if layers != LAYER_IDS:
        raise ValueError(f"trace layers {layers} do not match expected {LAYER_IDS}")
    positions = sorted({row["position"] for row in rows})
    for position in positions:
        rows_at_position = [row for row in rows if row["position"] == position]
        got = {row["layer_id"] for row in rows_at_position}
        if got != set(LAYER_IDS):
            raise ValueError(f"position {position} is missing layers: {sorted(set(LAYER_IDS)-got)}")
        lengths = {row["compressed_length"] for row in rows_at_position}
        if len(lengths) != 1:
            raise ValueError(
                f"position {position} has layer-dependent compressed_length values: {sorted(lengths)}"
            )
    return rows


@dataclass(frozen=True)
class Trace:
    rows: tuple[dict[str, Any], ...]
    positions: tuple[int, ...]
    layer_rows: dict[int, tuple[dict[str, Any], ...]]
    raw_span: int
    logical_blocks: int
    compressed_length: int
    synthetic_span: bool
    source_path: str


def _make_trace(rows: list[dict[str, Any]], span: int | None) -> Trace:
    source_last = max(row["position"] for row in rows)
    source_blocks = max(row["compressed_length"] for row in rows)
    target_span = source_last if span is None else int(span)
    if target_span < source_last:
        raise ValueError(f"--span {target_span} is below trace position {source_last}")
    target_blocks = (target_span + C4 - 1) // C4
    offset = target_blocks - source_blocks
    synthetic = target_span != source_last
    transformed: list[dict[str, Any]] = []
    for row in rows:
        new_row = dict(row)
        if synthetic:
            new_row["position"] = target_span - (source_last - row["position"])
            new_row["block_indices"] = [x + offset for x in row["block_indices"]]
            new_row["compressed_length"] = row["compressed_length"] + offset
        transformed.append(new_row)
    positions = tuple(sorted({row["position"] for row in transformed}))
    layer_rows = {
        layer: tuple(row for row in transformed if row["layer_id"] == layer)
        for layer in LAYER_IDS
    }
    return Trace(
        rows=tuple(transformed),
        positions=positions,
        layer_rows=layer_rows,
        raw_span=target_span,
        logical_blocks=target_blocks,
        compressed_length=target_blocks,
        synthetic_span=synthetic,
        source_path="",
    )


def _write_trace(path: Path, trace: Trace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in trace.rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")


def _plan(args: argparse.Namespace) -> dict[str, Any]:
    interpreter = sys.executable
    return {
        "schema_version": "qwen38-hisparse-gpu-microbench-v1",
        "status": "plan_only_no_gpu",
        "upstream": {
            "python": str(Path(args.sglang_root) / "python/sglang/kernels/ops/kvcache/hisparse.py"),
            "cuda": str(Path(args.sglang_root) / "python/sglang/kernels/jit/csrc/kvcacheio/hisparse.cuh"),
            "resolve": "load_cache_to_device_buffer_mla",
            "no_io": "skip_io=True compile-time kernel flag; resolve/LRU still runs",
        },
        "trace_contract": {
            "source": "rank0 JSONL positions 38185..38951 with raw compressed_length per row",
            "synthetic_wide": "--spans 261144 relocates the same selection pattern and preserves compressed_length deltas; not a real 261K trace",
        },
        "contract": {
            "layers_serial": len(LAYER_IDS),
            "layer_ids": list(LAYER_IDS),
            "hot_c4_blocks_per_layer_request": HOT_BLOCKS,
            "topk_c4_blocks": TOP_K,
            "raw_native_kv_shape": ["token", 1, HEAD_DIM],
            "raw_fp8_k_bytes_per_c4": C4 * HEAD_DIM,
            "raw_fp8_v_bytes_per_c4": C4 * HEAD_DIM,
            "raw_fp8_kv_bytes_per_c4_per_layer_per_gpu": RAW_KV_BYTES,
            "prototype_host_layout": "one [K4(1024B), V4(1024B)] packed MLA record; prototype design",
            "prototype_packed_record_bytes": ITEM_BYTES,
            "prototype_scale_storage": "separate non-unit layer-level K/V float32 constants; not in transfer record",
        },
        "arms": {
            "A": "full resident GPU packed K/V read + identical final gather",
            "B": "upstream GPU resolve/LRU + skip_io=True + identical final gather; bytes invalid by design",
            "C": "upstream GPU resolve/LRU + real pinned NUMA host K/V fetch + identical final gather",
        },
        "measurement": {
            "cuda_events": "one elapsed event pair per trace step around 12-layer graph replay",
            "metadata_copy": "top-k and seq-len device updates occur before event and are logged as excluded",
            "repeats": args.repeats,
            "first_steps_separate": args.warmup_steps,
            "dual_gpu": "one spawned process per GPU, barrier-synchronized",
            "numa_nodes": args.numa_nodes,
            "cold_cache_bytes": args.cold_bytes,
            "cold_cache_repeats": args.cold_repeats,
            "cold_cache_eviction": "same-device uint8 read/write before each event; synchronized and excluded",
            "gpu_miss_accounting": "RecordMissPlan output retained per layer/step; no miss plan is supplied to the kernel",
            "memory_accounting": "worker records CUDA allocated/reserved/peak after states and graphs and before cleanup, plus per-arm tensor nbytes",
            "physical_record_buffer": f"{HOT_BLOCKS}+{PAGE_SIZE} slots/layer; hot={HOT_BLOCKS * ITEM_BYTES * len(LAYER_IDS)}B all layers, tail={PAGE_SIZE * ITEM_BYTES * len(LAYER_IDS)}B all layers",
            "worker_timeout_s": args.worker_timeout_s,
            "barrier_timeout_s": BARRIER_TIMEOUT_S,
        },
        "source_commit": _source_revision(Path(args.sglang_root)),
        "dependencies": [
            "Python 3.10+",
            "PyTorch with CUDA",
            "the stated sglang checkout on PYTHONPATH",
            "CUDA-capable GPUs; optional libnuma.so.1 for in-process first-touch binding",
            "nvidia-smi for service-occupancy guard",
        ],
        "commands": {
            "toy": f"{interpreter} {Path(__file__).resolve()} toy-check",
            "plan": f"{interpreter} {Path(__file__).resolve()} plan",
            "run_source_after_gpu_window": (
                f"{interpreter} {Path(__file__).resolve()} run --allow-gpu --trace {args.trace} "
                f"--spans source --gpus 0,1 --out {args.out}"
            ),
            "run_261144_after_gpu_window": (
                f"{interpreter} {Path(__file__).resolve()} run --allow-gpu --trace {args.trace} "
                f"--spans 261144 --gpus 0,1 --out {args.out}"
            ),
        },
        "replay_policy": (
            "GPU resolve/LRU is authoritative for C. CPU never supplies a miss/victim plan. "
            "The toy CPU policy only checks hit/miss/evict shape; any future diagnostic "
            "miss-count/order mismatch with upstream parallel tie order is not a kernel bug by itself."
        ),
        "limitations": [
            "This is a synthetic FP8 byte payload and not model KV/output qualification.",
            "No real 261K QSA rank-0 trace is present; 261144 uses selection-pattern relocation and is marked synthetic.",
            "The packed [K4,V4] host record is a standalone prototype layout; native K/V-to-packed-to-gather byte equality is checked.",
            "The newest compressed block is staged into the extra page slot outside the timed region; generated-KV writeback/eviction is not a full lifecycle test.",
            "The cold-cache pass only touches a same-device eviction buffer before each event and does not simulate full model-weight traffic.",
            "GPU timings are unrun until the main agent opens the GPU window.",
        ],
    }


def _parse_span_list(spec: str) -> list[int | None]:
    out: list[int | None] = []
    for part in spec.split(","):
        part = part.strip().lower()
        if part in {"source", "trace", "native"}:
            out.append(None)
        else:
            value = int(part)
            if value <= 0:
                raise ValueError(f"invalid span {value}")
            out.append(value)
    if not out:
        raise ValueError("empty --spans")
    return out


def _numa_cpus(node: int) -> set[int]:
    path = Path(f"/sys/devices/system/node/node{node}/cpulist")
    if not path.exists():
        return set()
    cpus: set[int] = set()
    for part in path.read_text().strip().split(","):
        if not part:
            continue
        if "-" in part:
            first, last = (int(x) for x in part.split("-", 1))
            cpus.update(range(first, last + 1))
        else:
            cpus.add(int(part))
    return cpus


def _bind_numa(node: int) -> dict[str, Any]:
    result: dict[str, Any] = {"requested": node, "cpus": [], "libnuma": False}
    cpus = _numa_cpus(node)
    if cpus:
        os.sched_setaffinity(0, cpus)
        result["cpus"] = sorted(cpus)
    for libname in ("libnuma.so.1", "libnuma.so"):
        try:
            lib = ctypes.CDLL(libname)
            lib.numa_run_on_node(ctypes.c_int(node))
            lib.numa_set_preferred(ctypes.c_int(node))
            result["libnuma"] = True
            break
        except OSError:
            continue
    return result


def _observe_host_numa(tensors: Sequence[Any]) -> dict[str, Any]:
    """Record the numa_maps lines covering the pinned host-record pointers."""
    result: dict[str, Any] = {"status": "unavailable", "records": []}
    maps_path = Path("/proc/self/maps")
    numa_path = Path("/proc/self/numa_maps")
    if not maps_path.exists() or not numa_path.exists():
        result["reason"] = "procfs numa maps unavailable"
        return result
    try:
        vmas: list[tuple[int, int, int]] = []
        for line in maps_path.read_text().splitlines():
            address = line.split(None, 1)[0]
            first, last = (int(value, 16) for value in address.split("-", 1))
            vmas.append((first, last, first))
        numa_lines: dict[int, str] = {}
        for line in numa_path.read_text().splitlines():
            first = line.split(None, 1)[0]
            numa_lines[int(first, 16)] = line
        observations = []
        for tensor in tensors:
            pointer = int(tensor.data_ptr())
            vma_start = next((first for first, last, _ in vmas if first <= pointer < last), None)
            line = numa_lines.get(vma_start) if vma_start is not None else None
            nodes = {
                int(part[1:].split("=", 1)[0]): int(part.split("=", 1)[1])
                for part in (line or "").split()
                if part.startswith("N") and "=" in part
            }
            observations.append(
                {
                    "data_ptr": hex(pointer),
                    "vma_start": hex(vma_start) if vma_start is not None else None,
                    "numa_maps_line": line,
                    "node_pages": nodes,
                }
            )
        result["status"] = "observed" if all(item["numa_maps_line"] for item in observations) else "partial"
        result["records"] = observations
        return result
    except (OSError, ValueError) as exc:
        result["reason"] = repr(exc)
        return result


def _barrier_wait(barrier: Any) -> None:
    try:
        barrier.wait(timeout=BARRIER_TIMEOUT_S)
    except (threading.BrokenBarrierError, TimeoutError) as exc:
        raise RuntimeError(f"dual-rank barrier failed or timed out after {BARRIER_TIMEOUT_S}s") from exc


def _cuda_memory_snapshot(torch: Any, device: Any) -> dict[str, int]:
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def _state_memory_bytes(state: BenchState) -> dict[str, int]:
    nbytes = lambda tensors: sum(int(t.numel() * t.element_size()) for t in tensors)
    device_records_bytes = nbytes(state.device_records)
    metadata_bytes = sum(
        nbytes(tensors)
        for tensors in (
            state.device_tokens,
            [state.device_locs],
            [state.host_locs],
            state.lru_slots,
            state.out_locs,
            state.gather_indices,
            state.out_records,
            state.scale_k,
            state.scale_v,
            state.out_scale_k,
            state.out_scale_v,
            state.topk,
            state.topk_ids64,
            state.miss_src,
            state.miss_dst,
            state.miss_count,
            [state.seq_len, state.req_pool, state.real_reqs],
        )
    )
    resident_bytes = 0 if state.resident_records is None else nbytes(state.resident_records)
    return {
        "device_records_bytes": device_records_bytes,
        "metadata_scales_gather_workspace_bytes": metadata_bytes,
        "resident_records_bytes": resident_bytes,
        "state_tensor_bytes_total": device_records_bytes + metadata_bytes + resident_bytes,
    }


def _gpu_processes() -> list[dict[str, str]]:
    if shutil.which("nvidia-smi") is None:
        raise RuntimeError("nvidia-smi is required for the GPU occupancy guard")
    cmd = [
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,gpu_uuid",
        "--format=csv,noheader,nounits",
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"nvidia-smi occupancy query failed: {proc.stderr.strip()}")
    rows = []
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if parts and parts[0]:
            rows.append({"pid": parts[0], "process_name": parts[1] if len(parts) > 1 else "", "gpu_uuid": parts[2] if len(parts) > 2 else ""})
    return rows


def _source_revision(path: Path) -> str | None:
    proc = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    revision = proc.stdout.strip()
    return revision or None


def _guard_gpu(allow_gpu: bool) -> list[dict[str, str]]:
    if not allow_gpu:
        raise RuntimeError("GPU execution is gated; rerun only during the main agent's window with --allow-gpu")
    processes = _gpu_processes()
    if processes:
        raise RuntimeError(f"GPU compute processes present; refusing to touch CUDA: {processes}")
    return processes


def _import_runtime(sglang_root: Path):
    root = str(sglang_root / "python")
    if root not in sys.path:
        sys.path.insert(0, root)
    import torch  # type: ignore
    from sglang.kernels.ops.kvcache.hisparse import (  # type: ignore
        load_cache_to_device_buffer_mla,
    )

    return torch, load_cache_to_device_buffer_mla


def _make_payloads(torch: Any, trace: Trace, device: Any, node: int, rank: int) -> dict[str, Any]:
    """Build native K/V, then one pinned ``[K4,V4]`` record per C4 block."""
    numa = _bind_numa(node)
    try:
        torch.set_num_threads(1)
        numa["torch_num_threads"] = int(torch.get_num_threads())
    except (AttributeError, RuntimeError):
        numa["torch_num_threads"] = None
    blocks = trace.logical_blocks
    if blocks >= 1 << 20:
        raise ValueError(f"synthetic payload block-ID encoding supports <2^20 blocks, got {blocks}")
    tokens = blocks * C4
    host_records: list[Any] = []
    host_scale_k: list[Any] = []
    host_scale_v: list[Any] = []
    native_checks: list[dict[str, Any]] = []
    token_id = torch.arange(tokens, dtype=torch.int64)
    block_id = token_id // C4
    token_in_block = token_id.remainder(C4)
    lane = torch.arange(HEAD_DIM, dtype=torch.int64).view(1, 1, HEAD_DIM)
    # E4M3FN reserves exponent=0xF.  Keep all synthetic bytes below 0x70 so
    # every payload byte is a finite FP8 encoding, then encode the complete
    # logical C4 ID in four base-32 lanes to make wrong-row fetches obvious.
    block_digits = torch.stack([(torch.arange(blocks, dtype=torch.int64) >> (5 * i)) & 31 for i in range(4)], dim=1)
    for layer_pos, layer_id in enumerate(LAYER_IDS):
        base_k = ((token_id.view(tokens, 1, 1) * 17 + lane + layer_id * 3) % 112).to(torch.uint8)
        base_v = ((token_id.view(tokens, 1, 1) * 29 + lane * 3 + layer_id * 5 + 37) % 112).to(torch.uint8)
        native_k = base_k.clone()
        native_v = base_v.clone()
        native_k[:, 0, :4] = block_digits[block_id]
        native_v[:, 0, :4] = block_digits[block_id] + 32
        native_k[:, 0, 4] = token_in_block + 64
        native_v[:, 0, 4] = token_in_block + 68
        layer_digits = torch.tensor(
            [layer_id & 31, (layer_id >> 5) & 31], dtype=torch.uint8
        )
        native_k[:, 0, 5] = layer_digits[0]
        native_k[:, 0, 6] = layer_digits[1]
        native_k[:, 0, 7] = rank
        native_v[:, 0, 5] = layer_digits[0] + 32
        native_v[:, 0, 6] = layer_digits[1] + 32
        native_v[:, 0, 7] = rank + 64
        packed = torch.empty((blocks, ITEM_BYTES), dtype=torch.uint8, pin_memory=True)
        packed[:, :RAW_K_OR_V_BYTES].copy_(native_k.reshape(blocks, RAW_K_OR_V_BYTES))
        packed[:, RAW_K_OR_V_BYTES:].copy_(native_v.reshape(blocks, RAW_K_OR_V_BYTES))
        assert torch.equal(packed[:, :RAW_K_OR_V_BYTES], native_k.reshape(blocks, RAW_K_OR_V_BYTES))
        assert torch.equal(packed[:, RAW_K_OR_V_BYTES:], native_v.reshape(blocks, RAW_K_OR_V_BYTES))
        if int(packed.max().item()) >= 0x78:
            raise AssertionError("synthetic payload contains an E4M3FN reserved exponent encoding")
        k_scale = torch.tensor([0.125 + 0.03125 * (layer_pos + 1)], dtype=torch.float32)
        v_scale = torch.tensor([0.375 + 0.046875 * (layer_pos + 1)], dtype=torch.float32)
        host_records.append(packed)
        host_scale_k.append(k_scale)
        host_scale_v.append(v_scale)
        native_checks.append(
            {
                "layer_id": layer_id,
                "native_k_shape": [tokens, 1, HEAD_DIM],
                "native_v_shape": [tokens, 1, HEAD_DIM],
                "payload_identity_fields": {
                    "k_lanes_0_3": "logical C4 block ID in base-32 digits",
                    "v_lanes_0_3": "logical C4 block ID in base-32 digits plus 32",
                    "lanes_4": "token offset within C4 plus K/V tag",
                    "lanes_5_6": "layer ID in base-32 digits",
                    "lane_7": "rank tag",
                },
                "rank": rank,
                "k_scale_min": float(k_scale.min().item()),
                "v_scale_min": float(v_scale.min().item()),
                "scale_non_unit": True,
                "finite_fp8_bytes": True,
                "native_to_hostpacked": "passed",
                "packed_record_bytes": ITEM_BYTES,
            }
        )
        del base_k, base_v, native_k, native_v, k_scale, v_scale, layer_digits, packed
    del token_id, block_id, token_in_block, lane, block_digits
    numa["host_records_observed"] = _observe_host_numa(host_records)
    return {
        "host_records": host_records,
        "host_scale_k": host_scale_k,
        "host_scale_v": host_scale_v,
        "native_checks": native_checks,
        "numa": numa,
    }


def _row_map(trace: Trace) -> dict[tuple[int, int], list[int]]:
    return {(row["position"], row["layer_id"]): row["block_indices"] for row in trace.rows}


@dataclass
class BenchState:
    host_records: list[Any]
    host_scale_k: list[Any]
    host_scale_v: list[Any]
    device_records: list[Any]
    device_tokens: list[Any]
    device_locs: Any
    host_locs: Any
    lru_slots: list[Any]
    out_locs: list[Any]
    gather_indices: list[Any]
    out_records: list[Any]
    scale_k: list[Any]
    scale_v: list[Any]
    out_scale_k: list[Any]
    out_scale_v: list[Any]
    seq_len: Any
    req_pool: Any
    real_reqs: Any
    topk: list[Any]
    topk_ids64: list[Any]
    miss_src: list[Any]
    miss_dst: list[Any]
    miss_count: list[Any]
    resident_records: list[Any] | None = None


def _build_state(torch: Any, payload: dict[str, Any], trace: Trace, device: Any, resident: bool) -> BenchState:
    blocks = trace.logical_blocks
    physical = HOT_BLOCKS + PAGE_SIZE
    host_records = payload["host_records"]
    host_scale_k = payload["host_scale_k"]
    host_scale_v = payload["host_scale_v"]
    host_locs = torch.arange(blocks, dtype=torch.int64, device=device).view(1, -1)
    device_locs = torch.arange(physical, dtype=torch.int32, device=device).view(1, -1)
    req_pool = torch.zeros((1,), dtype=torch.int64, device=device)
    real_reqs = torch.ones((1,), dtype=torch.int32, device=device)
    seq_len = torch.tensor([blocks], dtype=torch.int32, device=device)
    device_records: list[Any] = []
    device_tokens: list[Any] = []
    lru_slots: list[Any] = []
    out_locs: list[Any] = []
    gather_indices: list[Any] = []
    out_records: list[Any] = []
    scale_k: list[Any] = []
    scale_v: list[Any] = []
    out_scale_k: list[Any] = []
    out_scale_v: list[Any] = []
    topk: list[Any] = []
    topk_ids64: list[Any] = []
    miss_src: list[Any] = []
    miss_dst: list[Any] = []
    miss_count: list[Any] = []
    for layer in range(len(LAYER_IDS)):
        record = torch.zeros((physical, ITEM_BYTES), dtype=torch.uint8, device=device)
        latest = blocks - 1
        toks = torch.full((1, physical), -1, dtype=torch.int32, device=device)
        toks[0, HOT_BLOCKS] = latest
        lru = torch.arange(HOT_BLOCKS, dtype=torch.int16, device=device).view(1, -1)
        out = torch.full((1, TOP_K), -1, dtype=torch.int32, device=device)
        idx64 = torch.empty((TOP_K,), dtype=torch.int64, device=device)
        idx = torch.empty((1, TOP_K), dtype=torch.int32, device=device)
        gidx = torch.empty((TOP_K,), dtype=torch.int64, device=device)
        out_record = torch.empty((TOP_K, ITEM_BYTES), dtype=torch.uint8, device=device)
        miss_src_layer = torch.empty((1, TOP_K), dtype=torch.int64, device=device)
        miss_dst_layer = torch.empty((1, TOP_K), dtype=torch.int32, device=device)
        miss_count_layer = torch.zeros((1,), dtype=torch.int32, device=device)
        layer_scale_k = host_scale_k[layer].to(device=device, non_blocking=True)
        layer_scale_v = host_scale_v[layer].to(device=device, non_blocking=True)
        out_k_scale = torch.empty((1,), dtype=torch.float32, device=device)
        out_v_scale = torch.empty((1,), dtype=torch.float32, device=device)
        device_records.append(record)
        device_tokens.append(toks)
        lru_slots.append(lru)
        out_locs.append(out)
        gather_indices.append(gidx)
        out_records.append(out_record)
        scale_k.append(layer_scale_k)
        scale_v.append(layer_scale_v)
        out_scale_k.append(out_k_scale)
        out_scale_v.append(out_v_scale)
        topk.append(idx)
        topk_ids64.append(idx64)
        miss_src.append(miss_src_layer)
        miss_dst.append(miss_dst_layer)
        miss_count.append(miss_count_layer)
    resident_records = None
    if resident:
        resident_records = [host_records[i].to(device=device, non_blocking=True) for i in range(len(LAYER_IDS))]
    torch.cuda.synchronize(device)
    return BenchState(
        host_records=host_records,
        host_scale_k=host_scale_k,
        host_scale_v=host_scale_v,
        device_records=device_records,
        device_tokens=device_tokens,
        device_locs=device_locs,
        host_locs=host_locs,
        lru_slots=lru_slots,
        out_locs=out_locs,
        gather_indices=gather_indices,
        out_records=out_records,
        scale_k=scale_k,
        scale_v=scale_v,
        out_scale_k=out_scale_k,
        out_scale_v=out_scale_v,
        seq_len=seq_len,
        req_pool=req_pool,
        real_reqs=real_reqs,
        topk=topk,
        topk_ids64=topk_ids64,
        miss_src=miss_src,
        miss_dst=miss_dst,
        miss_count=miss_count,
        resident_records=resident_records,
    )


def _reset_state(torch: Any, state: BenchState, trace: Trace, device: Any) -> None:
    slot_ids = torch.arange(HOT_BLOCKS, dtype=torch.int16, device=device)
    for layer in range(len(LAYER_IDS)):
        first = trace.layer_rows[LAYER_IDS[layer]][0]
        initial_ids = list(first["block_indices"])
        latest = int(first["compressed_length"]) - 1
        if latest in initial_ids:
            initial_ids.remove(latest)
        initial_ids = initial_ids[:HOT_BLOCKS]
        state.device_tokens[layer].fill_(-1)
        state.lru_slots[layer][0].copy_(slot_ids)
        state.miss_count[layer].zero_()
        if initial_ids:
            # The source rows live on pinned CPU memory.  Keep the host
            # index on CPU; only the logical IDs copied into GPU metadata need
            # a device tensor.
            ids_cpu = torch.tensor(initial_ids, dtype=torch.long)
            ids_device = ids_cpu.to(device=device)
            n = len(initial_ids)
            slots = torch.arange(n, dtype=torch.int32, device=device)
            state.device_tokens[layer][0, :n] = ids_device.to(torch.int32)
            state.lru_slots[layer][0, :n] = slots.to(torch.int16)
            state.device_records[layer][:n].copy_(
                state.host_records[layer].index_select(0, ids_cpu), non_blocking=True
            )
        state.device_tokens[layer][0, HOT_BLOCKS] = latest
        state.device_records[layer][HOT_BLOCKS].copy_(state.host_records[layer][latest], non_blocking=True)
        state.out_locs[layer].fill_(-1)
    state.seq_len.fill_(trace.layer_rows[LAYER_IDS[0]][0]["compressed_length"])
    torch.cuda.synchronize(device)


def _set_step_inputs(torch: Any, state: BenchState, trace: Trace, position: int, device: Any) -> None:
    rows = {layer: trace.layer_rows[layer] for layer in LAYER_IDS}
    # Positions and layers are fixed after parsing, so the lookup is small and
    # explicit.  Copies are outside the CUDA-event interval by contract.
    for layer_pos, layer_id in enumerate(LAYER_IDS):
        row = next(row for row in rows[layer_id] if row["position"] == position)
        ids = torch.tensor(row["block_indices"], dtype=torch.int32, device=device)
        state.topk[layer_pos].copy_(ids.view(1, -1))
        state.topk_ids64[layer_pos].copy_(ids.to(torch.int64))
        # The newest C4 block is produced locally in a real decode.  Keep the
        # extra page slot current outside the measured event; writeback and
        # eviction of generated KV remain outside this standalone benchmark.
        latest = int(row["compressed_length"]) - 1
        state.device_tokens[layer_pos][0, HOT_BLOCKS] = latest
        state.device_records[layer_pos][HOT_BLOCKS].copy_(state.host_records[layer_pos][latest], non_blocking=True)
    state.seq_len.fill_(int(next(row for row in rows[LAYER_IDS[0]] if row["position"] == position)["compressed_length"]))


def _run_graph_step(torch: Any, load_cache: Any, graph: Any, mode: str, state: BenchState, device: Any) -> None:
    graph.replay()


def _capture_graph(torch: Any, load_cache: Any, mode: str, state: BenchState, trace: Trace, device: Any) -> Any:
    # One uncaptured call forces JIT compilation and validates graph inputs.
    first = trace.positions[0]
    _set_step_inputs(torch, state, trace, first, device)
    if mode == "A":
        _run_a(torch, state)
    else:
        _run_resolve_gather(torch, load_cache, state, mode)
    torch.cuda.synchronize(device)
    _reset_state(torch, state, trace, device)
    graph = torch.cuda.CUDAGraph()
    _set_step_inputs(torch, state, trace, first, device)
    with torch.cuda.graph(graph):
        if mode == "A":
            _run_a(torch, state)
        else:
            _run_resolve_gather(torch, load_cache, state, mode)
    torch.cuda.synchronize(device)
    _reset_state(torch, state, trace, device)
    return graph


def _run_a(torch: Any, state: BenchState) -> None:
    assert state.resident_records is not None
    for layer in range(len(LAYER_IDS)):
        state.gather_indices[layer].copy_(state.topk_ids64[layer])
        torch.index_select(
            state.resident_records[layer],
            0,
            state.gather_indices[layer],
            out=state.out_records[layer],
        )
        state.out_scale_k[layer].copy_(state.scale_k[layer])
        state.out_scale_v[layer].copy_(state.scale_v[layer])


def _run_resolve_gather(torch: Any, load_cache: Any, state: BenchState, mode: str) -> None:
    for layer in range(len(LAYER_IDS)):
        load_cache(
            top_k_tokens=state.topk[layer],
            device_buffer_tokens=state.device_tokens[layer],
            host_cache_locs=state.host_locs,
            device_buffer_locs=state.device_locs,
            host_cache=state.host_records[layer],
            device_buffer=state.device_records[layer],
            top_k_device_locs=state.out_locs[layer],
            req_pool_indices=state.req_pool,
            seq_lens=state.seq_len,
            lru_slots=state.lru_slots[layer],
            item_size_bytes=ITEM_BYTES,
            num_top_k=TOP_K,
            hot_buffer_size=HOT_BLOCKS,
            page_size=PAGE_SIZE,
            block_size=BLOCK_SIZE,
            num_real_reqs=state.real_reqs,
            miss_src=state.miss_src[layer],
            miss_dst=state.miss_dst[layer],
            miss_count=state.miss_count[layer],
            skip_io=(mode == "B"),
        )
        state.gather_indices[layer].copy_(state.out_locs[layer].view(-1))
        # A/B/C all perform one identical final gather of the packed 2048-byte
        # [K4,V4] record.  B's bytes are intentionally invalid because skip_io
        # elides only the resolve transfer, while the resolve/LRU still ran.
        torch.index_select(
            state.device_records[layer],
            0,
            state.gather_indices[layer],
            out=state.out_records[layer],
        )
        state.out_scale_k[layer].copy_(state.scale_k[layer])
        state.out_scale_v[layer].copy_(state.scale_v[layer])


def _expected_rows(torch: Any, state: BenchState, trace: Trace, position: int, layer_pos: int) -> tuple[Any, Any, Any, Any]:
    row = next(row for row in trace.layer_rows[LAYER_IDS[layer_pos]] if row["position"] == position)
    ids = torch.tensor(row["block_indices"], dtype=torch.long)
    expected_records = state.host_records[layer_pos].index_select(0, ids)
    return (
        expected_records,
        state.host_scale_k[layer_pos],
        state.host_scale_v[layer_pos],
        row,
    )


def _validate_resolve_slots(
    torch: Any,
    state: BenchState,
    row: dict[str, Any],
    layer_pos: int,
) -> dict[str, Any]:
    """Check GPU-produced locs/tokens without prescribing GPU eviction order."""
    expected_ids = torch.tensor(row["block_indices"], dtype=torch.int32)
    locs_device = state.out_locs[layer_pos].view(-1).to(dtype=torch.long)
    locs = locs_device.detach().cpu()
    loc_values = [int(value) for value in locs.tolist()]
    if len(loc_values) != TOP_K or any(value < 0 or value >= HOT_BLOCKS + PAGE_SIZE for value in loc_values):
        raise AssertionError(f"GPU resolve returned invalid physical locs: {loc_values[:8]}")
    if len(set(loc_values)) != TOP_K:
        raise AssertionError("GPU resolve returned duplicate physical locs for unique selected IDs")
    selected_tokens = state.device_tokens[layer_pos].view(-1).index_select(0, locs_device).detach().cpu()
    if not torch.equal(selected_tokens, expected_ids):
        raise AssertionError(
            f"GPU selected-slot token mismatch at layer={LAYER_IDS[layer_pos]} "
            f"position={row['position']}"
        )

    lru_values = [int(value) for value in state.lru_slots[layer_pos].view(-1).detach().cpu().tolist()]
    if len(lru_values) != HOT_BLOCKS or sorted(lru_values) != list(range(HOT_BLOCKS)):
        raise AssertionError("GPU LRU must be a complete permutation of physical hot slots")
    live_slot_tensor = torch.tensor(
        lru_values, dtype=torch.long, device=state.device_tokens[layer_pos].device
    )
    live_tokens = state.device_tokens[layer_pos].view(-1).index_select(0, live_slot_tensor).detach().cpu()
    live_values = [int(value) for value in live_tokens.tolist() if int(value) >= 0]
    compressed_length = int(row["compressed_length"])
    if any(value >= compressed_length for value in live_values):
        raise AssertionError("GPU LRU contains a token beyond compressed_length")
    if len(set(live_values)) != len(live_values):
        raise AssertionError("GPU LRU contains duplicate live logical block IDs")
    newest = int(row["compressed_length"]) - 1
    newest_token = int(state.device_tokens[layer_pos][0, HOT_BLOCKS].item())
    if newest_token != newest:
        raise AssertionError(
            f"newest slot mismatch: got {newest_token}, expected compressed_length-1={newest}"
        )
    return {
        "selected_slot_values": "passed",
        "lru_slot_invariant": "passed",
        "newest_slot_token": newest,
        "gpu_miss_count": int(state.miss_count[layer_pos].item()),
        "gpu_miss_bytes": int(state.miss_count[layer_pos].item()) * ITEM_BYTES,
    }


def _validate_step(torch: Any, state: BenchState, trace: Trace, position: int, mode: str, device: Any) -> dict[str, Any]:
    result: dict[str, Any] = {"position": position, "mode": mode, "layers": len(LAYER_IDS), "valid": True}
    for layer_pos, layer_id in enumerate(LAYER_IDS):
        expected_records, expected_scale_k, expected_scale_v, row = _expected_rows(
            torch, state, trace, position, layer_pos
        )
        if mode != "A":
            result.setdefault("resolve_invariants", []).append(
                _validate_resolve_slots(torch, state, row, layer_pos)
            )
        if mode == "B":
            continue
        actual_records = state.out_records[layer_pos].detach().cpu()
        if not torch.equal(actual_records, expected_records):
            raise AssertionError(f"{mode} byte mismatch at position={position}, layer={layer_id}")
        actual_scale_k = state.out_scale_k[layer_pos].detach().cpu()
        actual_scale_v = state.out_scale_v[layer_pos].detach().cpu()
        if not torch.equal(actual_scale_k, expected_scale_k) or not torch.equal(
            actual_scale_v, expected_scale_v
        ):
            raise AssertionError(f"{mode} scale mismatch at position={position}, layer={layer_id}")
        result.setdefault("checks_detail", []).append(
            {
                "layer_id": layer_id,
                "compressed_length": int(row["compressed_length"]),
                "record_bytes": int(actual_records.numel() // TOP_K),
                "k_scale": float(actual_scale_k.item()),
                "v_scale": float(actual_scale_v.item()),
            }
        )
    if mode == "B":
        result["valid"] = False
        result["reason"] = "noIO timing control; resolve/LRU and slot metadata are checked, KV bytes are intentionally invalid"
    return result


def _evict_l2(torch: Any, cold_buffer: Any, device: Any) -> None:
    if cold_buffer is None:
        return
    # A same-device read/write over 128 MiB is deliberately outside the event
    # interval.  It is a sensitivity control, not a model-weight simulation.
    cold_buffer.add_(1)
    torch.cuda.synchronize(device)


def _run_mode(
    torch: Any,
    load_cache: Any,
    mode: str,
    state: BenchState,
    trace: Trace,
    device: Any,
    repeats: int,
    warmup_steps: int,
    check_steps: set[int],
    graph: Any | None = None,
    step_barrier: Any | None = None,
    cold_buffer: Any = None,
) -> dict[str, Any]:
    graph_memory = None
    if graph is None:
        graph = _capture_graph(torch, load_cache, mode, state, trace, device)
        graph_memory = _cuda_memory_snapshot(torch, device)
    # Do not start one rank's first event while the other rank is still
    # compiling/capturing.  The barrier is reused for every mode and repeat.
    if step_barrier is not None:
        _barrier_wait(step_barrier)
    repeat_results: list[dict[str, Any]] = []
    for repeat in range(repeats):
        _reset_state(torch, state, trace, device)
        if step_barrier is not None:
            _barrier_wait(step_barrier)
        samples: list[float] = []
        checks: list[dict[str, Any]] = []
        gpu_miss_counts: list[list[int]] = []
        gpu_miss_bytes: list[list[int]] = []
        for step, position in enumerate(trace.positions):
            _set_step_inputs(torch, state, trace, position, device)
            _evict_l2(torch, cold_buffer, device)
            if step_barrier is not None:
                _barrier_wait(step_barrier)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            if step_barrier is not None:
                _barrier_wait(step_barrier)
            elapsed_ms = float(start.elapsed_time(end))
            samples.append(elapsed_ms)
            counts = [
                0 if mode == "A" else int(miss_count.item())
                for miss_count in state.miss_count
            ]
            if any(count < 0 or count > TOP_K for count in counts):
                raise AssertionError(f"GPU miss_count outside [0,{TOP_K}] at step={step}: {counts}")
            gpu_miss_counts.append(counts)
            gpu_miss_bytes.append([count * ITEM_BYTES for count in counts])
            if step in check_steps and (mode != "B" or not checks):
                checks.append(_validate_step(torch, state, trace, position, mode, device))
        repeat_results.append(
            {
                "repeat": repeat,
                "all_steps": _percentiles(samples),
                "first_steps": _percentiles(samples[:warmup_steps]),
                "after_first_steps": _percentiles(samples[warmup_steps:]),
                "checks": checks,
                "gpu_miss_count_by_layer_step": gpu_miss_counts,
                "gpu_miss_bytes_by_layer_step": gpu_miss_bytes,
                "transfer_bytes_scope": "B=0 physical H2D due skip_io; C=GPU miss_count*2048; A=0 resolve misses",
                "samples_ms": samples,
            }
        )
    if step_barrier is not None:
        _barrier_wait(step_barrier)
    return {
        "mode": mode,
        "repeats": repeat_results,
        "graph_replay": True,
        "event_scope": "12-layer resolve/gather only",
        "memory_after_graph_creation": graph_memory,
    }


def _worker(
    rank: int,
    gpu: int,
    numa_node: int,
    args_dict: dict[str, Any],
    barrier: Any,
    queue: Any,
) -> None:
    try:
        args = argparse.Namespace(**args_dict)
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        torch, load_cache = _import_runtime(Path(args.sglang_root))
        torch.cuda.set_device(gpu)
        device = torch.device(f"cuda:{gpu}")
        torch.cuda.reset_peak_memory_stats(device)
        trace_rows = _read_trace(Path(args.trace))
        trace = _make_trace(trace_rows, args.span)
        trace = Trace(**{**trace.__dict__, "source_path": str(args.trace)})
        payload = _make_payloads(torch, trace, device, numa_node, rank)
        # A needs the full resident copy; B/C share the exact same packed input
        # dimensions and top-k cadence but own independent LRU/device state.
        states = {
            "A": _build_state(torch, payload, trace, device, resident=True),
            "B": _build_state(torch, payload, trace, device, resident=False),
            "C": _build_state(torch, payload, trace, device, resident=False),
        }
        torch.cuda.synchronize(device)
        state_memory = {mode: _state_memory_bytes(states[mode]) for mode in ("A", "B", "C")}
        memory_after_states = _cuda_memory_snapshot(torch, device)
        check_steps = {0, len(trace.positions) // 2, len(trace.positions) - 1}
        _barrier_wait(barrier)
        # Capture each arm before any timing starts so this snapshot is the
        # allocator state after all state tensors and CUDA graphs exist.
        graphs = {}
        graph_memory_by_arm = {}
        for mode in ("A", "B", "C"):
            graphs[mode] = _capture_graph(torch, load_cache, mode, states[mode], trace, device)
            graph_memory_by_arm[mode] = _cuda_memory_snapshot(torch, device)
        _barrier_wait(barrier)
        torch.cuda.synchronize(device)
        memory_after_graphs = _cuda_memory_snapshot(torch, device)
        modes = {}
        for mode in ("A", "B", "C"):
            modes[mode] = _run_mode(
                torch,
                load_cache,
                mode,
                states[mode],
                trace,
                device,
                args.repeats,
                args.warmup_steps,
                check_steps,
                graph=graphs[mode],
                step_barrier=barrier,
            )
        conditions: dict[str, Any] = {"hot": modes}
        if args.cold_bytes:
            cold_buffer = torch.empty((args.cold_bytes,), dtype=torch.uint8, device=device)
            cold_buffer.zero_()
            torch.cuda.synchronize(device)
            cold_modes = {}
            for mode in ("A", "B", "C"):
                cold_modes[mode] = _run_mode(
                    torch,
                    load_cache,
                    mode,
                    states[mode],
                    trace,
                    device,
                    args.cold_repeats,
                    args.warmup_steps,
                    check_steps,
                    graph=graphs[mode],
                    step_barrier=barrier,
                    cold_buffer=cold_buffer,
                )
            conditions["cold"] = cold_modes
        torch.cuda.synchronize(device)
        memory_end_before_cleanup = _cuda_memory_snapshot(torch, device)
        queue.put(
            {
                "rank": rank,
                "gpu": gpu,
                "numa": payload["numa"],
                "trace": {
                    "source": str(args.trace),
                    "raw_span": trace.raw_span,
                    "logical_blocks": trace.logical_blocks,
                    "first_compressed_length": trace.rows[0]["compressed_length"],
                    "last_compressed_length": trace.rows[-1]["compressed_length"],
                    "synthetic_span": trace.synthetic_span,
                    "positions": len(trace.positions),
                    "layers": list(LAYER_IDS),
                },
                "native_checks": payload["native_checks"],
                "memory": {
                    "states_created": memory_after_states,
                    "graphs_created": memory_after_graphs,
                    "graph_creation_by_arm": graph_memory_by_arm,
                    "end_before_cleanup": memory_end_before_cleanup,
                    "per_arm_tensor_bytes": state_memory,
                    "physical_slots_per_layer": HOT_BLOCKS + PAGE_SIZE,
                    "hot_slots_per_layer": HOT_BLOCKS,
                    "tail_slots_per_layer": PAGE_SIZE,
                    "packed_record_bytes": ITEM_BYTES,
                    "hot_device_records_bytes_all_layers": HOT_BLOCKS * ITEM_BYTES * len(LAYER_IDS),
                    "tail_device_records_bytes_all_layers": PAGE_SIZE * ITEM_BYTES * len(LAYER_IDS),
                    "cold_buffer_bytes": args.cold_bytes,
                    "interpretation": "allocated/reserved/peak are CUDA allocator values; tensor bytes are explicit nbytes",
                },
                "modes": modes,
                "conditions": conditions,
            }
        )
    except BaseException as exc:
        try:
            barrier.abort()
        except BaseException:
            pass
        queue.put(
            {
                "rank": rank,
                "gpu": gpu,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        try:
            if "torch" in locals():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except BaseException:
            pass


def _combine(results: list[dict[str, Any]], warmup_steps: int) -> dict[str, Any]:
    if any("error" in result for result in results):
        return {"status": "error", "ranks": results}
    paired_conditions: dict[str, Any] = {}
    for condition in results[0].get("conditions", {"hot": results[0]["modes"]}):
        mode_source = [r.get("conditions", {"hot": r["modes"]})[condition] for r in results]
        modes: dict[str, Any] = {}
        deltas: dict[str, list[dict[str, Any]]] = {"C_minus_A": [], "B_minus_A": [], "C_minus_B": []}
        for mode in ("A", "B", "C"):
            repeats = len(mode_source[0][mode]["repeats"])
            joined = []
            for repeat in range(repeats):
                rank_samples = [r[mode]["repeats"][repeat]["samples_ms"] for r in mode_source]
                paired = [max(values) for values in zip(*rank_samples)]
                skew = [abs(values[0] - values[1]) for values in zip(*rank_samples)]
                for window, lo, hi in (
                    ("all_steps", 0, len(paired)),
                    ("first_steps", 0, min(warmup_steps, len(paired))),
                    ("after_first_steps", min(warmup_steps, len(paired)), len(paired)),
                ):
                    joined.append(
                        {
                            "repeat": repeat,
                            "window": window,
                            "dual_rank_max_ms": _percentiles(paired[lo:hi]),
                            "rank_abs_skew_ms": _percentiles(skew[lo:hi]),
                        }
                    )
            modes[mode] = joined
        repeats = len(mode_source[0]["A"]["repeats"])
        for repeat in range(repeats):
            paired_samples = {
                mode: [
                    max(values)
                    for values in zip(*[r[mode]["repeats"][repeat]["samples_ms"] for r in mode_source])
                ]
                for mode in ("A", "B", "C")
            }
            for name, left, right in (("C_minus_A", "C", "A"), ("B_minus_A", "B", "A"), ("C_minus_B", "C", "B")):
                delta = [x - y for x, y in zip(paired_samples[left], paired_samples[right])]
                for window, lo, hi in (
                    ("all_steps", 0, len(delta)),
                    ("first_steps", 0, min(warmup_steps, len(delta))),
                    ("after_first_steps", min(warmup_steps, len(delta)), len(delta)),
                ):
                    deltas[name].append({"repeat": repeat, "window": window, "delta_ms": _percentiles(delta[lo:hi])})
        paired_conditions[condition] = {"paired_windows": modes, "paired_deltas": deltas}
    return {"status": "passed", "ranks": results, "paired_conditions": paired_conditions}


def run_gpu(args: argparse.Namespace) -> dict[str, Any]:
    occupancy = _guard_gpu(args.allow_gpu)
    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    nodes = [int(x) for x in args.numa_nodes.split(",") if x.strip()]
    if len(gpus) != 2 or len(nodes) != 2:
        raise ValueError("this protocol requires exactly two --gpus and two --numa-nodes")
    spans = _parse_span_list(args.spans)
    if len(spans) != 1:
        raise ValueError("run one span per invocation; repeat the command for source and 261144")
    rows = _read_trace(Path(args.trace))
    trace = _make_trace(rows, spans[0])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    normalized = out / ("trace_rank0.synthetic261k.jsonl" if trace.synthetic_span else "trace_rank0.jsonl")
    _write_trace(normalized, Trace(**{**trace.__dict__, "source_path": str(args.trace)}))
    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(2)
    queue = ctx.Queue()
    args_dict = vars(args).copy()
    args_dict["span"] = spans[0]
    children = [
        ctx.Process(target=_worker, args=(rank, gpu, node, args_dict, barrier, queue))
        for rank, (gpu, node) in enumerate(zip(gpus, nodes))
    ]
    for child in children:
        child.start()
    results: list[dict[str, Any]] = []
    result_deadline = time.monotonic() + args.worker_timeout_s
    try:
        for _ in children:
            try:
                remaining = max(0.1, result_deadline - time.monotonic())
                result = queue.get(timeout=remaining)
                results.append(result)
                if "error" in result:
                    break
            except queue_module.Empty:
                results.append(
                    {
                        "rank": -1,
                        "error": f"worker result timeout after {args.worker_timeout_s}s",
                        "traceback": "parent timed out waiting for the worker result",
                    }
                )
                break
    finally:
        # Do not leave a failed rank or a peer stuck behind its barrier.  This
        # is deliberately bounded so a failed E3 cannot hold the outer service
        # recovery window indefinitely.
        for child in children:
            child.join(timeout=30)
        for child in children:
            if child.is_alive():
                child.terminate()
        for child in children:
            child.join(timeout=30)
    received_ranks = {result.get("rank") for result in results}
    for rank, child in enumerate(children):
        if rank not in received_ranks:
            results.append(
                {
                    "rank": rank,
                    "gpu": gpus[rank],
                    "error": f"worker exited without a result (exitcode={child.exitcode})",
                    "traceback": "no result was placed on the multiprocessing queue",
                }
            )
    results.sort(key=lambda result: result.get("rank", -1))
    report = {
        "schema_version": "qwen38-hisparse-gpu-microbench-v1",
        "status": "passed" if all("error" not in result for result in results) else "error",
        "occupancy_before_spawn": occupancy,
        "source_commit": _source_revision(Path(args.sglang_root)),
        "contract": {
            "layers": list(LAYER_IDS),
            "hot_c4_blocks": HOT_BLOCKS,
            "topk": TOP_K,
            "raw_fp8_k_bytes_per_c4": RAW_K_OR_V_BYTES,
            "raw_fp8_v_bytes_per_c4": RAW_K_OR_V_BYTES,
            "raw_fp8_kv_bytes_per_c4": RAW_KV_BYTES,
            "prototype_packed_record_bytes": ITEM_BYTES,
            "prototype_packed_layout": "[K4(1024B), V4(1024B)] one MLA record",
            "scale_storage": "separate non-unit layer-level K/V float32 constants",
        },
        "trace": {
            "input": str(args.trace),
            "normalized": str(normalized),
            "raw_span": trace.raw_span,
            "logical_blocks": trace.logical_blocks,
            "synthetic_span": trace.synthetic_span,
            "positions": len(trace.positions),
            "selection_pattern": "rank0 trace block_indices; no CPU miss/victim plan supplied to C",
        },
        "controls": {
            "repeats": args.repeats,
            "first_steps_separate": args.warmup_steps,
            "event_scope": "per-token CUDA event around one 12-layer serial graph replay including final K/V gather",
            "metadata_updates_excluded": True,
            "metadata_updates": "top-k/seq-len and latest compressed_length-1 extra-slot staging are outside events",
            "initial_cache": "first step selection preloaded into hot slots except compressed_length-1; remaining slots empty",
            "rank_pairing": "same-step max across ranks before P50/P95/P99/max; rank skew retained",
            "gpu_miss_accounting": "upstream RecordMissPlan outputs are retained per layer/step; C physical bytes=miss_count*2048",
            "cpu_gpu_replay_policy": "CPU toy policy is diagnostic only; GPU group/tie eviction order and miss count are authoritative",
            "memory_accounting": "each rank records CUDA allocator states/graphs/end and per-arm tensor nbytes including physical tail slots",
            "B_correctness": "invalid by design (noIO only); resolve/LRU remains measured",
            "C_correctness": "full FP8 payload bytes and K/V scales independently checked at first/middle/last sampled steps",
            "cold_cache": {
                "bytes": args.cold_bytes,
                "repeats": args.cold_repeats,
                "eviction_excluded_from_events": True,
                "interpretation": "L2 sensitivity control only; not a model-weight simulation",
            },
            "worker_timeout_s": args.worker_timeout_s,
            "barrier_timeout_s": BARRIER_TIMEOUT_S,
        },
        "ranks": results,
    }
    if report["status"] == "passed":
        combined = _combine(results, args.warmup_steps)
        report["paired_conditions"] = combined["paired_conditions"]
    report_path = out / f"summary-span-{trace.raw_span}.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "toy-check", "run"))
    parser.add_argument("--sglang-root", default=str(DEFAULT_SGLANG_ROOT))
    parser.add_argument("--trace", default=str(DEFAULT_TRACE))
    parser.add_argument("--spans", default="source")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--numa-nodes", default="3,2")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-steps", type=int, default=128)
    parser.add_argument("--cold-bytes", type=int, default=128 * 1024 * 1024)
    parser.add_argument("--cold-repeats", type=int, default=1)
    parser.add_argument("--worker-timeout-s", type=int, default=600)
    parser.add_argument("--out", default=str(Path(__file__).resolve().parent / "gpu-runs"))
    parser.add_argument("--allow-gpu", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "toy-check":
        print(json.dumps(run_toy_check(), indent=2, sort_keys=True))
        return 0
    if args.command == "plan":
        print(json.dumps(_plan(args), indent=2, sort_keys=True))
        return 0
    if (
        args.repeats < 1
        or args.cold_repeats < 1
        or args.warmup_steps < 0
        or args.cold_bytes < 0
        or args.worker_timeout_s < 1
    ):
        raise ValueError(
            "--repeats/--cold-repeats/--worker-timeout-s must be >=1; "
            "--warmup-steps/--cold-bytes must be >=0"
        )
    report = run_gpu(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
