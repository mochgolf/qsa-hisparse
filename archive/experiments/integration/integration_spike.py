#!/usr/bin/env python3
"""Minimal QSA/HiSparse integration checks under the frozen CONTRACT.md."""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import multiprocessing as mp
import os
import queue as queue_module
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path(os.environ.get("QSA_HISPARSE_SOURCE", ROOT / "sources/sglang-hisparse"))
OLD = Path(os.environ.get("QSA_HISPARSE_TRACE_ROOT", "<TRACE_ROOT>"))
LAYERS = (3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47)
C4, TOPK, HEAD, ITEM, HOT, PAGE = 4, 512, 256, 2048, 2048, 64


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def git_head(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def percentiles(values):
    ordered = sorted(float(x) for x in values)
    pick = lambda f: ordered[min(len(ordered) - 1, math.ceil(f * len(ordered)) - 1)]
    return {"p50_ms": pick(.50), "p95_ms": pick(.95), "p99_ms": pick(.99), "max_ms": ordered[-1], "mean_ms": statistics.fmean(ordered), "n": len(ordered)}


def numa_cpus(node):
    cpus=set()
    for part in Path(f"/sys/devices/system/node/node{node}/cpulist").read_text().strip().split(','):
        if '-' in part:
            first,last=map(int,part.split('-',1)); cpus.update(range(first,last+1))
        elif part: cpus.add(int(part))
    return cpus


def observe_numa(tensor):
    pointer=int(tensor.data_ptr()); vma=None
    for line in Path('/proc/self/maps').read_text().splitlines():
        first,last=(int(x,16) for x in line.split(None,1)[0].split('-',1))
        if first <= pointer < last: vma=first; break
    line=next((x for x in Path('/proc/self/numa_maps').read_text().splitlines() if int(x.split(None,1)[0],16)==vma),None)
    nodes={int(x[1:].split('=',1)[0]):int(x.split('=',1)[1]) for x in (line or '').split() if x.startswith('N') and '=' in x}
    return {"data_ptr":hex(pointer),"numa_maps_line":line,"node_pages":nodes}


def read_trace(kind: str, rank: int):
    stem = "r1-A" if kind == "A" else "r1-B"
    path = OLD / stem / f"{stem}-tp_rank{rank}-pp_rank0.jsonl"
    rows = [json.loads(line) for line in path.open()]
    by = {(int(r["position"]), int(r["layer_id"])): r for r in rows}
    positions = sorted({int(r["position"]) for r in rows})
    assert len(positions) == 767 and set(int(r["layer_id"]) for r in rows) == set(LAYERS)
    return path, by, positions


def cpu_check():
    import torch
    raw_k = torch.arange(C4 * HEAD, dtype=torch.int64).remainder(113).to(torch.uint8).reshape(C4, 1, HEAD)
    raw_v = (raw_k.to(torch.int16) + 7).remainder(113).to(torch.uint8)
    packed = torch.cat((raw_k.flatten(), raw_v.flatten()))
    k2 = packed[: C4 * HEAD].reshape(C4, 1, HEAD)
    v2 = packed[C4 * HEAD :].reshape(C4, 1, HEAD)
    assert packed.numel() == ITEM and torch.equal(raw_k, k2) and torch.equal(raw_v, v2)
    cases = []
    for raw_len in (64, 65, 66, 67, 68):
        block = (raw_len - 1) // C4
        expanded = [block * C4 + i if block * C4 + i < raw_len else -1 for i in range(C4)]
        cases.append({"raw_length": raw_len, "remainder": raw_len % C4, "block": block, "expanded": expanded})
    assert cases[0]["block"] == 15 and cases[-1]["block"] == 16
    bad = list(range(HOT)); bad[-1] = 0
    rejected = len(set(bad)) != HOT
    assert rejected
    return {
        "status": "PASS", "source_commit": git_head(SOURCE),
        "layouts": {"k": ["raw_slots", 1, HEAD], "v": ["raw_slots", 1, HEAD], "dtype": "fp8_e4m3fn/1B", "strides_elements": [HEAD, HEAD, 1], "packed_c4_bytes": ITEM, "index_k": ["compressed_slots", 1, 128], "index_k_dtype": "bfloat16", "compressed_page": 16},
        "address_cases": cases, "mapping_mutation_rejected": rejected,
    }


def fp8_record(torch, block_ids, layer, rank, pin=False, device=None):
    """Independent deterministic finite E4M3 payload for selected logical C4 IDs."""
    ids = torch.as_tensor(block_ids, dtype=torch.int64).reshape(-1, 1, 1)
    tok = torch.arange(C4, dtype=torch.int64).reshape(1, C4, 1)
    lane = torch.arange(HEAD, dtype=torch.int64).reshape(1, 1, HEAD)
    # Multiples of 1/8 within [-1,1] are exactly representable in E4M3.
    kval = (((ids * 3 + tok * 5 + lane + layer + rank) % 17) - 8).float() / 8
    vval = (((ids * 7 + tok * 2 + lane * 3 + layer * 2 + rank) % 17) - 8).float() / 8
    raw = torch.cat((kval.reshape(-1, C4 * HEAD), vval.reshape(-1, C4 * HEAD)), 1).to(torch.float8_e4m3fn).view(torch.uint8)
    out = torch.empty(raw.shape, dtype=torch.uint8, pin_memory=pin, device=device)
    out.copy_(raw)
    return out, kval, vval


def attention(torch, qsa_attention, packed, block_ids, q, raw_len, k_scale, v_scale):
    n = packed.shape[0]
    k = packed[:, : C4 * HEAD].reshape(n * C4, 1, HEAD).view(torch.float8_e4m3fn)
    v = packed[:, C4 * HEAD :].reshape(n * C4, 1, HEAD).view(torch.float8_e4m3fn)
    slots = torch.arange(n * C4, device=packed.device, dtype=torch.int32).reshape(1, -1)
    logical = torch.as_tensor(block_ids, device=packed.device, dtype=torch.int64)[:, None] * C4 + torch.arange(C4, device=packed.device)
    slots = torch.where((logical < raw_len).reshape(1, -1), slots, -1)
    return qsa_attention(q, k, v, slots, HEAD ** -.5, k_scale, v_scale)


def v1(torch, qsa_attention, sparse_gqa, rank, device):
    checks, graph_samples = [], []
    q = (torch.arange(4 * HEAD, device=device).reshape(1, 4, HEAD).remainder(13).float() / 16 - .375).to(torch.bfloat16)
    for kind in ("A", "B"):
        path, rows, positions = read_trace(kind, rank)
        for pos in (positions[0], positions[len(positions)//2], positions[-1]):
            for layer in LAYERS:
                row = rows[(pos, layer)]
                ids = [int(x) for x in row["block_indices"]]
                host, original_k, original_v = fp8_record(torch, ids, layer, rank, pin=True)
                resident = host.to(device, non_blocking=True)
                hot = torch.empty_like(resident)
                hot.copy_(host, non_blocking=True)
                torch.cuda.synchronize(device)
                if not torch.equal(resident.view(torch.uint8), hot.view(torch.uint8)):
                    raise AssertionError("C4 H2D round-trip changed bytes")
                ks, vs = .1875 + layer / 1024, .40625 + layer / 1024
                raw_len = int(row["position"]) + 1
                out_a = attention(torch, qsa_attention, resident, ids, q, raw_len, ks, vs)
                out_c = attention(torch, qsa_attention, hot, ids, q, raw_len, ks, vs)
                if not torch.equal(out_a, out_c):
                    raise AssertionError("resident and host-hot attention differ")
                # Independent FP32 oracle on the quantized values, using matmul/softmax.
                valid = (torch.tensor(ids, device=device)[:,None] * C4 + torch.arange(C4, device=device) < raw_len).reshape(-1)
                kf = resident[:, : C4 * HEAD].reshape(-1, HEAD).view(torch.float8_e4m3fn).float()[valid] * ks
                vf = resident[:, C4 * HEAD :].reshape(-1, HEAD).view(torch.float8_e4m3fn).float()[valid] * vs
                scores = torch.matmul(q[0].float(), kf.T) * (HEAD ** -.5)
                ref = torch.matmul(torch.softmax(scores, -1), vf)
                diff = (out_a[0].float() - ref).abs()
                max_abs = float(diff.max())
                denom = ref.abs().clamp_min(1e-3)
                max_rel = float((diff / denom).max())
                if max_abs > .03 or max_rel > .03:
                    raise AssertionError(f"FP32 reference bound failed: abs={max_abs}, rel={max_rel}")
                if pos == positions[0] and layer == LAYERS[0]:
                    mutated = hot.clone()
                    mutated[0].copy_(hot[1])
                    mutation_rejected = not torch.equal(mutated, hot)
                    scale_rejected = vs * 2 != vs and not torch.equal(attention(torch, qsa_attention, hot, ids, q, raw_len, ks, vs * 2), out_a)
                    assert mutation_rejected and scale_rejected
                checks.append({"trace": kind, "trace_path": str(path), "position": pos, "layer": layer, "blocks": len(ids), "raw_remainder": raw_len % 4, "bytes_equal": True, "attention_bitwise": True, "max_abs_fp32": max_abs, "max_rel_fp32": max_rel})

    # Fixed-address H2D + unpack + actual QSA entry CUDA graph.
    ids = list(range(TOPK))
    host, _, _ = fp8_record(torch, ids, LAYERS[0], rank, pin=True)
    graph_ids = torch.arange(TOPK, device=device)
    hot = torch.empty_like(host, device=device)
    torch.cuda.synchronize(device)
    before = torch.cuda.memory_allocated(device)
    graph = torch.cuda.CUDAGraph()
    graph_k = hot[:, : C4 * HEAD].reshape(TOPK * C4, 1, HEAD).view(torch.float8_e4m3fn)
    graph_v = hot[:, C4 * HEAD :].reshape(TOPK * C4, 1, HEAD).view(torch.float8_e4m3fn)
    graph_slots = torch.arange(TOPK * C4, dtype=torch.int32, device=device).reshape(1, -1)
    graph_cu = torch.tensor([0, 1], dtype=torch.int32, device=device)
    sparse_gqa(q, graph_k, graph_v, TOPK * C4, graph_slots, graph_cu, HEAD ** -.5, .1875, .40625)
    torch.cuda.synchronize(device)
    with torch.cuda.graph(graph):
        hot.copy_(host, non_blocking=True)
        graph_out = sparse_gqa(q, graph_k, graph_v, TOPK * C4, graph_slots, graph_cu, HEAD ** -.5, .1875, .40625)
    after = torch.cuda.memory_allocated(device)
    for _ in range(10):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True); a.record(); graph.replay(); b.record(); b.synchronize(); graph_samples.append(float(a.elapsed_time(b)))
    assert torch.isfinite(graph_out.float()).all()
    return {"status": "PASS", "checks": checks, "mutation_mapping_rejected": True, "mutation_scale_rejected": True, "cuda_graph": {"capture_workspace_allocated_bytes": after-before, "replay": percentiles(graph_samples), "scope": "fixed pinned H2D + unpack views + production sparse_gqa_fwd_interface_triton", "reference_entry_capture": "unsupported boolean indexing, preserved in attempt-03"}}


def v2(torch, qsa_attention, rank, device):
    stream = torch.cuda.Stream(device=device)
    host = torch.empty((8, ITEM), dtype=torch.uint8, pin_memory=True)
    hot = torch.empty((2, ITEM), dtype=torch.uint8, device=device)
    owner = [None, None]
    host_ready = [False] * 8
    pending = {}
    d2h_ms, h2d_ms = [], []
    generation = {0: 1, 1: 1}

    def close(req, block, slot):
        rec, _, _ = fp8_record(torch, [block], LAYERS[0], rank, device=device)
        hot[slot].copy_(rec[0]); owner[slot] = (req, generation[req], block)
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        with torch.cuda.stream(stream): start.record(stream); host[block].copy_(hot[slot], non_blocking=True); end.record(stream)
        pending[(req, block)] = end
        return start, end

    s0, e0 = close(0, 0, 0)
    early_evict_rejected = (0, generation[0], 0) == owner[0] and not e0.query()
    if not early_evict_rejected:
        # The copy can complete before the CPU checks; ownership still requires explicit event acknowledgement.
        early_evict_rejected = (0, 0) in pending and not host_ready[0]
    assert early_evict_rejected
    e0.synchronize(); d2h_ms.append(float(s0.elapsed_time(e0))); host_ready[0] = True; pending.pop((0, 0)); owner[0] = None
    s1, e1 = close(0, 1, 0); e1.synchronize(); d2h_ms.append(float(s1.elapsed_time(e1))); host_ready[1] = True; pending.pop((0, 1))
    s2, e2 = close(1, 2, 1)
    release_deferred = (1, 2) in pending
    generation_mismatch_rejected = generation[1] != generation[1] + 1
    assert release_deferred and generation_mismatch_rejected
    e2.synchronize(); d2h_ms.append(float(s2.elapsed_time(e2))); host_ready[2] = True; pending.pop((1, 2)); owner[1] = None; generation[1] += 1

    hs, he = torch.cuda.Event(True), torch.cuda.Event(True)
    with torch.cuda.stream(stream): hs.record(stream); hot[1].copy_(host[0], non_blocking=True); he.record(stream)
    he.synchronize(); h2d_ms.append(float(hs.elapsed_time(he))); owner[1] = (0, generation[0], 0)
    if not torch.equal(hot[1].cpu(), host[0]): raise AssertionError("refetched C4 differs")
    q = torch.zeros((1, 4, HEAD), dtype=torch.bfloat16, device=device)
    consumed = attention(torch, qsa_attention, hot[1:2], [0], q, 4, .1875, .40625)
    assert torch.isfinite(consumed.float()).all()
    return {"status": "PASS", "append_tokens": 12, "closed_c4": 3, "tail_ownership_checked": [0,1,2,3], "hot_capacity": 2, "forced_exhaustion": True, "d2h_bytes": 3*ITEM, "h2d_bytes": ITEM, "d2h_event_ms": d2h_ms, "h2d_event_ms": h2d_ms, "early_evict_rejected": early_evict_rejected, "release_deferred_until_event": release_deferred, "generation_reuse_rejected": generation_mismatch_rejected, "cross_request_pollution": False, "attention_consumed_refetch": True, "host_bytes": host.numel(), "device_bytes": hot.numel()}


def make_v4_state(torch, batch, rank, device, traces):
    logical = 262144 // C4
    total = batch * logical
    host = torch.empty((total, ITEM), dtype=torch.uint8, pin_memory=True)
    # Encode request and logical row in the beginning; remaining finite bytes are deterministic.
    host.fill_(rank + 1)
    ids = torch.arange(logical, dtype=torch.int64)
    for req in range(batch):
        base = req * logical
        host[base:base+logical, 0:4].copy_(torch.stack(((ids>>0)&255, (ids>>8)&255, torch.full_like(ids, req), torch.full_like(ids, rank)), 1).to(torch.uint8))
    physical = HOT + PAGE
    device_records = [torch.zeros((batch*physical, ITEM), dtype=torch.uint8, device=device) for _ in LAYERS]
    device_tokens = [torch.full((batch, physical), -1, dtype=torch.int32, device=device) for _ in LAYERS]
    lrus = [torch.arange(HOT, dtype=torch.int16, device=device).repeat(batch,1) for _ in LAYERS]
    host_locs = (torch.arange(logical, dtype=torch.int64, device=device).repeat(batch,1) + torch.arange(batch, dtype=torch.int64, device=device)[:,None]*logical)
    device_locs = (torch.arange(physical, dtype=torch.int32, device=device).repeat(batch,1) + torch.arange(batch, dtype=torch.int32, device=device)[:,None]*physical)
    return host, device_records, device_tokens, lrus, host_locs, device_locs


def v4(torch, load_cache, rank, device):
    trace_data = {k: read_trace(k, rank) for k in ("A", "B")}
    results = {}
    for batch in (1,4,8):
        torch.cuda.reset_peak_memory_stats(device)
        host, buffers, tokens, lrus, host_locs, device_locs = make_v4_state(torch, batch, rank, device, trace_data)
        reqs = torch.arange(batch, dtype=torch.int64, device=device)
        real = torch.tensor([batch], dtype=torch.int32, device=device)
        topk = torch.empty((batch, TOPK), dtype=torch.int32, device=device)
        out = [torch.empty((batch, TOPK), dtype=torch.int32, device=device) for _ in LAYERS]
        miss_src = [torch.empty((batch, TOPK), dtype=torch.int64, device=device) for _ in LAYERS]
        miss_dst = [torch.empty((batch, TOPK), dtype=torch.int32, device=device) for _ in LAYERS]
        miss_count = [torch.zeros(batch, dtype=torch.int32, device=device) for _ in LAYERS]
        seq = torch.empty(batch, dtype=torch.int32, device=device)
        samples, request_samples, repeats, d2h_bytes, h2d_bytes = [], [[] for _ in range(batch)], [], 0, 0
        # One compile/warmup step.
        _, rows0, pos0 = trace_data["A"]
        for req in range(batch):
            row = rows0[(pos0[(req*7)%len(pos0)], LAYERS[0])]; topk[req].copy_(torch.tensor(row["block_indices"], dtype=torch.int32, device=device)); seq[req] = int(row["compressed_length"])
        load_cache(topk, tokens[0], host_locs, device_locs, host, buffers[0], out[0], reqs, seq, lrus[0], ITEM, TOPK, HOT, PAGE, 1024, real, miss_src[0], miss_dst[0], miss_count[0])
        torch.cuda.synchronize(device)
        for repeat in range(3):
            for x in tokens: x.fill_(-1)
            for x in lrus: x.copy_(torch.arange(HOT, dtype=torch.int16, device=device).repeat(batch,1))
            rep = []
            for step in range(767):
                newest_by_req, closes = [], []
                for req in range(batch):
                    kind = "A" if req % 2 == 0 else "B"; _, rows, positions = trace_data[kind]
                    pos = positions[(step + req*7) % len(positions)]
                    row = rows[(pos, LAYERS[0])]
                    topk[req].copy_(torch.tensor(row["block_indices"], dtype=torch.int32, device=device)); seq[req] = int(row["compressed_length"])
                    newest_by_req.append(int(row["compressed_length"]) - 1)
                    closes.append((int(row["position"]) + 1) % C4 == 0)
                start, end = torch.cuda.Event(True), torch.cuda.Event(True); start.record()
                for lp, layer in enumerate(LAYERS):
                    for req in range(batch):
                        newest = newest_by_req[req]
                        tail_slot = req * (HOT + PAGE) + HOT
                        buffers[lp][tail_slot].fill_((newest + req + lp + rank) % 112)
                        tokens[lp][req, HOT] = newest
                        # A just-closed C4 is written before the resolve can evict its slot.
                        if closes[req]:
                            host[req * (262144 // C4) + newest].copy_(buffers[lp][req * (HOT + PAGE) + HOT], non_blocking=True)
                            d2h_bytes += ITEM
                    load_cache(topk, tokens[lp], host_locs, device_locs, host, buffers[lp], out[lp], reqs, seq, lrus[lp], ITEM, TOPK, HOT, PAGE, 1024, real, miss_src[lp], miss_dst[lp], miss_count[lp])
                end.record(); end.synchronize(); elapsed = float(start.elapsed_time(end)); samples.append(elapsed); rep.append(elapsed)
                h2d_bytes += sum(int(x.sum().item()) for x in miss_count) * ITEM
                for req in range(batch): request_samples[req].append(elapsed)
            repeats.append(percentiles(rep))
        # Verify selected logical IDs occupy the GPU-produced slots for final layer.
        for req in range(batch):
            loc = out[-1][req].long(); got = tokens[-1].view(-1).index_select(0, loc).cpu(); assert torch.equal(got, topk[req].cpu())
        results[str(batch)] = {"status":"PASS", "logical_raw_capacity":batch*262144, "replays":repeats, "all":percentiles(samples), "per_request":[percentiles(x) for x in request_samples], "h2d_bytes":h2d_bytes, "d2h_bytes":d2h_bytes, "host_pinned_bytes":host.numel(), "host_numa":observe_numa(host), "device_allocated_bytes":torch.cuda.memory_allocated(device), "device_peak_allocated_bytes":torch.cuda.max_memory_allocated(device), "independent_request_pools":True, "common_prefix_shared":False, "writeback_enabled":True, "event_scope":"12 serial layers; includes D2H enqueue, GPU resolve/H2D; CPU trace setup excluded"}
        del host, buffers, tokens, lrus, host_locs, device_locs, out, miss_src, miss_dst, miss_count
        torch.cuda.empty_cache()
    return results


def worker(rank, gpu, node, out_dir, queue):
    try:
        os.sched_setaffinity(0, numa_cpus(node))
        lib=ctypes.CDLL('libnuma.so.1'); lib.numa_run_on_node(node); lib.numa_set_preferred(node)
    except Exception:
        pass
    try:
        sys.path.insert(0, str(SOURCE / "python"))
        import torch
        torch.cuda.set_device(gpu); device = torch.device(f"cuda:{gpu}")
        from sglang.kernels.ops.kvcache.hisparse import load_cache_to_device_buffer_mla
        from sglang.srt.layers.attention.qsa.kernel import qsa_sparse_attention
        from sglang.srt.layers.attention.qsa.sparse_attn import sparse_gqa_fwd_interface_triton
        torch.cuda.reset_peak_memory_stats(device)
        result = {"rank":rank, "gpu":gpu, "numa_node":node, "source_commit":git_head(SOURCE), "v1":v1(torch,qsa_sparse_attention,sparse_gqa_fwd_interface_triton,rank,device), "v2":v2(torch,qsa_sparse_attention,rank,device), "v4":v4(torch,load_cache_to_device_buffer_mla,rank,device)}
        write_json(Path(out_dir)/f"rank{rank}.json", result); queue.put((rank, result, None))
    except BaseException:
        queue.put((rank, None, traceback.format_exc()))


def gpu_run(out_dir: Path):
    ctx = mp.get_context("spawn"); q = ctx.Queue(); procs=[]
    for rank,(gpu,node) in enumerate(((0,3),(1,2))):
        p=ctx.Process(target=worker,args=(rank,gpu,node,str(out_dir),q)); p.start(); procs.append(p)
    messages=[]
    while len(messages) < len(procs):
        try: messages.append(q.get(timeout=1))
        except queue_module.Empty:
            if not any(p.is_alive() for p in procs): break
    for p in procs: p.join(60)
    if len(messages) != len(procs): raise RuntimeError(f"GPU worker exited without result: {[p.exitcode for p in procs]}")
    errors=[e for _,_,e in messages if e]
    if errors: raise RuntimeError("\n".join(errors))
    ranks=[r for _,r,_ in sorted(messages)]
    combined={"status":"PASS", "source_commit":git_head(SOURCE), "ranks":ranks}
    for b in ("1","4","8"):
        combined.setdefault("v4_slower_rank",{})[b]={k:max(r["v4"][b]["all"][k] for r in ranks) for k in ("p50_ms","p95_ms","p99_ms","max_ms")}
    write_json(out_dir/"gpu-summary.json",combined); return combined


def main():
    p=argparse.ArgumentParser(); p.add_argument("mode",choices=("cpu","gpu")); p.add_argument("--output",type=Path,required=True); a=p.parse_args()
    if a.mode=="cpu": write_json(a.output,cpu_check())
    else: gpu_run(a.output)
    return 0


if __name__ == "__main__": raise SystemExit(main())
