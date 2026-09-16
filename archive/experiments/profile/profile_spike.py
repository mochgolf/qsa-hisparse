#!/usr/bin/env python3
"""Bounded dual-rank V4 profiling arms; production source is never modified."""

from __future__ import annotations

import argparse
import ctypes
import importlib.util
import json
import multiprocessing as mp
import os
import queue
import sys
import time
import traceback
from contextlib import nullcontext
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE_PATH = HERE.parent / "qwen38-hisparse-integration-spike-20260908/corrective_spike.py"
SOURCE = ROOT / "sources/sglang-hisparse-spike-20260908"
DEADLINE_S = 3300
PROFILE_STEPS = range(72, 80)


def load_base():
    spec = importlib.util.spec_from_file_location("v4_profile_base", BASE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = load_base()


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def pair_samples(rows):
    grouped = {}
    for row in rows:
        key = (row["label"], row["trace"], row["request_count"], row["replay"], row["step"])
        ranks = grouped.setdefault(key, {})
        if row["rank"] in ranks:
            raise ValueError(f"duplicate rank sample {key + (row['rank'],)}")
        ranks[row["rank"]] = row
    if not grouped or any(set(ranks) != {0, 1} for ranks in grouped.values()):
        raise ValueError("each sample key must contain exactly ranks 0 and 1")
    paired = []
    for key, ranks in sorted(grouped.items()):
        slower = max(ranks.values(), key=lambda row: row["elapsed_ms"])
        paired.append({
            **slower,
            "elapsed_ms": max(row["elapsed_ms"] for row in ranks.values()),
            "wall_ms": max(row["wall_ms"] for row in ranks.values()),
            "slower_rank": slower["rank"],
            "rank_elapsed_ms": {str(rank): row["elapsed_ms"] for rank, row in ranks.items()},
            "rank_wall_ms": {str(rank): row["wall_ms"] for rank, row in ranks.items()},
        })
    return paired


def nvtx(torch, enabled: bool, name: str):
    return torch.cuda.nvtx.range(name) if enabled else nullcontext()


def unpack_indices(torch, batch: int, device):
    rows = torch.arange(batch * BASE.TOPK * 4, dtype=torch.int64, device=device)
    k = (rows // 4) * 8 + rows % 4
    return torch.cat((k, k + 4))


def reset_device(torch, buffers, tokens, lrus, device) -> None:
    lru = torch.arange(BASE.HOT, dtype=torch.int16, device=device).repeat(tokens[0].size(0), 1)
    for value in buffers:
        value.zero_()
    for value in tokens:
        value.fill_(-1)
    for value in lrus:
        value.copy_(lru)


def forced_lifecycle(torch, rank, batch, host, buffers, tokens, lrus, host_locs,
                     dev_locs, topk, seq, reqs, real, out, ms, md, mc,
                     copy_stream, load_cache, device) -> int:
    topk.copy_(torch.arange(BASE.TOPK, dtype=torch.int32, device=device).repeat(batch, 1))
    topk[:, -1] = BASE.LOGICAL - 2
    seq.fill_(BASE.LOGICAL)
    checks = 0
    for req in range(batch):
        for li in range(len(BASE.LAYERS)):
            block = BASE.LOGICAL - 2
            rec = torch.frombuffer(
                bytearray(BASE.packed_record_bytes(block, li, rank, req, 1)),
                dtype=torch.uint8,
            )
            tokens[li][req].fill_(-1)
            lrus[li][req].copy_(torch.arange(BASE.HOT, dtype=torch.int16, device=device))
            tail = req * (BASE.HOT + BASE.PAGE) + BASE.HOT
            buffers[li][tail].copy_(rec.to(device))
            tokens[li][req, BASE.HOT] = block
            producer, done = torch.cuda.Event(), torch.cuda.Event()
            producer.record()
            with torch.cuda.stream(copy_stream):
                copy_stream.wait_event(producer)
                host[req, li, block].copy_(buffers[li][tail], non_blocking=True)
                done.record(copy_stream)
            done.synchronize()
            BASE.validate_refetch_record(torch, host[req, li, block], rec)
            buffers[li][tail].fill_(0x55)
            tokens[li][req, BASE.HOT] = -1
            load_cache(
                topk, tokens[li], host_locs[li], dev_locs, host.view(-1, BASE.ITEM),
                buffers[li], out[li], reqs, seq, lrus[li], BASE.ITEM, BASE.TOPK,
                BASE.HOT, BASE.PAGE, 1024, real, ms[li], md[li], mc[li],
            )
            torch.cuda.synchronize(device)
            count = int(mc[li][req])
            src = ms[li][req, :count]
            expected_src = req * len(BASE.LAYERS) * BASE.LOGICAL + li * BASE.LOGICAL + block
            matches = (src == expected_src).nonzero().flatten()
            if count <= 0 or matches.numel() != 1:
                raise AssertionError("forced GPU miss plan lacks generated host block")
            col = int(matches[0])
            dest = int(md[li][req, col])
            loc = int(out[li][req, -1])
            low = req * (BASE.HOT + BASE.PAGE)
            if loc != dest or not low <= loc < low + BASE.HOT:
                raise AssertionError("forced refetch did not land in normal hot slot")
            BASE.validate_refetch_record(torch, buffers[li][loc].cpu(), rec)
            buffers[li][loc].fill_(0x55)
            try:
                BASE.validate_refetch_record(torch, buffers[li][loc].cpu(), rec)
            except AssertionError:
                pass
            else:
                raise AssertionError("poisoned refetch was accepted")
            checks += 1
    return checks


def expected_unpacked(torch, packed):
    return packed[:, :1024].reshape(-1, BASE.HEAD), packed[:, 1024:].reshape(-1, BASE.HEAD)


def validate_step(torch, rank, batch, step_data, generated, host, buffers, tokens,
                  out, gathered, unpack_k, unpack_v, scale_state, host_scales,
                  device_scales) -> tuple[int, int]:
    checks = byte_count = 0
    for li in range(len(BASE.LAYERS)):
        if scale_state[li] != (.1875 + li / 1024, .40625 + li / 1024):
            raise AssertionError("scale state")
        if not torch.equal(device_scales[:, li].cpu(), host_scales[:, li]):
            raise AssertionError("scale transfer")
        for req in range(batch):
            top_cpu, seq_cpu, _, _ = step_data[li]
            versions = dict.fromkeys(generated[req][li], 1)
            versions[int(seq_cpu[req]) - 1] = 1
            BASE.check_host_rows(
                torch, host, buffers[li], tokens[li], out[li], top_cpu,
                req, li, rank, versions,
            )
            blocks = top_cpu[req].tolist()
            expected = BASE.expected_rows(torch, blocks, li, rank, req)
            changed = [i for i, block in enumerate(blocks) if versions.get(int(block), 0) == 1]
            if changed:
                expected[changed] = BASE.expected_rows(
                    torch, [int(blocks[i]) for i in changed], li, rank, req, 1,
                )
            actual_packed = gathered[li][req].cpu()
            BASE.validate_refetch_record(torch, actual_packed, expected)
            expected_k, expected_v = expected_unpacked(torch, expected)
            actual_k = unpack_k[li][req].cpu()
            actual_v = unpack_v[li][req].cpu()
            if not torch.equal(actual_k, expected_k) or not torch.equal(actual_v, expected_v):
                raise AssertionError(
                    f"unpack bytes differ rank={rank} batch={batch} layer={BASE.LAYERS[li]} req={req}"
                )
            checks += 2
            byte_count += actual_k.numel() + actual_v.numel()
    return checks, byte_count


def run_arm(torch, rank: int, device, barrier, label: str, arm: str,
            batch: int, replays: int, steps: int, profile: bool):
    from sglang.kernels.ops.kvcache.hisparse import load_cache_to_device_buffer_mla

    if arm not in {"baseline", "event-reuse", "fused-unpack"}:
        raise ValueError(arm)
    traces = {kind: BASE.read_trace(kind, rank) for kind in ("A", "B")}
    host = BASE.init_host(torch, batch, rank)
    required = batch * len(BASE.LAYERS) * BASE.LOGICAL * BASE.ITEM
    if host.numel() != required:
        raise AssertionError("host allocation size")
    physical = BASE.HOT + BASE.PAGE
    buffers = [torch.empty((batch * physical, BASE.ITEM), dtype=torch.uint8, device=device) for _ in BASE.LAYERS]
    tokens = [torch.full((batch, physical), -1, dtype=torch.int32, device=device) for _ in BASE.LAYERS]
    lrus = [torch.arange(BASE.HOT, dtype=torch.int16, device=device).repeat(batch, 1) for _ in BASE.LAYERS]
    host_locs = [
        torch.arange(BASE.LOGICAL, device=device, dtype=torch.int64)[None, :]
        + torch.arange(batch, device=device, dtype=torch.int64)[:, None] * (len(BASE.LAYERS) * BASE.LOGICAL)
        + li * BASE.LOGICAL
        for li in range(len(BASE.LAYERS))
    ]
    dev_locs = (
        torch.arange(physical, device=device, dtype=torch.int32)[None, :]
        + torch.arange(batch, device=device, dtype=torch.int32)[:, None] * physical
    )
    topk = torch.empty((batch, BASE.TOPK), dtype=torch.int32, device=device)
    seq = torch.empty(batch, dtype=torch.int32, device=device)
    reqs = torch.arange(batch, dtype=torch.int64, device=device)
    real = torch.tensor([batch], dtype=torch.int32, device=device)
    out = [torch.empty((batch, BASE.TOPK), dtype=torch.int32, device=device) for _ in BASE.LAYERS]
    ms = [torch.empty((batch, BASE.TOPK), dtype=torch.int64, device=device) for _ in BASE.LAYERS]
    md = [torch.empty((batch, BASE.TOPK), dtype=torch.int32, device=device) for _ in BASE.LAYERS]
    mc = [torch.zeros(batch, dtype=torch.int32, device=device) for _ in BASE.LAYERS]
    gathered = [torch.empty((batch, BASE.TOPK, BASE.ITEM), dtype=torch.uint8, device=device) for _ in BASE.LAYERS]
    if arm == "fused-unpack":
        unpacked = [torch.empty((2, batch, BASE.TOPK * 4, BASE.HEAD), dtype=torch.uint8, device=device) for _ in BASE.LAYERS]
        unpack_k = [value[0] for value in unpacked]
        unpack_v = [value[1] for value in unpacked]
        indices = unpack_indices(torch, batch, device)
    else:
        unpacked = indices = None
        unpack_k = [torch.empty((batch, BASE.TOPK * 4, BASE.HEAD), dtype=torch.uint8, device=device) for _ in BASE.LAYERS]
        unpack_v = [torch.empty_like(unpack_k[0]) for _ in BASE.LAYERS]
    copy_stream = torch.cuda.Stream(device=device)
    dependency_events = (
        [[(torch.cuda.Event(), torch.cuda.Event()) for _ in range(batch)] for _ in BASE.LAYERS]
        if arm == "event-reuse" else None
    )
    scale_state = [(.1875 + li / 1024, .40625 + li / 1024) for li in range(len(BASE.LAYERS))]
    host_scales = torch.tensor([[scale_state[li] for li in range(len(BASE.LAYERS))] for _ in range(batch)], dtype=torch.float32, pin_memory=True)
    device_scales = host_scales.to(device)
    prepared = BASE.prepare_v4_steps(torch, traces, batch, rank)
    samples = []
    correctness = {"forced_lifecycle_checks": 0, "unpack_byte_checks": 0, "unpack_bytes_checked": 0}
    total_d2h = total_h2d = natural_refetches = 0
    torch.cuda.reset_peak_memory_stats(device)
    for replay in range(replays):
        generated = [[set() for _ in BASE.LAYERS] for _ in range(batch)]
        BASE.reset_host(torch, host, batch, rank)
        reset_device(torch, buffers, tokens, lrus, device)
        correctness["forced_lifecycle_checks"] += forced_lifecycle(
            torch, rank, batch, host, buffers, tokens, lrus, host_locs,
            dev_locs, topk, seq, reqs, real, out, ms, md, mc, copy_stream,
            load_cache_to_device_buffer_mla, device,
        )
        BASE.reset_host(torch, host, batch, rank)
        reset_device(torch, buffers, tokens, lrus, device)
        for step in range(steps):
            step_data = prepared[step]
            positions = step_data[0][3]
            close_count = sum((position + 1) % 4 == 0 for position in positions)
            page64_count = sum((position + 1) % 64 == 0 for position in positions)
            instrument = profile and step in PROFILE_STEPS
            barrier.wait(300)
            step_range = f"r{rank}/step{step}/{'close' if close_count else 'nonclose'}"
            with nvtx(torch, instrument, step_range):
                start, end = torch.cuda.Event(True), torch.cuda.Event(True)
                wall_start = time.perf_counter_ns()
                start.record()
                step_d2h = 0
                for li, lid in enumerate(BASE.LAYERS):
                    top_cpu, seq_cpu, records, layer_positions = step_data[li]
                    with nvtx(torch, instrument, f"r{rank}/step{step}/l{lid}/metadata-tail"):
                        topk.copy_(top_cpu, non_blocking=True)
                        seq.copy_(seq_cpu, non_blocking=True)
                        dones = []
                        for req in range(batch):
                            position = layer_positions[req]
                            newest = int(seq_cpu[req]) - 1
                            tail = req * physical + BASE.HOT
                            buffers[li][tail].copy_(records[req], non_blocking=True)
                            tokens[li][req, BASE.HOT] = newest
                            if (position + 1) % 4 == 0:
                                if dependency_events is None:
                                    producer, done = torch.cuda.Event(), torch.cuda.Event()
                                else:
                                    producer, done = dependency_events[li][req]
                                producer.record()
                                with torch.cuda.stream(copy_stream):
                                    copy_stream.wait_event(producer)
                                    host[req, li, newest].copy_(buffers[li][tail], non_blocking=True)
                                    done.record(copy_stream)
                                dones.append(done)
                                step_d2h += BASE.ITEM
                                generated[req][li].add(newest)
                    with nvtx(torch, instrument, f"r{rank}/step{step}/l{lid}/wait-d2h"):
                        for done in dones:
                            torch.cuda.current_stream(device).wait_event(done)
                    with nvtx(torch, instrument, f"r{rank}/step{step}/l{lid}/resolver"):
                        load_cache_to_device_buffer_mla(
                            topk, tokens[li], host_locs[li], dev_locs,
                            host.view(-1, BASE.ITEM), buffers[li], out[li], reqs,
                            seq, lrus[li], BASE.ITEM, BASE.TOPK, BASE.HOT,
                            BASE.PAGE, 1024, real, ms[li], md[li], mc[li],
                        )
                    with nvtx(torch, instrument, f"r{rank}/step{step}/l{lid}/gather"):
                        torch.index_select(
                            buffers[li], 0, out[li].reshape(-1).long(),
                            out=gathered[li].view(-1, BASE.ITEM),
                        )
                    with nvtx(torch, instrument, f"r{rank}/step{step}/l{lid}/unpack"):
                        if arm == "fused-unpack":
                            torch.index_select(
                                gathered[li].view(-1, BASE.HEAD), 0, indices,
                                out=unpacked[li].view(-1, BASE.HEAD),
                            )
                        else:
                            for member in range(4):
                                unpack_k[li][:, member::4].copy_(
                                    gathered[li][:, :, member * BASE.HEAD:(member + 1) * BASE.HEAD]
                                )
                                unpack_v[li][:, member::4].copy_(
                                    gathered[li][:, :, 1024 + member * BASE.HEAD:1024 + (member + 1) * BASE.HEAD]
                                )
                end.record()
                end.synchronize()
                wall_ms = (time.perf_counter_ns() - wall_start) / 1e6
                event_ms = float(start.elapsed_time(end))
            step_h2d = 0
            for li in range(len(BASE.LAYERS)):
                layer_misses = int(mc[li].sum().item())
                step_h2d += layer_misses * BASE.ITEM
                for req in range(batch):
                    count = int(mc[li][req].item())
                    base = req * len(BASE.LAYERS) * BASE.LOGICAL + li * BASE.LOGICAL
                    fetched = (ms[li][req, :count] - base).cpu().tolist()
                    natural_refetches += sum(int(value in generated[req][li]) for value in fetched)
            total_d2h += step_d2h
            total_h2d += step_h2d
            samples.append({
                "label": label, "arm": arm, "trace": "mixed-A-even-B-odd",
                "request_count": batch, "replay": replay, "step": step,
                "rank": rank, "positions": positions, "c4_close_count": close_count,
                "page64_boundary_count": page64_count, "elapsed_ms": event_ms,
                "wall_ms": wall_ms, "miss_count": step_h2d // BASE.ITEM,
                "d2h_bytes": step_d2h, "h2d_bytes": step_h2d,
            })
            if step in (0, BASE.STEPS // 2, BASE.STEPS - 1) or close_count or page64_count:
                checks, byte_count = validate_step(
                    torch, rank, batch, step_data, generated, host, buffers,
                    tokens, out, gathered, unpack_k, unpack_v, scale_state,
                    host_scales, device_scales,
                )
                correctness["unpack_byte_checks"] += checks
                correctness["unpack_bytes_checked"] += byte_count
            barrier.wait(300)
    result = {
        "label": label, "arm": arm, "status": "PASS_COMPONENT_CORRECTNESS",
        "batch": batch, "replays": replays, "steps": steps,
        "rank_event": BASE.stats([row["elapsed_ms"] for row in samples]),
        "rank_wall": BASE.stats([row["wall_ms"] for row in samples]),
        "correctness": correctness, "d2h_bytes": total_d2h,
        "h2d_bytes": total_h2d, "naturally_generated_refetches": natural_refetches,
        "host_pinned_bytes": host.numel(), "host_numa": BASE.observe_numa(host),
        "device_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "device_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "event_scope": "start/end enclose metadata and tail injection, producer-gated D2H, dependency waits, resolver/H2D, selected gather, and actual K/V unpack",
        "candidate_change": {
            "baseline": "none",
            "event-reuse": "preallocate and reuse dependency events outside the timed loop",
            "fused-unpack": "replace eight strided copy launches per layer with one pre-indexed index_select",
        }[arm],
    }
    del host, buffers, tokens, lrus, prepared, gathered, unpack_k, unpack_v
    torch.cuda.empty_cache()
    return result, samples


def worker(rank, output, barrier, result_queue, specs):
    try:
        node = 3 - rank
        lib = ctypes.CDLL("libnuma.so.1")
        lib.numa_run_on_node(node)
        lib.numa_set_preferred(node)
        sys.path.insert(0, str(SOURCE / "python"))
        import torch

        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
        identity = {
            "rank": rank, "gpu": rank, "numa_node": node,
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "source_commit": BASE.git_head(SOURCE),
        }
        results, all_samples = [], []
        for spec in specs:
            result, samples = run_arm(torch, rank, device, barrier, **spec)
            results.append(result)
            all_samples.extend(samples)
        save(Path(output) / f"rank{rank}-results.json", {"identity": identity, "arms": results})
        save(Path(output) / f"rank{rank}-samples.json", all_samples)
        result_queue.put((rank, identity, results, all_samples, None))
    except BaseException:
        try:
            barrier.abort()
        except BaseException:
            pass
        result_queue.put((rank, None, None, None, traceback.format_exc()))


def specs_for(args):
    if args.mode == "before":
        return [dict(label="baseline-before", arm="baseline", batch=1, replays=3, steps=BASE.STEPS, profile=False)]
    if args.mode == "profile":
        return [dict(label="timeline", arm="baseline", batch=1, replays=1, steps=80, profile=True)]
    if args.mode == "after":
        arms = [value for value in args.arms.split(",") if value]
        if any(value not in {"event-reuse", "fused-unpack"} for value in arms):
            raise ValueError(args.arms)
        return [
            *[dict(label=value, arm=value, batch=1, replays=3, steps=BASE.STEPS, profile=False) for value in arms],
            dict(label="baseline-after", arm="baseline", batch=1, replays=3, steps=BASE.STEPS, profile=False),
        ]
    if args.mode == "scale":
        if args.candidate not in {"event-reuse", "fused-unpack"}:
            raise ValueError(args.candidate)
        return [
            dict(label=f"{args.candidate}-b{batch}", arm=args.candidate, batch=batch, replays=3, steps=BASE.STEPS, profile=False)
            for batch in (4, 8)
        ]
    raise ValueError(args.mode)


def run(output: Path, specs) -> None:
    output.mkdir(parents=True, exist_ok=False)
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    barrier = ctx.Barrier(2, timeout=300)
    processes = [ctx.Process(target=worker, args=(rank, str(output), barrier, result_queue, specs)) for rank in range(2)]
    for process in processes:
        process.start()
    messages = []
    deadline = time.monotonic() + DEADLINE_S
    while len(messages) < 2 and time.monotonic() < deadline:
        try:
            messages.append(result_queue.get(timeout=min(5, max(.1, deadline - time.monotonic()))))
        except queue.Empty:
            if not any(process.is_alive() for process in processes):
                break
    for process in processes:
        process.join(10)
        if process.is_alive():
            process.terminate()
            process.join(5)
        if process.is_alive():
            process.kill()
            process.join(5)
    if len(messages) != 2:
        raise RuntimeError(f"worker missing result; exitcodes={[process.exitcode for process in processes]}")
    errors = [message[4] for message in messages if message[4]]
    if errors:
        raise RuntimeError("\n".join(errors))
    messages.sort()
    samples = sum((message[3] for message in messages), [])
    paired = pair_samples(samples)
    arm_results = {}
    for spec in specs:
        label = spec["label"]
        selected = [row for row in paired if row["label"] == label]
        arm_results[label] = {
            "arm": spec["arm"], "batch": spec["batch"], "paired_samples": len(selected),
            "paired_event": BASE.stats([row["elapsed_ms"] for row in selected]),
            "paired_wall": BASE.stats([row["wall_ms"] for row in selected]),
            "rank_results": [next(result for result in message[2] if result["label"] == label) for message in messages],
        }
    save(output / "all-rank-samples.json", samples)
    save(output / "paired-slower-rank-samples.json", paired)
    save(output / "summary.json", {
        "status": "PASS", "source_commit": BASE.git_head(SOURCE),
        "specs": specs, "arms": arm_results,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("before", "profile", "after", "scale"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arms", default="event-reuse,fused-unpack")
    parser.add_argument("--candidate")
    args = parser.parse_args()
    run(args.output, specs_for(args))


if __name__ == "__main__":
    main()
