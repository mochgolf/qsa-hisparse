#!/usr/bin/env python3
"""Reduce the selected NVTX steps from an Nsight Systems SQLite export."""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path

STEP = re.compile(r"r(?P<rank>[01])/step(?P<step>\d+)/(?P<kind>close|nonclose)$")
STAGE = re.compile(r"r(?P<rank>[01])/step(?P<step>\d+)/l(?P<layer>\d+)/(?P<stage>metadata-tail|wait-d2h|resolver|gather|unpack)$")


def save(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def merged_ns(intervals) -> int:
    total = 0
    end = None
    for start, stop in sorted(intervals):
        if end is None or start > end:
            total += stop - start
            end = stop
        elif stop > end:
            total += stop - end
            end = stop
    return total


def percentile(values, fraction):
    values = sorted(values)
    if not values:
        return None
    return values[round((len(values) - 1) * fraction)]


def stats(values):
    return {
        "n": len(values), "p50_us": percentile(values, .5),
        "p95_us": percentile(values, .95), "max_us": max(values) if values else None,
        "mean_us": sum(values) / len(values) if values else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    db = sqlite3.connect(args.sqlite)
    strings = dict(db.execute("select id,value from StringIds"))
    ranges = []
    for start, end, text, text_id, global_tid in db.execute(
        "select start,end,text,textId,globalTid from NVTX_EVENTS where end is not null"
    ):
        name = text or strings.get(text_id, "")
        match = STEP.fullmatch(name) or STAGE.fullmatch(name)
        if match:
            ranges.append({
                "name": name, "start": start, "end": end,
                "process": global_tid & ~0xFFFFFF, "fields": match.groupdict(),
                "type": "step" if STEP.fullmatch(name) else "stage",
            })
    if len([row for row in ranges if row["type"] == "step"]) != 16:
        raise RuntimeError(f"expected 16 rank-step ranges, got {len(ranges)} total ranges")

    runtimes = defaultdict(list)
    for start, end, global_tid, correlation, name_id in db.execute(
        "select start,end,globalTid,correlationId,nameId from CUPTI_ACTIVITY_KIND_RUNTIME"
    ):
        runtimes[global_tid & ~0xFFFFFF].append((start, end, correlation, strings.get(name_id, str(name_id))))
    gpu_by_correlation = defaultdict(list)
    for start, end, device, stream, correlation, global_pid, name_id in db.execute(
        "select start,end,deviceId,streamId,correlationId,globalPid,shortName from CUPTI_ACTIVITY_KIND_KERNEL"
    ):
        gpu_by_correlation[(global_pid, correlation)].append({
            "kind": "kernel", "name": strings.get(name_id, str(name_id)),
            "start": start, "end": end, "device": device, "stream": stream,
        })
    copy_kinds = dict(db.execute("select id,label from ENUM_CUDA_MEMCPY_OPER"))
    for start, end, device, stream, correlation, global_pid, byte_count, copy_kind in db.execute(
        "select start,end,deviceId,streamId,correlationId,globalPid,bytes,copyKind from CUPTI_ACTIVITY_KIND_MEMCPY"
    ):
        gpu_by_correlation[(global_pid, correlation)].append({
            "kind": "memcpy", "name": copy_kinds.get(copy_kind, str(copy_kind)),
            "start": start, "end": end, "device": device, "stream": stream,
            "bytes": byte_count,
        })

    detail = []
    for row in ranges:
        apis = [value for value in runtimes[row["process"]] if value[0] < row["end"] and value[1] > row["start"]]
        activities = []
        for api_start, api_end, correlation, api_name in apis:
            for activity in gpu_by_correlation.get((row["process"], correlation), ()):
                activities.append({**activity, "api_name": api_name, "correlation": correlation})
        intervals = [(value["start"], value["end"]) for value in activities]
        gpu_start = min((value[0] for value in intervals), default=None)
        gpu_end = max((value[1] for value in intervals), default=None)
        by_api = defaultdict(lambda: {"count": 0, "cpu_ns": 0})
        for start, end, _, name in apis:
            by_api[name]["count"] += 1
            by_api[name]["cpu_ns"] += end - start
        detail.append({
            **{key: (int(value) if key in {"rank", "step", "layer"} else value) for key, value in row["fields"].items()},
            "name": row["name"], "type": row["type"],
            "cpu_range_us": (row["end"] - row["start"]) / 1000,
            "cuda_api_us": sum(end - start for start, end, _, _ in apis) / 1000,
            "cuda_api": dict(sorted(by_api.items())),
            "gpu_activity_count": len(activities),
            "gpu_kernel_count": sum(value["kind"] == "kernel" for value in activities),
            "gpu_memcpy_count": sum(value["kind"] == "memcpy" for value in activities),
            "gpu_memcpy_bytes": sum(value.get("bytes", 0) for value in activities),
            "gpu_busy_union_us": merged_ns(intervals) / 1000,
            "gpu_envelope_us": (gpu_end - gpu_start) / 1000 if gpu_start is not None else 0,
            "gpu_idle_inside_envelope_us": ((gpu_end - gpu_start) - merged_ns(intervals)) / 1000 if gpu_start is not None else 0,
            "streams": sorted({value["stream"] for value in activities}),
            "activities": activities,
        })

    steps = [row for row in detail if row["type"] == "step"]
    stages = [row for row in detail if row["type"] == "stage"]
    by_kind = {}
    for kind in ("close", "nonclose"):
        selected = [row for row in steps if row["kind"] == kind]
        by_kind[kind] = {
            "cpu_range": stats([row["cpu_range_us"] for row in selected]),
            "cuda_api": stats([row["cuda_api_us"] for row in selected]),
            "gpu_busy_union": stats([row["gpu_busy_union_us"] for row in selected]),
            "gpu_envelope": stats([row["gpu_envelope_us"] for row in selected]),
            "gpu_idle_inside_envelope": stats([row["gpu_idle_inside_envelope_us"] for row in selected]),
            "gpu_kernel_count": stats([row["gpu_kernel_count"] for row in selected]),
            "gpu_memcpy_count": stats([row["gpu_memcpy_count"] for row in selected]),
        }
    by_stage = {}
    for stage in ("metadata-tail", "wait-d2h", "resolver", "gather", "unpack"):
        selected = [row for row in stages if row["stage"] == stage]
        by_stage[stage] = {
            "ranges": len(selected),
            "cuda_api_us_sum": sum(row["cuda_api_us"] for row in selected),
            "gpu_busy_union_us_sum": sum(row["gpu_busy_union_us"] for row in selected),
            "gpu_activity_count": sum(row["gpu_activity_count"] for row in selected),
            "gpu_memcpy_bytes": sum(row["gpu_memcpy_bytes"] for row in selected),
        }
    unpack_per_step = [
        sum(row["gpu_activity_count"] for row in stages if row["rank"] == step["rank"] and row["step"] == step["step"] and row["stage"] == "unpack")
        for step in steps
    ]
    event_names = ("EventCreate", "EventRecord", "StreamWaitEvent")
    close_event_api_us = []
    nonclose_event_api_us = []
    for step in steps:
        value = sum(
            item["cpu_ns"] for name, item in step["cuda_api"].items()
            if any(token in name for token in event_names)
        ) / 1000
        (close_event_api_us if step["kind"] == "close" else nonclose_event_api_us).append(value)
    evidence = {
        "event-reuse": {
            "supported": bool(close_event_api_us) and min(close_event_api_us) > max(nonclose_event_api_us, default=0),
            "close_event_api": stats(close_event_api_us),
            "nonclose_event_api": stats(nonclose_event_api_us),
            "hypothesis": "close steps add dependency-event creation/record/wait APIs inside the timed loop",
        },
        "fused-unpack": {
            "supported": bool(unpack_per_step) and min(unpack_per_step) >= 8 * len(range(3, 48, 4)),
            "activities_per_rank_step": stats(unpack_per_step),
            "hypothesis": "baseline emits eight unpack copy activities per layer (96 per rank-step)",
        },
    }
    result = {
        "status": "PASS", "sqlite": str(args.sqlite),
        "selected_steps": list(range(72, 80)), "range_count": len(ranges),
        "step_summary": by_kind, "stage_totals_not_additive": by_stage,
        "candidate_evidence": evidence,
        "interpretation_guard": "stage sums are work attribution only; the critical-path quantities are each step's GPU envelope, union busy time, and idle inside that envelope",
    }
    save(args.output / "timeline-analysis.json", result)
    with (args.output / "timeline-events.jsonl").open("w") as stream:
        for row in detail:
            stream.write(json.dumps(row) + "\n")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
