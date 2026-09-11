#!/usr/bin/env python3
"""Reproducible CPU reduction of execution-01 raw samples."""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = HERE / "execution-01"
BUDGET_MS = 2.427575
HISTORICAL_GAP_MS = .086089


def nearest(values, fraction):
    ordered = sorted(values)
    return ordered[math.ceil(fraction * len(ordered)) - 1]


def stats(values):
    return {
        "n": len(values), "p50": nearest(values, .5), "p95": nearest(values, .95),
        "p99": nearest(values, .99), "max": max(values),
        "mean": statistics.fmean(values),
    }


def read_rows(path, label):
    return [row for row in json.loads(path.read_text()) if row["label"] == label]


def arm_summary(rows):
    result = {}
    groups = {
        "all": rows,
        "close": [row for row in rows if row["c4_close_count"]],
        "nonclose": [row for row in rows if not row["c4_close_count"]],
        "page64": [row for row in rows if row["page64_boundary_count"]],
        "step0": [row for row in rows if row["step"] == 0],
        "steady_after_step0": [row for row in rows if row["step"] != 0],
    }
    for name, selected in groups.items():
        result[name] = {
            "event_ms": stats([row["elapsed_ms"] for row in selected]),
            "wall_ms": stats([row["wall_ms"] for row in selected]),
            "miss_count": stats([row["miss_count"] for row in selected]),
        }
    result["slower_rank_counts"] = dict(Counter(str(row["slower_rank"]) for row in rows))
    result["per_replay_event_ms"] = {
        str(replay): stats([row["elapsed_ms"] for row in rows if row["replay"] == replay])
        for replay in range(3)
    }
    return result


def delta_summary(control, candidate):
    control_by_key = {(row["replay"], row["step"]): row for row in control}
    candidate_by_key = {(row["replay"], row["step"]): row for row in candidate}
    if set(control_by_key) != set(candidate_by_key):
        raise RuntimeError("paired keys differ")
    result = {}
    for group, keys in {
        "all": candidate_by_key,
        "close": [key for key, row in candidate_by_key.items() if row["c4_close_count"]],
        "nonclose": [key for key, row in candidate_by_key.items() if not row["c4_close_count"]],
    }.items():
        selected = list(keys) if isinstance(keys, list) else list(keys.keys())
        result[group] = stats([
            control_by_key[key]["elapsed_ms"] - candidate_by_key[key]["elapsed_ms"]
            for key in selected
        ])
    return result


def main() -> None:
    arms = {
        "baseline-before": read_rows(RUN / "baseline-before/paired-slower-rank-samples.json", "baseline-before"),
        "event-reuse": read_rows(RUN / "candidates-and-baseline-after/paired-slower-rank-samples.json", "event-reuse"),
        "fused-unpack": read_rows(RUN / "candidates-and-baseline-after/paired-slower-rank-samples.json", "fused-unpack"),
        "baseline-after": read_rows(RUN / "candidates-and-baseline-after/paired-slower-rank-samples.json", "baseline-after"),
        "fused-unpack-b4": read_rows(RUN / "best-scaling/paired-slower-rank-samples.json", "fused-unpack-b4"),
        "fused-unpack-b8": read_rows(RUN / "best-scaling/paired-slower-rank-samples.json", "fused-unpack-b8"),
    }
    summaries = {name: arm_summary(rows) for name, rows in arms.items()}
    before_p95 = summaries["baseline-before"]["all"]["event_ms"]["p95"]
    after_p95 = summaries["baseline-after"]["all"]["event_ms"]["p95"]
    event_p95 = summaries["event-reuse"]["all"]["event_ms"]["p95"]
    fused_p95 = summaries["fused-unpack"]["all"]["event_ms"]["p95"]

    timeline = json.loads((RUN / "timeline/timeline-analysis.json").read_text())
    timeline_rows = [json.loads(line) for line in (RUN / "timeline/timeline-events.jsonl").read_text().splitlines()]
    stages = [row for row in timeline_rows if row["type"] == "stage"]
    stage_by_kind = {}
    for kind in ("close", "nonclose"):
        stage_by_kind[kind] = {}
        for stage in ("metadata-tail", "wait-d2h", "resolver", "gather", "unpack"):
            selected = [
                row for row in stages if row["stage"] == stage
                and ((row["step"] % 4 == 3) == (kind == "close"))
            ]
            stage_by_kind[kind][stage] = {
                "ranges": len(selected),
                "cpu_range_us_per_layer": statistics.fmean(row["cpu_range_us"] for row in selected),
                "cuda_api_us_per_layer": statistics.fmean(row["cuda_api_us"] for row in selected),
                "gpu_busy_union_us_per_layer": statistics.fmean(row["gpu_busy_union_us"] for row in selected),
                "gpu_activities_per_layer": statistics.fmean(row["gpu_activity_count"] for row in selected),
                "gpu_memcpy_bytes_per_layer": statistics.fmean(row["gpu_memcpy_bytes"] for row in selected),
            }

    profiled = read_rows(RUN / "timeline-run/paired-slower-rank-samples.json", "timeline")
    profile_overhead = {}
    for kind in ("close", "nonclose"):
        values = []
        for row in profiled:
            if row["step"] < 72 or ((row["c4_close_count"] > 0) != (kind == "close")):
                continue
            reference = statistics.median(
                value["elapsed_ms"] for value in arms["baseline-before"] if value["step"] == row["step"]
            )
            values.append(row["elapsed_ms"] - reference)
        profile_overhead[kind] = stats(values)

    correctness = {}
    for source, labels in {
        "baseline-before": (RUN / "baseline-before/summary.json", ["baseline-before"]),
        "candidate-arms": (RUN / "candidates-and-baseline-after/summary.json", ["event-reuse", "fused-unpack", "baseline-after"]),
        "best-scaling": (RUN / "best-scaling/summary.json", ["fused-unpack-b4", "fused-unpack-b8"]),
    }.items():
        data = json.loads(labels[0].read_text())
        correctness[source] = {
            label: {
                "statuses": [row["status"] for row in data["arms"][label]["rank_results"]],
                "forced_lifecycle_checks": sum(row["correctness"]["forced_lifecycle_checks"] for row in data["arms"][label]["rank_results"]),
                "unpack_byte_checks": sum(row["correctness"]["unpack_byte_checks"] for row in data["arms"][label]["rank_results"]),
                "unpack_bytes_checked": sum(row["correctness"]["unpack_bytes_checked"] for row in data["arms"][label]["rank_results"]),
            }
            for label in labels[1]
        }

    result = {
        "budget": {
            "b1_p95_ms": BUDGET_MS, "historical_gap_ms": HISTORICAL_GAP_MS,
            "baseline_before_margin_ms": BUDGET_MS - before_p95,
            "baseline_after_margin_ms": BUDGET_MS - after_p95,
            "event_reuse_margin_ms": BUDGET_MS - event_p95,
            "fused_unpack_margin_ms": BUDGET_MS - fused_p95,
        },
        "arms": summaries,
        "baseline_drift": {
            "p95_after_minus_before_ms": after_p95 - before_p95,
            "fraction_of_historical_gap": (after_p95 - before_p95) / HISTORICAL_GAP_MS,
            "before_minus_attempt19_ms": before_p95 - 2.513664,
            "after_minus_attempt19_ms": after_p95 - 2.513664,
        },
        "candidate_deltas_ms": {
            candidate: {
                "vs_baseline_before": delta_summary(arms["baseline-before"], arms[candidate]),
                "vs_baseline_after": delta_summary(arms["baseline-after"], arms[candidate]),
            }
            for candidate in ("event-reuse", "fused-unpack")
        },
        "candidate_p95_gain": {
            "event-reuse-conservative-ms": min(before_p95, after_p95) - event_p95,
            "event-reuse-fraction-of-historical-gap": (min(before_p95, after_p95) - event_p95) / HISTORICAL_GAP_MS,
            "fused-unpack-conservative-ms": min(before_p95, after_p95) - fused_p95,
            "fused-unpack-fraction-of-historical-gap": (min(before_p95, after_p95) - fused_p95) / HISTORICAL_GAP_MS,
        },
        "historical_scaling_comparison": {
            "b4": {"attempt19_p95_ms": 4.363936, "candidate_p95_ms": summaries["fused-unpack-b4"]["all"]["event_ms"]["p95"]},
            "b8": {"attempt19_p95_ms": 6.790784, "candidate_p95_ms": summaries["fused-unpack-b8"]["all"]["event_ms"]["p95"]},
            "same_window_baseline": False,
        },
        "timeline": {
            "selected_steps": timeline["selected_steps"],
            "step_summary": timeline["step_summary"],
            "stage_by_kind_nonadditive": stage_by_kind,
            "candidate_evidence": timeline["candidate_evidence"],
            "instrumentation_event_ms_overhead_vs_baseline_before": profile_overhead,
        },
        "correctness": correctness,
    }
    (RUN / "analysis-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    with (RUN / "raw-samples.jsonl").open("w") as stream:
        for path in (
            RUN / "baseline-before/all-rank-samples.json",
            RUN / "timeline-run/all-rank-samples.json",
            RUN / "candidates-and-baseline-after/all-rank-samples.json",
            RUN / "best-scaling/all-rank-samples.json",
        ):
            for row in json.loads(path.read_text()):
                stream.write(json.dumps(row) + "\n")
    with (RUN / "paired-samples.jsonl").open("w") as stream:
        for rows in arms.values():
            for row in rows:
                stream.write(json.dumps(row) + "\n")
    print(json.dumps({"budget": result["budget"], "gain": result["candidate_p95_gain"]}, indent=2))


if __name__ == "__main__":
    main()
