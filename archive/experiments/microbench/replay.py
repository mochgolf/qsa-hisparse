#!/usr/bin/env python3
"""CPU-only, setwise LRU replay for old and new QSA rank-0 traces.

Each trace is JSONL in the r1 format: one normal graph-decode row per QSA
layer and absolute position.  The replay reports C4 selection misses and
separately labels prompt-history misses, newly complete generated C4 demand,
initial loading, and incomplete tail tokens.  It does not model CUDA, H2D
latency, overlap, or a real allocator.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict, defaultdict
import json
import math
from pathlib import Path
from statistics import fmean


LAYERS = (3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47)
CAPACITIES = (512, 1024, 1536, 2048, 4096)
FULL_PAGE_TOKENS = 64
COMPRESS_RATIO = 4
C4_SLOTS_PER_PAGE = FULL_PAGE_TOKENS // COMPRESS_RATIO
C4_BYTES = 2 * 256 * 4  # 12 QSA layers are counted separately; one C4/layer.
EXPECTED_OLD_MISSES_512 = 1_579_601
EXPECTED_OLD_DENOMINATOR = 4_706_304


def percentile(values: list[int], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return float(ordered[lower])
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def stats(values: list[int]) -> dict[str, int | float | None]:
    return {
        "n": len(values),
        "mean": fmean(values) if values else None,
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values) if values else None,
    }


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def load_trace(path: Path) -> tuple[dict[int, dict[int, set[int]]], dict[str, object]]:
    by_layer: dict[int, dict[int, set[int]]] = defaultdict(dict)
    compressed_by_position: dict[int, int] = {}
    line_count = 0
    request_ids: set[str] = set()
    for line_no, line in enumerate(path.open(encoding="utf-8"), 1):
        if not line.strip():
            continue
        line_count += 1
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_no}: expected an object")
        if row.get("rank") != 0 or row.get("forward_mode") != "decode" or row.get("cuda_graph") is not True:
            raise ValueError(f"{path}:{line_no}: expected rank-0 graph decode row")
        layer = row.get("layer_id")
        position = row.get("position")
        compressed_length = row.get("compressed_length")
        blocks = row.get("block_indices")
        if layer not in LAYERS or not _is_int(position) or not _is_int(compressed_length):
            raise ValueError(f"{path}:{line_no}: invalid layer/position/compressed_length")
        if not isinstance(blocks, list) or len(blocks) != 512 or any(not _is_int(x) for x in blocks):
            raise ValueError(f"{path}:{line_no}: expected 512 integer block IDs")
        selected = {x for x in blocks if 0 <= x < compressed_length}
        if len(selected) != 512:
            raise ValueError(f"{path}:{line_no}: invalid or duplicate selected block IDs")
        if position in by_layer[layer]:
            raise ValueError(f"{path}:{line_no}: duplicate layer/position")
        by_layer[layer][position] = selected
        previous_length = compressed_by_position.setdefault(position, compressed_length)
        if previous_length != compressed_length:
            raise ValueError(f"{path}:{line_no}: compressed_length differs across layers")
        request_id = row.get("request_id")
        if isinstance(request_id, str):
            request_ids.add(request_id)

    if tuple(sorted(by_layer)) != LAYERS:
        raise ValueError(f"{path}: expected layers {LAYERS}, got {tuple(sorted(by_layer))}")
    positions = sorted(by_layer[LAYERS[0]])
    if not positions or any(b != a + 1 for a, b in zip(positions, positions[1:])):
        raise ValueError(f"{path}: positions must be contiguous")
    if any(sorted(by_layer[layer]) != positions for layer in LAYERS):
        raise ValueError(f"{path}: layer position sets differ")
    return by_layer, {
        "path": str(path.resolve()),
        "rows": line_count,
        "request_ids": sorted(request_ids),
        "layers": list(LAYERS),
        "positions": [positions[0], positions[-1]],
        "position_count": len(positions),
        "prompt_tokens": positions[0],
        "measured_transition_count": max(0, len(positions) - 1),
        "compressed_length_by_position": {
            str(position): compressed_by_position[position] for position in positions
        },
    }


def update_cache(
    cache: OrderedDict[int, None], current: set[int], capacity: int
) -> tuple[set[int], OrderedDict[int, None]]:
    """Apply one protected set update; IDs break ties in ascending order."""

    if len(current) > capacity:
        raise ValueError(f"selected set ({len(current)}) exceeds C4 capacity {capacity}")
    cached = set(cache)
    hits = current & cached
    misses = current - cached
    old_nonselected = [block for block in cache if block not in current]
    keep_count = max(0, capacity - len(current))
    retained = old_nonselected[-keep_count:] if keep_count else []
    # retained is oldest-to-newest; misses precede hits, so hits are more MRU.
    order = retained + sorted(misses) + sorted(hits)
    if len(order) != len(set(order)) or len(order) > capacity:
        raise AssertionError("LRU update lost uniqueness or exceeded capacity")
    return misses, OrderedDict((block, None) for block in order)


def self_check() -> None:
    cache = OrderedDict((x, None) for x in (1, 2))
    misses, cache = update_cache(cache, {1, 3}, 3)
    assert misses == {3}
    assert list(cache) == [2, 3, 1]
    misses, cache = update_cache(cache, {1, 4}, 3)
    assert misses == {4}
    assert list(cache) == [3, 4, 1]
    protected = OrderedDict((x, None) for x in (1, 2, 3))
    misses, protected = update_cache(protected, {1, 2, 4}, 3)
    assert misses == {4} and set(protected) == {1, 2, 4}
    misses, protected = update_cache(protected, {1, 2, 4}, 3)
    assert not misses and set(protected) == {1, 2, 4}
    print("synthetic self-check: PASS (hit priority, deterministic IDs, set protection)")


def _empty_metric() -> dict[str, list[int]]:
    return defaultdict(list)


def _window(values: list[int], start: int, stop: int) -> dict[str, object]:
    return stats(values[start:stop])


def _metric_stats(
    layer_values: dict[int, dict[str, list[int]]],
    token_values: dict[str, list[int]],
    start: int,
    stop: int,
) -> dict[str, object]:
    metrics = ("miss", "history_miss", "generated_complete_miss", "generated_complete_demand")
    return {
        "transition_count": max(0, stop - start),
        "layers": {
            str(layer): {
                metric: _window(values[metric], start, stop)
                for metric in metrics
            }
            for layer, values in layer_values.items()
        },
        "same_token_12_layer_sum": {
            metric: _window(token_values[metric], start, stop) for metric in metrics
        },
        "same_token_12_layer_sum_mib": {
            "selection_miss": _window(
                [value * C4_BYTES / 2**20 for value in token_values["miss"]], start, stop
            ),
            "history_h2d_candidate": _window(
                [value * C4_BYTES / 2**20 for value in token_values["history_miss"]], start, stop
            ),
        },
    }


def _page_report(
    by_layer: dict[int, dict[int, set[int]]], positions: list[int], capacity: int
) -> dict[str, object]:
    page_capacity = capacity // C4_SLOTS_PER_PAGE
    all_steps: list[int] = []
    layer_steps: dict[int, list[int]] = {layer: [] for layer in LAYERS}
    infeasible: list[dict[str, int]] = []
    for position in positions:
        max_pages = 0
        for layer in LAYERS:
            pages = len({block // C4_SLOTS_PER_PAGE for block in by_layer[layer][position]})
            layer_steps[layer].append(pages)
            max_pages = max(max_pages, pages)
        all_steps.append(max_pages)
        if max_pages > page_capacity:
            infeasible.append({"position": position, "required_pages_max_layer": max_pages})
    return {
        "page_size_tokens": FULL_PAGE_TOKENS,
        "c4_slots_per_page": C4_SLOTS_PER_PAGE,
        "same_byte_budget_page_capacity": page_capacity,
        "complete_selected_set_feasible_all_layers": not infeasible,
        "infeasible_step_count": len(infeasible),
        "infeasible_examples": infeasible[:8],
        "max_required_pages_across_layers": max(all_steps) if all_steps else None,
        "required_pages_max_layer_stats": stats(all_steps),
        "per_layer_required_pages": {
            str(layer): stats(values) for layer, values in layer_steps.items()
        },
        "interpretation": (
            "page feasibility only; no page-LRU miss rate or virtual H2D is "
            "computed when a complete selected set does not fit"
        ),
    }


def replay_trace(
    name: str,
    by_layer: dict[int, dict[int, set[int]]],
    trace_info: dict[str, object],
    miss_writer,
) -> dict[str, object]:
    positions = sorted(by_layer[LAYERS[0]])
    warm_position = positions[0]
    measured = positions[1:]
    prompt_tokens = int(trace_info["prompt_tokens"])
    prompt_complete_c4 = prompt_tokens // COMPRESS_RATIO
    initial_by_layer: dict[str, dict[str, int]] = {}
    for layer in LAYERS:
        initial = by_layer[layer][warm_position]
        history = {block for block in initial if block < prompt_complete_c4}
        generated = initial - history
        initial_by_layer[str(layer)] = {
            "selected_c4_blocks": len(initial),
            "prompt_history_blocks": len(history),
            "generated_complete_c4_blocks": len(generated),
        }
        miss_writer.write(
            json.dumps(
                {
                    "trace": name,
                    "event": "initial_load",
                    "capacity": None,
                    "position": warm_position,
                    "layer_id": layer,
                    "initial_load_ids": sorted(initial),
                    "prompt_history_ids": sorted(history),
                    "generated_complete_ids": sorted(generated),
                    "h2d_eligible": False,
                },
                separators=(",", ":"),
            )
            + "\n"
        )

    tails = []
    for position in positions:
        compressed_length = int(trace_info["compressed_length_by_position"][str(position)])
        tail_tokens = position + 1 - COMPRESS_RATIO * compressed_length
        if not 0 <= tail_tokens < COMPRESS_RATIO:
            raise ValueError(
                f"{name}: position {position} has invalid incomplete C4 tail {tail_tokens}"
            )
        tails.append({"position": position, "tail_tokens": tail_tokens})

    capacity_reports: dict[str, object] = {}
    for capacity in CAPACITIES:
        caches = {
            layer: OrderedDict((block, None) for block in sorted(by_layer[layer][warm_position]))
            for layer in LAYERS
        }
        layer_values: dict[int, dict[str, list[int]]] = {
            layer: {metric: [] for metric in (
                "miss", "history_miss", "generated_complete_miss", "generated_complete_demand"
            )}
            for layer in LAYERS
        }
        token_values = {metric: [] for metric in (
            "miss", "history_miss", "generated_complete_miss", "generated_complete_demand"
        )}
        for position in measured:
            totals = {metric: 0 for metric in token_values}
            for layer in LAYERS:
                current = by_layer[layer][position]
                misses, caches[layer] = update_cache(caches[layer], current, capacity)
                history_misses = {block for block in misses if block < prompt_complete_c4}
                generated_misses = misses - history_misses
                generated_demand = {block for block in current if block >= prompt_complete_c4}
                values = {
                    "miss": len(misses),
                    "history_miss": len(history_misses),
                    "generated_complete_miss": len(generated_misses),
                    "generated_complete_demand": len(generated_demand),
                }
                for metric, value in values.items():
                    layer_values[layer][metric].append(value)
                    totals[metric] += value
                miss_writer.write(
                    json.dumps(
                        {
                            "trace": name,
                            "event": "transition",
                            "capacity": capacity,
                            "position": position,
                            "layer_id": layer,
                            "miss_ids": sorted(misses),
                            "history_miss_ids": sorted(history_misses),
                            "generated_complete_miss_ids": sorted(generated_misses),
                            "h2d_eligible": False,
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )
            for metric, value in totals.items():
                token_values[metric].append(value)

        full_stop = len(measured)
        first_stop = min(128, full_stop)
        windows = {
            "full": _metric_stats(layer_values, token_values, 0, full_stop),
            "first_128_steps": _metric_stats(layer_values, token_values, 0, first_stop),
            "remaining_steps": _metric_stats(layer_values, token_values, first_stop, full_stop),
        }
        total_miss = sum(token_values["miss"])
        total_history = sum(token_values["history_miss"])
        total_generated_miss = sum(token_values["generated_complete_miss"])
        total_generated_demand = sum(token_values["generated_complete_demand"])
        capacity_reports[str(capacity)] = {
            "capacity_c4_slots_per_layer": capacity,
            "warm_position": warm_position,
            "measured_positions": [measured[0], measured[-1]] if measured else [],
            "measured_transition_count": len(measured),
            "miss_blocks_total": total_miss,
            "miss_blocks_denominator": len(measured) * len(LAYERS) * 512,
            "miss_fraction": total_miss / (len(measured) * len(LAYERS) * 512)
            if measured
            else None,
            "accounting": {
                "initial_load": {
                    "selected_c4_blocks_total": len(LAYERS) * 512,
                    "prompt_history_blocks_total": sum(
                        item["prompt_history_blocks"] for item in initial_by_layer.values()
                    ),
                    "generated_complete_c4_blocks_total": sum(
                        item["generated_complete_c4_blocks"] for item in initial_by_layer.values()
                    ),
                    "h2d_bytes_measured": 0,
                    "note": "initialization is demand/accounting only; no H2D timing is inferred",
                },
                "prompt_history_miss": {
                    "c4_blocks_total": total_history,
                    "h2d_candidate_bytes": total_history * C4_BYTES,
                },
                "generated_new_complete_c4_demand": {
                    "selected_demand_blocks_total": total_generated_demand,
                    "miss_blocks_total": total_generated_miss,
                    "h2d_candidate_bytes": 0,
                    "note": "new generated KV locality and later writeback/re-fetch are not modeled; no H2D is inferred here",
                },
            },
            "windows": windows,
            "page_feasibility": _page_report(by_layer, positions, capacity),
        }

    identity = None
    if name.lower() in {"old", "old38k", "old-38k", "legacy"}:
        old = capacity_reports["512"]
        identity = {
            "capacity": 512,
            "misses": old["miss_blocks_total"],
            "denominator": old["miss_blocks_denominator"],
            "expected_misses": EXPECTED_OLD_MISSES_512,
            "expected_denominator": EXPECTED_OLD_DENOMINATOR,
            "passed": old["miss_blocks_total"] == EXPECTED_OLD_MISSES_512
            and old["miss_blocks_denominator"] == EXPECTED_OLD_DENOMINATOR,
        }
        if not identity["passed"]:
            raise AssertionError(f"old 1x identity failed: {identity}")

    tail_tokens = [item["tail_tokens"] for item in tails]
    return {
        "trace": trace_info,
        "policy": {
            "capacity_unit": "compressed QSA C4 block slots per layer",
            "cache_scope": "one independent setwise LRU per QSA layer",
            "update": "current selected set protected; retained old entries, then misses, then hits",
            "tie_break": "ascending C4 block ID within miss and hit groups",
            "warmup": "first position initialized; no empty-cache miss counted",
            "prompt_history_boundary": f"C4 block ID < prompt_tokens//4 ({prompt_complete_c4})",
            "c4_bytes_per_layer_block": C4_BYTES,
        },
        "initial_load_by_layer": initial_by_layer,
        "tail_incomplete_c4": {
            "formula": "(absolute_position + 1) - 4 * floor((absolute_position + 1) / 4)",
            "observed_tail_tokens_by_position": tail_tokens,
            "positions_with_incomplete_tail": sum(value > 0 for value in tail_tokens),
            "tail_tokens_total": sum(tail_tokens),
            "tail_tokens_max": max(tail_tokens) if tail_tokens else 0,
            "h2d_bytes_measured": 0,
            "note": "incomplete C4 tails are reported separately and never virtualized as H2D",
        },
        "old_1x_identity": identity,
        "capacities": capacity_reports,
    }


def parse_trace_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("trace must be NAME=PATH")
    name, raw_path = value.split("=", 1)
    if not name or not raw_path:
        raise argparse.ArgumentTypeError("trace must be NAME=PATH")
    return name, Path(raw_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--trace", action="append", type=parse_trace_spec, metavar="NAME=PATH")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--source-commit", default="6f25e04479152cfe68e76a16b5c00236a6f360f8")
    parser.add_argument(
        "--source-path",
        default="<SGLANG_WORKTREE>",
    )
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return 0
    if not args.trace:
        parser.error("at least one --trace NAME=PATH is required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    miss_path = args.output_dir / "miss-ids.jsonl"
    summaries = {}
    with miss_path.open("w", encoding="utf-8") as miss_writer:
        for name, path in args.trace:
            if name in summaries:
                raise SystemExit(f"duplicate trace name: {name}")
            by_layer, info = load_trace(path)
            summaries[name] = replay_trace(name, by_layer, info, miss_writer)
    output = {
        "schema_version": "qwen38-hisparse-cpu-replay-v2",
        "kind": "CPU trace replay; not a GPU, PCIe, H2D, or throughput benchmark",
        "source": {"source_commit": args.source_commit, "source_path": args.source_path},
        "traces": summaries,
        "miss_ids_path": str(miss_path.resolve()),
        "miss_id_semantics": {
            "history_miss_ids": "prompt-history C4 IDs and the only H2D microbenchmark candidates",
            "generated_complete_miss_ids": "generated-complete IDs in this replay; new-KV locality and later writeback/re-fetch are not modeled, so no H2D is inferred",
            "initial_load_ids": "first selected set per layer; initialization only, no H2D timing",
        },
        "limitations": [
            "setwise LRU accounting only; no CUDA resolve, NUMA, PCIe, copy scheduling, or overlap",
            "C4 bytes are layout arithmetic, not measured transfer bytes",
            "64-token page output is complete-selected-set feasibility only; no page-LRU miss rate",
            "trace input must contain normal rank-0 graph-decode rows with 512 unique valid IDs",
        ],
    }
    summary_path = args.output_dir / "replay-summary.json"
    summary_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    identity = {
        name: report["old_1x_identity"]
        for name, report in summaries.items()
        if report["old_1x_identity"] is not None
    }
    print(json.dumps({"summary": str(summary_path), "miss_ids": str(miss_path), "old_1x_identity": identity}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
