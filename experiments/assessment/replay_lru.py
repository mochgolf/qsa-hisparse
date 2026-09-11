#!/usr/bin/env python3
"""Replay QSA compressed-block selections with a deterministic set-at-once LRU.

This is a CPU-only accounting model.  It does not model CUDA, PCIe, copy time,
or a real HiSparse allocator.  One independent cache is used per QSA layer.
The first trace position warms each layer; measured steps are the 766 following
positions.  For each step, current selections are handled as one set:

* old entries outside the current set are retained from oldest to newest;
* new misses are inserted in ascending block-ID order;
* hits are inserted after misses (therefore more MRU), also by block ID.

The current set is protected while the update is made, so every selected block
is resident after the update.  This makes capacity 512 exactly the adjacent
set-difference replay and provides the expected 1,579,601/4,706,304 check.
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict, defaultdict
from pathlib import Path
from statistics import fmean


LAYERS = (3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47)
CAPACITIES = (512, 1024, 1536, 2048, 4096)
FULL_PAGE_TOKENS = 64
COMPRESS_RATIO = 4
COMPRESSED_SLOTS_PER_PAGE = FULL_PAGE_TOKENS // COMPRESS_RATIO
EXPECTED_MISSES_512 = 1_579_601
EXPECTED_BLOCKS = 4_706_304


def percentile(values: list[int], fraction: float) -> float:
    """Linear-interpolated percentile, matching the QSA r1 analyzer."""
    if not values:
        raise ValueError("percentile of empty values")
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def stats(values: list[int]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "min": min(values),
        "mean": fmean(values),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values),
    }


def load_trace(path: Path) -> dict[int, dict[int, set[int]]]:
    by_layer: dict[int, dict[int, set[int]]] = defaultdict(dict)
    line_count = 0
    with path.open() as source:
        for line_no, line in enumerate(source, 1):
            if not line.strip():
                continue
            line_count += 1
            row = json.loads(line)
            if (
                row.get("rank") != 0
                or row.get("forward_mode") != "decode"
                or row.get("cuda_graph") is not True
            ):
                raise ValueError(f"line {line_no}: not a rank0 graph decode row")
            if row.get("layer_id") not in LAYERS:
                raise ValueError(f"line {line_no}: unexpected layer")
            blocks = row.get("block_indices")
            if not isinstance(blocks, list) or len(blocks) != 512:
                raise ValueError(f"line {line_no}: expected 512 block IDs")
            compressed_length = row.get("compressed_length")
            selected = {block for block in blocks if 0 <= block < compressed_length}
            if len(selected) != 512:
                raise ValueError(f"line {line_no}: invalid or duplicate block IDs")
            position = row.get("position")
            if position in by_layer[row["layer_id"]]:
                raise ValueError(f"line {line_no}: duplicate layer/position")
            by_layer[row["layer_id"]][position] = selected

    if tuple(sorted(by_layer)) != LAYERS:
        raise ValueError(f"expected layers {LAYERS}, got {tuple(sorted(by_layer))}")
    positions = sorted(by_layer[LAYERS[0]])
    if len(positions) != 767 or positions[-1] - positions[0] != 766:
        raise ValueError("expected 767 contiguous trace positions")
    if any(sorted(by_layer[layer]) != positions for layer in LAYERS):
        raise ValueError("layer position sets differ")
    if line_count != 9204:
        raise ValueError(f"expected 9204 rank0 rows, got {line_count}")
    return by_layer


def replay(
    by_layer: dict[int, dict[int, set[int]]], capacity: int
) -> tuple[list[int], list[int], list[int]]:
    positions = sorted(by_layer[LAYERS[0]])
    # Warm each layer with the first full selected set.
    caches = {
        layer: OrderedDict((block, None) for block in sorted(by_layer[layer][positions[0]]))
        for layer in LAYERS
    }
    token_misses: list[int] = []
    token_miss_pages: list[int] = []
    layer_misses: list[int] = []

    for position in positions[1:]:
        total_miss = 0
        total_miss_pages = 0
        for layer in LAYERS:
            current = by_layer[layer][position]
            cache = caches[layer]
            cached = set(cache)
            hits = current & cached
            misses = current - cached
            old_nonselected = [block for block in cache if block not in current]
            keep_count = max(0, capacity - len(current))
            retained = old_nonselected[-keep_count:] if keep_count else []
            order = retained + sorted(misses) + sorted(hits)
            if len(order) != len(set(order)) or len(order) > capacity:
                raise AssertionError("LRU update lost uniqueness or exceeded capacity")
            caches[layer] = OrderedDict((block, None) for block in order)
            total_miss += len(misses)
            total_miss_pages += len({block // COMPRESSED_SLOTS_PER_PAGE for block in misses})
            layer_misses.append(len(misses))
        token_misses.append(total_miss)
        token_miss_pages.append(total_miss_pages)

    return token_misses, token_miss_pages, layer_misses


def page_demand(by_layer: dict[int, dict[int, set[int]]]) -> dict[str, object]:
    positions = sorted(by_layer[LAYERS[0]])[1:]
    per_layer_step = []
    per_token_sum = []
    per_token_max = []
    per_token_union = []
    for position in positions:
        counts = [
            len({block // COMPRESSED_SLOTS_PER_PAGE for block in by_layer[layer][position]})
            for layer in LAYERS
        ]
        per_layer_step.extend(counts)
        per_token_sum.append(sum(counts))
        per_token_max.append(max(counts))
        per_token_union.append(
            len(
                set().union(
                    *(
                        {block // COMPRESSED_SLOTS_PER_PAGE for block in by_layer[layer][position]}
                        for layer in LAYERS
                    )
                )
            )
        )
    return {
        "geometry": {
            "full_page_tokens": FULL_PAGE_TOKENS,
            "compress_ratio": COMPRESS_RATIO,
            "compressed_slots_per_full_page": COMPRESSED_SLOTS_PER_PAGE,
            "capacity_pages_for_compressed_slots": {
                str(capacity): capacity // COMPRESSED_SLOTS_PER_PAGE
                for capacity in CAPACITIES
            },
        },
        "selected_pages_per_layer_step": stats(per_layer_step),
        "selected_pages_per_token_sum_of_12_layers": stats(per_token_sum),
        "selected_pages_per_token_max_layer": stats(per_token_max),
        "selected_pages_per_token_union_across_layers": stats(per_token_union),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    # A hit must outlive a newly fetched miss when later selections need room.
    small = {layer: dict(enumerate(({1, 2, 3}, {2, 4}, {5, 6}, {2}))) for layer in LAYERS}
    assert replay(small, 3)[0] == [12, 24, 0]
    protected = {layer: dict(enumerate(({1, 2, 3}, {1, 2, 4}, {1, 2, 4}))) for layer in LAYERS}
    assert replay(protected, 3)[0] == [12, 0]

    by_layer = load_trace(args.trace)
    positions = sorted(by_layer[LAYERS[0]])
    results = {}
    for capacity in CAPACITIES:
        token_misses, token_miss_pages, layer_misses = replay(by_layer, capacity)
        results[str(capacity)] = {
            "warm_position": positions[0],
            "measured_positions": [positions[1], positions[-1]],
            "measured_tokens": len(token_misses),
            "selected_blocks_per_token_12_layers": 12 * 512,
            "miss_blocks_total": sum(token_misses),
            "miss_blocks_denominator": len(token_misses) * 12 * 512,
            "miss_fraction": sum(token_misses) / (len(token_misses) * 12 * 512),
            "miss_blocks_per_token_12_layers": stats(token_misses),
            "miss_blocks_per_layer_step": stats(layer_misses),
            "miss_pages_per_token_sum_of_12_layers": stats(token_miss_pages),
        }

    check = {
        "synthetic_hit_priority_and_selection_protection": "passed",
        "capacity_512_misses": results["512"]["miss_blocks_total"],
        "capacity_512_denominator": results["512"]["miss_blocks_denominator"],
        "expected": {
            "misses": EXPECTED_MISSES_512,
            "denominator": EXPECTED_BLOCKS,
        },
        "passed": (
            results["512"]["miss_blocks_total"] == EXPECTED_MISSES_512
            and results["512"]["miss_blocks_denominator"] == EXPECTED_BLOCKS
        ),
    }
    if not check["passed"]:
        raise AssertionError(f"capacity-512 check failed: {check}")

    output = {
        "schema_version": "qwen38-hisparse-cpu-lru-v1",
        "kind": "CPU trace replay; not a GPU or transfer benchmark",
        "trace": {
            "source_commit": "6f25e04479152cfe68e76a16b5c00236a6f360f8",
            "path": str(args.trace.resolve()),
            "rank": 0,
            "layers": list(LAYERS),
            "positions": [positions[0], positions[-1]],
            "position_count": len(positions),
            "warm_position": positions[0],
            "measured_transition_count": len(positions) - 1,
            "rows": len(positions) * len(LAYERS),
        },
        "policy": {
            "capacity_unit": "compressed QSA block slots per layer",
            "cache_scope": "one independent LRU per QSA layer",
            "update": "set-at-once; current selection protected; hits more MRU than new misses",
            "tie_break": "ascending block ID within hit and miss groups",
            "warmup": "first position preloaded; no empty-cache miss counted",
        },
        "self_check": check,
        "capacities": results,
        "page_demand": page_demand(by_layer),
        "limitations": [
            "single deterministic 38185-token prompt and 768-token completion",
            "does not model host/device bandwidth, copy scheduling, or overlap",
            "page counts use QSA full page 64 tokens and ratio 4, so 16 compressed slots/page",
            "one QSA trace does not establish quality, concurrency, or steady service throughput",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "self_check": check, "capacities": results}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
