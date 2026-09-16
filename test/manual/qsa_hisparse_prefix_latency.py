#!/usr/bin/env python3
"""Measure complete one-token HTTP requests, including host prefix restoration.

This is a timing and reuse check. Exact generated-token comparisons belong to
qsa_hisparse_prefix_acceptance.py with the deterministic serving configuration.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

from qsa_hisparse_prefix_acceptance import gpu_snapshot, request_json, write_json


def measure(url, ids, salt):
    start = time.monotonic()
    response = request_json(
        url,
        "/generate",
        {
            "input_ids": ids,
            "cache_salt": salt,
            "sampling_params": {"temperature": 0, "max_new_tokens": 1},
        },
    )
    elapsed = time.monotonic() - start
    info = response["meta_info"]
    assert info["completion_tokens"] == 1, response
    assert info["finish_reason"]["type"] not in ("abort", "error"), response
    return {
        "wall_seconds": elapsed,
        "cached_tokens": info.get("cached_tokens", 0),
        "prompt_tokens": info["prompt_tokens"],
        "output_ids": response["output_ids"],
        "gpu_after": gpu_snapshot(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8081")
    parser.add_argument("--fixtures", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--lengths", nargs="+", type=int, default=[8192, 65536, 262016])
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    fixtures = json.loads(args.fixtures.read_text())["cases"]
    namespace = f"prefix-latency-{time.time_ns()}"
    report = {"passed": False, "gpu_before": gpu_snapshot(), "cases": []}
    try:
        for length in args.lengths:
            case = next(c for c in fixtures if c["name"] == f"prefix-{length}-copy")
            row = {"prefix_length": length, "pairs": []}
            report["cases"].append(row)
            for repeat in range(args.repeats):
                salt = f"{namespace}-{length}-{repeat}"
                cold = measure(args.url, case["input_ids"], salt)
                warm = measure(args.url, case["input_ids"], salt)
                row["pairs"].append({"cold": cold, "warm": warm})
                assert cold["cached_tokens"] == 0, cold
                assert warm["cached_tokens"] == length, warm
                print(
                    f"{length} tokens: cold {cold['wall_seconds']:.3f}s, "
                    f"warm {warm['wall_seconds']:.3f}s",
                    flush=True,
                )
                write_json(args.output, report)
            for kind in ("cold", "warm"):
                row[f"median_{kind}_seconds"] = statistics.median(
                    pair[kind]["wall_seconds"] for pair in row["pairs"]
                )
            row["observed_ratio"] = (
                row["median_cold_seconds"] / row["median_warm_seconds"]
            )
        report["gpu_after"] = gpu_snapshot()
        report["passed"] = True
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        write_json(args.output, report)


if __name__ == "__main__":
    main()
