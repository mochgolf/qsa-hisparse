#!/usr/bin/env python3
"""Qualify actual B2/B8 graph execution against repeated cold token oracles.

Ignore EOS for a fixed 128 steps so short answers cannot finish before all
restored requests enter decode. Per-rank event logs must prove the graph sizes.
"""

import argparse
import concurrent.futures
import hashlib
import json
import threading
import time
from pathlib import Path

from qsa_hisparse_prefix_acceptance import request_json, write_json


def generate(url, ids, salt, rid, count=128):
    result = request_json(
        url,
        "/generate",
        {
            "rid": rid,
            "input_ids": ids,
            "cache_salt": salt,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": count,
                "ignore_eos": True,
            },
        },
    )
    assert len(result["output_ids"]) == count, result["meta_info"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:30000")
    parser.add_argument("--fixtures", required=True, type=Path)
    parser.add_argument("--events", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prefix-length", type=int, default=65536)
    args = parser.parse_args()
    case = next(
        c
        for c in json.loads(args.fixtures.read_text())["cases"]
        if c["name"] == f"prefix-{args.prefix_length}-copy"
    )
    namespace = f"actual-batch-{time.time_ns()}"
    report = {
        "passed": False,
        "namespace": namespace,
        "prefix_length": args.prefix_length,
        "fixtures_sha256": hashlib.sha256(args.fixtures.read_bytes()).hexdigest(),
        "cold": [],
        "warm": {},
        "graph_evidence": {},
    }
    try:
        for repeat in range(2):
            result = generate(
                args.url,
                case["input_ids"],
                f"{namespace}-cold-{repeat}",
                f"{namespace}-cold-{repeat}",
            )
            assert result["meta_info"]["cached_tokens"] == 0
            report["cold"].append(result)
            write_json(args.output, report)
        reference = report["cold"][0]["output_ids"]
        assert reference == report["cold"][1]["output_ids"], "Cold oracle differs"
        salt = f"{namespace}-shared"
        report["seed"] = generate(
            args.url, case["prefix_ids"], salt, f"{namespace}-seed", 1
        )
        for batch in (2, 8):
            barrier = threading.Barrier(batch)

            def submit(index):
                barrier.wait(timeout=30)
                return generate(
                    args.url,
                    case["input_ids"],
                    salt,
                    f"{namespace}-warm{batch}-{index}",
                )

            with concurrent.futures.ThreadPoolExecutor(max_workers=batch) as pool:
                results = list(pool.map(submit, range(batch)))
            report["warm"][str(batch)] = results
            for result in results:
                assert result["meta_info"]["cached_tokens"] == args.prefix_length
                assert result["output_ids"] == reference, (
                    f"B{batch} token oracle differs"
                )
            print(
                f"B{batch}: all 128 output IDs exact; all prefixes reused", flush=True
            )
            write_json(args.output, report)
        paths = sorted(args.events.glob("rank-*.jsonl"))
        assert len(paths) == 2, "Expected TP2 ledger files"
        for path in paths:
            found = set()
            # A final unrelated event can still be writing; inspect complete lines.
            for line in path.read_text().splitlines(keepends=True):
                if not line.endswith("\n"):
                    continue
                row = json.loads(line)
                batch = row.get("batch_size")
                if row["event"] != "graph_replay" or batch not in (2, 8):
                    continue
                leases = row["ordered_leases"]
                if len(leases) == batch and all(
                    lease[2].startswith(f"{namespace}-warm{batch}-") for lease in leases
                ):
                    found.add(batch)
            report["graph_evidence"][path.name] = sorted(found)
            assert found == {2, 8}, f"Missing actual B2/B8 graph evidence: {path}"
        report["passed"] = True
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        write_json(args.output, report)


if __name__ == "__main__":
    main()
