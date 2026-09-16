#!/usr/bin/env python3
"""Freeze token fixtures and qualify a live QSA prefix cache through its HTTP API.

No model implementation is imported during measurement. Run freeze before
candidate measurements; baseline records repeated uncached outputs, while
qualify compares isolated cold, seeded warm, and branching requests.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import subprocess
import time
import urllib.request
from pathlib import Path


def request_json(base, endpoint, payload=None, timeout=1800):
    encoded = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        base.rstrip("/") + endpoint,
        data=encoded,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read()
    return json.loads(raw) if raw else None


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def freeze(args):
    from transformers import AutoTokenizer

    if args.fixtures.exists():
        raise RuntimeError("Refusing to overwrite frozen fixtures")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
    header = encode("Reference document. Keep the following records unchanged.\n")
    record = encode("Record: amber is 17, cyan is 25, and violet is 42.\n")
    footer = encode("\nEnd of reference document.\n")
    suffixes = {
        "copy": encode("Question: Reply with the single word READY.\nAnswer:"),
        "arithmetic": encode(
            "Question: What is 17 plus 25? Reply with only the number.\nAnswer:"
        ),
    }
    cases = []
    for length in args.lengths:
        count = length - len(header) - len(footer)
        if count < 0:
            raise ValueError("Fixture prefix is too short")
        prefix = (
            header
            + (record * ((count + len(record) - 1) // len(record)))[:count]
            + footer
        )
        assert len(prefix) == length
        for name, suffix in suffixes.items():
            cases.append(
                {
                    "name": f"prefix-{length}-{name}",
                    "group": f"prefix-{length}",
                    "prefix_length": length,
                    "prefix_ids": prefix,
                    "input_ids": prefix + suffix,
                    "max_new_tokens": 32,
                }
            )
    token_file = Path(args.model) / "tokenizer.json"
    data = {
        "format": 1,
        "created_at": time.time(),
        "tokenizer_sha256": hashlib.sha256(token_file.read_bytes()).hexdigest(),
        "acceptance": "Exact output token IDs; nonzero warm cached tokens; zero salted cold cached tokens.",
        "cases": cases,
    }
    write_json(args.fixtures, data)
    print(f"Frozen {len(cases)} cases in {args.fixtures}", flush=True)


def generate(base, ids, salt, count=32):
    start = time.monotonic()
    data = request_json(
        base,
        "/generate",
        {
            "input_ids": ids,
            "cache_salt": salt,
            "sampling_params": {"temperature": 0, "max_new_tokens": count},
            "return_logprob": True,
            "logprob_start_len": -1,
            "top_logprobs_num": 5,
        },
    )
    if not isinstance(data, dict) or not isinstance(data.get("output_ids"), list):
        raise RuntimeError(f"Invalid generation response: {data}")
    info = data.get("meta_info", {})
    if info.get("finish_reason", {}).get("type") in {"abort", "error"}:
        raise RuntimeError(f"Generation failed: {info}")
    entered = info.get("forward_entry_time")
    prefilled = info.get("prefill_finished_time")
    return {
        "output_ids": data["output_ids"],
        "text": data.get("text"),
        "prompt_tokens": info.get("prompt_tokens"),
        "cached_tokens": info.get("cached_tokens", 0),
        "prefill_seconds": None
        if entered is None or prefilled is None
        else prefilled - entered,
        "wall_seconds": time.monotonic() - start,
        "meta_info": info,
    }


def gpu_snapshot():
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return [
        dict(zip(("gpu", "used_mib", "free_mib"), map(int, line.split(","))))
        for line in result.stdout.splitlines()
    ]


def measure(args):
    manifest = json.loads(args.fixtures.read_text())
    cases = [
        case for case in manifest["cases"] if case["prefix_length"] <= args.max_prefix
    ]
    if args.min_prefix:
        cases = [case for case in cases if case["prefix_length"] >= args.min_prefix]
    if not cases:
        raise ValueError("No cases selected")
    namespace = f"qsa-accept-{time.time_ns()}"
    report = {
        "phase": args.command,
        "fixtures_sha256": hashlib.sha256(args.fixtures.read_bytes()).hexdigest(),
        "started_at": time.time(),
        "base_url": args.url,
        "gpu_before": gpu_snapshot(),
        "cases": [],
        "output_mismatches": [],
        "passed": False,
    }

    def mismatch(message, row=None):
        report["output_mismatches"].append(message)
        if row is not None:
            row.setdefault("output_mismatches", []).append(message)
        if not args.continue_on_mismatch:
            raise AssertionError(message)

    write_json(args.output, report)
    try:
        for index, case in enumerate(cases):
            print(f"{args.command}: {case['name']}", flush=True)
            row = {"name": case["name"], "prefix_length": case["prefix_length"]}
            report["cases"].append(row)
            for repeat in range(2):
                row[f"cold_{repeat}"] = generate(
                    args.url,
                    case["input_ids"],
                    f"{namespace}-cold-{index}-{repeat}",
                    case["max_new_tokens"],
                )
                if row[f"cold_{repeat}"]["cached_tokens"] != 0:
                    raise AssertionError("Unique cache salt unexpectedly hit a prefix")
            if row["cold_0"]["output_ids"] != row["cold_1"]["output_ids"]:
                mismatch(f"Cold repetitions differ: {case['name']}", row)
            if args.command == "qualify":
                salt = f"{namespace}-warm-{case['group']}"
                row["seed"] = generate(args.url, case["prefix_ids"], salt, 1)
                row["warm"] = generate(
                    args.url, case["input_ids"], salt, case["max_new_tokens"]
                )
                if row["warm"]["cached_tokens"] <= 0:
                    raise AssertionError(f"No real prefix hit: {case['name']}")
                if row["warm"]["output_ids"] != row["cold_0"]["output_ids"]:
                    mismatch(f"Warm and cold outputs differ: {case['name']}", row)
                row["repeated_warm"] = generate(
                    args.url, case["input_ids"], salt, case["max_new_tokens"]
                )
                if row["repeated_warm"]["output_ids"] != row["cold_0"]["output_ids"]:
                    mismatch(f"Repeated warm output differs: {case['name']}", row)
            row["passed"] = not row.get("output_mismatches")
            row["gpu_after"] = gpu_snapshot()
            write_json(args.output, report)
            print(
                f"{'Passed' if row['passed'] else 'Mismatch'} {case['name']}; "
                f"warm reused {row.get('warm', {}).get('cached_tokens', 0)} tokens",
                flush=True,
            )
        if args.command == "qualify" and args.concurrency:
            # Use the longest selected prefix that leaves room for decode.
            case = max(
                (c for c in cases if len(c["input_ids"]) < 260000),
                key=lambda c: c["prefix_length"],
            )
            salt = f"{namespace}-concurrent"
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=args.concurrency
            ) as pool:
                cold = list(
                    pool.map(
                        lambda i: generate(
                            args.url, case["input_ids"], f"{namespace}-ccold-{i}", 128
                        ),
                        range(args.concurrency),
                    )
                )
            # Populate immediately before the warm batch: cold controls may
            # legitimately evict an older seed under the configured budget.
            seed = generate(args.url, case["prefix_ids"], salt, 1)
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=args.concurrency
            ) as pool:
                warm = list(
                    pool.map(
                        lambda i: generate(args.url, case["input_ids"], salt, 128),
                        range(args.concurrency),
                    )
                )
            report["concurrent"] = {
                "count": args.concurrency,
                "case": case["name"],
                "seed": seed,
                "cold": cold,
                "warm": warm,
            }
            for a, b in zip(cold, warm):
                if a["cached_tokens"] != 0 or b["cached_tokens"] <= 0:
                    raise AssertionError("Concurrent cold/warm cache accounting failed")
                if a["output_ids"] != b["output_ids"]:
                    mismatch("Concurrent cold/warm output IDs differ")
        report["gpu_after"] = gpu_snapshot()
        if report["output_mismatches"]:
            raise AssertionError(
                f"{len(report['output_mismatches'])} exact-output comparisons failed; "
                "continued measurements do not waive these failures"
            )
        report["passed"] = True
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["finished_at"] = time.time()
        write_json(args.output, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["freeze", "baseline", "qualify"])
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument(
        "--lengths",
        nargs="+",
        type=int,
        default=[64, 128, 2048, 4096, 8192, 32768, 65536, 262016],
    )
    parser.add_argument("--url", default="http://127.0.0.1:8082")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--min-prefix", type=int, default=0)
    parser.add_argument("--max-prefix", type=int, default=8192)
    parser.add_argument("--concurrency", type=int, choices=[0, 2, 8], default=0)
    parser.add_argument(
        "--continue-on-mismatch",
        action="store_true",
        help="Collect remaining measurements, retaining a failed report and exit status",
    )
    args = parser.parse_args()
    if args.command == "freeze":
        if not args.model:
            parser.error("freeze requires --model")
        freeze(args)
    else:
        if not args.output:
            parser.error("measurements require --output")
        measure(args)


if __name__ == "__main__":
    main()
