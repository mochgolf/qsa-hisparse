#!/usr/bin/env python3
"""Exercise host-prefix flush, abort, logprobs and reuse against an idle server.

This sends real requests and flushes the server's cache. It checks lifecycle
and accounting, not a numerical tolerance or deterministic model output.
"""

import argparse
import concurrent.futures
import json
import time
import urllib.request

from qsa_hisparse_prefix_acceptance import generate, request_json, write_json


def main():
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:30000")
    parser.add_argument("--fixtures", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    case = next(
        c
        for c in json.loads(args.fixtures.read_text())["cases"]
        if c["name"] == "prefix-8192-copy"
    )
    salt = f"lifecycle-{time.time_ns()}"
    report = {"passed": False}
    try:
        report["seed"] = generate(args.url, case["prefix_ids"], salt, 1)
        report["warm"] = generate(args.url, case["input_ids"], salt, 2)
        assert report["warm"]["cached_tokens"] == 8192
        request = urllib.request.Request(
            args.url + "/flush_cache", data=b"", method="POST"
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            report["flush"] = {
                "status": response.status,
                "body": response.read().decode(),
            }
        report["after_flush"] = generate(args.url, case["input_ids"], salt, 2)
        assert report["after_flush"]["cached_tokens"] == 0
        report["after_reseed"] = generate(args.url, case["input_ids"], salt, 2)
        assert report["after_reseed"]["cached_tokens"] == 8192
        print("Flush, miss and refill passed", flush=True)

        rid = f"abort-{time.time_ns()}"
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                request_json,
                args.url,
                "/generate",
                {
                    "rid": rid,
                    "input_ids": case["input_ids"],
                    "cache_salt": salt,
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 4096,
                        "ignore_eos": True,
                    },
                },
            )
            # Only coordinates this deliberate cancellation; no retry on failure.
            time.sleep(0.5)
            report["abort_result"] = request_json(
                args.url, "/abort_request", {"rid": rid}
            )
            report["aborted_response"] = future.result(timeout=60)
        assert (
            report["aborted_response"]["meta_info"]["finish_reason"]["type"] == "abort"
        )
        report["after_abort"] = generate(args.url, case["input_ids"], salt, 2)
        assert report["after_abort"]["cached_tokens"] == 8192
        print("Abort and subsequent reuse passed", flush=True)

        report["logprob"] = []
        for start in [0, 64, 2048, 4096]:
            result = request_json(
                args.url,
                "/generate",
                {
                    "input_ids": case["input_ids"],
                    "cache_salt": salt,
                    "sampling_params": {"temperature": 0, "max_new_tokens": 1},
                    "return_logprob": True,
                    "logprob_start_len": start,
                },
            )
            info = result["meta_info"]
            assert info["cached_tokens"] <= start
            assert len(info["input_token_logprobs"]) == len(case["input_ids"]) - start
            report["logprob"].append({"start": start, "meta_info": info})
        print("Requested input logprobs preserved", flush=True)
        report["passed"] = True
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        write_json(args.output, report)


if __name__ == "__main__":
    main()
