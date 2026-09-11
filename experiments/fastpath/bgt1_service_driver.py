#!/usr/bin/env python3
"""Guarded short B2/B8 service A/B for the HiSparse ragged-FA2 path."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import os
from pathlib import Path
import re
import statistics
import sys
import threading
import time
from typing import Any, Mapping, Sequence

import psutil


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = Path(os.environ.get("QSA_HISPARSE_BASE_SOURCE", ROOT / "sources/sglang"))
CANDIDATE = Path(
    os.environ.get("QSA_HISPARSE_CANDIDATE_SOURCE", ROOT / "sources/sglang-hisparse")
)
BASE_COMMIT = "5401b6e65925422acd3c9a015935267970acbf4b"
CANDIDATE_COMMIT = "687a39bbe0af00240fecb61872e8643ccb076cb7"
B1_DRIVER = HERE / "b1_driver.py"
B1_RESULT = Path(os.environ.get("QSA_HISPARSE_B1_RESULT", "<B1_RESULT_JSON>"))
COMPONENT_RESULT = HERE / "component-execution-01/manifest.json"
REQUEST = Path(os.environ.get("QSA_HISPARSE_REQUEST_JSON", "<REQUEST_JSON>"))
REQUEST_METADATA = Path(
    os.environ.get("QSA_HISPARSE_REQUEST_METADATA", "<REQUEST_METADATA_JSON>")
)
CHAT_TAIL = 9
PROMPT_LENGTHS = {
    2: (4096, 4099),
    8: (4096, 4097, 4098, 4099, 4096, 4097, 4098, 4099),
}
OUTPUT = 256


def load(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("bgt1_service_b1", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


b1 = load(B1_DRIVER)
v3 = b1.v3


class ValidationError(RuntimeError):
    pass


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def request_ids() -> list[int]:
    value = json.loads(REQUEST.read_text())
    metadata = json.loads(REQUEST_METADATA.read_text())
    ids = value.get("input_ids")
    if not isinstance(ids, list) or len(ids) != 261120:
        raise ValidationError("frozen request-A token count differs")
    if metadata.get("token_splice", {}).get("chat_tail_tokens") != CHAT_TAIL:
        raise ValidationError("frozen request-A chat tail differs")
    return ids


def payload(source_ids: Sequence[int], length: int, rid: str) -> dict[str, Any]:
    tail = list(source_ids[-CHAT_TAIL:])
    return {
        "rid": rid,
        "input_ids": list(source_ids[: length - len(tail)]) + tail,
        "sampling_params": {
            "temperature": 0,
            "top_p": 1.0,
            "top_k": 1,
            "max_new_tokens": OUTPUT,
            "ignore_eos": True,
        },
        "stream": True,
        "log_metrics": True,
    }


def stream_one(
    body: Mapping[str, Any],
    directory: Path,
    label: str,
    prompt_length: int,
    timeout: float,
    barrier: threading.Barrier,
) -> dict[str, Any]:
    barrier.wait()
    started_ns = time.monotonic_ns()
    result = v3.request_stream(
        30001, body, directory, label, prompt_length, OUTPUT, timeout
    )
    rows = [
        json.loads(line)
        for line in Path(result["events_path"]).read_text().splitlines()
    ]
    output_ids: list[int] = []
    times_ns: list[int] = []
    for row in rows:
        observed = v3.runner._event_output_ids(row)
        if observed is None:
            continue
        before = len(output_ids)
        output_ids = v3.runner._merge_stream_output_ids(output_ids, observed)
        if len(output_ids) != before + 1:
            raise ValidationError(f"{label}: SSE did not add exactly one token")
        times_ns.append(started_ns + round(float(row["_elapsed_s"]) * 1e9))
    if len(output_ids) != OUTPUT or len(times_ns) != OUTPUT:
        raise ValidationError(f"{label}: incomplete output timeline")
    result.update(
        output_ids=output_ids,
        times_ns=times_ns,
        steady_65_256_tok_s=(OUTPUT - 65) * 1e9 / (times_ns[-1] - times_ns[64]),
    )
    return result


def run_group(
    batch: int,
    source_ids: Sequence[int],
    directory: Path,
    arm: str,
    timeout: float,
) -> dict[str, Any]:
    lengths = PROMPT_LENGTHS[batch]
    barrier = threading.Barrier(batch)
    started_ns = time.monotonic_ns()
    with ThreadPoolExecutor(max_workers=batch) as pool:
        futures = [
            pool.submit(
                stream_one,
                payload(source_ids, length, f"bgt1-{arm}-b{batch}-{index}"),
                directory,
                f"b{batch}-{index}",
                length,
                timeout,
                barrier,
            )
            for index, length in enumerate(lengths)
        ]
        rows = [future.result() for future in futures]
    ended_ns = time.monotonic_ns()
    common_start = max(row["times_ns"][64] for row in rows)
    common_end = min(row["times_ns"][-1] for row in rows)
    if common_end <= common_start:
        raise ValidationError(f"B{batch}: no common steady decode interval")
    common_tokens = sum(
        sum(common_start < point <= common_end for point in row["times_ns"])
        for row in rows
    )
    if common_tokens < batch * 64:
        raise ValidationError(f"B{batch}: common interval is too short")
    result = {
        "batch": batch,
        "prompt_lengths": list(lengths),
        "requests_complete": len(rows),
        "mean_request_steady_tok_s": statistics.fmean(
            row["steady_65_256_tok_s"] for row in rows
        ),
        "common_steady_tokens": common_tokens,
        "common_steady_seconds": (common_end - common_start) / 1e9,
        "common_steady_aggregate_tok_s": common_tokens
        * 1e9
        / (common_end - common_start),
        "wall_output_tok_s": batch * OUTPUT * 1e9 / (ended_ns - started_ns),
        "requests": [
            {
                k: value
                for k, value in row.items()
                if k not in ("output_ids", "times_ns")
            }
            for row in rows
        ],
        "output_ids": [row["output_ids"] for row in rows],
    }
    write(directory / f"b{batch}-result.json", result)
    with (directory / "requests.jsonl").open("a") as sink:
        for row in result["requests"]:
            sink.write(json.dumps(row) + "\n")
    return result


def validate_log(path: Path, candidate: bool) -> dict[str, Any]:
    text = path.read_text(errors="replace")
    decode = [
        (int(batch), graph == "True")
        for batch, graph in re.findall(
            r"Decode batch.*?#running-req:\s*(\d+).*?cuda graph:\s*(True|False)",
            text,
        )
    ]
    active: dict[str, set[str]] = {}
    for rank, batch in re.findall(
        r"TP([01]).*QSA HiSparse SM89 ragged FA2 B([1-8]) active", text
    ):
        active.setdefault(batch, set()).add(rank)
    fatal = re.findall(
        r"(?im)^.*(?:CUDA out of memory|device-side assert|graph capture failed|"
        r"falling back.*graph|worker.*exited unexpectedly|UnexpectedEOF).*$",
        text,
    )
    required = {"2": {"0", "1"}, "8": {"0", "1"}}
    seen = {batch for batch, graph in decode if graph}
    if not {2, 8} <= seen or any(not graph for _, graph in decode):
        raise ValidationError(f"natural B2/B8 graph decode is missing: {decode}")
    if fatal or (candidate and any(active.get(k) != v for k, v in required.items())):
        raise ValidationError(f"server log differs: active={active}, fatal={fatal[:1]}")
    if not candidate and active:
        raise ValidationError(f"baseline unexpectedly used ragged FA2: {active}")
    return {
        "active_tp_ranks": {key: sorted(value) for key, value in active.items()},
        "natural_graph_batch_sizes": sorted(seen),
        "fatal_patterns": fatal,
    }


def run_arm(
    name: str,
    path: Path,
    commit: str,
    argv: Sequence[str],
    environment: Mapping[str, str],
    source_ids: Sequence[int],
    attempt: Path,
    timeout: float,
) -> dict[str, Any]:
    directory = attempt / name
    directory.mkdir()
    log = directory / "server.log"
    process = None
    try:
        process = v3.runner.launch_server(
            v3.capture, argv, path, environment, log, 30001, 900
        )
        write(directory / "server-info.json", v3.capture.http(30001, "/server_info", timeout=30))
        before = b1.metrics(directory / "metrics-before.prom")
        groups = {
            str(batch): run_group(batch, source_ids, directory, name, timeout)
            for batch in (2, 8)
        }
        after, recovery = b1.wait_recovered(directory, before, log)
        return {
            "name": name,
            "source": str(path),
            "source_commit": commit,
            "groups": groups,
            "route": b1.validate_route(before, after),
            "recovery": recovery,
            "log": validate_log(log, name == "candidate"),
        }
    finally:
        if process is not None:
            v3.runner.stop_tree(v3.capture, process.pid)
        v3.runner.assert_test_free(30001)
        owners = v3.profile.gpu_owner_pids()
        if owners:
            raise ValidationError(f"GPU owners remain after {name}: {sorted(owners)}")


def preflight() -> dict[str, Any]:
    b1_result = json.loads(B1_RESULT.read_text())
    component = json.loads(COMPONENT_RESULT.read_text())
    if (
        b1_result.get("status") != "PASS_B1_FAST_PATH_RESTORED"
        or b1_result.get("comparison", {}).get("integration_performance_gate") is not True
    ):
        raise ValidationError("accepted B1 execution is missing")
    if (
        component.get("status") != "PASS_COMPONENT_RESTORED"
        or component.get("component", {}).get("source_commit") != CANDIDATE_COMMIT
    ):
        raise ValidationError("accepted B>1 component gate is missing")
    return {
        "baseline": b1.source(BASE, BASE_COMMIT),
        "candidate": b1.source(CANDIDATE, CANDIDATE_COMMIT),
        "b1_status": b1_result["status"],
        "component_status": component["status"],
    }


def run(args: argparse.Namespace) -> int:
    if not args.allow_gpu:
        raise ValidationError("pass --allow-gpu after review")
    attempt = args.attempt.resolve()
    attempt.mkdir(parents=True, exist_ok=False)
    state: dict[str, Any] = {
        "status": "PREFLIGHT",
        "started_at": v3.stamp(),
        "arms": {},
    }
    snapshot = environment = models = None
    stopped = False
    error = None
    try:
        state["preflight"] = preflight()
        source_ids = request_ids()
        pid = v3.runner.listener_pid(30000)
        snapshot, environment = v3.runner.snapshot_service(pid, BASE)
        models = v3.capture.http(30000, "/v1/models", timeout=30)
        argv = b1.test_argv(snapshot["argv"])
        if v3.runner.arg_values(argv, "--cuda-graph-bs-decode") != [
            str(i) for i in range(1, 9)
        ]:
            raise ValidationError("production argv lacks exact B1-B8 graphs")
        state.update(
            production_snapshot=snapshot,
            idle=v3.runner.idle_gate(v3.capture, 30000, b1.PRODUCTION_LOG, 3, 180),
            test_argv=argv,
        )
        v3.runner.assert_test_free(30001)
        write(attempt / "manifest.json", state)
        stopped = True
        v3.runner.stop_tree(v3.capture, pid)
        if v3.profile.gpu_owner_pids():
            raise ValidationError("GPU owners remain after stopping production")
        for name, path, commit in (
            ("baseline", BASE, BASE_COMMIT),
            ("candidate", CANDIDATE, CANDIDATE_COMMIT),
        ):
            state["status"] = f"RUNNING_{name.upper()}"
            write(attempt / "manifest.json", state)
            state["arms"][name] = run_arm(
                name,
                path,
                commit,
                argv,
                b1.arm_environment(environment, path),
                source_ids,
                attempt,
                args.timeout,
            )
        gains = {}
        for batch in (2, 8):
            baseline = state["arms"]["baseline"]["groups"][str(batch)]
            candidate = state["arms"]["candidate"]["groups"][str(batch)]
            gains[str(batch)] = (
                candidate["common_steady_aggregate_tok_s"]
                / baseline["common_steady_aggregate_tok_s"]
                - 1
            )
        state["comparison"] = {
            "common_steady_gain_fraction": gains,
            "performance_gate": all(value >= 0.03 for value in gains.values()),
            "cross_arm_output_ids_equal_descriptive": {
                str(batch): state["arms"]["baseline"]["groups"][str(batch)][
                    "output_ids"
                ]
                == state["arms"]["candidate"]["groups"][str(batch)]["output_ids"]
                for batch in (2, 8)
            },
        }
        if not state["comparison"]["performance_gate"]:
            raise ValidationError(f"B>1 gain is below 3%: {gains}")
        state["status"] = "PASS_BGT1_SERVICE"
    except BaseException as exc:
        error = exc
        state.update(status="FAIL_RESTORE_PENDING", error=f"{type(exc).__name__}: {exc}")
    finally:
        if stopped and snapshot is not None and environment is not None and models is not None:
            try:
                state["restore"] = v3.restore_production(
                    snapshot, environment, models, attempt
                )
                v3.runner.assert_test_free(30001)
                pid = int(state["restore"]["pid"])
                family = {
                    pid,
                    *(child.pid for child in psutil.Process(pid).children(recursive=True)),
                }
                owners = v3.profile.gpu_owner_pids()
                if len(owners) != 2 or owners - family:
                    raise ValidationError(
                        f"post-restore GPU owners differ: {sorted(owners)}"
                    )
                state["post_restore"] = {
                    "test_port_free": True,
                    "gpu_owners": sorted(owners),
                    "all_owned_by_production": True,
                }
                state["status"] = (
                    "PASS_BGT1_SERVICE_RESTORED"
                    if state["status"] == "PASS_BGT1_SERVICE"
                    else "FAIL_RESTORED"
                )
            except BaseException as restore_error:
                error = error or restore_error
                state.update(
                    status="BLOCKED_RESTORE",
                    restore_error=f"{type(restore_error).__name__}: {restore_error}",
                )
        state["ended_at"] = v3.stamp()
        write(attempt / "manifest.json", state)
    if error is not None:
        raise error
    return 0


def self_check() -> None:
    source_ids = request_ids()
    for batch, lengths in PROMPT_LENGTHS.items():
        assert len(lengths) == batch
        if batch == 8:
            assert {length % 4 for length in lengths} == set(range(4))
        for index, length in enumerate(lengths):
            body = payload(source_ids, length, f"cpu-{batch}-{index}")
            assert len(body["input_ids"]) == length
    print(json.dumps({"status": "PASS_CPU_SELF_CHECK", "gpu_used": False}))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("self-check")
    execute = sub.add_parser("run")
    execute.add_argument("--allow-gpu", action="store_true")
    execute.add_argument("--attempt", type=Path, required=True)
    execute.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()
    if args.command == "self-check":
        self_check()
    else:
        raise SystemExit(run(args))


if __name__ == "__main__":
    main()
