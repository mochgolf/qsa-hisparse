#!/usr/bin/env python3
"""One guarded latest-upstream B1/B2/B4/B8 256K scaling run."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from datetime import datetime
from itertools import pairwise
from pathlib import Path
from typing import Any
from unittest.mock import patch

import psutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
LAB = Path(os.environ.get("QSA_HISPARSE_HARNESS_ROOT", "<HARNESS_ROOT>"))
SOURCE = Path(os.environ.get("QSA_HISPARSE_SOURCE", ROOT / "sources/sglang-hisparse"))
SOURCE_COMMIT = "8606d4b2df088512157c8df9615cff10e11028ae"
UPSTREAM_BASE = "4309c7ce19dc42fb42cc9e7d883691c8dd8bda10"
BASE_DRIVER = HERE / "b8_production_driver.py"
REQUEST = Path(os.environ.get("QSA_HISPARSE_REQUEST_JSON", "<REQUEST_JSON>"))
REQUEST_METADATA = Path(
    os.environ.get("QSA_HISPARSE_REQUEST_METADATA", "<REQUEST_METADATA_JSON>")
)
REQUEST_COMMIT = "06fca7b97a0d322d18938c933cab6a0f214eeb77"
CONTRACT = HERE / "CONTRACT.md"
CPU_AUDIT = HERE / "CPU-PREP.json"
BATCHES = (1, 2, 4, 8)
PROMPT = 261_120
OUTPUT = 1_022
TOTAL = 2_097_152
COMMON_SKIP = 64
COMMON_MIN = 32
COMMON_PREFERRED = 64


def load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = load("qsa_latest_upstream_scaling_base", BASE_DRIVER)
prod, p2, v3 = base.prod, base.prod.p2, base.v3


class ValidationError(RuntimeError):
    pass


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")


def git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(SOURCE), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def clean_source() -> dict[str, Any]:
    value = v3.require_clean_source(SOURCE)
    if value.get("commit") != SOURCE_COMMIT:
        raise ValidationError(f"candidate source {value.get('commit')} != {SOURCE_COMMIT}")
    if git("merge-base", "--is-ancestor", UPSTREAM_BASE, SOURCE_COMMIT) != "":
        raise ValidationError("unexpected merge-base output")
    if int(git("rev-list", "--count", f"{UPSTREAM_BASE}..{SOURCE_COMMIT}")) != 11:
        raise ValidationError("latest-upstream replay is not the frozen 11-commit stack")
    return value


def frozen_request() -> dict[str, Any]:
    value = json.loads(REQUEST.read_text())
    metadata = json.loads(REQUEST_METADATA.read_text())
    sampling = value.get("sampling_params", {})
    if (
        len(value.get("input_ids", [])) != PROMPT
        or metadata.get("token_splice", {}).get("chat_tail_tokens") != 9
        or sampling.get("temperature") != 0
        or sampling.get("top_p") != 1.0
        or sampling.get("top_k") != 1
        or sampling.get("ignore_eos") is not True
    ):
        raise ValidationError("frozen request-A differs")
    return value


def payloads(batch: int) -> list[dict[str, Any]]:
    frozen = frozen_request()
    result = []
    for index in range(batch):
        value = dict(frozen)
        value["sampling_params"] = dict(frozen["sampling_params"])
        value["sampling_params"]["max_new_tokens"] = OUTPUT
        value.update(
            rid=f"latest-upstream-256k-b{batch}-{index}",
            stream=True,
            log_metrics=True,
        )
        result.append(value)
    return result


class BarrierStreamWorker(p2.StreamWorker):
    def __init__(
        self,
        payload: Mapping[str, Any],
        output: Path,
        timeout: float,
        barrier: threading.Barrier,
    ):
        self.barrier = barrier
        self.submitted_at_ns: int | None = None
        super().__init__(payload, output, timeout)

    def _run(self) -> None:
        try:
            self.barrier.wait(timeout=30)
            self.submitted_at_ns = time.time_ns()
        except BaseException as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self.finished.set()
            write(self.result_path, self.snapshot())
            return
        super()._run()


def percentile(values: Sequence[float], q: float) -> float | None:
    return prod.percentile(values, q) if values else None


def stats_ms(values_ns: Sequence[int]) -> dict[str, Any]:
    values = [value / 1e6 for value in values_ns]
    return {
        "count": len(values),
        "mean_ms": statistics.fmean(values) if values else None,
        "p50_ms": percentile(values, 0.50),
        "p95_ms": percentile(values, 0.95),
        "max_ms": max(values, default=None),
    }


def stream_evidence(path: Path) -> dict[str, Any]:
    output_ids: list[int] = []
    times_ns: list[int] = []
    meta: Mapping[str, Any] = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        candidate = row.get("meta_info") or row.get("usage")
        if isinstance(candidate, Mapping):
            meta = candidate
        observed = v3.runner._event_output_ids(row)
        if observed is None:
            continue
        before = len(output_ids)
        output_ids = v3.runner._merge_stream_output_ids(output_ids, observed)
        if len(output_ids) != before + 1:
            raise ValidationError(f"{path}: SSE event did not add exactly one token")
        times_ns.append(int(row["_wall_time_ns"]))
    forward, finished = meta.get("forward_entry_time"), meta.get("prefill_finished_time")
    if (
        len(output_ids) != OUTPUT
        or len(times_ns) != OUTPUT
        or any(right <= left for left, right in pairwise(times_ns))
        or meta.get("prompt_tokens") != PROMPT
        or meta.get("completion_tokens") != OUTPUT
        or not isinstance(forward, (int, float))
        or not isinstance(finished, (int, float))
        or finished <= forward
    ):
        raise ValidationError(f"{path}: stream geometry/timing differs")
    return {
        "output_ids": output_ids,
        "times_ns": times_ns,
        "meta": dict(meta),
        "prefill_duration_ms": (finished - forward) * 1000,
        "prefill_tokens_per_s": PROMPT / (finished - forward),
    }


def metrics_gate(
    before: Sequence[Mapping[str, Any]],
    after: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    graph = prod.counter_delta(before, after, "decode_cuda_graph")
    eager = prod.counter_delta(before, after, "decode_none")
    prefill = prod.counter_delta(before, after, "prefill_none")
    prefill_graph = prod.counter_delta(before, after, "prefill_cuda_graph")
    ranks = {dict(labels).get("tp_rank") for labels, value in graph.items() if value > 0}
    if (
        ranks != {"0", "1"}
        or sum(graph.values()) <= 0
        or sum(eager.values()) != 0
        or sum(prefill.values()) <= 0
        or sum(prefill_graph.values()) != 0
    ):
        raise ValidationError("CUDA-graph/prefill metric route differs")
    expected = {
        "sglang:kv_available_tokens": float(TOTAL),
        "sglang:mamba_available_tokens": 40.0,
        "sglang:num_running_reqs": 0.0,
        "sglang:num_queue_reqs": 0.0,
    }
    capacities = {}
    for name, target in expected.items():
        start, end = prod.only_value(before, name), prod.only_value(after, name)
        if start != target or end != target:
            raise ValidationError(f"{name} did not recover: {start} -> {end}")
        capacities[name] = {"before": start, "after": end}
    return {
        "decode_cuda_graph_delta": sum(graph.values()),
        "decode_cuda_graph_tp_ranks": sorted(ranks),
        "decode_none_delta": sum(eager.values()),
        "prefill_none_delta": sum(prefill.values()),
        "prefill_cuda_graph_delta": sum(prefill_graph.values()),
        "capacities": capacities,
    }


def wait_recovered(directory: Path, process: Any) -> list[dict[str, Any]]:
    p2.wait_final_idle(directory / "final-idle.json", process, seconds=3, timeout=90)
    deadline = time.monotonic() + 45
    samples = []
    while True:
        rows = prod.metrics_snapshot(directory / "metrics-after.prom")
        state = {
            name: prod.only_value(rows, name)
            for name in (
                "sglang:kv_available_tokens",
                "sglang:mamba_available_tokens",
                "sglang:num_running_reqs",
                "sglang:num_queue_reqs",
            )
        }
        samples.append(state)
        if state == {
            "sglang:kv_available_tokens": float(TOTAL),
            "sglang:mamba_available_tokens": 40.0,
            "sglang:num_running_reqs": 0.0,
            "sglang:num_queue_reqs": 0.0,
        }:
            write(directory / "recovery-samples.json", samples)
            return rows
        if time.monotonic() >= deadline:
            raise ValidationError(f"capacity did not recover: {state}")
        time.sleep(1)


def log_gate(text: str, batch: int) -> dict[str, Any]:
    decode = [
        (int(count), graph == "True")
        for count, graph in re.findall(
            r"Decode batch.*?#running-req:\s*(\d+).*?cuda graph:\s*(True|False)", text
        )
    ]
    fatal = re.findall(
        r"(?im)^.*(?:CUDA out of memory|graph capture failed|falling back.*graph|"
        r"device-side assert|worker.*exited unexpectedly|UnexpectedEOF).*$",
        text,
    )
    if fatal or any(not graph for _, graph in decode):
        raise ValidationError(f"B{batch} used eager decode")
    return {
        "decode_rows": len(decode),
        "sampled_batch_sizes": sorted({size for size, _ in decode}),
        "cuda_graph_only_in_sampled_rows": True,
        "fatal_patterns": [],
    }


def summarize_group(
    batch: int,
    workers: Sequence[BarrierStreamWorker],
    before: Sequence[Mapping[str, Any]],
    after: Sequence[Mapping[str, Any]],
    log_text: str,
    started_wall_ns: int,
    ended_wall_ns: int,
) -> dict[str, Any]:
    streams = {
        worker.payload["rid"]: stream_evidence(worker.output) for worker in workers
    }
    submitted = {
        worker.payload["rid"]: worker.submitted_at_ns for worker in workers
    }
    if any(value is None for value in submitted.values()):
        raise ValidationError(f"B{batch} lacks barrier submission timestamp")
    start = max(row["times_ns"][COMMON_SKIP] for row in streams.values())
    end = min(row["times_ns"][-1] for row in streams.values())
    if end <= start:
        raise ValidationError(f"B{batch} common decode window is empty")
    arrivals = {
        rid: [point for point in row["times_ns"] if start < point <= end]
        for rid, row in streams.items()
    }
    if min(map(len, arrivals.values())) < COMMON_MIN:
        raise ValidationError(
            f"B{batch} common tokens below {COMMON_MIN}: "
            f"{ {rid: len(points) for rid, points in arrivals.items()} }"
        )
    count = sum(map(len, arrivals.values()))
    aggregate = count * 1e9 / (end - start)
    forward = [float(row["meta"]["forward_entry_time"]) for row in streams.values()]
    finished = [float(row["meta"]["prefill_finished_time"]) for row in streams.values()]
    per_request = {}
    for rid, row in streams.items():
        points = arrivals[rid]
        per_request[rid] = {
            "ttft_ms": (row["times_ns"][0] - int(submitted[rid])) / 1e6,
            "prefill_duration_ms": row["prefill_duration_ms"],
            "prefill_tokens_per_s": row["prefill_tokens_per_s"],
            "common_tokens": len(points),
            "common_tpot": stats_ms(
                [right - left for left, right in pairwise(points)]
            ),
            "output_count": len(row["output_ids"]),
        }
    return {
        "batch": batch,
        "prompt_tokens_each": PROMPT,
        "output_tokens_each": OUTPUT,
        "requests_complete": len(streams),
        "barrier_submitted": True,
        "wall_start_ns": started_wall_ns,
        "wall_end_ns": ended_wall_ns,
        "group_wall_seconds": (ended_wall_ns - started_wall_ns) / 1e9,
        "all_prefill_wall_seconds": max(finished) - min(forward),
        "all_prefill_aggregate_tokens_per_s": batch
        * PROMPT
        / (max(finished) - min(forward)),
        "common_window": {
            "skip_tokens_per_request": COMMON_SKIP,
            "start_ns": start,
            "end_ns": end,
            "seconds": (end - start) / 1e9,
            "per_request_tokens": {
                rid: len(points) for rid, points in arrivals.items()
            },
            "minimum_tokens_per_request": min(map(len, arrivals.values())),
            "minimum_gate": COMMON_MIN,
            "preferred_64_tokens": min(map(len, arrivals.values()))
            >= COMMON_PREFERRED,
            "aggregate_tokens": count,
            "aggregate_decode_tokens_per_s": aggregate,
            "per_request_decode_tokens_per_s": aggregate / batch,
        },
        "per_request": per_request,
        "metrics": metrics_gate(before, after),
        "log": log_gate(log_text, batch),
    }


def run_group(
    service_directory: Path,
    batch: int,
    timeout: float,
    process: Any,
) -> dict[str, Any]:
    directory = service_directory / f"b{batch}"
    directory.mkdir()
    before = prod.metrics_snapshot(directory / "metrics-before.prom")
    server_log = service_directory / "server.log"
    offset = server_log.stat().st_size
    barrier = threading.Barrier(batch)
    workers = [
        BarrierStreamWorker(
            value,
            directory / f"client-{value['rid']}.jsonl",
            timeout,
            barrier,
        )
        for value in payloads(batch)
    ]
    started_wall_ns = time.time_ns()
    for worker in workers:
        worker.start()
    results = {
        worker.payload["rid"]: worker.join(timeout, OUTPUT, process, workers)
        for worker in workers
    }
    ended_wall_ns = time.time_ns()
    after = wait_recovered(directory, process)
    log_text = server_log.read_bytes()[offset:].decode(errors="replace")
    (directory / "server-window.log").write_text(log_text)
    write(directory / "client-results.json", results)
    write(
        directory / "request-timeline.json",
        {
            "submitted_at_ns": {
                worker.payload["rid"]: worker.submitted_at_ns for worker in workers
            }
        },
    )
    result = summarize_group(
        batch,
        workers,
        before,
        after,
        log_text,
        started_wall_ns,
        ended_wall_ns,
    )
    write(directory / "reduced.json", result)
    return result


def run_groups(
    directory: Path,
    _strict_cancel: bool,
    timeout: float,
    process: Any,
) -> dict[str, Any]:
    groups = {
        str(batch): run_group(directory, batch, timeout, process) for batch in BATCHES
    }
    write(directory / "groups.json", groups)
    return groups


def run_candidate_service(
    directory: Path,
    argv: Sequence[str],
    environment: Mapping[str, str],
    timeout: float,
    gc_timeout: float,
) -> None:
    old = base.SOURCE, base.SOURCE_COMMIT, prod.run_requests
    base.SOURCE, base.SOURCE_COMMIT, prod.run_requests = (
        SOURCE,
        SOURCE_COMMIT,
        run_groups,
    )
    try:
        base.run_service(directory, argv, environment, timeout, gc_timeout)
    finally:
        base.SOURCE, base.SOURCE_COMMIT, prod.run_requests = old


def nvml_by_group(path: Path, groups: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    rows = []
    for line in path.read_text().splitlines():
        row = json.loads(line)
        row["time_ns"] = int(datetime.fromisoformat(row["at"]).timestamp() * 1e9)
        rows.append(row)
    result = {}
    for label, group in groups.items():
        selected = [
            row
            for row in rows
            if group["wall_start_ns"] <= row["time_ns"] <= group["wall_end_ns"]
        ]
        if not selected:
            raise ValidationError(f"B{label} has no NVML samples")
        result[label] = {}
        for index in (0, 1):
            samples = [
                gpu
                for row in selected
                for gpu in row.get("gpus", [])
                if gpu["index"] == index
            ]
            if not samples:
                raise ValidationError(f"B{label} GPU{index} has no NVML samples")
            result[label][str(index)] = {
                "samples": len(samples),
                "first_mib": samples[0]["used_mib"],
                "peak_mib": max(row["used_mib"] for row in samples),
                "final_mib": samples[-1]["used_mib"],
                "peak_utilization": max(row["utilization"] for row in samples),
            }
    return result


def reduce_service(directory: Path) -> dict[str, Any]:
    groups = json.loads((directory / "groups.json").read_text())
    log_text = (directory / "server.log").read_text(errors="replace")
    launch_env = json.loads((directory / "launch.json").read_text()).get(
        "environment", {}
    )
    forbidden_env = {
        "SGLANG_QSA_HISPARSE_V3_EVENTS",
        "SGLANG_QSA_HISPARSE_V3_CAPTURE",
        "SGLANG_QSA_P2_VALIDATE_INITIAL_PAIR",
    }
    event_dir = directory / "source-events"
    if forbidden_env & set(launch_env) or (
        event_dir.is_dir() and any(event_dir.iterdir())
    ):
        raise ValidationError("diagnostic source instrumentation leaked into service")
    active: dict[str, set[str]] = {}
    for rank, batch in re.findall(
        r"TP([01]).*QSA HiSparse SM89 ragged FA2 B([1-8]) active", log_text
    ):
        active.setdefault(batch, set()).add(rank)
    required = {str(batch): {"0", "1"} for batch in BATCHES}
    fatal = re.findall(
        r"(?im)^.*(?:CUDA out of memory|graph capture failed|falling back.*graph|"
        r"device-side assert|worker.*exited unexpectedly|UnexpectedEOF).*$",
        log_text,
    )
    if fatal or any(active.get(batch) != ranks for batch, ranks in required.items()):
        raise ValidationError(f"service log differs: active={active}, fatal={fatal[:1]}")
    nvml = v3.nvml_summary(directory / "nvml.jsonl")
    if nvml.get("sample_count", 0) <= 0 or nvml.get("errors"):
        raise ValidationError("NVML evidence missing")
    nvml["terminal_plateau"] = prod.validate_nvml_plateau(directory / "nvml.jsonl")
    b1 = groups["1"]["common_window"]["aggregate_decode_tokens_per_s"]
    scaling = {}
    for batch in BATCHES:
        rate = groups[str(batch)]["common_window"]["aggregate_decode_tokens_per_s"]
        scaling[str(batch)] = {
            "aggregate_decode_tokens_per_s": rate,
            "per_request_decode_tokens_per_s": rate / batch,
            "speedup_vs_b1": rate / b1,
            "scaling_efficiency_vs_b1": rate / (batch * b1),
        }
    return {
        "schema": "qsa-hisparse-latest-upstream-256k-scaling-v1",
        "status": "PASS_LATEST_UPSTREAM_256K_SCALING",
        "source_commit": SOURCE_COMMIT,
        "upstream_base": UPSTREAM_BASE,
        "groups": groups,
        "scaling": scaling,
        "active_tp_ranks": {key: sorted(value) for key, value in active.items()},
        "nvml": nvml,
        "nvml_group_global_bounds": nvml_by_group(directory / "nvml.jsonl", groups),
        "gc_gate": prod.p4.validate_gc(directory),
        "diagnostic_source_events": False,
        "fixed_slo": False,
        "production_go": False,
    }


def preflight() -> dict[str, Any]:
    audit = json.loads(CPU_AUDIT.read_text())
    if (
        audit.get("status") != "PASS_CPU_PREP"
        or audit.get("source_commit") != SOURCE_COMMIT
        or audit.get("upstream_base") != UPSTREAM_BASE
    ):
        raise ValidationError("source-bound CPU audit is missing")
    return {
        "source": clean_source(),
        "upstream_base": UPSTREAM_BASE,
        "source_stack_commits": 11,
        "contract": base.tracked(CONTRACT),
        "driver": base.tracked(Path(__file__).resolve()),
        "cpu_audit": base.tracked(CPU_AUDIT),
        "request": prod.tracked_at(REQUEST_COMMIT, REQUEST),
        "request_metadata": prod.tracked_at(REQUEST_COMMIT, REQUEST_METADATA),
        "prompt_tokens_each": PROMPT,
        "output_tokens_each": OUTPUT,
        "batches": list(BATCHES),
    }


def post_restore_check(restore: Mapping[str, Any]) -> dict[str, Any]:
    v3.runner.assert_test_free(v3.TEST_PORT)
    pid = int(restore["pid"])
    family = {
        pid,
        *(child.pid for child in psutil.Process(pid).children(recursive=True)),
    }
    owners = v3.profile.gpu_owner_pids()
    if len(owners) != 2 or owners - family:
        raise ValidationError(f"post-restore GPU owners differ: {sorted(owners)}")
    return {"test_port_free": True, "gpu_owners": sorted(owners)}


def run(args: argparse.Namespace) -> int:
    if not args.allow_gpu:
        raise ValidationError("pass reviewed --allow-gpu")
    attempt = args.attempt.resolve()
    attempt.mkdir(parents=True, exist_ok=False)
    state: dict[str, Any] = {
        "schema": "qsa-hisparse-latest-upstream-scaling-driver-v1",
        "status": "PREFLIGHT",
        "source_required": SOURCE_COMMIT,
        "production_go": False,
    }
    snapshot = environment = models = reduced = None
    stopped = False
    primary: BaseException | None = None
    try:
        state["preflight"] = preflight()
        pid = args.production_pid or v3.runner.listener_pid(v3.PRODUCTION_PORT)
        snapshot, environment = v3.runner.snapshot_service(pid, v3.PRODUCTION_SOURCE)
        v3.capture.http(v3.PRODUCTION_PORT, "/health", timeout=10)
        models = v3.capture.http(v3.PRODUCTION_PORT, "/v1/models", timeout=30)
        v3.runner.assert_test_free(v3.TEST_PORT)
        argv = base.arm_argv(snapshot["argv"])
        state.update(
            production_snapshot=snapshot,
            idle_gate=v3.runner.idle_gate(
                v3.capture,
                v3.PRODUCTION_PORT,
                args.production_log,
                3,
                180,
            ),
            candidate_argv=argv,
        )
        write(attempt / "manifest.json", state)
        stopped = True
        v3.runner.stop_tree(v3.capture, pid)
        if v3.profile.gpu_owner_pids():
            raise ValidationError("GPU owners remain after production stop")
        service = attempt / "candidate"
        run_candidate_service(
            service, argv, environment, args.request_timeout, args.gc_gate_timeout
        )
        reduced = reduce_service(service)
        reduced["post_run_source"] = clean_source()
        write(attempt / "reduced.json", reduced)
        state["status"] = "GPU_COMPLETE"
    except BaseException as exc:
        primary = exc
        state.update(status="FAIL_RESTORE_PENDING", error=f"{type(exc).__name__}: {exc}")
    finally:
        if stopped and snapshot is not None and environment is not None and models is not None:
            try:
                restore = v3.restore_production(snapshot, environment, models, attempt)
                state["restore"] = restore
                state["post_restore"] = post_restore_check(restore)
                state["status"] = (
                    "PASS_LATEST_UPSTREAM_256K_SCALING_RESTORED"
                    if primary is None
                    else "FAIL_RESTORED"
                )
            except BaseException as exc:
                primary = primary or exc
                state.update(
                    status="BLOCKED_RESTORE",
                    restore_error=f"{type(exc).__name__}: {exc}",
                )
        write(attempt / "manifest.json", state)
    if primary is not None:
        raise primary
    return 0


def self_check() -> None:
    clean_source()
    assert PROMPT + OUTPUT == 262_142
    assert 8 * (PROMPT + OUTPUT) == 2_097_136 <= TOTAL
    assert [len(payloads(batch)) for batch in BATCHES] == list(BATCHES)
    assert all(
        row["sampling_params"]["max_new_tokens"] == OUTPUT
        for batch in BATCHES
        for row in payloads(batch)
    )
    fixture = json.loads(prod.p5a.p3b.ARGV_FIXTURE.read_text())["argv"]
    argv = base.arm_argv(fixture)
    assert v3.runner.arg_values(argv, "--cuda-graph-bs-decode") == [
        str(index) for index in range(1, 9)
    ]
    print(json.dumps({"status": "PASS_CPU_SELF_CHECK", "gpu_used": False}))


def failure_check() -> None:
    failure = RuntimeError("injected latest-upstream failure")
    with tempfile.TemporaryDirectory() as temp:
        args = argparse.Namespace(
            allow_gpu=True,
            attempt=Path(temp) / "attempt",
            production_pid=1,
            production_log=Path("unused"),
            request_timeout=1,
            gc_gate_timeout=1,
        )
        with (
            patch.object(sys.modules[__name__], "preflight", return_value={}),
            patch.object(v3.runner, "snapshot_service", return_value=({"argv": []}, {})),
            patch.object(v3.capture, "http", side_effect=[{}, {"data": []}]),
            patch.object(v3.runner, "assert_test_free"),
            patch.object(v3.runner, "idle_gate", return_value={}),
            patch.object(base, "arm_argv", return_value=[]),
            patch.object(v3.runner, "stop_tree"),
            patch.object(v3.profile, "gpu_owner_pids", return_value=[]),
            patch.object(sys.modules[__name__], "run_candidate_service", side_effect=failure),
            patch.object(v3, "restore_production", return_value={"pid": 1}) as restore,
            patch.object(sys.modules[__name__], "post_restore_check", return_value={}),
        ):
            try:
                run(args)
            except RuntimeError as exc:
                assert exc is failure
            else:
                raise AssertionError("injected failure was swallowed")
        manifest = json.loads((args.attempt / "manifest.json").read_text())
        assert restore.call_count == 1 and manifest["status"] == "FAIL_RESTORED"
    print(json.dumps({"status": "PASS_CPU_FAILURE_RESTORE", "gpu_used": False}))


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("self-check")
    sub.add_parser("failure-check")
    execute = sub.add_parser("run")
    execute.add_argument("--allow-gpu", action="store_true")
    execute.add_argument("--attempt", type=Path, required=True)
    execute.add_argument("--production-pid", type=int)
    execute.add_argument("--production-log", type=Path, default=v3.PRODUCTION_LOG)
    execute.add_argument("--request-timeout", type=float, default=7200)
    execute.add_argument("--gc-gate-timeout", type=float, default=60)
    args = parser.parse_args()
    if args.command == "self-check":
        self_check()
        return 0
    if args.command == "failure-check":
        failure_check()
        return 0

    def interrupted(signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt(f"signal {signum}; restore required")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
