#!/usr/bin/env python3
"""One guarded production-like B8 run, reusing the accepted B4 harness."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
LAB = Path(os.environ.get("QSA_HISPARSE_HARNESS_ROOT", "<HARNESS_ROOT>"))
LOCAL_SOURCE = Path(
    os.environ.get("QSA_HISPARSE_LOCAL_SOURCE", ROOT / "sources/sglang-hisparse")
)
LOCAL_SOURCE_COMMIT = "5401b6e65925422acd3c9a015935267970acbf4b"
UPSTREAM_SOURCE = Path(
    os.environ.get("QSA_HISPARSE_UPSTREAM_SOURCE", ROOT / "sources/sglang-hisparse-upstream")
)
UPSTREAM_SOURCE_COMMIT = "7360188c29ad8260a6819a1beffc8c7b6f79abd8"
UPSTREAM = os.environ.get("QSA_B8_UPSTREAM") == "1"
SOURCE = UPSTREAM_SOURCE if UPSTREAM else LOCAL_SOURCE
SOURCE_COMMIT = UPSTREAM_SOURCE_COMMIT if UPSTREAM else LOCAL_SOURCE_COMMIT
PROD_DRIVER = HERE / "production_readiness_driver.py"
CONTRACT = HERE / ("UPSTREAM-B8-CONTRACT.md" if UPSTREAM else "B8-PRODUCTION-CONTRACT.md")
SHAPE_AUDIT = HERE / "B8-SHAPE-EXECUTION-02-MAIN-AUDIT.json"
LIFECYCLE_AUDIT = HERE / "B8-EXECUTION-02-MAIN-AUDIT.json"
HC_RESULT = HERE / "hc-kernel-execution-01/test.stdout.log"
SUPERVISOR = HERE / ("upstream-b8-supervisor.sh" if UPSTREAM else "b8-production-supervisor.sh")
UPSTREAM_CPU_AUDIT = HERE / "UPSTREAM-PORT-CPU-AUDIT.json"
REQUEST = Path(os.environ.get("QSA_HISPARSE_REQUEST_JSON", "<REQUEST_JSON>"))
REQUEST_COMMIT = "06fca7b97a0d322d18938c933cab6a0f214eeb77"

BATCH = 8
PROMPT = 261_120
OUTPUT = 768
CHUNK = 2048
TOTAL = 2_097_152
RIDS = tuple(f"b8-production-A{i}" for i in range(BATCH))


def load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


prod = load("qsa_b8_production_base", PROD_DRIVER)
v3 = prod.v3


class ValidationError(RuntimeError):
    pass


def write_json(path: Path, value: Any) -> None:
    prod.write_json(path, value)


def tracked(path: Path) -> dict[str, Any]:
    relative = path.relative_to(LAB).as_posix()
    commit = subprocess.run(
        ["git", "-C", str(LAB), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    result = subprocess.run(
        ["git", "-C", str(LAB), "show", f"{commit}:{relative}"],
        capture_output=True,
    )
    if result.returncode or result.stdout != path.read_bytes():
        raise ValidationError(f"Git bytes differ: {relative}@{commit}")
    return {"path": str(path.resolve()), "lab_commit": commit, "git_byte_equal": True}


def clean_source() -> dict[str, Any]:
    value = v3.require_clean_source(SOURCE)
    if value.get("commit") != SOURCE_COMMIT:
        raise ValidationError(f"candidate source {value.get('commit')} != {SOURCE_COMMIT}")
    return value


def requests() -> list[dict[str, Any]]:
    frozen = json.loads(REQUEST.read_text())
    if (len(frozen.get("input_ids", [])) != PROMPT
            or frozen.get("sampling_params", {}).get("max_new_tokens") != OUTPUT):
        raise ValidationError("frozen request geometry differs")
    result = []
    for rid in RIDS:
        value = dict(frozen)
        value["sampling_params"] = dict(frozen["sampling_params"])
        value.update(rid=rid, stream=True, log_metrics=True)
        result.append(value)
    return result


def preflight(require_supervisor: bool) -> dict[str, Any]:
    shape = json.loads(SHAPE_AUDIT.read_text())
    lifecycle = json.loads(LIFECYCLE_AUDIT.read_text())
    if (shape.get("status") != "PASS_DETERMINISTIC_HC_BATCH_INVARIANCE"
            or (not UPSTREAM and shape.get("final_source_commit") != SOURCE_COMMIT)):
        raise ValidationError("deterministic HC audit is not accepted")
    if (lifecycle.get("event_lifecycle", {}).get("status") != "PASS_B8_EVENT_LIFECYCLE"
            or lifecycle.get("event_lifecycle", {}).get("tp_schedule_equal") is not True):
        raise ValidationError("prior B8 lifecycle evidence is not accepted")
    if "12 passed" not in HC_RESULT.read_text():
        raise ValidationError("HC kernel result differs")
    helpers = {"driver": tracked(Path(__file__).resolve()), "base_driver": tracked(PROD_DRIVER)}
    if require_supervisor:
        helpers["supervisor"] = tracked(SUPERVISOR)
    result = {
        "source": clean_source(), "contract": tracked(CONTRACT), "helpers": helpers,
        "request": prod.tracked_at(REQUEST_COMMIT, REQUEST),
        "shape_audit": tracked(SHAPE_AUDIT), "lifecycle_audit": tracked(LIFECYCLE_AUDIT),
        "hc_kernel_result": tracked(HC_RESULT), "request_count": BATCH,
        "prompt_tokens_each": PROMPT, "output_tokens_each": OUTPUT,
        "source_events_enabled": False, "deterministic_inference": False,
    }
    if UPSTREAM:
        audit = json.loads(UPSTREAM_CPU_AUDIT.read_text())
        if (audit.get("status") != "PASS_CPU_PORT"
                or audit.get("source_commit") != SOURCE_COMMIT):
            raise ValidationError("upstream CPU port audit is not accepted")
        result["upstream_cpu_audit"] = tracked(UPSTREAM_CPU_AUDIT)
    return result


def arm_argv(original: Sequence[str]) -> list[str]:
    argv = prod.p5a.p5_argv(original)
    argv = v3.runner.remove_option(argv, "--enable-deterministic-inference")
    for name, values in (
        ("--max-running-requests", (str(BATCH),)),
        ("--max-total-tokens", (str(TOTAL),)),
        ("--chunked-prefill-size", (str(CHUNK),)),
        ("--cuda-graph-max-bs-decode", (str(BATCH),)),
        ("--cuda-graph-bs-decode", tuple(str(i) for i in range(1, BATCH + 1))),
    ):
        argv = v3.runner.replace_option(argv, name, values)
    required = {
        "--max-running-requests": str(BATCH), "--max-total-tokens": str(TOTAL),
        "--chunked-prefill-size": str(CHUNK), "--prefill-decode-interval": "1",
        "--cuda-graph-max-bs-decode": str(BATCH),
        "--cuda-graph-backend-decode": "full",
        "--cuda-graph-backend-prefill": "disabled",
    }
    if any(v3.arg_value(argv, key) != value for key, value in required.items()):
        raise ValidationError("B8 argv differs")
    if v3.runner.arg_values(argv, "--cuda-graph-bs-decode") != [str(i) for i in range(1, 9)]:
        raise ValidationError("B8 graph sizes differ")
    forbidden = {"--enable-deterministic-inference", "--enable-mixed-chunk",
                 "--enable-priority-scheduling"}
    if any(flag in argv for flag in forbidden) or v3.runner.strip_speculative_options(argv) != argv:
        raise ValidationError("B8 forbidden runtime option present")
    return argv


def validate_runtime(info: Mapping[str, Any], _chunk: int) -> None:
    resolved = info.get("server_args", info)
    config = resolved.get("cuda_graph_config", {})
    decode, prefill = config.get("decode", {}), config.get("prefill", {})
    required = {
        "tp_size": 2, "chunked_prefill_size": CHUNK, "prefill_decode_interval": 1,
        "max_running_requests": BATCH, "max_total_tokens": TOTAL,
        "context_length": 262_144, "disable_cuda_graph_padding": True,
        "disable_radix_cache": True, "disable_overlap_schedule": True,
        "enable_mixed_chunk": False, "enable_priority_scheduling": False,
        "speculative_algorithm": None, "enable_deterministic_inference": False,
        "stream_interval": 1, "enable_metrics": True,
    }
    if any(resolved.get(key) != value for key, value in required.items()):
        raise ValidationError("resolved B8 runtime differs")
    if (decode.get("backend") != "full" or decode.get("bs") != list(range(1, 9))
            or decode.get("max_bs") != BATCH or prefill.get("backend") != "disabled"):
        raise ValidationError("resolved B8 graph configuration differs")


def run_service(directory: Path, argv: Sequence[str], base: Mapping[str, str],
                timeout: float, gc_timeout: float) -> None:
    old = prod.SOURCE, prod.SOURCE_COMMIT, prod.p4.SOURCE, prod.requests, prod.validate_runtime
    prod.SOURCE, prod.SOURCE_COMMIT, prod.p4.SOURCE = SOURCE, SOURCE_COMMIT, SOURCE
    prod.requests, prod.validate_runtime = requests, validate_runtime
    try:
        prod.run_service(directory, argv, base, CHUNK, timeout, gc_timeout)
    finally:
        prod.SOURCE, prod.SOURCE_COMMIT, prod.p4.SOURCE, prod.requests, prod.validate_runtime = old


def service_log(path: Path) -> dict[str, Any]:
    text = path.read_text(errors="replace")
    decode = [(int(count), graph == "True") for count, graph in re.findall(
        r"Decode batch.*?#running-req:\s*(\d+).*?cuda graph:\s*(True|False)", text)]
    if not decode or any(not graph for _, graph in decode):
        raise ValidationError("decode did not stay on CUDA graphs")
    bad = re.findall(
        r"(?im)^.*(?:CUDA out of memory|graph capture failed|falling back.*graph|"
        r"pinn(?:ed|ing) memory.*fail|worker.*exited unexpectedly).*$", text,
    )
    if bad:
        raise ValidationError(f"server log fatal pattern: {bad[0]}")
    return {"decode_log_count": len(decode), "batch_sizes_seen": sorted({x for x, _ in decode}),
            "max_natural_decode_batch": max(x for x, _ in decode), "fatal_patterns": []}


def reduce_result(directory: Path) -> dict[str, Any]:
    timeline = json.loads((directory / "request-timeline.json").read_text())["submitted_at_ns"]
    results = json.loads((directory / "client-results.json").read_text())
    streams = {rid: prod.stream_evidence(directory / f"client-{rid}.jsonl") for rid in results}
    if (set(streams) != set(RIDS) or set(timeline) != set(RIDS)
            or any(row.get("status") != "complete" or row.get("done") is not True
                   or row.get("error") is not None or row.get("output_count") != OUTPUT
                   for row in results.values())):
        raise ValidationError("eight-request completion differs")
    metrics = prod.validate_metrics(directory)
    capacities = metrics["capacities"]
    if (capacities["sglang:kv_available_tokens"]["before"] < BATCH * (PROMPT + OUTPUT)
            or capacities["sglang:mamba_available_tokens"]["before"] < BATCH):
        raise ValidationError("B8 recovered capacity is below request geometry")
    gc = prod.p4.validate_gc(directory)
    nvml = v3.nvml_summary(directory / "nvml.jsonl")
    if nvml.get("sample_count", 0) <= 0 or nvml.get("errors"):
        raise ValidationError("B8 NVML evidence missing")
    nvml["terminal_plateau"] = prod.validate_nvml_plateau(directory / "nvml.jsonl")
    log = service_log(directory / "server.log")
    launch_env = json.loads((directory / "launch.json").read_text()).get("environment", {})
    forbidden_env = {"SGLANG_QSA_HISPARSE_V3_EVENTS", "SGLANG_QSA_HISPARSE_V3_CAPTURE",
                     "SGLANG_QSA_P2_VALIDATE_INITIAL_PAIR"}
    if forbidden_env & set(launch_env):
        raise ValidationError("diagnostic environment leaked into B8 production arm")
    event_dir = directory / "source-events"
    if event_dir.is_dir() and any(event_dir.iterdir()):
        raise ValidationError("source event files were written")
    first_common = max(row["times_ns"][0] for row in streams.values())
    last_common = min(row["times_ns"][-1] for row in streams.values())
    common = None
    if last_common > first_common:
        arrivals = {rid: sum(first_common < point <= last_common for point in row["times_ns"])
                    for rid, row in streams.items()}
        common = {"start_ns": first_common, "end_ns": last_common,
                  "aggregate_tokens": sum(arrivals.values()),
                  "aggregate_tokens_per_s": sum(arrivals.values()) * 1e9 /
                      (last_common - first_common), "per_request_token_arrivals": arrivals}
    start, end = min(timeline.values()), max(row["times_ns"][-1] for row in streams.values())
    outputs = {rid: row["output_ids"] for rid, row in streams.items()}
    reference = outputs[RIDS[0]]
    equal = {rid: values == reference for rid, values in outputs.items()}
    final_idle = json.loads((directory / "final-idle.json").read_text())
    if final_idle.get("status") != "PASS":
        raise ValidationError("B8 final idle gate differs")
    return {
        "schema": "qsa-hisparse-b8-production-measurement-v1",
        "status": "PASS_B8_PRODUCTION_MEASUREMENT", "source_commit": SOURCE_COMMIT,
        "requests_complete": BATCH, "prompt_tokens_each": PROMPT, "output_tokens_each": OUTPUT,
        "per_request": {rid: {"ttft_ms": (row["times_ns"][0] - timeline[rid]) / 1e6,
                              "e2e_ms": (row["times_ns"][-1] - timeline[rid]) / 1e6,
                              "tpot": row["inter_token"],
                              "prefill_tokens_per_s": row["prefill_tokens_per_s"]}
                        for rid, row in streams.items()},
        "natural_all_request_decode_intersection": common,
        "aggregate_all_output_tokens_per_s": BATCH * OUTPUT * 1e9 / (end - start),
        "native_output_equality_descriptive": equal, "token_oracle_qualification": False,
        "service_log_gate": log, "metrics_gate": metrics, "gc_gate": gc, "nvml": nvml,
        "final_idle": final_idle, "fixed_slo": False, "production_go": False,
    }


def run(args: argparse.Namespace) -> int:
    if not args.allow_gpu:
        raise ValidationError("GPU/service path requires reviewed --allow-gpu")
    attempt = args.attempt.resolve()
    attempt.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, Any] = {
        "schema": "qsa-hisparse-b8-production-driver-v1", "status": "preflight",
        "source_required": SOURCE_COMMIT, "request_count": BATCH,
        "source_profile": "upstream" if UPSTREAM else "local-candidate",
        "deterministic_inference": False, "source_events": False, "production_go": False,
    }
    write_json(attempt / "manifest.json", manifest)
    snapshot = base = models = reduced = None
    stopped = False
    primary: BaseException | None = None
    try:
        prep = preflight(True)
        pid = args.production_pid or v3.runner.listener_pid(v3.PRODUCTION_PORT)
        snapshot, base = v3.runner.snapshot_service(pid, v3.PRODUCTION_SOURCE)
        v3.capture.http(v3.PRODUCTION_PORT, "/health", timeout=10)
        models = v3.capture.http(v3.PRODUCTION_PORT, "/v1/models", timeout=30)
        v3.runner.assert_test_free(v3.TEST_PORT)
        idle = v3.runner.idle_gate(v3.capture, v3.PRODUCTION_PORT, args.production_log,
                                   args.idle_seconds, args.gate_timeout)
        argv = arm_argv(snapshot["argv"])
        manifest.update(status="ready-to-stop", preflight=prep, production_snapshot=snapshot,
                        idle_gate=idle, candidate_argv=argv)
        write_json(attempt / "manifest.json", manifest)
        stopped = True
        v3.runner.stop_tree(v3.capture, pid)
        if v3.profile.gpu_owner_pids():
            raise ValidationError("GPU owners remain after production stop")
        arm = attempt / "b8-production"
        run_service(arm, argv, base, args.request_timeout, args.gc_gate_timeout)
        reduced = reduce_result(arm)
        reduced["post_arm_source"] = clean_source()
        write_json(attempt / "reduced.json", reduced)
        manifest["status"] = "gpu-complete"
    except BaseException as exc:
        primary = exc
        manifest.update(status="FAIL", error=f"{type(exc).__name__}: {exc}")
    finally:
        if stopped and snapshot is not None and base is not None and models is not None:
            try:
                manifest["restore"] = v3.restore_production(snapshot, base, models, attempt)
                if reduced is not None:
                    reduced["status"] = ("FAIL_RESTORED" if primary
                                         else "PASS_B8_PRODUCTION_MEASUREMENT")
                    write_json(attempt / "reduced.json", reduced)
                manifest["status"] = (reduced["status"] if manifest["status"] == "gpu-complete"
                                      else "FAIL_RESTORED")
            except BaseException as exc:
                primary = primary or exc
                manifest.update(status="BLOCKED_RESTORE",
                                restore_error=f"{type(exc).__name__}: {exc}")
        write_json(attempt / "manifest.json", manifest)
    if primary:
        raise primary
    return 0


def self_check() -> None:
    clean_source()
    assert len(requests()) == BATCH and len({row["rid"] for row in requests()}) == BATCH
    fixture = json.loads(prod.p5a.p3b.ARGV_FIXTURE.read_text())["argv"]
    argv = arm_argv(fixture)
    assert v3.runner.arg_values(argv, "--cuda-graph-bs-decode") == [str(i) for i in range(1, 9)]
    env = prod.service_environment({"PYTHONPATH": "old", "SGLANG_QSA_HISPARSE_V3_EVENTS": "bad"},
                                   "", Path("unused"))
    assert not ({"SGLANG_QSA_HISPARSE_V3_EVENTS", "SGLANG_QSA_HISPARSE_V3_CAPTURE",
                 "SGLANG_QSA_P2_VALIDATE_INITIAL_PAIR"} & set(env))
    print(f"self-check ok: {'upstream' if UPSTREAM else 'local'} source, B8 argv, "
          "frozen requests and no diagnostic env")


def failure_check() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValidationError("failure-check requires CUDA_VISIBLE_DEVICES=''")
    failure, restored = RuntimeError("injected B8 failure"), {"health_http": 200}
    with tempfile.TemporaryDirectory() as temp:
        args = argparse.Namespace(allow_gpu=True, attempt=Path(temp) / "attempt",
            production_pid=1, production_log=Path("unused"), idle_seconds=0,
            gate_timeout=1, request_timeout=1, gc_gate_timeout=1)
        with patch.object(sys.modules[__name__], "preflight", return_value={}), \
                patch.object(v3.runner, "snapshot_service", return_value=({"argv": []}, {})), \
                patch.object(v3.capture, "http", side_effect=[{}, {"data": []}]), \
                patch.object(v3.runner, "assert_test_free"), \
                patch.object(v3.runner, "idle_gate", return_value={}), \
                patch.object(sys.modules[__name__], "arm_argv", return_value=[]), \
                patch.object(v3.runner, "stop_tree"), \
                patch.object(v3.profile, "gpu_owner_pids", return_value=[]), \
                patch.object(sys.modules[__name__], "run_service", side_effect=failure), \
                patch.object(v3, "restore_production", return_value=restored) as restore:
            try:
                run(args)
            except RuntimeError as exc:
                assert exc is failure
            else:
                raise AssertionError("injected B8 failure was swallowed")
        manifest = json.loads((args.attempt / "manifest.json").read_text())
        assert restore.call_count == 1 and manifest["status"] == "FAIL_RESTORED"
    print("failure-check ok: B8 failure restores production once")


def prepare(output: Path) -> int:
    value: dict[str, Any] = {"schema": "qsa-hisparse-b8-production-cpu-prep-v1",
                             "gpu_used": False, "service_used": False,
                             "source_profile": "upstream" if UPSTREAM else "local-candidate",
                             "source_commit": SOURCE_COMMIT}
    try:
        self_check()
        value.update(status="PASS_CPU_PREP", preflight=preflight(True),
                     future_gpu_arm="b8-production", production_go=False)
    except BaseException as exc:
        value.update(status="FAIL_CPU_PREP", error=f"{type(exc).__name__}: {exc}")
    write_json(output, value)
    return 0 if value["status"] == "PASS_CPU_PREP" else 2


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    sub = value.add_subparsers(dest="command", required=True)
    sub.add_parser("self-check")
    sub.add_parser("failure-check")
    prep = sub.add_parser("prepare")
    prep.add_argument("--output", type=Path, required=True)
    runp = sub.add_parser("run")
    runp.add_argument("--allow-gpu", action="store_true")
    runp.add_argument("--attempt", type=Path, required=True)
    runp.add_argument("--production-pid", type=int)
    runp.add_argument("--production-log", type=Path, default=v3.PRODUCTION_LOG)
    runp.add_argument("--idle-seconds", type=float, default=3)
    runp.add_argument("--gate-timeout", type=float, default=180)
    runp.add_argument("--request-timeout", type=float, default=7200)
    runp.add_argument("--gc-gate-timeout", type=float, default=60)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "self-check":
        self_check()
        return 0
    if args.command == "failure-check":
        failure_check()
        return 0
    if args.command == "prepare":
        return prepare(args.output)
    def interrupted(signum: int, frame: Any) -> None:
        raise KeyboardInterrupt(f"signal {signum}; restore required")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
