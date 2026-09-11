#!/usr/bin/env python3
"""One guarded production-stop window for the frozen V4 profiling plan."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
HARNESS_ROOT = Path(os.environ.get("QSA_HISPARSE_HARNESS_ROOT", "<HARNESS_ROOT>"))
OLD = Path(os.environ.get("QSA_HISPARSE_RUNNER_ROOT", "<RUNNER_ROOT>"))
SOURCE = Path(os.environ.get("QSA_HISPARSE_SOURCE", ROOT / "sources/sglang-hisparse"))
PROD = Path(os.environ.get("QSA_HISPARSE_BASE_SOURCE", ROOT / "sources/sglang"))
PROD_LOG = Path(os.environ.get("QSA_HISPARSE_BASELINE_LOG", "<BASELINE_LOG>"))
PYTHON = Path(os.environ.get("QSA_HISPARSE_PYTHON", sys.executable))
PORT, TEST_PORT = 30000, 30001
GPU_DEADLINE_S = 3900
USEFUL_MIN_MS = .025


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = load("v4_profile_runner", OLD / "run_tests.py")
capture = runner.load_module("v4_profile_capture", runner.CAPTURE_PATH)


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def command_output(command) -> str:
    return subprocess.run(command, capture_output=True, text=True, timeout=10, check=True).stdout


def stable_models(value):
    return [
        {key: item.get(key) for key in ("id", "object", "owned_by", "root", "parent", "max_model_len")}
        for item in value.get("data", [])
    ]


def telemetry() -> dict:
    return {
        "at": runner.stamp(),
        "gpu": command_output([
            "nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw,clocks.sm,clocks.mem",
            "--format=csv,noheader,nounits",
        ]).splitlines(),
        "compute_apps": command_output([
            "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ]).splitlines(),
    }


def gpu_owner_pids() -> set[int]:
    result = set()
    for line in telemetry()["compute_apps"]:
        fields = [value.strip() for value in line.split(",")]
        if len(fields) >= 2 and fields[1].isdigit():
            result.add(int(fields[1]))
    return result


def memory_gate() -> dict:
    meminfo = {
        line.split(":", 1)[0]: int(line.split()[1]) * 1024
        for line in Path("/proc/meminfo").read_text().splitlines()
        if ":" in line and line.split()[1].isdigit()
    }
    nodes = {}
    for node in (2, 3):
        rows = Path(f"/sys/devices/system/node/node{node}/meminfo").read_text().splitlines()
        fields = {row.split()[2].rstrip(":"): int(row.split()[3]) * 1024 for row in rows if len(row.split()) >= 4 and row.split()[3].isdigit()}
        reclaimable = fields["MemFree"] + fields["Active(file)"] + fields["Inactive(file)"] + fields["SReclaimable"] - fields["Dirty"]
        nodes[str(node)] = {"reclaimable_estimate_bytes": reclaimable, "fields": fields}
    if meminfo.get("MemAvailable", 0) < 32 * 2**30 or min(value["reclaimable_estimate_bytes"] for value in nodes.values()) < 13 * 2**30:
        raise RuntimeError(f"insufficient host/NUMA memory: available={meminfo.get('MemAvailable')}, nodes={nodes}")
    return {"mem_available_bytes": meminfo["MemAvailable"], "nodes": nodes, "required_total_bytes": 32 * 2**30, "required_per_node_bytes": 13 * 2**30}


def bounded(command, cwd: Path, log_path: Path, timeout: float) -> int:
    with log_path.open("w") as log:
        process = subprocess.Popen(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, text=True, start_new_session=True)
        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(10)
            raise RuntimeError(f"GPU subprocess exceeded remaining {timeout:.1f}s")


def paired_rows(path: Path, label: str):
    return [row for row in json.loads(path.read_text()) if row["label"] == label]


def nearest(values, fraction):
    values = sorted(values)
    return values[round((len(values) - 1) * fraction)]


def select_candidate(before: Path, after: Path, candidates) -> tuple[str | None, dict]:
    baseline_before = paired_rows(before / "paired-slower-rank-samples.json", "baseline-before")
    baseline_after = paired_rows(after / "paired-slower-rank-samples.json", "baseline-after")
    before_by_key = {(row["replay"], row["step"]): row["elapsed_ms"] for row in baseline_before}
    after_by_key = {(row["replay"], row["step"]): row["elapsed_ms"] for row in baseline_after}
    baseline_p95 = {
        "before": nearest(list(before_by_key.values()), .95),
        "after": nearest(list(after_by_key.values()), .95),
    }
    decisions = {}
    for candidate in candidates:
        rows = paired_rows(after / "paired-slower-rank-samples.json", candidate)
        values = {(row["replay"], row["step"]): row["elapsed_ms"] for row in rows}
        if set(values) != set(before_by_key) or set(values) != set(after_by_key):
            raise RuntimeError(f"paired keys differ for {candidate}")
        p95 = nearest(list(values.values()), .95)
        close_keys = {
            (row["replay"], row["step"])
            for row in rows if row["c4_close_count"] > 0
        }
        delta_before = [before_by_key[key] - values[key] for key in close_keys]
        delta_after = [after_by_key[key] - values[key] for key in close_keys]
        conservative_p95_gain = min(baseline_p95.values()) - p95
        median_before = statistics.median(delta_before)
        median_after = statistics.median(delta_after)
        useful = conservative_p95_gain >= USEFUL_MIN_MS and min(median_before, median_after) >= USEFUL_MIN_MS
        decisions[candidate] = {
            "paired_p95_ms": p95,
            "conservative_p95_gain_ms": conservative_p95_gain,
            "same_key_close_median_gain_vs_before_ms": median_before,
            "same_key_close_median_gain_vs_after_ms": median_after,
            "useful": useful,
        }
    useful = [candidate for candidate in candidates if decisions[candidate]["useful"]]
    best = max(useful, key=lambda value: decisions[value]["conservative_p95_gain_ms"], default=None)
    return best, {
        "predeclared_useful_min_ms": USEFUL_MIN_MS,
        "rule": "candidate total P95 and same-key close-step median must improve both bracketing baselines by at least 0.025 ms",
        "baseline_p95_ms": baseline_p95, "candidates": decisions, "best": best,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", required=True)
    args = parser.parse_args()
    output = HERE / args.attempt
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "started_at": runner.stamp(), "status": "preflight",
        "gpu_deadline_s": GPU_DEADLINE_S,
        "script_commit": runner.git_commit(HARNESS_ROOT),
        "source_commit": runner.git_commit(SOURCE),
        "candidate_useful_min_ms": USEFUL_MIN_MS,
        "frozen_order": ["baseline-before", "timeline", "event-reuse", "fused-unpack", "baseline-after", "best-b4", "best-b8"],
    }
    save(output / "manifest.json", manifest)
    prod_pid = runner.listener_pid(PORT)
    snapshot, prod_env = runner.snapshot_service(prod_pid, PROD)
    initial_health = capture.http(PORT, "/health", timeout=10)
    initial_models = capture.http(PORT, "/v1/models", timeout=30)
    save(output / "service-snapshot.json", {**snapshot, "health": initial_health, "models": initial_models})
    if runner.listener_pid(PORT) != prod_pid:
        raise RuntimeError("production listener changed during preflight")
    family = {prod_pid, *[child.pid for child in __import__("psutil").Process(prod_pid).children(recursive=True)]}
    model_others = [value for value in runner.server_processes() if value["pid"] not in family]
    cuda_others = sorted(gpu_owner_pids() - family)
    if model_others or cuda_others:
        raise RuntimeError(f"unexpected GPU/model owners: model={model_others}, cuda_pids={cuda_others}")
    runner.assert_test_free(TEST_PORT)
    manifest["idle_gate"] = runner.idle_gate(capture, PORT, PROD_LOG, 3.0, 120.0)
    manifest["memory_gate"] = memory_gate()
    manifest["host_state"] = {
        "free_b": command_output(["free", "-b"]),
        "numactl_hardware": command_output(["numactl", "--hardware"]),
        "nvidia_topology": command_output(["nvidia-smi", "topo", "-m"]),
        "controller_cpu_affinity": sorted(os.sched_getaffinity(0)),
    }
    manifest["telemetry_before_stop"] = telemetry()
    manifest["production_pid_discovered"] = prod_pid
    manifest["status"] = "ready-to-stop"
    save(output / "manifest.json", manifest)
    stopped = False
    primary = None
    gpu_started = None
    try:
        stopped = True
        runner.stop_tree(capture, prod_pid)
        if gpu_owner_pids():
            raise RuntimeError(f"GPU owners remain after stopping production: {sorted(gpu_owner_pids())}")
        gpu_started = time.monotonic()

        def run_phase(label, command):
            remaining = GPU_DEADLINE_S - (time.monotonic() - gpu_started)
            if remaining <= 30:
                raise RuntimeError("GPU window deadline exhausted")
            manifest["status"] = label
            manifest.setdefault("phase_telemetry", {})[label + "-before"] = telemetry()
            save(output / "manifest.json", manifest)
            rc = bounded(command, HARNESS_ROOT, output / f"{label}.log", remaining)
            manifest.setdefault("returncodes", {})[label] = rc
            manifest["phase_telemetry"][label + "-after"] = telemetry()
            save(output / "manifest.json", manifest)
            if rc:
                raise RuntimeError(f"{label} failed rc={rc}")

        run_phase("baseline-before", [str(PYTHON), str(HERE / "profile_spike.py"), "before", "--output", str(output / "baseline-before")])
        timeline_base = output / "timeline/b1"
        timeline_base.parent.mkdir(parents=True, exist_ok=True)
        run_phase("timeline", [
            "nsys", "profile", "--trace=cuda,nvtx", "--sample=none", "--cpuctxsw=none",
            "--gpu-metrics-devices=all", "--gpu-metrics-frequency=1000",
            "--force-overwrite=true", "--export=sqlite", "--output", str(timeline_base),
            str(PYTHON), str(HERE / "profile_spike.py"), "profile", "--output", str(output / "timeline-run"),
        ])
        sqlite_path = timeline_base.with_suffix(".sqlite")
        parse_rc = bounded(
            [str(PYTHON), str(HERE / "parse_timeline.py"), str(sqlite_path), "--output", str(output / "timeline")],
            HARNESS_ROOT, output / "timeline-parse.log", 120,
        )
        if parse_rc:
            raise RuntimeError(f"timeline parser failed rc={parse_rc}")
        timeline = json.loads((output / "timeline/timeline-analysis.json").read_text())
        candidates = [name for name, value in timeline["candidate_evidence"].items() if value["supported"]]
        manifest["profile_supported_candidates"] = candidates
        if candidates:
            run_phase("candidates-and-baseline-after", [
                str(PYTHON), str(HERE / "profile_spike.py"), "after",
                "--output", str(output / "candidates-and-baseline-after"),
                "--arms", ",".join(candidates),
            ])
        else:
            run_phase("baseline-after", [
                str(PYTHON), str(HERE / "profile_spike.py"), "after",
                "--output", str(output / "candidates-and-baseline-after"), "--arms", "",
            ])
        best, selection = select_candidate(
            output / "baseline-before", output / "candidates-and-baseline-after", candidates,
        )
        save(output / "candidate-selection.json", selection)
        manifest["best_candidate"] = best
        if best:
            run_phase("best-scaling", [
                str(PYTHON), str(HERE / "profile_spike.py"), "scale",
                "--output", str(output / "best-scaling"), "--candidate", best,
            ])
        manifest["gpu_elapsed_s"] = time.monotonic() - gpu_started
        manifest["status"] = "gpu-complete"
    except BaseException as exc:
        primary = exc
        manifest["status"] = "FAIL"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if stopped:
            try:
                try:
                    existing = runner.listener_pid(PORT)
                except Exception:
                    existing = None
                if existing is not None:
                    actual, actual_env = runner.snapshot_service(existing, Path(snapshot["cwd"]))
                    if actual["argv"] != snapshot["argv"] or actual["source_commit"] != snapshot["source_commit"]:
                        raise RuntimeError(f"unexpected replacement service on 30000: {actual}")
                    pid = existing
                else:
                    pid = runner.launch_server(
                        capture, snapshot["argv"], Path(snapshot["cwd"]), prod_env,
                        PROD_LOG, PORT, 900,
                    ).pid
                    actual, actual_env = runner.snapshot_service(pid, Path(snapshot["cwd"]))
                health = capture.http(PORT, "/health", timeout=10)
                models = capture.http(PORT, "/v1/models", timeout=30)
                restore = {
                    "checked_at": runner.stamp(), "pid": pid,
                    "health_http": 200 if health is not None else None,
                    "models": models,
                    "model_matches": stable_models(models) == stable_models(initial_models),
                    "argv_matches": actual["argv"] == snapshot["argv"],
                    "cwd_matches": actual["cwd"] == snapshot["cwd"],
                    "source_commit": actual["source_commit"],
                    "source_matches": actual["source_commit"] == snapshot["source_commit"],
                    "environment_matches": actual_env == prod_env,
                }
                if not all((restore["health_http"] == 200, restore["model_matches"], restore["argv_matches"], restore["cwd_matches"], restore["source_matches"], restore["environment_matches"])):
                    raise RuntimeError(f"restore mismatch: {restore}")
                save(output / "restore-verification.json", restore)
                manifest["restore"] = restore
            except BaseException as exc:
                manifest["status"] = "BLOCKED/RESTORE"
                manifest["restore_error"] = f"{type(exc).__name__}: {exc}"
                primary = primary or exc
        manifest["ended_at"] = runner.stamp()
        save(output / "manifest.json", manifest)
    if primary:
        raise primary


if __name__ == "__main__":
    main()
