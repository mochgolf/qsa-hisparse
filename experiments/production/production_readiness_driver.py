#!/usr/bin/env python3
"""Guarded two-arm B4 production-readiness measurement."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import statistics
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
LAB = Path(os.environ.get("QSA_HISPARSE_HARNESS_ROOT", "<HARNESS_ROOT>"))
SOURCE = Path(os.environ.get("QSA_HISPARSE_SOURCE", ROOT / "sources/sglang-hisparse"))
SOURCE_COMMIT = "7c37f8ee93ed988bfcad82bff077255431f63bc2"
CONTRACT = HERE / "CONTRACT.md"
CONTRACT_COMMIT = "e1a469b83350f8a49a559cd62e50cd72acd3e315"
P5A_PATH = Path(os.environ.get("QSA_HISPARSE_P5A_DRIVER", "<P5A_DRIVER>"))
SUPERVISOR = HERE / "production-readiness-supervisor.sh"
REQUEST_DIR = Path(os.environ.get("QSA_HISPARSE_REQUEST_DIR", "<REQUEST_DIR>"))
REQUEST_COMMIT = "06fca7b97a0d322d18938c933cab6a0f214eeb77"
ARMS = (("baseline-chunk4096", 4096), ("treatment-chunk2048", 2048))
PROMPT = 261_120
OUTPUT = 768
METRIC_RE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{([^}]*)\})?\s+([-+0-9.eE]+)\s*$")
LABEL_RE = re.compile(r'(\w+)="([^"]*)"')


def load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


p5a = load("qsa_production_readiness_p5a", P5A_PATH)
p4, p3a, p2, v3 = p5a.p4, p5a.p3a, p5a.p2, p5a.v3


class ValidationError(RuntimeError):
    pass


def write_json(path: Path, value: Any) -> None:
    p5a.write_json(path, value)


def tracked_at(commit: str, path: Path) -> dict[str, Any]:
    return p5a.tracked_at(commit, path)


def clean_source() -> dict[str, Any]:
    value = v3.require_clean_source(SOURCE)
    if value.get("commit") != SOURCE_COMMIT:
        raise ValidationError(f"candidate source {value.get('commit')} != {SOURCE_COMMIT}")
    return value


def requests() -> list[dict[str, Any]]:
    result = []
    for label in ("A", "B", "A", "B"):
        path = REQUEST_DIR / f"request-{label}.json"
        value = json.loads(path.read_text())
        value["rid"] = f"production-readiness-{label}{len(result)}"
        value["stream"] = True
        if (len(value.get("input_ids", [])) != PROMPT
                or value.get("sampling_params", {}).get("max_new_tokens") != OUTPUT):
            raise ValidationError(f"request {label} geometry differs")
        result.append(value)
    return result


def preflight(require_supervisor: bool) -> dict[str, Any]:
    helpers = {"driver": p5a.tracked(Path(__file__).resolve()),
               "p5a_driver": p5a.tracked(P5A_PATH)}
    if require_supervisor:
        helpers["supervisor"] = p5a.tracked(SUPERVISOR)
    return {"source": clean_source(), "contract": tracked_at(CONTRACT_COMMIT, CONTRACT),
            "helpers": helpers,
            "requests": {name: tracked_at(REQUEST_COMMIT, REQUEST_DIR / f"request-{name}.json")
                         for name in ("A", "B")},
            "request_count": len(requests()), "prompt_tokens_each": PROMPT,
            "output_tokens_each": OUTPUT, "source_events_enabled": False}


def arm_argv(original: Sequence[str], chunk: int) -> list[str]:
    argv = p5a.p5_argv(original)
    argv = v3.runner.remove_option(argv, "--enable-deterministic-inference")
    argv = v3.runner.replace_option(argv, "--chunked-prefill-size", (str(chunk),))
    required = {"--tp-size": "2", "--chunked-prefill-size": str(chunk),
                "--prefill-decode-interval": "1", "--max-running-requests": "4",
                "--max-total-tokens": "1048576", "--context-length": "262144",
                "--cuda-graph-backend-decode": "full",
                "--cuda-graph-backend-prefill": "disabled",
                "--cuda-graph-max-bs-decode": "4"}
    if any(v3.arg_value(argv, key) != value for key, value in required.items()):
        raise ValidationError(f"chunk{chunk}: candidate argv differs")
    if v3.runner.arg_values(argv, "--cuda-graph-bs-decode") != ["1", "2", "3", "4"]:
        raise ValidationError("decode graph sizes differ")
    required_flags = {"--disable-cuda-graph-padding", "--disable-radix-cache",
                      "--disable-overlap-schedule"}
    forbidden = {"--enable-deterministic-inference", "--enable-mixed-chunk",
                 "--enable-priority-scheduling"}
    if (not required_flags.issubset(argv) or any(flag in argv for flag in forbidden)
            or v3.runner.strip_speculative_options(argv) != argv):
        raise ValidationError(f"chunk{chunk}: runtime flags differ")
    return argv


def validate_arms(left: Sequence[str], right: Sequence[str]) -> None:
    if (v3.runner.remove_option(left, "--chunked-prefill-size") !=
            v3.runner.remove_option(right, "--chunked-prefill-size")):
        raise ValidationError("arms differ outside chunked-prefill-size")


def service_environment(base: Mapping[str, str], observe: str, events: Path) -> dict[str, str]:
    del observe, events
    env = dict(base)
    env["PYTHONPATH"] = str((SOURCE / "python").resolve()) + os.pathsep + base.get("PYTHONPATH", "")
    env["PWD"] = str(SOURCE.resolve())
    env["PYTHONUNBUFFERED"] = "1"
    env["SGLANG_QSA_HISPARSE_V3"] = "p2-offload"
    env["SGLANG_QSA_HISPARSE_V3_OBSERVE"] = "light"
    for key in ("SGLANG_QSA_HISPARSE_V3_EVENTS", "SGLANG_QSA_HISPARSE_V3_CAPTURE",
                "SGLANG_QSA_P2_VALIDATE_INITIAL_PAIR"):
        env.pop(key, None)
    return env


def validate_runtime(info: Mapping[str, Any], chunk: int) -> None:
    resolved = info.get("server_args", info)
    config = resolved.get("cuda_graph_config", {})
    decode, prefill = config.get("decode", {}), config.get("prefill", {})
    required = {"tp_size": 2, "chunked_prefill_size": chunk,
                "prefill_decode_interval": 1, "max_running_requests": 4,
                "max_total_tokens": 1_048_576, "context_length": 262_144,
                "disable_cuda_graph_padding": True, "disable_radix_cache": True,
                "disable_overlap_schedule": True, "enable_mixed_chunk": False,
                "enable_priority_scheduling": False,
                "speculative_algorithm": None, "enable_deterministic_inference": False,
                "stream_interval": 1, "enable_metrics": True}
    if any(resolved.get(key) != value for key, value in required.items()):
        raise ValidationError(f"chunk{chunk}: resolved runtime differs")
    if (decode.get("backend") != "full" or decode.get("bs") != [1, 2, 3, 4]
            or decode.get("max_bs") != 4 or prefill.get("backend") != "disabled"):
            raise ValidationError(f"chunk{chunk}: resolved graph configuration differs")


def metrics_snapshot(path: Path) -> list[dict[str, Any]]:
    text = str(v3.capture.http(v3.TEST_PORT, "/metrics", timeout=10))
    path.write_text(text)
    rows = []
    for line in text.splitlines():
        match = METRIC_RE.match(line.strip())
        if match:
            rows.append({"name": match.group(1),
                         "labels": dict(LABEL_RE.findall(match.group(2) or "")),
                         "value": float(match.group(3))})
    return rows


def metric_map(rows: Sequence[Mapping[str, Any]], name: str,
               mode: str | None = None) -> dict[tuple[tuple[str, str], ...], float]:
    result = {}
    for row in rows:
        labels = row.get("labels", {})
        if row.get("name") == name and (mode is None or labels.get("mode") == mode):
            result[tuple(sorted(labels.items()))] = float(row["value"])
    return result


def counter_delta(before: Sequence[Mapping[str, Any]], after: Sequence[Mapping[str, Any]],
                  mode: str) -> dict[tuple[tuple[str, str], ...], float]:
    old = metric_map(before, "sglang:cuda_graph_passes_total", mode)
    new = metric_map(after, "sglang:cuda_graph_passes_total", mode)
    keys = set(old) | set(new)
    result = {key: new.get(key, 0.0) - old.get(key, 0.0) for key in keys}
    if any(value < 0 for value in result.values()):
        raise ValidationError(f"metrics counter reset for {mode}")
    return result


def only_value(rows: Sequence[Mapping[str, Any]], name: str) -> float:
    values = [float(row["value"]) for row in rows
              if row.get("name") == name and "priority" not in row.get("labels", {})]
    if not values or len(set(values)) != 1:
        raise ValidationError(f"metric {name} missing or TP values differ: {values}")
    return values[0]


def run_requests(directory: Path, strict_cancel: bool, timeout: float,
                 process: Any) -> dict[str, Any]:
    del strict_cancel
    before = metrics_snapshot(directory / "metrics-before.prom")
    workers: list[Any] = []
    submitted: dict[str, int] = {}
    for value in requests():
        worker = p2.StreamWorker(value, directory / f"client-{value['rid']}.jsonl", timeout)
        workers.append(worker)
        submitted[value["rid"]] = time.time_ns()
        worker.start()
    results: dict[str, Any] = {}
    try:
        for worker in workers:
            results[worker.payload["rid"]] = worker.join(timeout, OUTPUT, process, workers)
        p2.wait_final_idle(directory / "final-idle.json", process)
        after = metrics_snapshot(directory / "metrics-after.prom")
        health = v3.capture.http(v3.TEST_PORT, "/health", timeout=10)
        write_json(directory / "request-timeline.json", {"submitted_at_ns": submitted})
        write_json(directory / "client-results.json", results)
        write_json(directory / "metrics-snapshots.json", {"before": before, "after": after})
        write_json(directory / "post-request-health.json", {"http": 200, "response": health})
        return results
    finally:
        write_json(directory / "client-results-diagnostic.json",
                   {worker.payload["rid"]: worker.snapshot() for worker in workers})


def run_service(directory: Path, argv: Sequence[str], base: Mapping[str, str], chunk: int,
                timeout: float, gc_timeout: float) -> None:
    old = (p4.run_requests, p4.service_environment, p4.validate_runtime, p4.SOURCE_COMMIT)
    p4.run_requests = run_requests
    p4.service_environment = service_environment
    p4.validate_runtime = lambda info, interval: validate_runtime(info, chunk)
    p4.SOURCE_COMMIT = SOURCE_COMMIT
    try:
        p4.run_service(directory, argv, base, 1, "light", False, timeout, gc_timeout)
    finally:
        (p4.run_requests, p4.service_environment, p4.validate_runtime, p4.SOURCE_COMMIT) = old


def percentile(values: Sequence[float], q: float) -> float | None:
    return p3a.percentile(values, q) if values else None


def stats_ns(values: Sequence[int]) -> dict[str, Any]:
    milliseconds = [value / 1e6 for value in values]
    return {"count": len(milliseconds), "mean_ms": statistics.fmean(milliseconds) if milliseconds else None,
            "p50_ms": percentile(milliseconds, .50), "p95_ms": percentile(milliseconds, .95),
            "p99_ms": percentile(milliseconds, .99), "max_ms": max(milliseconds, default=None)}


def stream_evidence(path: Path) -> dict[str, Any]:
    output_ids: list[int] = []
    times: list[int] = []
    meta: Mapping[str, Any] = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        candidate = row.get("meta_info") or row.get("usage")
        if isinstance(candidate, Mapping):
            meta = candidate
        observed = v3.runner._event_output_ids(row)
        if observed is not None:
            before = len(output_ids)
            output_ids = v3.runner._merge_stream_output_ids(output_ids, observed)
            if len(output_ids) - before != 1:
                raise ValidationError(f"SSE row is not one token: {path}")
            times.append(int(row["_wall_time_ns"]))
    if (len(output_ids) != OUTPUT or len(times) != OUTPUT
            or meta.get("prompt_tokens") != PROMPT or meta.get("completion_tokens") != OUTPUT
            or any(right <= left for left, right in zip(times, times[1:]))):
        raise ValidationError(f"SSE evidence differs: {path}")
    forward, finished = meta.get("forward_entry_time"), meta.get("prefill_finished_time")
    if not isinstance(forward, (int, float)) or not isinstance(finished, (int, float)) or finished <= forward:
        raise ValidationError(f"server prefill timing missing: {path}")
    gaps = [right - left for left, right in zip(times, times[1:])]
    return {"output_ids": output_ids, "times_ns": times,
            "ttft_anchor_ns": times[0], "last_output_ns": times[-1],
            "inter_token": stats_ns(gaps), "prefill_duration_ms": (finished - forward) * 1000,
            "prefill_tokens_per_s": PROMPT / (finished - forward), "meta": dict(meta)}


def validate_metrics(directory: Path) -> dict[str, Any]:
    snapshots = json.loads((directory / "metrics-snapshots.json").read_text())
    before, after = snapshots["before"], snapshots["after"]
    graph = counter_delta(before, after, "decode_cuda_graph")
    graph_ranks = {dict(key).get("tp_rank") for key, value in graph.items() if value > 0}
    eager = counter_delta(before, after, "decode_none")
    prefill = counter_delta(before, after, "prefill_none")
    prefill_graph = counter_delta(before, after, "prefill_cuda_graph")
    if (graph_ranks != {"0", "1"} or sum(graph.values()) <= 0
            or sum(eager.values()) != 0 or sum(prefill.values()) <= 0
            or sum(prefill_graph.values()) != 0):
        raise ValidationError("actual graph/prefill metrics route differs")
    capacities = {}
    for name, minimum in (("sglang:kv_available_tokens", 4 * (PROMPT + OUTPUT)),
                          ("sglang:mamba_available_tokens", 4),
                          ("sglang:num_running_reqs", 0),
                          ("sglang:num_queue_reqs", 0)):
        start, end = only_value(before, name), only_value(after, name)
        if end != start or start < minimum:
            raise ValidationError(f"metric {name} did not recover: {start} -> {end}")
        capacities[name] = {"before": start, "after": end}
    return {"decode_cuda_graph_delta": sum(graph.values()),
            "decode_cuda_graph_tp_ranks": sorted(graph_ranks),
            "decode_none_delta": sum(eager.values()),
            "prefill_none_delta": sum(prefill.values()),
            "prefill_cuda_graph_delta": sum(prefill_graph.values()),
            "capacities": capacities}


def validate_service_log(path: Path) -> dict[str, Any]:
    text = path.read_text(errors="replace")
    decode = [(int(count), graph == "True") for count, graph in re.findall(
        r"Decode batch.*?#running-req:\s*(\d+).*?cuda graph:\s*(True|False)", text)]
    if not decode or not any(count == 4 and graph for count, graph in decode):
        raise ValidationError("server log has no real B4 CUDA-graph decode")
    if any(not graph for _, graph in decode):
        raise ValidationError("server log contains eager decode fallback")
    bad = re.findall(r"(?im)^.*(?:CUDA out of memory|graph capture failed|falling back.*graph|pinn(?:ed|ing) memory.*fail|worker.*exited unexpectedly).*$", text)
    if bad:
        raise ValidationError(f"server log fatal pattern: {bad[0]}")
    return {"decode_log_count": len(decode), "batch_sizes_seen": sorted({x for x, _ in decode}),
            "real_B4_cuda_graph": True, "fatal_patterns": []}


def validate_nvml_plateau(path: Path) -> dict[str, Any]:
    samples: dict[int, list[int]] = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        for gpu in row.get("gpus", []):
            samples.setdefault(int(gpu["index"]), []).append(int(gpu["used_mib"]))
    if set(samples) != {0, 1} or any(len(values) < 8 for values in samples.values()):
        raise ValidationError("NVML per-device terminal samples missing")
    result = {}
    for index, values in sorted(samples.items()):
        tail = values[-8:]
        if max(tail) - min(tail) > 16:
            raise ValidationError(f"GPU{index} memory did not reach terminal plateau")
        result[str(index)] = {"first_mib": values[0], "peak_mib": max(values),
                              "last_mib": values[-1], "tail_samples": len(tail),
                              "tail_span_mib": max(tail) - min(tail)}
    return result


def reduce_arm(directory: Path, chunk: int) -> dict[str, Any]:
    timeline = json.loads((directory / "request-timeline.json").read_text())["submitted_at_ns"]
    results = json.loads((directory / "client-results.json").read_text())
    streams = {rid: stream_evidence(directory / f"client-{rid}.jsonl") for rid in results}
    if (set(streams) != set(timeline) or any(row.get("status") != "complete"
            or row.get("done") is not True or row.get("error") is not None
            or row.get("output_count") != OUTPUT for row in results.values())):
        raise ValidationError("four-request completion differs")
    first, last = max(row["times_ns"][0] for row in streams.values()), \
                  min(row["times_ns"][-1] for row in streams.values())
    steady = {rid: [right - left for left, right in zip(row["times_ns"], row["times_ns"][1:])
                    if first <= left < right <= last] for rid, row in streams.items()}
    arrivals = {rid: sum(first < point <= last for point in row["times_ns"])
                for rid, row in streams.items()}
    count = sum(arrivals.values())
    if last <= first or any(not values for values in steady.values()) or any(value <= 0 for value in arrivals.values()):
        raise ValidationError("B4 concurrency-intersection window is empty")
    ordered = sorted(timeline, key=timeline.get)
    first_token_order = sorted(streams, key=lambda rid: streams[rid]["times_ns"][0])
    survivor: list[dict[str, Any]] = []
    for index, rid in enumerate(ordered[1:], 1):
        start, end = timeline[rid], streams[rid]["times_ns"][0]
        gaps, crossings = [], []
        arrivals_in_window = 0
        for prior in ordered[:index]:
            points = streams[prior]["times_ns"]
            arrivals_in_window += sum(start < point <= end for point in points)
            gaps.extend(right - left for left, right in zip(points, points[1:])
                        if start <= left < right <= end)
            crossings.extend(right - left for left, right in zip(points, points[1:])
                             if (left < start < right) or (left < end < right))
        survivor.append({"incoming_rid": rid, "observer_window": "submit_to_first_token",
                         "window_ms": (end - start) / 1e6,
                         "survivor_token_arrivals": arrivals_in_window,
                         "fully_contained_survivor_gaps": stats_ns(gaps),
                         "boundary_crossing_survivor_gaps": stats_ns(crossings)})
    start = min(timeline.values()); end = max(row["times_ns"][-1] for row in streams.values())
    gc = p4.validate_gc(directory)
    nvml = v3.nvml_summary(directory / "nvml.jsonl")
    if nvml.get("sample_count", 0) <= 0 or nvml.get("errors"):
        raise ValidationError("NVML evidence missing")
    nvml["terminal_plateau"] = validate_nvml_plateau(directory / "nvml.jsonl")
    metrics = validate_metrics(directory)
    service_log = validate_service_log(directory / "server.log")
    launch = json.loads((directory / "launch.json").read_text())
    public = launch.get("environment", {})
    if any(key in public for key in ("SGLANG_QSA_HISPARSE_V3_EVENTS",
                                      "SGLANG_QSA_HISPARSE_V3_CAPTURE",
                                      "SGLANG_QSA_P2_VALIDATE_INITIAL_PAIR")):
        raise ValidationError("source EVENTS/capture/validation barrier unexpectedly enabled")
    event_dir = directory / "source-events"
    if event_dir.is_dir() and any(event_dir.iterdir()):
        raise ValidationError("source EVENTS files were written")
    return {"status": "PASS_ARM", "chunked_prefill_size": chunk,
            "requests_complete": 4, "token_oracle_qualification": False,
            "per_request": {rid: {"ttft_ms": (row["times_ns"][0] - timeline[rid]) / 1e6,
                                  "e2e_ms": (row["times_ns"][-1] - timeline[rid]) / 1e6,
                                  "tpot": row["inter_token"],
                                  "prefill_duration_ms": row["prefill_duration_ms"],
                                  "prefill_tokens_per_s": row["prefill_tokens_per_s"]}
                            for rid, row in streams.items()},
            "steady_B4_external_intersection": {"start_ns": first, "end_ns": last,
                "per_request": {rid: stats_ns(values) for rid, values in steady.items()},
                "per_request_token_arrivals": arrivals,
                "aggregate_tokens": count, "aggregate_tokens_per_s": count * 1e9 / (last - first)},
            "subsequent_request_observer_windows": survivor,
            "submission_order": ordered, "first_token_order": first_token_order,
            "all_output_tokens_per_s_including_prefill": 4 * OUTPUT * 1e9 / (end - start),
            "gc_gate": gc, "nvml": nvml, "metrics_gate": metrics,
            "service_log_gate": service_log,
            "final_idle": json.loads((directory / "final-idle.json").read_text()),
            "output_ids": {rid: row["output_ids"] for rid, row in streams.items()}}


def output_comparison(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    if set(left) != set(right):
        raise ValidationError("arm output request identities differ")
    pairs = zip(sorted(left), sorted(right))
    rows = []
    for old, new in pairs:
        a, b = left[old], right[new]
        differences = [index for index, (x, y) in enumerate(zip(a, b)) if x != y]
        rows.append({"baseline_rid": old, "treatment_rid": new,
                     "equal": not differences, "difference_count": len(differences),
                     "first_difference": differences[0] if differences else None})
    return {"descriptive_only": True, "acceptance_gate": False, "pairs": rows}


def run(args: argparse.Namespace) -> int:
    if not args.allow_gpu:
        raise ValidationError("GPU/service path requires reviewed --allow-gpu")
    attempt = args.attempt.resolve(); attempt.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, Any] = {"schema": "qsa-hisparse-production-readiness-v1",
        "status": "preflight", "source_required": SOURCE_COMMIT,
        "arms": [name for name, _ in ARMS], "token_oracle_qualification": False,
        "chunk_performance_qualification": False, "fixed_slo": False, "production_go": False}
    write_json(attempt / "manifest.json", manifest)
    snapshot = base = models = None
    reduced: dict[str, Any] = {"schema": "qsa-hisparse-production-readiness-reduced-v1"}
    stopped = False; primary: BaseException | None = None
    try:
        prep = preflight(True)
        pid = args.production_pid or v3.runner.listener_pid(v3.PRODUCTION_PORT)
        snapshot, base = v3.runner.snapshot_service(pid, v3.PRODUCTION_SOURCE)
        v3.capture.http(v3.PRODUCTION_PORT, "/health", timeout=10)
        models = v3.capture.http(v3.PRODUCTION_PORT, "/v1/models", timeout=30)
        v3.runner.assert_test_free(v3.TEST_PORT)
        idle = v3.runner.idle_gate(v3.capture, v3.PRODUCTION_PORT, args.production_log,
                                   args.idle_seconds, args.gate_timeout)
        argvs = {name: arm_argv(snapshot["argv"], chunk) for name, chunk in ARMS}
        validate_arms(argvs[ARMS[0][0]], argvs[ARMS[1][0]])
        manifest.update(status="ready-to-stop", preflight=prep, production_snapshot=snapshot,
                        idle_gate=idle, candidate_argv=argvs)
        write_json(attempt / "manifest.json", manifest)
        stopped = True; v3.runner.stop_tree(v3.capture, pid)
        if v3.profile.gpu_owner_pids():
            raise ValidationError("GPU owners remain after production stop")
        for name, chunk in ARMS:
            arm = attempt / name
            run_service(arm, argvs[name], base, chunk, args.request_timeout, args.gc_gate_timeout)
            reduced[name] = reduce_arm(arm, chunk)
            reduced[name]["post_arm_source"] = clean_source()
            write_json(attempt / "reduced.json", reduced)
        reduced.update(status="PASS_B4_PRODUCTION_READINESS_COLLECTION",
                       output_comparison=output_comparison(
                           reduced[ARMS[0][0]]["output_ids"], reduced[ARMS[1][0]]["output_ids"]),
                       token_oracle_qualification=False, chunk_performance_qualification=False,
                       fixed_slo=False, production_go=False)
        write_json(attempt / "reduced.json", reduced); manifest["status"] = "gpu-complete"
    except BaseException as exc:
        primary = exc; manifest.update(status="FAIL", error=f"{type(exc).__name__}: {exc}")
    finally:
        if stopped and snapshot is not None and base is not None and models is not None:
            try:
                manifest["restore"] = v3.restore_production(snapshot, base, models, attempt)
                final = "FAIL_RESTORED" if primary else "PASS_B4_PRODUCTION_READINESS_COLLECTION"
                manifest["status"] = final; reduced["status"] = final
                write_json(attempt / "reduced.json", reduced)
            except BaseException as exc:
                primary = primary or exc
                manifest.update(status="BLOCKED_RESTORE", restore_error=f"{type(exc).__name__}: {exc}")
        write_json(attempt / "manifest.json", manifest)
    if primary:
        raise primary
    return 0


def self_check() -> None:
    clean_source(); requests()
    fixture = json.loads(p5a.p3b.ARGV_FIXTURE.read_text())["argv"]
    left, right = arm_argv(fixture, 4096), arm_argv(fixture, 2048)
    validate_arms(left, right)
    assert "--enable-deterministic-inference" not in left
    assert stats_ns([1_000_000, 2_000_000])["mean_ms"] == 1.5
    env = service_environment({"PYTHONPATH": "old", "SGLANG_QSA_HISPARSE_V3_EVENTS": "bad"}, "", Path("x"))
    assert not any(key in env for key in ("SGLANG_QSA_HISPARSE_V3_EVENTS",
                                           "SGLANG_QSA_HISPARSE_V3_CAPTURE",
                                           "SGLANG_QSA_P2_VALIDATE_INITIAL_PAIR"))
    labels = lambda rank, mode: {"tp_rank": str(rank), "mode": mode}
    before = [{"name": "sglang:cuda_graph_passes_total", "labels": labels(rank, "decode_cuda_graph"), "value": 0}
              for rank in (0, 1)]
    after = [{"name": "sglang:cuda_graph_passes_total", "labels": labels(rank, "decode_cuda_graph"), "value": 3}
             for rank in (0, 1)]
    assert sum(counter_delta(before, after, "decode_cuda_graph").values()) == 6
    with tempfile.TemporaryDirectory() as temporary:
        log = Path(temporary) / "server.log"
        log.write_text("Decode batch, #running-req: 4, cuda graph: True\n")
        assert validate_service_log(log)["real_B4_cuda_graph"] is True
        nvml = Path(temporary) / "nvml.jsonl"
        nvml.write_text("".join(json.dumps({"gpus": [
            {"index": 0, "used_mib": 10}, {"index": 1, "used_mib": 11}]}) + "\n"
            for _ in range(8)))
        assert validate_nvml_plateau(nvml)["0"]["tail_span_mib"] == 0
    print("self-check ok: clean source, exact two-arm argv, no EVENTS, external timing")


def failure_check() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValidationError("failure-check requires CUDA_VISIBLE_DEVICES=''")
    failure, restored = RuntimeError("injected baseline failure"), {"health_http": 200}
    with tempfile.TemporaryDirectory() as temp:
        args = argparse.Namespace(allow_gpu=True, attempt=Path(temp) / "attempt", production_pid=1,
            production_log=Path("unused"), idle_seconds=0, gate_timeout=1,
            request_timeout=1, gc_gate_timeout=1)
        with patch.object(sys.modules[__name__], "preflight", return_value={}), \
                patch.object(v3.runner, "snapshot_service", return_value=({"argv": []}, {})), \
                patch.object(v3.capture, "http", side_effect=[{}, {"data": []}]), \
                patch.object(v3.runner, "assert_test_free"), \
                patch.object(v3.runner, "idle_gate", return_value={}), \
                patch.object(sys.modules[__name__], "arm_argv", return_value=[]), \
                patch.object(v3.runner, "stop_tree"), \
                patch.object(v3.profile, "gpu_owner_pids", return_value=[]), \
                patch.object(sys.modules[__name__], "run_service", side_effect=failure) as service, \
                patch.object(v3, "restore_production", return_value=restored) as restore:
            try:
                run(args)
            except RuntimeError as exc:
                assert exc is failure
            else:
                raise AssertionError("baseline failure was swallowed")
        manifest = json.loads((args.attempt / "manifest.json").read_text())
        assert service.call_count == 1 and restore.call_count == 1 and manifest["status"] == "FAIL_RESTORED"
    print("failure-check ok: baseline failure blocks treatment and restores production once")


def prepare(output: Path) -> int:
    result: dict[str, Any] = {"schema": "qsa-hisparse-production-readiness-cpu-prep-v1",
                             "gpu_used": False, "service_used": False,
                             "source_commit": SOURCE_COMMIT}
    try:
        self_check()
        result.update(status="PASS_CPU_PREP", preflight=preflight(True),
                      future_gpu_arms=[name for name, _ in ARMS],
                      token_oracle_qualification=False,
                      chunk_performance_qualification=False, production_go=False)
    except BaseException as exc:
        result.update(status="FAIL_CPU_PREP", error=f"{type(exc).__name__}: {exc}")
    write_json(output, result)
    return 0 if result["status"] == "PASS_CPU_PREP" else 2


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(); sub = value.add_subparsers(dest="command", required=True)
    sub.add_parser("self-check"); sub.add_parser("failure-check")
    prep = sub.add_parser("prepare"); prep.add_argument("--output", type=Path, required=True)
    runp = sub.add_parser("run"); runp.add_argument("--allow-gpu", action="store_true")
    runp.add_argument("--attempt", type=Path, required=True); runp.add_argument("--production-pid", type=int)
    runp.add_argument("--production-log", type=Path, default=v3.PRODUCTION_LOG)
    runp.add_argument("--idle-seconds", type=float, default=3); runp.add_argument("--gate-timeout", type=float, default=180)
    runp.add_argument("--request-timeout", type=float, default=7200)
    runp.add_argument("--gc-gate-timeout", type=float, default=60)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "self-check": self_check(); return 0
    if args.command == "failure-check": failure_check(); return 0
    if args.command == "prepare": return prepare(args.output)
    def interrupted(signum: int, frame: Any) -> None:
        raise KeyboardInterrupt(f"signal {signum}; restore required")
    signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
