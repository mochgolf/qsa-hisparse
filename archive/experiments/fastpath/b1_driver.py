#!/usr/bin/env python3
"""Guarded two-arm B1 HiSparse ragged-FA2 acceptance."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
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
CANDIDATE_COMMIT = "a8987a6fd3057edab2bef029fa36ac420e550dba"
V3_PATH = Path(os.environ.get("QSA_HISPARSE_V3_DRIVER", "<V3_DRIVER>"))
PROD_PATH = Path(
    os.environ.get("QSA_HISPARSE_PRODUCTION_DRIVER", "<PRODUCTION_DRIVER>")
)
REQUEST_PATH = Path(os.environ.get("QSA_HISPARSE_REQUEST_JSON", "<REQUEST_JSON>"))
COMPONENT = HERE / "b1_component_check.py"
PRODUCTION_LOG = Path(os.environ.get("QSA_HISPARSE_BASELINE_LOG", "<BASELINE_LOG>"))
PROMPT, OUTPUT = 38185, 768
HISTORICAL_RATE = 104.931471


def load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


v3 = load("b1_fast_v3", V3_PATH)
prod = load("b1_fast_prod", PROD_PATH)


class ValidationError(RuntimeError):
    pass


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")


def source(path: Path, commit: str) -> dict[str, Any]:
    value = v3.require_clean_source(path)
    if value.get("commit") != commit:
        raise ValidationError(f"{path}: {value.get('commit')} != {commit}")
    return value


def prompt_ids(model: str) -> list[int]:
    from transformers import AutoTokenizer

    request = json.loads(REQUEST_PATH.read_text())
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    rendered = tokenizer.apply_chat_template(
        request["messages"],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    ids = tokenizer.encode(rendered, add_special_tokens=False)
    if len(ids) != PROMPT:
        raise ValidationError(
            f"rendered prompt has {len(ids)} tokens, expected {PROMPT}"
        )
    return ids


def payload(ids: Sequence[int], rid: str) -> dict[str, Any]:
    return {
        "rid": rid,
        "input_ids": list(ids),
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


def test_argv(original: Sequence[str]) -> list[str]:
    result = v3.runner.replace_option(original, "--port", ("30001",))
    if (
        v3.arg_value(result, "--tp-size") != "2"
        or v3.arg_value(result, "--cuda-graph-backend-decode") != "full"
        or v3.arg_value(result, "--cuda-graph-backend-prefill") != "disabled"
        or "1" not in v3.runner.arg_values(result, "--cuda-graph-bs-decode")
        or v3.runner.strip_speculative_options(result) != result
    ):
        raise ValidationError("production argv is not TP2/no-MTP/full-graph B1")
    return result


def arm_environment(original: Mapping[str, str], path: Path) -> dict[str, str]:
    result = dict(original)
    result.update(PYTHONPATH=str(path / "python"), PWD=str(path))
    if (
        result.get("SGLANG_QSA_HISPARSE_V3") != "p2-offload"
        or result.get("SGLANG_QSA_HISPARSE_V3_OBSERVE") != "light"
    ):
        raise ValidationError("production environment is not p2-offload/light")
    return result


def metrics(path: Path) -> list[dict[str, Any]]:
    return prod.metrics_snapshot(path)


def stream(
    port: int, body: Mapping[str, Any], directory: Path, label: str, timeout: float
) -> dict[str, Any]:
    result = v3.request_stream(port, body, directory, label, PROMPT, OUTPUT, timeout)
    rows = [
        json.loads(line)
        for line in Path(result["events_path"]).read_text().splitlines()
    ]
    ids: list[int] = []
    times: list[float] = []
    for row in rows:
        observed = v3.runner._event_output_ids(row)
        if observed is None:
            continue
        before = len(ids)
        ids = v3.runner._merge_stream_output_ids(ids, observed)
        if len(ids) != before + 1:
            raise ValidationError(f"{label}: SSE did not add exactly one token")
        times.append(float(row["_elapsed_s"]))
    if (
        len(ids) != OUTPUT
        or len(times) != OUTPUT
        or any(b <= a for a, b in zip(times, times[1:]))
    ):
        raise ValidationError(f"{label}: invalid cumulative IDs/timestamps")
    full_rate = (OUTPUT - 1) / (times[-1] - times[0])
    steady_rate = (OUTPUT - 65) / (times[-1] - times[64])
    result.update(
        full_decode_tok_s=full_rate, steady_65_768_tok_s=steady_rate, output_ids=ids
    )
    write(directory / f"{label}-result.json", result)
    with (directory / "requests.jsonl").open("a") as sink:
        sink.write(
            json.dumps({k: v for k, v in result.items() if k != "output_ids"}) + "\n"
        )
    return result


def wait_recovered(
    directory: Path, before: Sequence[Mapping[str, Any]], log: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    idle = v3.runner.idle_gate(v3.capture, 30001, log, 1, 60)
    names = (
        "sglang:kv_available_tokens",
        "sglang:mamba_available_tokens",
        "sglang:num_running_reqs",
        "sglang:num_queue_reqs",
    )
    expected = {name: prod.only_value(before, name) for name in names}
    deadline = time.monotonic() + 40
    while True:
        after = metrics(directory / "metrics-after.prom")
        actual = {name: prod.only_value(after, name) for name in names}
        if actual == expected:
            return after, {"idle": idle, "before": expected, "after": actual}
        if time.monotonic() >= deadline:
            raise ValidationError(
                f"capacity metrics did not recover: {actual} != {expected}"
            )
        time.sleep(1)


def validate_route(
    before: Sequence[Mapping[str, Any]], after: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    graph = prod.counter_delta(before, after, "decode_cuda_graph")
    eager = prod.counter_delta(before, after, "decode_none")
    ranks = {dict(key).get("tp_rank") for key, value in graph.items() if value > 0}
    if ranks != {"0", "1"} or sum(graph.values()) <= 0 or sum(eager.values()) != 0:
        raise ValidationError(
            f"decode route differs: ranks={ranks}, graph={graph}, eager={eager}"
        )
    return {
        "decode_cuda_graph_delta": sum(graph.values()),
        "decode_cuda_graph_tp_ranks": sorted(ranks),
        "decode_none_delta": sum(eager.values()),
    }


def validate_log(path: Path, candidate: bool) -> dict[str, Any]:
    text = path.read_text(errors="replace")
    active = set(re.findall(r"TP([01]).*QSA HiSparse SM89 ragged FA2 B1 active", text))
    fatal = re.findall(
        r"(?im)^.*(?:CUDA out of memory|device-side assert|graph capture failed|"
        r"falling back.*graph|worker.*exited unexpectedly|UnexpectedEOF).*$",
        text,
    )
    if fatal or (candidate and active != {"0", "1"}) or (not candidate and active):
        raise ValidationError(
            f"server log validation failed: active={active}, fatal={fatal[:1]}"
        )
    return {"fast_path_active_tp_ranks": sorted(active), "fatal_patterns": fatal}


def run_arm(
    name: str,
    path: Path,
    commit: str,
    argv: Sequence[str],
    environment: Mapping[str, str],
    ids: Sequence[int],
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
        info = v3.capture.http(30001, "/server_info", timeout=30)
        write(directory / "server-info.json", info)
        before = metrics(directory / "metrics-before.prom")
        warmup = stream(
            30001, payload(ids, f"b1-fast-{name}-warmup"), directory, "warmup", timeout
        )
        measured = [
            stream(
                30001,
                payload(ids, f"b1-fast-{name}-measured-{index}"),
                directory,
                f"measured-{index}",
                timeout,
            )
            for index in (1, 2)
        ]
        after, recovery = wait_recovered(directory, before, log)
        rates = [row["steady_65_768_tok_s"] for row in measured]
        return {
            "name": name,
            "source": str(path),
            "source_commit": commit,
            "warmup": {
                "full_decode_tok_s": warmup["full_decode_tok_s"],
                "steady_65_768_tok_s": warmup["steady_65_768_tok_s"],
            },
            "measured": measured,
            "mean_steady_65_768_tok_s": statistics.fmean(rates),
            "mean_full_decode_tok_s": statistics.fmean(
                row["full_decode_tok_s"] for row in measured
            ),
            "within_arm_ids_equal": measured[0]["output_ids"]
            == measured[1]["output_ids"],
            "route": validate_route(before, after),
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


def run(args: argparse.Namespace) -> int:
    if not args.allow_gpu:
        raise ValidationError("pass --allow-gpu after review")
    attempt = Path(args.attempt).resolve()
    attempt.mkdir(parents=True, exist_ok=False)
    state: dict[str, Any] = {
        "status": "PREFLIGHT",
        "started_at": v3.stamp(),
        "arms": {},
    }
    snapshot = original_env = initial_models = None
    stopped = False
    error: BaseException | None = None
    try:
        state["sources"] = {
            "baseline": source(BASE, BASE_COMMIT),
            "candidate": source(CANDIDATE, CANDIDATE_COMMIT),
        }
        pid = v3.runner.listener_pid(30000)
        snapshot, original_env = v3.runner.snapshot_service(pid, BASE)
        initial_models = v3.capture.http(30000, "/v1/models", timeout=30)
        model = v3.arg_value(snapshot["argv"], "--model-path")
        if not model:
            raise ValidationError("production model path is missing")
        ids = prompt_ids(model)
        write(
            attempt / "prompt-info.json",
            {"tokens": len(ids), "source": str(REQUEST_PATH)},
        )
        argv = test_argv(snapshot["argv"])
        state.update(
            production_snapshot=snapshot,
            idle=v3.runner.idle_gate(v3.capture, 30000, PRODUCTION_LOG, 3, 180),
            test_argv=argv,
        )
        v3.runner.assert_test_free(30001)
        write(attempt / "manifest.json", state)
        stopped = True
        v3.runner.stop_tree(v3.capture, pid)
        if v3.profile.gpu_owner_pids():
            raise ValidationError("GPU owners remain after stopping production")
        component_path = attempt / "component.json"
        subprocess.run(
            [
                str(v3.PYTHON),
                str(COMPONENT),
                "--source",
                str(CANDIDATE),
                "--output",
                str(component_path),
            ],
            check=True,
            env={**original_env, "CUDA_VISIBLE_DEVICES": "0"},
        )
        state["component"] = json.loads(component_path.read_text())
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
                arm_environment(original_env, path),
                ids,
                attempt,
                args.timeout,
            )
        baseline, candidate = state["arms"]["baseline"], state["arms"]["candidate"]
        gain = (
            candidate["mean_steady_65_768_tok_s"] / baseline["mean_steady_65_768_tok_s"]
            - 1
        )
        state["comparison"] = {
            "steady_gain_fraction": gain,
            "candidate_vs_historical_fraction": candidate["mean_steady_65_768_tok_s"]
            / HISTORICAL_RATE
            - 1,
            "historical_reference_tok_s": HISTORICAL_RATE,
            "cross_arm_ids_equal": [
                left["output_ids"] == right["output_ids"]
                for left, right in zip(baseline["measured"], candidate["measured"])
            ],
            "integration_performance_gate": gain >= 0.10,
            "historical_parity_gate": candidate["mean_steady_65_768_tok_s"]
            >= 0.95 * HISTORICAL_RATE,
        }
        if gain < 0.10:
            raise ValidationError(f"candidate B1 gain {gain:.2%} is below 10%")
        state["status"] = "PASS_B1_FAST_PATH"
    except BaseException as exc:
        error = exc
        state.update(
            status="FAIL_RESTORE_PENDING", error=f"{type(exc).__name__}: {exc}"
        )
    finally:
        if (
            stopped
            and snapshot is not None
            and original_env is not None
            and initial_models is not None
        ):
            try:
                state["restore"] = v3.restore_production(
                    snapshot, original_env, initial_models, attempt
                )
                v3.runner.assert_test_free(30001)
                pid = int(state["restore"]["pid"])
                family = {
                    pid,
                    *(
                        child.pid
                        for child in psutil.Process(pid).children(recursive=True)
                    ),
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
                if state["status"] == "PASS_B1_FAST_PATH":
                    state["status"] = "PASS_B1_FAST_PATH_RESTORED"
                else:
                    state["status"] = "FAIL_RESTORED"
            except BaseException as restore_error:
                state.update(
                    status="BLOCKED_RESTORE",
                    restore_error=f"{type(restore_error).__name__}: {restore_error}",
                )
                error = error or restore_error
        state["ended_at"] = v3.stamp()
        write(attempt / "manifest.json", state)
    if error is not None:
        raise error
    return 0


def self_check() -> None:
    argv = test_argv(
        [
            "python",
            "-m",
            "sglang.launch_server",
            "--port",
            "30000",
            "--tp-size",
            "2",
            "--cuda-graph-backend-decode",
            "full",
            "--cuda-graph-backend-prefill",
            "disabled",
            "--cuda-graph-bs-decode",
            "1",
            "2",
        ]
    )
    assert v3.arg_value(argv, "--port") == "30001"
    assert len(payload([1, 2], "x")["input_ids"]) == 2
    print(json.dumps({"status": "PASS_CPU_SELF_CHECK", "gpu_used": False}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("self-check")
    execute = sub.add_parser("run")
    execute.add_argument("--allow-gpu", action="store_true")
    execute.add_argument("--attempt", required=True)
    execute.add_argument("--timeout", type=float, default=600)
    options = parser.parse_args()
    if options.command == "self-check":
        self_check()
    else:
        raise SystemExit(run(options))
