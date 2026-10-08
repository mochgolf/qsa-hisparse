#!/usr/bin/env python3
"""G2-3 native checks (docs/baseline.md step 16): production against the plugin.

    run_g23.py run --arm {fork,plugin} --base-profile <production service profile>
                   --fixtures <frozen fixtures.json> --output <new arm directory>
                   [--port 8082] [--ready-timeout SECONDS] [--dry-run]
    run_g23.py compare <fork arm directory> <plugin arm directory>

The private base profile is production's native service profile (production
weights, no deterministic inference, light observation, NUMA interleave). Its
command may be wrapped (``numactl --interleave=all <python> -m
sglang.launch_server ...``); the wrapper, arguments and environment are kept,
except ``--port`` (the arm port; production's 8081 is refused). The fork arm
runs production's code (run_g2.FORK_ROOT/python); the plugin arm runs
``<plugin>/src:<pin>/python`` through the launcher with
``SGLANG_QSA_MODEL_COMPAT=1``. Both use run_g2.PYTHON, ``PYTHONDONTWRITEBYTECODE=1``
and ``SGLANG_CACHE_DIR`` inside the arm directory.

``run``: checks that the fixtures' tokenizer hash equals the profile model's
``tokenizer.json``; refuses a busy GPU or port (run_compat.preflight); starts
the server in its own session; waits for health and, for the plugin arm, the
launcher's ``ready.json``; saves ``/get_server_info``; records the numactl
arguments of the scheduler wrappers SGLang wrote to
``/tmp/sglang_temp_file_*.sh`` since the start (``numactl.json``); runs
production's latency and lifecycle harnesses, the 8x128-token native
concurrency check and the OpenAI smoke (the logic of the historical
``final-native-concurrency.py`` and ``final-openai-smoke.py``, URL and model
from the profile); stops its process group. ``g23.json`` holds each check's
status; the exit status is 0 only if all passed.

``compare``: both arms passed every check; equal prompt and cached token
counts per request; equal numactl arguments; latency medians side by side
(no threshold). Native outputs are not compared.
"""

import argparse
import concurrent.futures
import glob
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_g2  # noqa: E402
from run_compat import preflight, stop, wait_ready  # noqa: E402

SERVER = ["-m", "sglang.launch_server"]
NUMACTL_SCRIPTS = "/tmp/sglang_temp_file_*.sh"  # numa_utils._create_numactl_executable
SMOKE_SYSTEM = "Answer concisely. Reference records:\n" + "amber=17; cyan=25; violet=42.\n" * 128


def server(arm, base, output, port):
    """(argv, env, ready file) of one arm from the production profile."""
    command = base["command"]
    start = next(
        (i for i in range(len(command) - 2) if command[i + 1 : i + 3] == SERVER), None
    )
    if start is None:
        raise SystemExit("base profile command does not run '<python> -m sglang.launch_server'")
    wrapper, args = command[:start], list(command[start + 3 :])
    env = {
        k: v
        for k, v in {**os.environ, **base["environment"]}.items()
        if k not in ("SGLANG_PLUGINS", "SGLANG_QSA_MODEL_COMPAT")
    }
    if env.get("SGLANG_QSA_HISPARSE_V3") != "p2-offload" or "--enable-deterministic-inference" in args:
        raise SystemExit("base profile is not the native p2-offload profile")
    if args.count("--port") != 1:
        raise SystemExit("base profile must contain --port once")
    args[args.index("--port") + 1] = str(port)
    env.update(PYTHONDONTWRITEBYTECODE="1", SGLANG_CACHE_DIR=str(output / "cache"))
    python = str(run_g2.PYTHON)
    if arm == "fork":
        env["PYTHONPATH"] = str(run_g2.FORK_ROOT / "python")
        return [*wrapper, python, *SERVER, *args], env, None
    env["SGLANG_QSA_MODEL_COMPAT"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(run_g2.PLUGIN_ROOT / "src"), str(run_g2.PIN_ROOT / "python")]
    )
    run_dir = output / "launcher"
    launcher = [python, "-m", "sglang_qsa_hisparse.launch", "--run-dir", str(run_dir), "--"]
    return [*wrapper, *launcher, *args], env, run_dir / "ready.json"


def check_tokenizer(base, fixtures):
    args = base["command"]
    model = Path(args[args.index("--model-path") + 1])
    actual = hashlib.sha256((model / "tokenizer.json").read_bytes()).hexdigest()
    if actual != fixtures["tokenizer_sha256"]:
        raise SystemExit("the fixtures were not tokenized with the profile model's tokenizer")


def numactl_arguments(since):
    """numactl arguments of the wrapper scripts SGLang wrote since ``since``."""
    found = []
    for path in glob.glob(NUMACTL_SCRIPTS):
        if os.path.getmtime(path) >= since:
            for line in Path(path).read_text().splitlines():
                if line.startswith("exec numactl "):
                    found.append(" ".join(line.split()[2:-2]))  # Drop the python and "$@".
    return sorted(found)


def post(url, path, payload, timeout):
    request = urllib.request.Request(
        url + path, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def native_concurrency(url, cases):
    """Seed prefix-65536-copy, then 8 barrier-aligned 128-token requests."""
    case = next(c for c in cases if c["name"] == "prefix-65536-copy")
    namespace = f"native-concurrency-{time.time_ns()}"

    def generate(ids, count, rid):
        payload = {"input_ids": ids, "cache_salt": namespace, "rid": rid,
                   "sampling_params": {"temperature": 0, "max_new_tokens": count, "ignore_eos": True}}  # fmt: skip
        result = post(url, "/generate", payload, 300)
        assert len(result["output_ids"]) == count, result
        assert result["meta_info"]["finish_reason"]["type"] not in ("abort", "error"), result
        return result

    report = {"seed": generate(case["prefix_ids"], 1, namespace + "-seed")}
    barrier = threading.Barrier(8)

    def submit(i):
        barrier.wait(timeout=30)
        return generate(case["input_ids"], 128, f"{namespace}-{i}")

    start = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        report["responses"] = list(pool.map(submit, range(8)))
    report["wall_seconds"] = time.monotonic() - start
    assert all(r["meta_info"]["cached_tokens"] == 65536 for r in report["responses"]), report
    return report


def openai_smoke(url, model):
    """A cold and a warm chat request; both answer 42; only the warm one reuses."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SMOKE_SYSTEM},
            {"role": "user", "content": "What is 17 plus 25? Reply with only the number."},
        ],
        "temperature": 0,
        "max_tokens": 64,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    report = {"request": payload, "responses": []}
    for label in ("cold", "warm"):
        start = time.monotonic()
        result = post(url, "/v1/chat/completions", payload, 120)
        report["responses"].append(
            {"label": label, "wall_seconds": time.monotonic() - start, "response": result}
        )
        assert result["choices"][0]["message"]["content"].strip() == "42", result
        assert result["choices"][0]["finish_reason"] == "stop", result
        details = result["usage"]["prompt_tokens_details"]
        reused = 0 if details is None else details["cached_tokens"]
        assert (reused == 0) if label == "cold" else (reused > 0), result
    return report


def checked(output, name, function, *args):
    """Run an in-process check; write its report; return 0 or 1."""
    report = {"passed": False}
    try:
        report.update(function(*args))
        report["passed"] = True
    except Exception as error:  # Recorded; the arm's status fails.
        report["error"] = repr(error)
    (output / f"{name}.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["passed"] else 1


def harness(output, url, fixtures, name):
    manual = run_g2.FORK_ROOT / "test" / "manual"
    with (output / f"{name}.log").open("w") as log:
        return subprocess.run(
            [str(run_g2.PYTHON), str(manual / f"qsa_hisparse_prefix_{name}.py"), "--url", url,
             "--fixtures", str(fixtures), "--output", str(output / f"{name}.json")],
            cwd=manual, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            stdout=log, stderr=subprocess.STDOUT,
        ).returncode  # fmt: skip


def run(args):
    base = json.loads(args.base_profile.read_text())
    fixtures = json.loads(args.fixtures.read_text())
    output = args.output.resolve()
    command, env, ready_file = server(args.arm, base, output, args.port)
    readiness = dict(base["readiness"])
    if args.ready_timeout:
        readiness["timeout_seconds"] = args.ready_timeout
    if args.dry_run:
        print(json.dumps({"arm": args.arm, "argv": command, "PYTHONPATH": env["PYTHONPATH"]}, indent=2))
        return 0
    check_tokenizer(base, fixtures)
    host = base["listen"]["host"]
    url = f"http://{host}:{args.port}"
    preflight(host, args.port)
    output.mkdir(parents=True)  # Refuses an existing arm directory.
    (output / "run.json").write_text(
        json.dumps(run_g2.provenance(args.arm, command, env), indent=2, sort_keys=True)
    )
    started = time.time()
    with (output / "server.log").open("w") as log:
        process = subprocess.Popen(
            command, env=env, cwd=output, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )  # fmt: skip
    status = {}
    try:
        wait_ready(process, url + readiness["path"], readiness, ready_file)
        with urllib.request.urlopen(url + "/get_server_info", timeout=30) as response:
            (output / "server_info.json").write_bytes(response.read())
        (output / "numactl.json").write_text(json.dumps(numactl_arguments(started), indent=2) + "\n")
        status["latency"] = harness(output, url, args.fixtures.resolve(), "latency")
        status["lifecycle"] = harness(output, url, args.fixtures.resolve(), "lifecycle")
        status["concurrency"] = checked(output, "concurrency", native_concurrency, url, fixtures["cases"])
        status["smoke"] = checked(output, "smoke", openai_smoke, url, base["readiness"]["model"])
    finally:
        stop(process, base["stop_timeout_seconds"])
        (output / "g23.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status))
    return 0 if status and not any(status.values()) else 1


def accounting(arm):
    """Prompt and cached token counts of every request, in harness order."""
    load = lambda name: json.loads((arm / f"{name}.json").read_text())  # noqa: E731
    counts = {}
    for case in load("latency")["cases"]:
        for i, pair in enumerate(case["pairs"]):
            for kind in ("cold", "warm"):
                counts[f"latency/{case['prefix_length']}/{i}/{kind}"] = (
                    pair[kind]["prompt_tokens"], pair[kind]["cached_tokens"])  # fmt: skip
    lifecycle = load("lifecycle")
    for key in ("seed", "warm", "after_flush", "after_reseed", "after_abort"):
        counts[f"lifecycle/{key}"] = (lifecycle[key]["prompt_tokens"], lifecycle[key]["cached_tokens"])
    for item in lifecycle["logprob"]:
        counts[f"lifecycle/logprob/{item['start']}"] = (
            item["meta_info"]["prompt_tokens"], item["meta_info"]["cached_tokens"])  # fmt: skip
    for i, response in enumerate(load("concurrency")["responses"]):
        info = response["meta_info"]
        counts[f"concurrency/{i}"] = (info["prompt_tokens"], info["cached_tokens"])
    for item in load("smoke")["responses"]:
        usage = item["response"]["usage"]
        details = usage["prompt_tokens_details"]
        counts[f"smoke/{item['label']}"] = (
            usage["prompt_tokens"], 0 if details is None else details["cached_tokens"])  # fmt: skip
    return counts


def compare(fork, plugin):
    problems = []
    for arm in (fork, plugin):
        status = json.loads((arm / "g23.json").read_text())
        if set(status) != {"latency", "lifecycle", "concurrency", "smoke"} or any(status.values()):
            problems.append(f"{arm.name}: checks did not all pass: {status}")
    if problems:
        print("\n".join(f"DIFF {p}" for p in problems) + f"\nFAIL: {len(problems)} problem(s)")
        return 1
    f, p = accounting(fork), accounting(plugin)
    problems += [f"{key}: F {f.get(key)} P {p.get(key)}" for key in sorted(f.keys() | p.keys())
                 if f.get(key) != p.get(key)]  # fmt: skip
    numactl = [json.loads((arm / "numactl.json").read_text()) for arm in (fork, plugin)]
    if numactl[0] != numactl[1]:
        problems.append(f"numactl arguments: F {numactl[0]} P {numactl[1]}")
    print(f"numactl (both arms): {numactl[0]}")
    print("latency medians (s), no threshold: length  F cold  P cold  F warm  P warm")
    cases = [json.loads((arm / "latency.json").read_text())["cases"] for arm in (fork, plugin)]
    for fc, pc in zip(*cases):
        print(f"  {fc['prefix_length']:>7}  {fc['median_cold_seconds']:.3f}  {pc['median_cold_seconds']:.3f}"
              f"  {fc['median_warm_seconds']:.3f}  {pc['median_warm_seconds']:.3f}")  # fmt: skip
    print(f"{len(f)} requests' token accounting compared")
    for problem in problems:
        print("DIFF", problem)
    print("PASS" if not problems else f"FAIL: {len(problems)} problem(s)")
    return 0 if not problems else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("run")
    one.add_argument("--arm", choices=("fork", "plugin"), required=True)
    one.add_argument("--base-profile", type=Path, required=True)
    one.add_argument("--fixtures", type=Path, required=True)
    one.add_argument("--output", type=Path, required=True)
    one.add_argument("--port", type=int, default=8082)
    one.add_argument("--ready-timeout", type=float, help="seconds (default: the profile's)")
    one.add_argument("--dry-run", action="store_true")
    both = sub.add_parser("compare")
    both.add_argument("fork", type=Path)
    both.add_argument("plugin", type=Path)
    args = parser.parse_args(argv)
    if args.command == "compare":
        return compare(args.fork, args.plugin)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
