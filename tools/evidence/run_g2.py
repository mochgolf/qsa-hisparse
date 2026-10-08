#!/usr/bin/env python3
"""Run one arm of the G2-2 deterministic HiSparse comparison (baseline.md steps 9–14).

    run_g2.py --arm {fork,plugin} --base-profile <service/candidate-acceptance.json>
              --fixtures <frozen fixtures.json> --output <new arm directory>

The private base profile supplies the unchanged server command and
environment (deterministic, p2-offload, strict observation, 8 GiB host
prefixes). Per arm: fork runs ``<fork>/python``; plugin runs
``<plugin>/src:<pin>/python`` through the launcher with
``SGLANG_QSA_MODEL_COMPAT=1``. Both arms load the W8 observer and write
events, observer records, server info and the four fork harness reports into
the arm directory. Compare afterwards with
``compare.py <fork> <plugin> --require-observer 2``.

Since Phase 5 the fork arm is production's code (``FORK_ROOT``, production
``897286b12a``, whose ``test/manual`` harnesses equal the fork's), both arms
use production's interpreter, and the plugin arm runs on the pin
``35f3c96ff4``.
"""

import argparse
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_compat import preflight, stop, wait_ready  # noqa: E402

QWEN = Path(__file__).resolve().parents[3]
PLUGIN_ROOT = Path(__file__).resolve().parents[2]
FORK_ROOT = QWEN / ".worktrees" / "sglang-dsh-production-20261004"  # production
PIN_ROOT = QWEN / ".worktrees" / "sglang-main-35f3c96ff4"
PYTHON = QWEN / "results" / "dsh-maintenance-20261004" / "upstream-runtime-env" / "bin" / "python"


def server(arm, base, output):
    command = base["command"]
    if command[1:3] != ["-m", "sglang.launch_server"]:
        raise SystemExit("base profile command is not '<python> -m sglang.launch_server'")
    args = command[3:]
    env = {k: v for k, v in {**os.environ, **base["environment"]}.items() if k != "SGLANG_PLUGINS"}
    if env.get("SGLANG_QSA_HISPARSE_V3") != "p2-offload" or "--enable-deterministic-inference" not in args:
        raise SystemExit("base profile is not the deterministic p2-offload profile")
    env.pop("SGLANG_QSA_MODEL_COMPAT", None)
    site = PLUGIN_ROOT / "tools" / "evidence" / "site"
    env.update(
        SGLANG_QSA_HISPARSE_V3_EVENTS=str(output / "events"),
        QSA_EVIDENCE_OBSERVER_DIR=str(output / "observer"),
        PYTHONDONTWRITEBYTECODE="1",
        SGLANG_CACHE_DIR=str(output / "cache"),
    )
    if arm == "fork":
        env["PYTHONPATH"] = os.pathsep.join([str(site), str(FORK_ROOT / "python")])
        return [str(PYTHON), "-m", "sglang.launch_server", *args], env, None
    env["SGLANG_QSA_MODEL_COMPAT"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(site), str(PLUGIN_ROOT / "src"), str(PIN_ROOT / "python")]
    )
    run_dir = output / "launcher"
    argv = [str(PYTHON), "-m", "sglang_qsa_hisparse.launch", "--run-dir", str(run_dir), "--", *args]
    return argv, env, run_dir / "ready.json"


def _tree(root, path):
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", f"HEAD:{path}"], capture_output=True, text=True
    )
    dirty = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--", path], capture_output=True, text=True
    ).stdout.strip()
    return {"tree": result.stdout.strip(), "dirty": bool(dirty)}


def provenance(arm, command, env):
    """Command, relevant environment and source trees of this arm."""
    keep = ("SGLANG_", "QSA_", "CUDA", "PYTHON")
    return {
        "arm": arm,
        "argv": command,
        "environment": {k: v for k, v in sorted(env.items()) if k.startswith(keep) or k == "PATH"},
        "sources": {
            "fork_python": _tree(FORK_ROOT, "python"),
            "pin_python": _tree(PIN_ROOT, "python"),
            "plugin_src": _tree(PLUGIN_ROOT, "src"),
        },
    }


def harness(output, url, fixtures, name, *args):
    manual = FORK_ROOT / "test" / "manual"
    with (output / f"{name}.log").open("w") as log:
        return subprocess.run(
            [str(PYTHON), str(manual / f"qsa_hisparse_prefix_{name}.py"), *args],
            cwd=manual,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            stdout=log,
            stderr=subprocess.STDOUT,
        ).returncode


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--arm", choices=("fork", "plugin"), required=True)
    parser.add_argument("--base-profile", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    base = json.loads(args.base_profile.read_text())
    output = args.output.resolve()
    output.mkdir(parents=True)  # Refuses an existing arm directory.
    (output / "events").mkdir()
    (output / "observer").mkdir()
    host, port = base["listen"]["host"], base["listen"]["port"]
    url = f"http://{host}:{port}"
    preflight(host, port)
    command, env, ready_file = server(args.arm, base, output)
    (output / "run.json").write_text(json.dumps(provenance(args.arm, command, env), indent=2, sort_keys=True))

    with (output / "server.log").open("w") as log:
        process = subprocess.Popen(
            command, env=env, cwd=output, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    results = {}
    try:
        wait_ready(process, url + base["readiness"]["path"], base["readiness"], ready_file)
        with urllib.request.urlopen(url + "/get_server_info", timeout=30) as response:
            (output / "server_info.json").write_bytes(response.read())
        fixtures = str(args.fixtures)
        events = str(output / "events")
        results["concurrency"] = harness(
            output, url, fixtures, "concurrency", "--url", url, "--fixtures", fixtures,
            "--events", events, "--output", str(output / "actual-concurrency.json"))
        results["lifecycle"] = harness(
            output, url, fixtures, "lifecycle", "--url", url, "--fixtures", fixtures,
            "--output", str(output / "lifecycle.json"))
        results["acceptance"] = harness(
            output, url, fixtures, "acceptance", "qualify", "--url", url, "--fixtures", fixtures,
            "--max-prefix", "262016", "--concurrency", "8", "--continue-on-mismatch",
            "--output", str(output / "qualification.json"))
        results["ledger"] = harness(
            output, url, fixtures, "ledger", events, "--output", str(output / "ledger.json"))
    finally:
        stop(process, base["stop_timeout_seconds"])
        (output / "harness-status.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results))
    return 0 if results and not any(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
