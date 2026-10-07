#!/usr/bin/env python3
"""Run one arm of the reduced deterministic compat-only profile (G2-2).

    run_compat.py --arm {fork,plugin} --base-profile <service/candidate-acceptance.json>
                  --fixtures <frozen fixtures.json> --output <new arm directory>
                  [--python ...] [--fork-root ...] [--pin-root ...] [--dry-run]

compat_profile.json fixes the profile; the private base profile supplies the
model path. Run the fork arm, then the plugin arm, into sibling directories,
then ``compare.py <fork dir> <plugin dir>``. Each run:

1. checks the fixture file's SHA-256 and that the profile's rule keeps
   exactly the recorded cases, which the harness's --max-prefix selects;
2. builds the server command: the base command with set_flags applied after
   checking the base values the derivation relies on; every SGLANG_QSA_*
   variable and SGLANG_PLUGINS removed, then the arm's variables set; the
   arm's PYTHONPATH (fork: <fork>/python; plugin: <plugin>/src:<pin>/python
   through the W7 launcher); PYTHONDONTWRITEBYTECODE=1 and SGLANG_CACHE_DIR
   inside the arm directory;
3. refuses to start if the port is production's 8081 or in use, or if
   nvidia-smi lists any compute process;
4. starts the server in its own session, logging to server.log, waits for the
   base profile's readiness path, saves /get_server_info to server_info.json,
   and runs the fork harness ``baseline`` (two salted cold requests per case)
   into compat-cold.json;
5. stops only the process group it started (SIGTERM, SIGKILL after the base
   profile's stop timeout). Exits with the harness status.
"""

import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parents[2]
PROFILE = Path(__file__).resolve().with_name("compat_profile.json")
HARNESS = Path("test/manual/qsa_hisparse_prefix_acceptance.py")
PRODUCTION_PORT = 8081


def edited_args(command, profile):
    """Base server arguments with set_flags applied; refuse an unexpected base."""
    if command[1:3] != ["-m", "sglang.launch_server"]:
        raise SystemExit("base profile command is not '<python> -m sglang.launch_server'")
    args = list(command[3:])
    for flag, value in profile["base_profile"]["requires"].items():
        where = [i for i, arg in enumerate(args) if arg == flag]
        if len(where) != 1 or (
            value is not None and args[where[0] + 1 : where[0] + 2] != [value]
        ):
            raise SystemExit(f"base profile must contain {flag} {value or ''} once")
    for flag, value in profile["set_flags"].items():
        args[args.index(flag) + 1] = value
    return args


def server_command(arm, base, profile, *, python, roots, output):
    args = edited_args(base["command"], profile)
    env = {
        key: value
        for key, value in {**os.environ, **base["environment"]}.items()
        if not key.startswith("SGLANG_QSA_") and key != "SGLANG_PLUGINS"
    }
    env.update(profile["environment"][arm])
    env.update(PYTHONDONTWRITEBYTECODE="1", SGLANG_CACHE_DIR=str(output / "cache"))
    if arm == "fork":
        env["PYTHONPATH"] = str(roots["fork"] / "python")
        argv = [python, "-m", "sglang.launch_server", *args]
    else:
        env["PYTHONPATH"] = os.pathsep.join(
            [str(roots["plugin"] / "src"), str(roots["pin"] / "python")]
        )
        argv = [python, "-m", "sglang_qsa_hisparse.launch", "--", *args]
    return argv, env


def checked_fixtures(path, profile):
    spec = profile["fixtures"]
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != spec["sha256"]:
        raise SystemExit(f"{path} is not the frozen fixture file")
    cases = json.loads(data)["cases"]
    context = profile["geometry"]["context_length"]
    kept = [c for c in cases if len(c["input_ids"]) + c["max_new_tokens"] <= context]
    selected = [c for c in cases if c["prefix_length"] <= spec["max_prefix"]]
    largest = max(len(c["input_ids"]) + c["max_new_tokens"] for c in kept)
    if (
        kept != selected
        or len(kept) != spec["kept_cases"]
        or largest != spec["largest_request_tokens"]
    ):
        raise SystemExit("fixture rule, harness selection and profile record differ")
    return kept


def option(args, flag):
    return args[args.index(flag) + 1]


def preflight(host, port):
    if port == PRODUCTION_PORT:
        raise SystemExit(f"refusing the production port {PRODUCTION_PORT}")
    with socket.socket() as probe:
        probe.settimeout(1)
        if probe.connect_ex((host, port)) == 0:
            raise SystemExit(f"port {port} is in use")
    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if apps:
        raise SystemExit(f"GPU compute processes are running:\n{apps}")


def wait_ready(process, url, readiness):
    deadline = time.monotonic() + readiness["timeout_seconds"]
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise SystemExit(f"server exited with {process.returncode} before ready")
        try:
            with urllib.request.urlopen(url, timeout=readiness["request_timeout_seconds"]):
                return
        except OSError:
            time.sleep(readiness["poll_seconds"])
    raise SystemExit(f"server not ready after {readiness['timeout_seconds']} s")


def stop(process, timeout):
    """Stop the process group this script started, and nothing else."""
    group = process.pid
    try:
        os.killpg(group, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        process.poll()
        try:
            os.killpg(group, 0)
        except ProcessLookupError:
            return
        time.sleep(0.2)
    os.killpg(group, signal.SIGKILL)
    process.wait()


def parse(argv):
    parent = PLUGIN_ROOT.parent
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--arm", choices=("fork", "plugin"), required=True)
    parser.add_argument("--base-profile", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--python",
        default=os.environ.get(
            "QSA_PYTHON", str(parent / "service/runtime-env-sglang-20260923/bin/python")
        ),
    )
    parser.add_argument(
        "--fork-root", type=Path, default=parent / ".worktrees/qsa-fork-ref-ee8fe158d6"
    )
    parser.add_argument(
        "--pin-root", type=Path, default=parent / ".worktrees/sglang-pin-76e06febab"
    )
    parser.add_argument("--profile", type=Path, default=PROFILE)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse(argv)
    profile = json.loads(args.profile.read_text())
    base = json.loads(args.base_profile.read_text())
    fixtures = args.fixtures.resolve()
    checked_fixtures(fixtures, profile)
    output = args.output.resolve()
    roots = {
        "fork": args.fork_root.resolve(),
        "plugin": PLUGIN_ROOT,
        "pin": args.pin_root.resolve(),
    }
    server, env = server_command(
        args.arm, base, profile, python=args.python, roots=roots, output=output
    )
    host, port = option(server, "--host"), int(option(server, "--port"))
    url = f"http://{host}:{port}"
    harness = [
        args.python,
        str(roots["fork"] / HARNESS),
        "baseline",
        "--url",
        url,
        "--fixtures",
        str(fixtures),
        "--max-prefix",
        str(profile["fixtures"]["max_prefix"]),
        "--output",
        str(output / profile["fixtures"]["output"]),
    ]
    plan = {
        "arm": args.arm,
        "profile": profile["name"],
        "status": profile["status"],
        "server": server,
        "env_set": {k: v for k, v in env.items() if os.environ.get(k) != v},
        "env_removed": sorted((os.environ.keys() | base["environment"].keys()) - env.keys()),
        "harness": harness,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    preflight(host, port)
    output.mkdir(parents=True)
    (output / "run.json").write_text(json.dumps(plan, indent=2) + "\n")
    with (output / "server.log").open("wb") as log:
        process = subprocess.Popen(
            server, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            wait_ready(process, url + base["readiness"]["path"], base["readiness"])
            with urllib.request.urlopen(url + "/get_server_info", timeout=60) as response:
                (output / "server_info.json").write_bytes(response.read())
            status = subprocess.run(harness, cwd=roots["fork"] / HARNESS.parent).returncode
        finally:
            stop(process, base["stop_timeout_seconds"])
    print(f"{args.arm} arm: harness exit {status}; evidence in {output}")
    return status


if __name__ == "__main__":
    sys.exit(main())
