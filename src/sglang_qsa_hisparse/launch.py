"""Start an SGLang server with this plugin active in every scheduler process.

Usage::

    python -m sglang_qsa_hisparse.launch [--run-dir DIR] [--timeout SECONDS] \\
        -- <sglang.launch_server arguments>

Feature switches (``SGLANG_QSA_MODEL_COMPAT``, ``SGLANG_QSA_HISPARSE_V3``) come
from the environment and must request at least one feature. The launcher is
the deployment half of PLAN.md's activation guarantees:

1. ``SGLANG_PLUGINS`` must be unset or exactly ``qsa_hisparse``; the server
   gets ``SGLANG_PLUGINS=qsa_hisparse``, so no other general plugin loads.
2. This package's dist-info (entry point ``qsa_hisparse``) is written to a
   private directory that is prepended to ``PYTHONPATH``, which the server
   and every process it spawns inherit.
3. A child interpreter with the server's environment must find exactly one
   ``qsa_hisparse`` entry point (this package, from this file's directory),
   parse the server arguments with SGLang's own parser (single node, one
   pipeline stage, one data-parallel replica: one scheduler per TP rank),
   and report native library versions equal to ``docs/environment.lock.json``.
4. ``SGLANG_QSA_ACTIVATION_DIR`` points at an empty private directory and the
   server starts in its own process group.
5. Ready (a stdout line and ``<run-dir>/ready.json``) is reported only once
   the server answers ``/health`` and the activation directory holds exactly
   one scheduler record per TP rank, each with the requested features and the
   locked versions. Any other outcome (exit, timeout, missing or wrong
   records) stops the server and exits non-zero.

After readiness the launcher supervises the server. SIGTERM or SIGINT stops
the server and ends the launcher by that signal; a server that stops on its
own (before readiness: with a non-zero status) ends the launcher the same way
(exit status or signal), so supervisors see the server's outcome. The
launcher only ever signals the process group it created for the server, and
only while the server's process id is unreaped (so the group id cannot have
been reused).
"""

import argparse
import http.client
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request
from collections import Counter
from importlib import metadata
from pathlib import Path

import sglang_qsa_hisparse
from sglang_qsa_hisparse.errors import PluginConfigError
from sglang_qsa_hisparse.features import Features, read_features
from sglang_qsa_hisparse.patching import ACTIVATION_DIR_ENV, DIST_NAME, PLUGIN_NAME

ENTRY_POINT_GROUP = "sglang.srt.plugins"
ENTRY_POINT = "sglang_qsa_hisparse.plugin:load"
# A private dist-info; its version is not a release.
DIST_INFO_VERSION = "0+launcher"
LOCKED_DISTRIBUTIONS = ("torch", "sglang-kernel", "flashinfer-python", "triton")
LOCK_FILE = Path(__file__).resolve().parents[2] / "docs" / "environment.lock.json"
SERVER_MODULE = "sglang.launch_server"
POLL_SECONDS = 0.5
STOP_TIMEOUT_SECONDS = 30.0
_MARKER = "QSA_LAUNCH_PREFLIGHT "


class LaunchError(RuntimeError):
    def __init__(self, message: str, status: int = 1):
        super().__init__(message)
        self.status = status


def native_versions() -> dict[str, str | None]:
    """Installed versions of the native libraries the plugin is validated with."""
    versions: dict[str, str | None] = {}
    for name in LOCKED_DISTRIBUTIONS:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def record_details() -> dict:
    """Fields ``patching.verify_final`` adds to every activation record."""
    return {"native_versions": native_versions()}


def load_lock(path: Path = LOCK_FILE) -> dict[str, str]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))["packages"]
    except (OSError, ValueError, KeyError) as error:
        raise LaunchError(f"Cannot read the version lock {path}: {error!r}") from error


def write_dist_info(directory: Path) -> Path:
    """Make ``directory`` expose this package's ``sglang.srt.plugins`` entry point."""
    dist = directory / f"sglang_qsa_hisparse-{DIST_INFO_VERSION}.dist-info"
    dist.mkdir(parents=True)
    (dist / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {DIST_NAME}\nVersion: {DIST_INFO_VERSION}\n"
    )
    (dist / "entry_points.txt").write_text(
        f"[{ENTRY_POINT_GROUP}]\n{PLUGIN_NAME} = {ENTRY_POINT}\n"
    )
    return dist


def server_environment(base: dict, site: Path, activation_dir: Path) -> dict:
    env = dict(base)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(site)] + ([base["PYTHONPATH"]] if base.get("PYTHONPATH") else [])
    )
    env["SGLANG_PLUGINS"] = PLUGIN_NAME
    env[ACTIVATION_DIR_ENV] = str(activation_dir)
    return env


_PREFLIGHT = f"""
import json, sys
from importlib.metadata import entry_points
from pathlib import Path
import sglang_qsa_hisparse
from sglang_qsa_hisparse.launch import native_versions
from sglang.srt.server_args import prepare_server_args

args = prepare_server_args(sys.argv[1:])
facts = {{
    "entry_points": [
        [ep.value, ep.dist.name if ep.dist else None]
        for ep in entry_points(group={ENTRY_POINT_GROUP!r})
        if ep.name == {PLUGIN_NAME!r}
    ],
    "package": str(Path(sglang_qsa_hisparse.__file__).resolve()),
    "native_versions": native_versions(),
}}
for name in ("nnodes", "pp_size", "dp_size", "tp_size", "host", "port"):
    facts[name] = getattr(args, name)
print({_MARKER!r} + json.dumps(facts), flush=True)
"""


def preflight(env: dict, server_args: list[str]) -> dict:
    """Facts about the server's environment, as seen by a fresh interpreter."""
    result = subprocess.run(
        [sys.executable, "-c", _PREFLIGHT, *server_args],
        env=env,
        capture_output=True,
        text=True,
    )
    lines = [l for l in result.stdout.splitlines() if l.startswith(_MARKER)]
    if result.returncode or len(lines) != 1:
        raise LaunchError(
            f"Preflight interpreter failed (status {result.returncode}):\n"
            f"{result.stderr.strip()[-3000:]}"
        )
    return json.loads(lines[0][len(_MARKER) :])


def preflight_problems(facts: dict, lock: dict) -> list[str]:
    problems = []
    if facts["entry_points"] != [[ENTRY_POINT, DIST_NAME]]:
        problems.append(
            f"entry points named {PLUGIN_NAME!r} in {ENTRY_POINT_GROUP!r} are "
            f"{facts['entry_points']}, expected exactly [{ENTRY_POINT!r}, {DIST_NAME!r}]"
        )
    package = str(Path(sglang_qsa_hisparse.__file__).resolve())
    if facts["package"] != package:
        problems.append(f"the server would import {facts['package']} instead of {package}")
    if facts["native_versions"] != lock:
        problems.append(
            f"native library versions {facts['native_versions']} differ from the lock {lock}"
        )
    topology = {name: facts[name] for name in ("nnodes", "pp_size", "dp_size")}
    if topology != {"nnodes": 1, "pp_size": 1, "dp_size": 1}:
        problems.append(
            f"unsupported topology {topology}: the launcher verifies one "
            f"scheduler per TP rank on a single node"
        )
    return problems


def check_records(
    directory: Path, tp_size: int, features: Features, lock: dict
) -> list[str]:
    """Problems with the scheduler activation records; empty when complete."""
    problems = []
    ranks: Counter = Counter()
    for path in sorted(directory.iterdir()):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            problems.append(f"{path.name}: unreadable ({error})")
            continue
        if record.get("role") != "scheduler":
            problems.append(f"{path.name}: role {record.get('role')!r}")
        activated = (record.get("features"), record.get("hisparse_mode"))
        if activated != (list(features.active), features.hisparse_mode):
            problems.append(
                f"{path.name}: features {activated}, requested "
                f"{(list(features.active), features.hisparse_mode)}"
            )
        if record.get("native_versions") != lock:
            problems.append(
                f"{path.name}: native library versions {record.get('native_versions')} "
                f"differ from the lock {lock}"
            )
        ranks[repr(record.get("tp_rank"))] += 1
    expected = Counter(repr(rank) for rank in range(tp_size))
    if ranks != expected:
        problems.append(
            f"scheduler records per TP rank {dict(sorted(ranks.items()))}, expected "
            f"one for each rank of tp_size {tp_size}"
        )
    return problems


def _health_url(host: str, port: int) -> str:
    host = {"": "127.0.0.1", "0.0.0.0": "127.0.0.1", "::": "::1"}.get(host, host)
    return f"http://[{host}]:{port}/health" if ":" in host else f"http://{host}:{port}/health"


def _healthy(url: str) -> bool:
    # A local server must not be reached through inherited proxy settings.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(url, timeout=5) as response:
            return response.status == 200
    except (OSError, http.client.HTTPException):
        return False


def _exit_state(pid: int, *, block: bool = False):
    """The child's exit state without reaping it (None while it runs)."""
    flags = os.WEXITED | os.WNOWAIT | (0 if block else os.WNOHANG)
    return os.waitid(os.P_PID, pid, flags)


def _describe(state) -> str:
    if state.si_code == os.CLD_EXITED:
        return f"exit status {state.si_status}"
    return f"signal {state.si_status}"


def _status(state) -> int:
    """Exit status, or the negated signal number (as ``Popen.returncode``)."""
    return state.si_status if state.si_code == os.CLD_EXITED else -state.si_status


_signals: list[int] = []


def _on_signal(signum, frame) -> None:
    _signals.append(signum)


def _check_signals() -> None:
    if _signals:
        raise _Stopped(_signals[0])


class _Stopped(Exception):
    def __init__(self, signum: int):
        super().__init__(f"stopped by signal {signum}")
        self.signum = signum


def stop_server(server: subprocess.Popen) -> None:
    """Stop the server's process group, then reap the server."""
    for signum in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(server.pid, signum)
        except ProcessLookupError:
            break
        deadline = time.monotonic() + STOP_TIMEOUT_SECONDS
        while _exit_state(server.pid) is None and time.monotonic() < deadline:
            time.sleep(0.1)
    server.wait()


def _wait_until_ready(server, url, activation_dir, tp_size, features, lock, timeout):
    deadline = time.monotonic() + timeout
    while not _healthy(url):
        _check_signals()
        state = _exit_state(server.pid)
        if state is not None:
            raise LaunchError(
                f"The server stopped before readiness ({_describe(state)})",
                status=_status(state) or 1,
            )
        if time.monotonic() > deadline:
            raise LaunchError(f"The server was not ready within {timeout:g} s")
        time.sleep(POLL_SECONDS)
    problems = check_records(activation_dir, tp_size, features, lock)
    if problems:
        raise LaunchError(
            "The server answered /health without valid activation records: "
            + "; ".join(problems)
        )


def _make_run_dir(path: Path | None) -> Path:
    if path is None:
        return Path(tempfile.mkdtemp(prefix="qsa-launch-"))
    try:
        path.mkdir(mode=0o700, parents=True)
    except FileExistsError as error:
        raise LaunchError(f"Run directory {path} already exists") from error
    return path.resolve()


def launch(run_dir: Path | None, timeout: float, server_module: str, server_args: list[str]) -> int:
    try:
        features = read_features()
    except PluginConfigError as error:
        raise LaunchError(str(error)) from error
    if not features.active:
        raise LaunchError(
            "No feature requested: set SGLANG_QSA_MODEL_COMPAT=1 "
            "(and optionally SGLANG_QSA_HISPARSE_V3)"
        )
    selected = os.environ.get("SGLANG_PLUGINS", "")
    if selected and {n.strip() for n in selected.split(",") if n.strip()} != {PLUGIN_NAME}:
        raise LaunchError(
            f"SGLANG_PLUGINS={selected!r}: only {PLUGIN_NAME!r} may load in a served process"
        )
    lock = load_lock()
    run_dir = _make_run_dir(run_dir)
    site = run_dir / "site"
    write_dist_info(site)
    activation_dir = run_dir / "activation"
    activation_dir.mkdir()
    env = server_environment(dict(os.environ), site, activation_dir)
    facts = preflight(env, server_args)
    problems = preflight_problems(facts, lock)
    if problems:
        raise LaunchError("Preflight failed: " + "; ".join(problems))

    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, _on_signal)
    server = subprocess.Popen(
        [sys.executable, "-m", server_module, *server_args],
        env=env,
        start_new_session=True,
    )
    print(f"QSA launcher: server pid {server.pid}, run directory {run_dir}", flush=True)
    try:
        _wait_until_ready(
            server,
            _health_url(facts["host"], facts["port"]),
            activation_dir,
            facts["tp_size"],
            features,
            lock,
            timeout,
        )
        ready = {
            "server_pid": server.pid,
            "features": list(features.active),
            "hisparse_mode": features.hisparse_mode,
            "tp_size": facts["tp_size"],
            "records": sorted(p.name for p in activation_dir.iterdir()),
        }
        partial = run_dir / "ready.json.tmp"
        partial.write_text(json.dumps(ready, indent=2) + "\n")
        partial.replace(run_dir / "ready.json")
        print(
            f"QSA launcher: ready; {facts['tp_size']} scheduler activation "
            f"record(s) verified for {list(features.active)}",
            flush=True,
        )
        while True:
            _check_signals()
            state = _exit_state(server.pid)
            if state is not None:
                print(f"QSA launcher: server stopped ({_describe(state)})", flush=True)
                return _status(state)
            time.sleep(POLL_SECONDS)
    except _Stopped as stopped:
        print(f"QSA launcher: {stopped}; stopping the server", file=sys.stderr, flush=True)
        return -stopped.signum
    finally:
        stop_server(server)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    split = argv.index("--") if "--" in argv else len(argv)
    parser = argparse.ArgumentParser(
        prog="python -m sglang_qsa_hisparse.launch",
        usage="%(prog)s [--run-dir DIR] [--timeout SECONDS] -- <sglang.launch_server args>",
        description="Start sglang.launch_server with the QSA HiSparse plugin verified active.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="new directory for the private dist-info, activation records and "
        "ready.json (default: a fresh temporary directory)",
    )
    parser.add_argument("--timeout", type=float, default=1800.0, help="seconds to wait for readiness")
    # Tests substitute a fake server; deployments always run sglang.launch_server.
    parser.add_argument("--server-module", default=SERVER_MODULE, help=argparse.SUPPRESS)
    options = parser.parse_args(argv[:split])
    if split == len(argv):
        parser.error("missing '--' before the server arguments")
    try:
        status = launch(options.run_dir, options.timeout, options.server_module, argv[split + 1 :])
    except LaunchError as error:
        print(f"QSA launcher: {error}", file=sys.stderr, flush=True)
        status = error.status
    if status < 0:
        # End by the same signal as the server (or the launcher's stop signal),
        # so supervisors see the same outcome as without the launcher.
        if -status != signal.SIGKILL:
            signal.signal(-status, signal.SIG_DFL)
        os.kill(os.getpid(), -status)
        return 128 - status  # Reached only if that signal's default is to ignore.
    return status


if __name__ == "__main__":
    sys.exit(main())
