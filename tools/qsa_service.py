#!/usr/bin/env python3
"""Manage a local model server in a systemd user cgroup; no SGLang imports."""

import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid


class ServiceError(RuntimeError):
    pass


def command(argv, *, check=True):
    result = subprocess.run(argv, text=True, capture_output=True)
    if check and result.returncode:
        raise ServiceError(
            f"{argv[0]} failed ({result.returncode}): "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    return result


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_profile(path):
    try:
        profile = json.loads(path.read_text())
        if profile.get("version") != 1:
            raise ValueError("version must be 1")
        name = profile["name"]
        if not isinstance(name, str) or not re.fullmatch(r"[a-zA-Z0-9_-]+", name):
            raise ValueError("name must contain only letters, numbers, _ or -")
        argv = profile["command"]
        if (
            not isinstance(argv, list)
            or not argv
            or not all(isinstance(arg, str) and "\0" not in arg for arg in argv)
        ):
            raise ValueError("command must be a nonempty list of strings")
        if not Path(argv[0]).is_absolute():
            raise ValueError("command executable must be an absolute path")
        for key in ("cwd", "state_dir"):
            if (
                not isinstance(profile[key], str)
                or not Path(profile[key]).is_absolute()
            ):
                raise ValueError(f"{key} must be an absolute path")
        environment = profile.get("environment", {})
        if not isinstance(environment, dict) or not all(
            isinstance(key, str)
            and re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*", key)
            and isinstance(value, str)
            and "\0" not in value
            for key, value in environment.items()
        ):
            raise ValueError("environment must map environment names to strings")
        if "QSA_SERVICE_INSTANCE" in environment:
            raise ValueError("QSA_SERVICE_INSTANCE is reserved for service ownership")
        listen = profile["listen"]
        if listen["host"] not in ("127.0.0.1", "::1"):
            raise ValueError("listen.host must be a numeric loopback address")
        if type(listen["port"]) is not int or not 1 <= listen["port"] <= 65535:
            raise ValueError("listen.port must be an integer between 1 and 65535")
        readiness = profile.setdefault("readiness", {})
        readiness.setdefault("path", "/health")
        if not isinstance(readiness["path"], str) or not readiness["path"].startswith(
            "/"
        ):
            raise ValueError("readiness.path must start with /")
        if "model" in readiness and not isinstance(readiness["model"], str):
            raise ValueError("readiness.model must be a string")
        for key, default in (
            ("timeout_seconds", 600),
            ("poll_seconds", 1),
            ("request_timeout_seconds", 2),
        ):
            value = readiness.setdefault(key, default)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"readiness.{key} must be a positive finite number")
        value = profile.setdefault("stop_timeout_seconds", 45)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError("stop_timeout_seconds must be a positive finite number")
        return profile
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ServiceError(f"Invalid profile {path}: {error}") from error


def unit_name(profile):
    namespace = str(Path(profile["state_dir"]).resolve())
    digest = hashlib.sha256(namespace.encode()).hexdigest()[:12]
    return f"qsa-{profile['name']}-{digest}.service"


def unit_info(unit):
    result = command(
        [
            "systemctl",
            "--user",
            "show",
            unit,
            "--no-pager",
            "--property=LoadState,ActiveState,SubState,InvocationID,MainPID,ControlGroup,Result,ExecMainStatus,Environment",
        ],
        check=False,
    )
    info = dict(
        line.split("=", 1) for line in result.stdout.splitlines() if "=" in line
    )
    if not info:
        raise ServiceError(
            f"Cannot query the systemd user manager: {result.stderr.strip()}"
        )
    return info


def read_state(state_dir):
    path = state_dir / "state.json"
    try:
        state = json.loads(path.read_text()) if path.exists() else None
        if state is not None and not isinstance(state, dict):
            raise ValueError("ownership record must be a JSON object")
        return state
    except (OSError, ValueError) as error:
        raise ServiceError(f"Unreadable ownership record {path}: {error}") from error


def owned_info(profile, state, *, persist_recovery=True):
    unit = unit_name(profile)
    info = unit_info(unit)
    if info.get("LoadState") == "not-found":
        return info
    if not state or state.get("unit") != unit:
        raise ServiceError(
            f"Refusing to adopt existing unit {unit}: no matching ownership record"
        )
    if (
        state.get("invocation_id")
        and info.get("InvocationID") != state["invocation_id"]
    ):
        raise ServiceError(
            f"Refusing to control {unit}: stale ownership record (invocation ID differs)"
        )
    marker = f"QSA_SERVICE_INSTANCE={state.get('instance', '')}"
    if not state.get("instance") or marker not in shlex.split(
        info.get("Environment", "")
    ):
        raise ServiceError(
            f"Refusing to control {unit}: service ownership marker differs"
        )
    if not state.get("invocation_id"):
        # A controller may die after dispatch and before saving InvocationID.
        # Its precommitted unique unit marker safely closes that transaction.
        if not info.get("InvocationID"):
            raise ServiceError(f"Cannot establish invocation identity for {unit}")
        state["invocation_id"] = info["InvocationID"]
        if persist_recovery:
            write_json(Path(profile["state_dir"]).resolve() / "state.json", state)
    return info


def listener_inodes(profile):
    port = profile["listen"]["port"]
    inodes = set()
    for table in ("tcp", "tcp6"):
        with open(f"/proc/net/{table}") as handle:
            next(handle)
            for line in handle:
                fields = line.split()
                if fields[3] == "0A" and int(fields[1].split(":")[1], 16) == port:
                    inodes.add(fields[9])
    return inodes


def listener_owned(profile, info):
    """Verify all listeners on this port belong to the unit, not a racing server."""
    inodes = listener_inodes(profile)
    if not inodes or not info.get("ControlGroup"):
        return False
    group = Path("/sys/fs/cgroup") / info["ControlGroup"].lstrip("/")
    owned_inodes = set()
    try:
        for process_file in group.rglob("cgroup.procs"):
            for pid in process_file.read_text().split():
                for descriptor in (Path("/proc") / pid / "fd").iterdir():
                    try:
                        link = os.readlink(descriptor)
                        if link.startswith("socket:["):
                            owned_inodes.add(link[8:-1])
                    except (FileNotFoundError, PermissionError):
                        pass
    except (FileNotFoundError, ProcessLookupError):
        return False
    return inodes <= owned_inodes


def ensure_port_free(profile):
    listen = profile["listen"]
    family = socket.AF_INET6 if ":" in listen["host"] else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((listen["host"], listen["port"]))
        except OSError as error:
            raise ServiceError(
                f"Listener conflict at {listen['host']}:{listen['port']}; "
                "the existing process was not adopted or stopped"
            ) from error


def health(profile):
    listen = profile["listen"]
    host = f"[{listen['host']}]" if ":" in listen["host"] else listen["host"]
    base = f"http://{host}:{listen['port']}"
    ready = profile["readiness"]
    # A local service must not accidentally use inherited HTTP proxy settings.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(
            base + ready["path"], timeout=ready["request_timeout_seconds"]
        ) as response:
            if response.status != 200:
                return False, f"health returned HTTP {response.status}"
        if ready.get("model"):
            with opener.open(
                base + "/v1/models", timeout=ready["request_timeout_seconds"]
            ) as response:
                models = json.load(response)
            if ready["model"] not in {
                model.get("id") for model in models.get("data", [])
            }:
                return False, f"expected served model {ready['model']!r} is missing"
        return True, "ready"
    except (OSError, ValueError, KeyError, TypeError, urllib.error.URLError) as error:
        return False, str(error)


def tail(path, lines):
    try:
        with path.open(errors="replace") as handle:
            from collections import deque

            return "".join(deque(handle, maxlen=lines))
    except FileNotFoundError:
        return "(no log output yet)\n"


@contextlib.contextmanager
def lifecycle_lock(state_dir):
    state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    with (state_dir / "control.lock").open("a+") as handle:
        os.chmod(handle.name, 0o600)
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def stop(profile, state_dir):
    state = read_state(state_dir)
    info = owned_info(profile, state)
    if info.get("LoadState") != "not-found":
        command(["systemctl", "--user", "stop", unit_name(profile)])
        command(
            ["systemctl", "--user", "reset-failed", unit_name(profile)], check=False
        )
    if state:
        state["last_action"] = "stopped"
        write_json(state_dir / "state.json", state)
    print(f"Stopped {profile['name']} (all workers in the managed cgroup)")


def start(profile, profile_path, state_dir):
    state = read_state(state_dir)
    info = owned_info(profile, state)
    fingerprint = hashlib.sha256(
        json.dumps(profile, sort_keys=True).encode()
    ).hexdigest()
    if info.get("ActiveState") in ("active", "activating", "reloading"):
        if state.get("profile_fingerprint") != fingerprint:
            raise ServiceError(
                "Service already runs a different profile; use restart to apply this profile"
            )
        ready, reason = health(profile)
        if ready and listener_owned(profile, info):
            print(
                f"Already ready: {profile['name']} (PID {info['MainPID']}, {info['ControlGroup']})"
            )
            return
        raise ServiceError(
            f"Managed service is already running but not ready: {reason}; use status/logs or restart"
        )
    if info.get("LoadState") != "not-found":
        command(["systemctl", "--user", "stop", unit_name(profile)])
        command(
            ["systemctl", "--user", "reset-failed", unit_name(profile)], check=False
        )
    ensure_port_free(profile)
    if not Path(profile["cwd"]).is_dir():
        raise ServiceError(f"Working directory does not exist: {profile['cwd']}")
    if not os.access(profile["command"][0], os.X_OK):
        raise ServiceError(f"Executable is unavailable: {profile['command'][0]}")

    instance = uuid.uuid4().hex
    snapshot = state_dir / f"launch-{instance}.json"
    write_json(snapshot, profile)
    log_path = state_dir / "service.log"
    with log_path.open("a") as log:
        os.chmod(log_path, 0o600)
        log.write(
            f"\n=== Starting {profile['name']} {instance} from {profile_path} at {time.strftime('%Y-%m-%d %H:%M:%S %z')} ===\n"
        )
    argv = [
        "systemd-run",
        "--user",
        "--quiet",
        f"--unit={unit_name(profile)}",
        "--property=Type=exec",
        "--property=Restart=no",
        "--property=KillMode=control-group",
        "--property=SendSIGKILL=yes",
        f"--property=TimeoutStopSec={profile['stop_timeout_seconds']}s",
        f"--property=StandardOutput=append:{log_path}",
        f"--property=StandardError=append:{log_path}",
        f"--setenv=QSA_SERVICE_INSTANCE={instance}",
        sys.executable,
        str(Path(__file__).resolve()),
        "_exec",
        "--profile",
        str(snapshot),
    ]
    state = {
        "unit": unit_name(profile),
        "invocation_id": None,
        "instance": instance,
        "profile_fingerprint": fingerprint,
        "profile_path": str(profile_path),
        "snapshot_path": str(snapshot),
        "log_path": str(log_path),
        "last_action": "starting",
        "started_at": time.time(),
    }
    # Commit a unique marker before dispatch. A subsequent controller can recover
    # the exact invocation even if this process dies immediately after launch.
    write_json(state_dir / "state.json", state)
    try:
        command(argv)
        info = owned_info(profile, state)
    except ServiceError as error:
        # Only the matching unique marker can authorize this cleanup.
        stop(profile, state_dir)
        state["last_action"] = "failed"
        state["failure"] = str(error)
        write_json(state_dir / "state.json", state)
        raise ServiceError(f"{error}\nRecent logs:\n{tail(log_path, 30)}") from error
    deadline = time.monotonic() + profile["readiness"]["timeout_seconds"]
    reason = "waiting for the managed listener"
    try:
        while time.monotonic() < deadline:
            info = owned_info(profile, state)
            if info.get("ActiveState") not in ("active", "activating"):
                raise ServiceError(
                    f"Server exited before readiness: state={info.get('ActiveState')}, "
                    f"result={info.get('Result')}, exit={info.get('ExecMainStatus')}"
                )
            if listener_owned(profile, info):
                ready, reason = health(profile)
                if ready:
                    state["last_action"] = "ready"
                    write_json(state_dir / "state.json", state)
                    print(
                        f"Ready: {profile['name']} at {profile['listen']['host']}:{profile['listen']['port']} "
                        f"(PID {info['MainPID']}, unit {state['unit']})\nLogs: {log_path}"
                    )
                    return
            time.sleep(
                min(
                    profile["readiness"]["poll_seconds"],
                    max(0, deadline - time.monotonic()),
                )
            )
        raise ServiceError(
            f"Readiness timed out after {profile['readiness']['timeout_seconds']}s: {reason}"
        )
    except (ServiceError, KeyboardInterrupt) as error:
        # Fail closed and release workers even when startup is interrupted.
        stop(profile, state_dir)
        state["last_action"] = "failed"
        state["failure"] = str(error) or "Startup interrupted"
        write_json(state_dir / "state.json", state)
        raise ServiceError(
            f"{state['failure']}\nRecent logs:\n{tail(log_path, 30)}"
        ) from error


def status(profile, state_dir, as_json):
    state = read_state(state_dir)
    info = owned_info(profile, state, persist_recovery=False)
    active = info.get("ActiveState") in ("active", "activating", "reloading")
    actual = load_profile(Path(state["snapshot_path"])) if active else profile
    ready, reason = health(actual) if active else (False, "stopped")
    if active and not listener_owned(actual, info):
        ready, reason = False, "waiting for a listener owned by the managed cgroup"
    failed = info.get("ActiveState") == "failed"
    conflict = not active and bool(listener_inodes(actual))
    if failed:
        reason = f"server exited: result={info.get('Result')}, exit={info.get('ExecMainStatus')}"
    elif conflict:
        reason = "an unmanaged listener occupies the configured port; it was not adopted or stopped"
    result = {
        "name": profile["name"],
        "status": "ready"
        if ready
        else "unhealthy"
        if active
        else "failed"
        if failed
        else "conflict"
        if conflict
        else "stopped",
        "unit": unit_name(profile),
        "pid": int(info.get("MainPID", "0")),
        "systemd_state": info.get("ActiveState", "inactive"),
        "detail": reason,
        "profile": state.get("profile_path") if state else None,
        "logs": state.get("log_path") if state else str(state_dir / "service.log"),
    }
    if state and state.get("failure") and not active:
        result["last_failure"] = state["failure"]
    print(
        json.dumps(result, indent=2)
        if as_json
        else f"{result['name']}: {result['status']} (PID {result['pid']}, {result['unit']})\n"
        f"Profile: {result['profile']}\nLogs: {result['logs']}\n{reason}"
        + (
            f"\nLast failure: {result['last_failure']}"
            if "last_failure" in result
            else ""
        )
    )
    return 0 if ready else 4 if active else 1 if failed or conflict else 3


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("start", "stop", "restart", "status", "logs", "_exec")
    )
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--json", action="store_true", help="JSON status output")
    parser.add_argument(
        "--follow", "-f", action="store_true", help="Follow append-only logs"
    )
    parser.add_argument("--lines", "-n", type=int, default=80)
    args = parser.parse_args(argv)
    try:
        profile_path = args.profile.resolve()
        profile = load_profile(profile_path)
        if args.action == "_exec":
            os.chdir(profile["cwd"])
            environment = dict(os.environ)
            environment.update(profile.get("environment", {}))
            os.execvpe(profile["command"][0], profile["command"], environment)
        if args.lines <= 0:
            raise ServiceError("--lines must be positive")
        state_dir = Path(profile["state_dir"]).resolve()
        if args.action == "logs":
            log = state_dir / "service.log"
            if args.follow:
                log.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                log.touch(mode=0o600, exist_ok=True)
                os.execvp("tail", ["tail", "-n", str(args.lines), "-F", str(log)])
            print(tail(log, args.lines), end="")
            return 0
        if args.action == "status":
            # Status must remain available while start waits for model loading.
            # Atomic records permit read-only inspection without the control lock.
            return status(profile, state_dir, args.json)
        with lifecycle_lock(state_dir):
            if args.action in ("stop", "restart"):
                stop(profile, state_dir)
            if args.action in ("start", "restart"):
                start(profile, profile_path, state_dir)
        return 0
    except (ServiceError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
