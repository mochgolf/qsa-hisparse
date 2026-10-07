"""Launcher preflight, activation-record readiness and cleanup (CPU, fake server).

Servers here are ``qsa_fake_sglang_server`` processes on ephemeral loopback
ports, started by the launcher under test; nothing else is signalled.
"""

import json
import os
import selectors
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from sglang_qsa_hisparse import launch
from sglang_qsa_hisparse.features import Features

HERE = Path(__file__).resolve().parent
COMPAT = Features(model_compat=True)
BOTH = Features(model_compat=True, hisparse_mode="p2-offload")
LOCK = launch.load_lock()


def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def process_alive(pid):
    try:
        # A zombie holds no resources; it only waits for its parent.
        return Path(f"/proc/{pid}/stat").read_text().split(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


def assert_processes_gone(directory):
    processes = json.loads((directory / "processes.json").read_text())
    pids = [processes["server"], *processes["ranks"]]
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and any(map(process_alive, pids)):
        time.sleep(0.05)
    assert not [pid for pid in pids if process_alive(pid)], processes


def launcher_env(tmp_path, mode, **extra):
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("SGLANG_QSA_") and key != "SGLANG_PLUGINS"
    }
    env.update(
        SGLANG_QSA_MODEL_COMPAT="1",
        QSA_FAKE_SERVER_MODE=mode,
        QSA_FAKE_SERVER_DIR=str(tmp_path),
        PYTHONPATH=os.pathsep.join([str(HERE), env.get("PYTHONPATH", "")]),
    )
    env.update(extra)
    return env


def start_launcher(tmp_path, mode, *server_args, **extra_env):
    argv = [
        sys.executable, "-m", "sglang_qsa_hisparse.launch",
        "--run-dir", str(tmp_path / "run"), "--timeout", "120",
        "--server-module", "qsa_fake_sglang_server",
        "--", "--model-path", str(tmp_path), "--host", "127.0.0.1",
        "--port", str(free_port()), *server_args,
    ]  # fmt: skip
    return subprocess.Popen(
        argv,
        env=launcher_env(tmp_path, mode, **extra_env),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def run_launcher(tmp_path, mode, *server_args, **extra_env):
    process = start_launcher(tmp_path, mode, *server_args, **extra_env)
    stdout, stderr = process.communicate(timeout=180)
    return process.returncode, stdout, stderr


def wait_for_line(process, text, timeout=180):
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    deadline = time.monotonic() + timeout
    seen = []
    while time.monotonic() < deadline:
        if not selector.select(timeout=1):
            continue
        line = process.stdout.readline()
        if not line:
            break
        seen.append(line)
        if text in line:
            return seen
    process.kill()
    raise AssertionError(f"{text!r} not printed; stdout {seen}; stderr {process.stderr.read()}")


# Units ------------------------------------------------------------------------


def test_lock_lists_the_locked_distributions():
    assert sorted(LOCK) == sorted(launch.LOCKED_DISTRIBUTIONS)
    assert None not in LOCK.values()


def test_dist_info_and_server_environment(tmp_path):
    dist = launch.write_dist_info(tmp_path / "site")
    assert (dist / "entry_points.txt").read_text() == (
        "[sglang.srt.plugins]\nqsa_hisparse = sglang_qsa_hisparse.plugin:load\n"
    )
    assert "Name: sglang-qsa-hisparse\n" in (dist / "METADATA").read_text()
    env = launch.server_environment(
        {"PYTHONPATH": "/a:/b", "SGLANG_PLUGINS": "qsa_hisparse"}, tmp_path / "site", tmp_path / "act"
    )
    assert env["PYTHONPATH"] == f"{tmp_path / 'site'}{os.pathsep}/a:/b"
    assert env["SGLANG_PLUGINS"] == "qsa_hisparse"
    assert env["SGLANG_QSA_ACTIVATION_DIR"] == str(tmp_path / "act")


def good_facts():
    return {
        "entry_points": [[launch.ENTRY_POINT, "sglang-qsa-hisparse"]],
        "package": str(Path(launch.sglang_qsa_hisparse.__file__).resolve()),
        "native_versions": dict(LOCK),
        "nnodes": 1, "pp_size": 1, "dp_size": 1, "tp_size": 2,
        "host": "127.0.0.1", "port": 1,
    }  # fmt: skip


@pytest.mark.parametrize(
    "change, message",
    [
        ({"entry_points": []}, "entry points"),
        ({"entry_points": [[launch.ENTRY_POINT, "sglang-qsa-hisparse"], ["other:load", "other"]]}, "entry points"),
        ({"entry_points": [["other:load", "sglang-qsa-hisparse"]]}, "entry points"),
        ({"package": "/elsewhere/sglang_qsa_hisparse/__init__.py"}, "would import"),
        ({"native_versions": {**LOCK, "triton": "0"}}, "differ from the lock"),
        ({"pp_size": 2}, "unsupported topology"),
        ({"dp_size": 2}, "unsupported topology"),
        ({"nnodes": 2}, "unsupported topology"),
    ],
)
def test_preflight_problems(change, message):
    assert launch.preflight_problems(good_facts(), LOCK) == []
    problems = launch.preflight_problems({**good_facts(), **change}, LOCK)
    assert len(problems) == 1 and message in problems[0], problems


def write_record(directory, name, **fields):
    record = {
        "role": "scheduler",
        "features": ["model_compat"],
        "hisparse_mode": None,
        "native_versions": dict(LOCK),
        **fields,
    }
    (directory / name).write_text(json.dumps(record))


def test_complete_records_pass(tmp_path):
    write_record(tmp_path, "scheduler-1.json", tp_rank=0)
    write_record(tmp_path, "scheduler-2.json", tp_rank=1)
    assert launch.check_records(tmp_path, 2, COMPAT, LOCK) == []
    write_record(
        tmp_path, "scheduler-1.json", tp_rank=0,
        features=["model_compat", "hisparse"], hisparse_mode="p2-offload",
    )  # fmt: skip
    write_record(
        tmp_path, "scheduler-2.json", tp_rank=1,
        features=["model_compat", "hisparse"], hisparse_mode="p2-offload",
    )  # fmt: skip
    assert launch.check_records(tmp_path, 2, BOTH, LOCK) == []


@pytest.mark.parametrize(
    "records, message",
    [
        ([{"tp_rank": 0}], "per TP rank"),
        ([{"tp_rank": 0}, {"tp_rank": 0}], "per TP rank"),
        ([{"tp_rank": 0}, {"tp_rank": 1}, {"tp_rank": 2}], "per TP rank"),
        ([{"tp_rank": 0}, {"tp_rank": None}], "per TP rank"),
        ([{"tp_rank": 0}, {"tp_rank": 1, "features": ["model_compat", "hisparse"]}], "features"),
        ([{"tp_rank": 0}, {"tp_rank": 1, "hisparse_mode": "offload"}], "features"),
        ([{"tp_rank": 0}, {"tp_rank": 1, "native_versions": {**LOCK, "torch": "0"}}], "lock"),
        ([{"tp_rank": 0}, {"tp_rank": 1, "role": "test"}], "role"),
    ],
)
def test_incomplete_or_wrong_records_fail(tmp_path, records, message):
    for index, fields in enumerate(records):
        write_record(tmp_path, f"scheduler-{index}.json", **fields)
    problems = launch.check_records(tmp_path, 2, COMPAT, LOCK)
    assert problems and message in "; ".join(problems), problems


def test_unreadable_record_fails(tmp_path):
    write_record(tmp_path, "scheduler-1.json", tp_rank=0)
    (tmp_path / "scheduler-2.json").write_text('{"role": "sched')
    problems = launch.check_records(tmp_path, 1, COMPAT, LOCK)
    assert any("unreadable" in p for p in problems), problems


@pytest.mark.parametrize(
    "host, url",
    [
        ("127.0.0.1", "http://127.0.0.1:8/health"),
        ("0.0.0.0", "http://127.0.0.1:8/health"),
        ("::", "http://[::1]:8/health"),
        ("::1", "http://[::1]:8/health"),
    ],
)
def test_health_url(host, url):
    assert launch._health_url(host, 8) == url


# Refusals before any server starts ------------------------------------------------


@pytest.mark.parametrize("plugins", ["other", "qsa_hisparse,other", "other,qsa_hisparse"])
def test_other_general_plugins_are_refused(tmp_path, plugins):
    code, _, stderr = run_launcher(tmp_path, "records", SGLANG_PLUGINS=plugins)
    assert code == 1
    assert "only 'qsa_hisparse' may load" in stderr
    assert not (tmp_path / "processes.json").exists()


def test_no_feature_is_refused(tmp_path):
    code, _, stderr = run_launcher(tmp_path, "records", SGLANG_QSA_MODEL_COMPAT="0")
    assert code == 1
    assert "No feature requested" in stderr
    assert not (tmp_path / "processes.json").exists()


def test_invalid_switches_are_refused(tmp_path):
    code, _, stderr = run_launcher(tmp_path, "records", SGLANG_QSA_HISPARSE_V3="p3")
    assert code == 1
    assert "SGLANG_QSA_HISPARSE_V3 must be one of" in stderr


def test_conflicting_entry_point_is_refused(tmp_path):
    # Another distribution claiming the plugin name would win SGLang's loader
    # (later entry points overwrite earlier ones with the same name).
    other = tmp_path / "other-site" / "other_plugin-1.0.dist-info"
    other.mkdir(parents=True)
    (other / "METADATA").write_text("Metadata-Version: 2.1\nName: other-plugin\nVersion: 1.0\n")
    (other / "entry_points.txt").write_text("[sglang.srt.plugins]\nqsa_hisparse = other:load\n")
    env_path = os.pathsep.join([str(tmp_path / "other-site"), os.environ.get("PYTHONPATH", "")])
    code, _, stderr = run_launcher(tmp_path, "records", PYTHONPATH=os.pathsep.join([str(HERE), env_path]))
    assert code == 1
    assert "Preflight failed" in stderr and "other:load" in stderr
    assert not (tmp_path / "processes.json").exists()


def test_invalid_server_arguments_are_refused(tmp_path):
    code, _, stderr = run_launcher(tmp_path, "records", "--no-such-sglang-flag")
    assert code == 1
    assert "Preflight interpreter failed" in stderr and "--no-such-sglang-flag" in stderr
    assert not (tmp_path / "processes.json").exists()


def test_existing_run_dir_is_refused(tmp_path):
    (tmp_path / "run").mkdir()
    code, _, stderr = run_launcher(tmp_path, "records")
    assert code == 1
    assert "already exists" in stderr


# Readiness with a fake server -------------------------------------------------------


def test_ready_only_with_records_from_the_real_loader(tmp_path):
    """Spawned ranks find the private entry point and write verified records."""
    launcher = start_launcher(tmp_path, "loader", "--tp-size", "2")
    try:
        wait_for_line(launcher, "QSA launcher: ready")
        run = tmp_path / "run"
        ready = json.loads((run / "ready.json").read_text())
        assert ready["tp_size"] == 2 and ready["features"] == ["model_compat"]
        records = [json.loads(p.read_text()) for p in sorted((run / "activation").iterdir())]
        assert sorted(r["tp_rank"] for r in records) == [0, 1]
        processes = json.loads((tmp_path / "processes.json").read_text())
        assert sorted(r["pid"] for r in records) == sorted(processes["ranks"])
        for record in records:
            assert record["native_versions"] == LOCK
            assert record["features"] == ["model_compat"]
            assert "sglang.srt.managers.scheduler.configure_scheduler_process" in record["patches"]
        launcher.send_signal(signal.SIGTERM)
        assert launcher.wait(timeout=60) == 128 + signal.SIGTERM
    finally:
        if launcher.poll() is None:
            launcher.kill()
    assert_processes_gone(tmp_path)


def test_missing_rank_record_stops_the_server(tmp_path):
    code, stdout, stderr = run_launcher(tmp_path, "missing-rank", "--tp-size", "2")
    assert code == 1, stdout + stderr
    assert "without valid activation records" in stderr and "per TP rank" in stderr
    assert "ready" not in stdout.replace("readiness", "")
    assert not (tmp_path / "run" / "ready.json").exists()
    assert_processes_gone(tmp_path)


def test_version_mismatch_stops_the_server(tmp_path):
    code, _, stderr = run_launcher(tmp_path, "wrong-versions", "--tp-size", "2")
    assert code == 1
    assert "differ from the lock" in stderr
    assert not (tmp_path / "run" / "ready.json").exists()
    assert_processes_gone(tmp_path)


def test_server_exit_before_readiness_is_reported(tmp_path):
    code, _, stderr = run_launcher(tmp_path, "exit", "--tp-size", "2")
    assert code == 1
    assert "stopped before readiness (exit status 17)" in stderr
    assert_processes_gone(tmp_path)


def test_sigint_before_readiness_stops_the_server(tmp_path):
    launcher = start_launcher(tmp_path, "loader", "--tp-size", "1")
    try:
        wait_for_line(launcher, "QSA launcher: server pid")
        deadline = time.monotonic() + 30
        while not (tmp_path / "processes.json").exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        launcher.send_signal(signal.SIGINT)
        assert launcher.wait(timeout=60) == 128 + signal.SIGINT
    finally:
        if launcher.poll() is None:
            launcher.kill()
    assert "stopped by signal" in launcher.stderr.read()
    assert not (tmp_path / "run" / "ready.json").exists()
    assert_processes_gone(tmp_path)
