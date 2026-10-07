"""Real CPU lifecycle checks against the user systemd manager (no model/GPU).

Port of the fork's ``test/qsa_hisparse/test_service_control.py`` to
``tools/qsa_service.py`` and the plugin launcher: the fake service is the
server module the launcher starts (``--server-module``), it writes the
scheduler activation record the launcher requires, and its mode, directory and
arguments travel through the profile environment and the SGLang arguments.
Assertions are the fork's; waits that now include the launcher's preflight
(a child interpreter importing SGLang's argument parser, about 5 s here) are
longer.

``ServiceLifecycleTests`` starts transient systemd user units, so automated
runs deselect it (PLAN.md rule 8, ``tests/conftest.py``). Run it on request
from the repository root with the CPU runner's environment::

    QSA_SERVICE_LIFECYCLE_TESTS=1 PYTHONPATH=$PWD/src:<pin>/python \\
        CUDA_VISIBLE_DEVICES=99 PYTHONDONTWRITEBYTECODE=1 \\
        <python> -m pytest -p no:cacheprovider tests/service
"""

import concurrent.futures
import importlib.util
import json
import os
from pathlib import Path
import socket
import signal
import subprocess
import sys
import tempfile
import time
import unittest


CONTROLLER = Path(__file__).resolve().parents[2] / "tools" / "qsa_service.py"
spec = importlib.util.spec_from_file_location("qsa_service_control", CONTROLLER)
control = importlib.util.module_from_spec(spec)
spec.loader.exec_module(control)

FAKE_SERVICE = r"""
import argparse
import http.server
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from sglang_qsa_hisparse.features import read_features
from sglang_qsa_hisparse.launch import record_details

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int)
port = parser.parse_known_args()[0].port
directory, mode = Path(os.environ["FAKE_DIR"]), os.environ["FAKE_MODE"]
worker = subprocess.Popen([sys.executable, "-c", "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(120)"], start_new_session=True)
(directory / "processes.json").write_text(json.dumps({"parent": os.getpid(), "worker": worker.pid, "args": sys.argv[1:], "env": os.environ.get("FAKE_EXACT_ENV"), "cwd": os.getcwd()}))
print("fake service launched", mode, flush=True)
if mode == "exit":
    print("intentional startup failure", flush=True)
    raise SystemExit(17)
# Stand-in for the activation record of the only scheduler (TP rank 0).
features = read_features()
Path(os.environ["SGLANG_QSA_ACTIVATION_DIR"], f"scheduler-{os.getpid()}.json").write_text(json.dumps({"role": "scheduler", "features": list(features.active), "hisparse_mode": features.hisparse_mode, "tp_rank": 0, **record_details()}))
started = time.monotonic()
class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self.send_response(503 if mode == "never-ready" or time.monotonic() - started < .1 else 200)
            self.end_headers()
            self.wfile.write(b"healthy")
        elif self.path == "/v1/models":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({"data": [{"id": "wrong-model" if mode == "wrong-model" else "fake-model"}]}).encode())
        else:
            self.send_response(404)
            self.end_headers()
    def log_message(self, *args):
        pass
http.server.HTTPServer(("127.0.0.1", port), Handler).serve_forever()
"""


LAUNCH = [sys.executable, "-m", "sglang_qsa_hisparse.launch"]


def process_alive(pid):
    try:
        # A zombie has no executable or sockets and cannot use the GPU/CPU.
        return Path(f"/proc/{pid}/stat").read_text().split(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


class ProfileValidationTests(unittest.TestCase):
    def test_rejects_remote_listener_and_invalid_environment(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "profile.json"
            profile = {
                "version": 1,
                "name": "test",
                "cwd": temporary,
                "state_dir": temporary + "/state",
                "command": [*LAUNCH, "--", "--model-path", temporary],
                "listen": {"host": "0.0.0.0", "port": 12345},
            }
            path.write_text(json.dumps(profile))
            with self.assertRaisesRegex(control.ServiceError, "loopback"):
                control.load_profile(path)
            profile["listen"]["host"] = "127.0.0.1"
            profile["environment"] = {"QSA_SERVICE_INSTANCE": "not-controller-owned"}
            path.write_text(json.dumps(profile))
            with self.assertRaisesRegex(control.ServiceError, "reserved"):
                control.load_profile(path)

    def test_command_must_run_the_launcher(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "profile.json"
            profile = {
                "version": 1,
                "name": "test",
                "cwd": temporary,
                "state_dir": temporary + "/state",
                "listen": {"host": "127.0.0.1", "port": 12345},
            }
            for argv, message in (
                ([sys.executable, "-m", "sglang.launch_server", "--model-path", "m"], "launcher"),
                ([*LAUNCH, "--model-path", "m"], "launcher"),
                ([*LAUNCH, "--run-dir", "/tmp/x", "--", "--model-path", "m"], "reserved"),
                ([*LAUNCH, "--run-dir=/tmp/x", "--", "--model-path", "m"], "reserved"),
            ):
                path.write_text(json.dumps(dict(profile, command=argv)))
                with self.assertRaisesRegex(control.ServiceError, message):
                    control.load_profile(path)
            argv = [*LAUNCH, "--timeout", "60", "--", "--model-path", "m", "--run-dir", "x"]
            path.write_text(json.dumps(dict(profile, command=argv)))
            loaded = control.load_profile(path)
            run = str(Path(temporary).resolve() / "state" / "run-abc")
            self.assertEqual(
                control.launcher_argv(loaded, "abc"),
                [*LAUNCH, "--run-dir", run, "--timeout", "60", "--", "--model-path", "m", "--run-dir", "x"],
            )

    def test_ready_requires_the_launcher_marker(self):
        with tempfile.TemporaryDirectory() as temporary:
            profile = {"state_dir": temporary}
            state = {"instance": "abc"}
            self.assertFalse(control.launcher_ready(profile, None))
            self.assertFalse(control.launcher_ready(profile, {"instance": None}))
            self.assertFalse(control.launcher_ready(profile, state))
            (Path(temporary) / "run-abc").mkdir()
            self.assertFalse(control.launcher_ready(profile, state))
            (Path(temporary) / "run-abc" / "ready.json").write_text("{}")
            self.assertTrue(control.launcher_ready(profile, state))
            self.assertFalse(control.launcher_ready(profile, {"instance": "other"}))


@unittest.skipUnless(
    Path("/sys/fs/cgroup/cgroup.controllers").exists()
    and subprocess.run(
        ["systemctl", "--user", "show", "--property=Version"], capture_output=True
    ).returncode
    == 0,
    "Requires Linux cgroup v2 and a working systemd user manager",
)
class ServiceLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="qsa-service-test-")
        self.directory = Path(self.temporary.name)
        self.fake = self.directory / "qsa_fake_service.py"
        self.fake.write_text(FAKE_SERVICE)
        with socket.socket() as socket_probe:
            socket_probe.bind(("127.0.0.1", 0))
            port = socket_probe.getsockname()[1]
        self.server_args = [
            "--model-path",
            str(self.directory),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--served-model-name",
            "literal $() `words`",
            "--tokenizer-path",
            "space arg",
        ]
        python_path = [self.directory] + [
            Path(p).resolve() for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p
        ]
        self.profile = {
            "version": 1,
            "name": "cpu-test",
            "cwd": str(self.directory),
            "state_dir": str(self.directory / "state"),
            "command": [
                *LAUNCH,
                "--server-module",
                "qsa_fake_service",
                "--",
                *self.server_args,
            ],
            "environment": {
                "FAKE_EXACT_ENV": "literal $() `value`",
                "FAKE_MODE": "ready",
                "FAKE_DIR": str(self.directory),
                # The unit gets the user manager's environment, not this one.
                "PYTHONPATH": os.pathsep.join(map(str, python_path)),
                "SGLANG_QSA_MODEL_COMPAT": "1",
                "CUDA_VISIBLE_DEVICES": "99",
                "PYTHONDONTWRITEBYTECODE": "1",
            },
            "listen": {"host": "127.0.0.1", "port": port},
            "readiness": {
                "path": "/health",
                "model": "fake-model",
                # Shared inference hosts can pause the CPU fixture between its
                # startup log and socket bind, and the launcher's preflight
                # imports SGLang (about 5 s). Failure cases override this.
                "timeout_seconds": 60,
                "poll_seconds": 0.03,
                "request_timeout_seconds": 0.2,
            },
            "stop_timeout_seconds": 0.3,
        }
        self.path = self.directory / "profile.json"
        self.save_profile()

    def save_profile(self):
        self.path.write_text(json.dumps(self.profile))

    def run_control(self, action, *args):
        return subprocess.run(
            [
                sys.executable,
                str(CONTROLLER),
                action,
                "--profile",
                str(self.path),
                *args,
            ],
            capture_output=True,
            text=True,
            timeout=90,
        )

    def assert_success(self, result):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def processes(self):
        return json.loads((self.directory / "processes.json").read_text())

    def assert_processes_gone(self, processes):
        # The launcher notices a dead server within its poll interval.
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and any(
            process_alive(processes[key]) for key in ("parent", "worker")
        ):
            time.sleep(0.02)
        for key in ("parent", "worker"):
            self.assertFalse(
                process_alive(processes[key]),
                f"{key} survived cleanup: {processes[key]}",
            )

    def tearDown(self):
        # Tests corrupt ownership intentionally; restore it only from a record
        # captured by this test, never adopt a foreign unit during cleanup.
        record = self.directory / "state" / "state.json"
        if hasattr(self, "original_state"):
            record.write_text(json.dumps(self.original_state))
        result = self.run_control("stop")
        self.assert_success(result)
        self.temporary.cleanup()

    def test_lifecycle_environment_and_detached_worker_cleanup(self):
        self.assert_success(self.run_control("start"))
        processes = self.processes()
        self.assertEqual(processes["args"], self.server_args)
        self.assertEqual(processes["env"], "literal $() `value`")
        self.assertEqual(processes["cwd"], str(self.directory))
        self.assertTrue(process_alive(processes["parent"]))
        self.assertTrue(process_alive(processes["worker"]))
        # The controller subprocess already ended; systemd owns the service.
        status = self.run_control("status", "--json")
        self.assert_success(status)
        self.assertEqual(json.loads(status.stdout)["status"], "ready")
        self.assert_success(self.run_control("start"))
        self.assertEqual(self.processes()["parent"], processes["parent"])
        self.assert_success(self.run_control("restart"))
        self.assert_processes_gone(processes)
        restarted = self.processes()
        self.assertNotEqual(restarted["parent"], processes["parent"])
        log_result = self.run_control("logs", "--lines", "20")
        self.assert_success(log_result)
        self.assertIn("fake service launched ready", log_result.stdout)
        self.assert_success(self.run_control("stop"))
        self.assert_processes_gone(restarted)
        self.assert_success(self.run_control("stop"))
        self.assertEqual(self.run_control("status").returncode, 3)

    def test_concurrent_starts_are_idempotent(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _: self.run_control("start"), range(2)))
        for result in results:
            self.assert_success(result)
        self.assertEqual(sum("Already ready" in result.stdout for result in results), 1)

    def test_health_timeout_cleans_detached_workers_and_reports_logs(self):
        self.profile["environment"]["FAKE_MODE"] = "never-ready"
        # Longer than the launcher's preflight, so the fake has started.
        self.profile["readiness"]["timeout_seconds"] = 20
        self.save_profile()
        result = self.run_control("start")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("Readiness timed out", result.stderr)
        self.assertIn("fake service launched never-ready", result.stderr)
        self.assert_processes_gone(self.processes())
        status = self.run_control("status", "--json")
        self.assertEqual(status.returncode, 3)
        self.assertIn("Readiness timed out", json.loads(status.stdout)["last_failure"])

    def test_early_exit_cleans_detached_worker(self):
        self.profile["environment"]["FAKE_MODE"] = "exit"
        self.save_profile()
        result = self.run_control("start")
        self.assertEqual(result.returncode, 1)
        self.assertIn("intentional startup failure", result.stderr)
        self.assertIn("exit=17", result.stderr)
        self.assert_processes_gone(self.processes())

    def test_model_identity_is_required(self):
        self.profile["environment"]["FAKE_MODE"] = "wrong-model"
        self.profile["readiness"]["timeout_seconds"] = 20
        self.save_profile()
        result = self.run_control("start")
        self.assertEqual(result.returncode, 1)
        self.assertIn("expected served model", result.stderr)
        self.assert_processes_gone(self.processes())

    def test_foreign_listener_is_never_adopted_or_stopped(self):
        foreign = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "http.server",
                str(self.profile["listen"]["port"]),
                "--bind",
                "127.0.0.1",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            deadline = time.monotonic() + 2
            while (
                not control.listener_inodes(self.profile)
                and time.monotonic() < deadline
            ):
                time.sleep(0.02)
            result = self.run_control("start")
            self.assertEqual(result.returncode, 1)
            self.assertIn("Listener conflict", result.stderr)
            status = self.run_control("status", "--json")
            self.assertEqual(status.returncode, 1)
            self.assertEqual(json.loads(status.stdout)["status"], "conflict")
            self.assert_success(self.run_control("stop"))
            self.assertIsNone(foreign.poll())
        finally:
            foreign.terminate()
            foreign.wait(timeout=3)

    def test_interrupted_startup_cleans_all_workers(self):
        self.profile["environment"]["FAKE_MODE"] = "never-ready"
        self.save_profile()
        starter = subprocess.Popen(
            [sys.executable, str(CONTROLLER), "start", "--profile", str(self.path)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        deadline = time.monotonic() + 30
        while (
            not (self.directory / "processes.json").exists()
            and time.monotonic() < deadline
        ):
            time.sleep(0.02)
        self.assertTrue((self.directory / "processes.json").exists())
        before = time.monotonic()
        status = self.run_control("status", "--json")
        self.assertEqual(status.returncode, 4, status.stdout + status.stderr)
        self.assertLess(time.monotonic() - before, 1)
        starter.send_signal(signal.SIGINT)
        stdout, stderr = starter.communicate(timeout=5)
        self.assertEqual(starter.returncode, 1, stdout + stderr)
        self.assertIn("Startup interrupted", stderr)
        self.assert_processes_gone(self.processes())

    def test_runtime_crash_is_reported_without_autorestart(self):
        self.assert_success(self.run_control("start"))
        processes = self.processes()
        os.kill(processes["parent"], signal.SIGKILL)
        self.assert_processes_gone(processes)
        result = self.run_control("status", "--json")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        status = json.loads(result.stdout)
        self.assertEqual(status["status"], "failed")
        self.assertIn("result=signal", status["detail"])
        self.assertEqual(self.processes()["parent"], processes["parent"])

    def test_stale_identity_cannot_stop_or_restart_service(self):
        self.assert_success(self.run_control("start"))
        record = self.directory / "state" / "state.json"
        self.original_state = json.loads(record.read_text())
        stale = dict(self.original_state, invocation_id="0" * 32, pid=os.getpid())
        record.write_text(json.dumps(stale))
        for action in ("start", "stop", "restart", "status"):
            result = self.run_control(action)
            self.assertEqual(result.returncode, 1)
            self.assertIn("stale ownership record", result.stderr)
            self.assertTrue(process_alive(self.processes()["parent"]))

    def test_changed_profile_requires_restart_and_status_uses_snapshot(self):
        self.assert_success(self.run_control("start"))
        self.profile["readiness"]["model"] = "other-model"
        self.save_profile()
        result = self.run_control("start")
        self.assertEqual(result.returncode, 1)
        self.assertIn("different profile", result.stderr)
        self.assert_success(self.run_control("status"))

    def test_missing_ownership_record_cannot_adopt_unit(self):
        self.assert_success(self.run_control("start"))
        record = self.directory / "state" / "state.json"
        self.original_state = json.loads(record.read_text())
        record.unlink()
        result = self.run_control("stop")
        self.assertEqual(result.returncode, 1)
        self.assertIn("no matching ownership record", result.stderr)
        self.assertTrue(process_alive(self.processes()["parent"]))

    def test_precommitted_marker_recovers_interrupted_ownership_write(self):
        self.assert_success(self.run_control("start"))
        record = self.directory / "state" / "state.json"
        original = json.loads(record.read_text())
        record.write_text(
            json.dumps(dict(original, invocation_id=None, last_action="starting"))
        )
        self.assert_success(self.run_control("start"))
        self.assertEqual(
            json.loads(record.read_text())["invocation_id"], original["invocation_id"]
        )

    def test_wrong_marker_cannot_recover_missing_invocation(self):
        self.assert_success(self.run_control("start"))
        record = self.directory / "state" / "state.json"
        self.original_state = json.loads(record.read_text())
        record.write_text(
            json.dumps(dict(self.original_state, invocation_id=None, instance="0" * 32))
        )
        result = self.run_control("stop")
        self.assertEqual(result.returncode, 1)
        self.assertIn("ownership marker differs", result.stderr)
        self.assertTrue(process_alive(self.processes()["parent"]))


class ExecWithoutSystemdTests(unittest.TestCase):
    """The lifecycle fixture through ``_exec`` and the launcher, without systemd."""

    setUp = ServiceLifecycleTests.setUp
    save_profile = ServiceLifecycleTests.save_profile
    processes = ServiceLifecycleTests.processes

    def tearDown(self):
        self.temporary.cleanup()

    def test_exec_starts_the_profile_through_the_launcher(self):
        instance = "0" * 32
        ready = control.run_dir(self.profile, instance) / "ready.json"
        launcher = subprocess.Popen(
            [sys.executable, str(CONTROLLER), "_exec", "--profile", str(self.path)],
            env=dict(os.environ, QSA_SERVICE_INSTANCE=instance),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        processes = None
        try:
            deadline = time.monotonic() + 60
            while not ready.exists() and launcher.poll() is None and time.monotonic() < deadline:
                time.sleep(0.05)
            self.assertTrue(ready.exists(), launcher.poll())
            processes = self.processes()
            self.assertEqual(processes["args"], self.server_args)
            self.assertEqual(processes["env"], "literal $() `value`")
            self.assertEqual(processes["cwd"], str(self.directory))
            self.assertEqual(control.health(control.load_profile(self.path)), (True, "ready"))
            launcher.send_signal(signal.SIGTERM)
            self.assertEqual(launcher.wait(timeout=60), -signal.SIGTERM)
            deadline = time.monotonic() + 10
            while process_alive(processes["parent"]) and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertFalse(process_alive(processes["parent"]))
        finally:
            if launcher.poll() is None:
                launcher.kill()
                launcher.wait()
            # Under systemd the unit's cgroup ends the detached worker.
            if processes and process_alive(processes["worker"]):
                os.kill(processes["worker"], signal.SIGKILL)


if __name__ == "__main__":
    unittest.main()
