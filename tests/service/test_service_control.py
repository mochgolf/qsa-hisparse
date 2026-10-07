"""Real CPU lifecycle checks against the user systemd manager (no model/GPU)."""

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


CONTROLLER = Path(__file__).resolve().parents[2] / "scripts" / "qsa_service.py"
spec = importlib.util.spec_from_file_location("qsa_service_control", CONTROLLER)
control = importlib.util.module_from_spec(spec)
spec.loader.exec_module(control)

FAKE_SERVICE = r"""
import http.server
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

port, directory, mode = int(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
worker = subprocess.Popen([sys.executable, "-c", "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(120)"], start_new_session=True)
(directory / "processes.json").write_text(json.dumps({"parent": os.getpid(), "worker": worker.pid, "args": sys.argv[4:], "env": os.environ.get("FAKE_EXACT_ENV"), "cwd": os.getcwd()}))
print("fake service launched", mode, flush=True)
if mode == "exit":
    print("intentional startup failure", flush=True)
    raise SystemExit(17)
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
                "command": [sys.executable, "-c", "pass"],
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
        self.fake = self.directory / "fake.py"
        self.fake.write_text(FAKE_SERVICE)
        with socket.socket() as socket_probe:
            socket_probe.bind(("127.0.0.1", 0))
            port = socket_probe.getsockname()[1]
        self.profile = {
            "version": 1,
            "name": "cpu-test",
            "cwd": str(self.directory),
            "state_dir": str(self.directory / "state"),
            "command": [
                sys.executable,
                str(self.fake),
                str(port),
                str(self.directory),
                "ready",
                "literal $() `words`",
                "space arg",
            ],
            "environment": {"FAKE_EXACT_ENV": "literal $() `value`"},
            "listen": {"host": "127.0.0.1", "port": port},
            "readiness": {
                "path": "/health",
                "model": "fake-model",
                # Shared inference hosts can pause the CPU fixture between its
                # startup log and socket bind. Failure cases override this.
                "timeout_seconds": 10,
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
            timeout=25,
        )

    def assert_success(self, result):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def processes(self):
        return json.loads((self.directory / "processes.json").read_text())

    def assert_processes_gone(self, processes):
        deadline = time.monotonic() + 2
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
        self.assertEqual(processes["args"], ["literal $() `words`", "space arg"])
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
        self.profile["command"][4] = "never-ready"
        self.profile["readiness"]["timeout_seconds"] = 0.4
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
        self.profile["command"][4] = "exit"
        self.save_profile()
        result = self.run_control("start")
        self.assertEqual(result.returncode, 1)
        self.assertIn("intentional startup failure", result.stderr)
        self.assertIn("exit=17", result.stderr)
        self.assert_processes_gone(self.processes())

    def test_model_identity_is_required(self):
        self.profile["command"][4] = "wrong-model"
        self.profile["readiness"]["timeout_seconds"] = 0.4
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
        self.profile["command"][4] = "never-ready"
        self.save_profile()
        starter = subprocess.Popen(
            [sys.executable, str(CONTROLLER), "start", "--profile", str(self.path)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        deadline = time.monotonic() + 2
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


if __name__ == "__main__":
    unittest.main()
