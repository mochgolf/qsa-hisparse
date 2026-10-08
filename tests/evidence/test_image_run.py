"""run_image.py (I-C) on CPU: a fake launcher serves /health and writes
observer files, nvidia-smi is a stub, and the harness is replaced by a
recorder (image_prefix_harness.py has its own tests)."""

import hashlib
import json
import os
import socket
import sys
import urllib.request
from pathlib import Path

import pytest

FAKE_LAUNCHER = """
import http.server, json, os, sys
from pathlib import Path
args = sys.argv[1:]
run_dir = Path(args[args.index("--run-dir") + 1])
run_dir.mkdir()
args = args[args.index("--") + 1 :]
print("argv", json.dumps(args), flush=True)
print("env", json.dumps(dict(os.environ)), flush=True)
state = {"tokens": 64, "token_sha256": "t", "segments": [], "pending": [], "rope": "r", "mamba": []}
for rank in (0, 1):
    restored = dict(state, rope="other") if os.environ.get("FAKE_OBSERVER") == "bad" else state
    rows = [dict(state, event="capture", rank=rank, checkpoint=0, rid="a"),
            dict(restored, event="restore", rank=rank, checkpoint=0, rid="b")]
    Path(os.environ["QSA_EVIDENCE_OBSERVER_DIR"], f"rank-{rank}.jsonl").write_text(
        "".join(json.dumps(row) + "\\n" for row in rows))
    if os.environ.get("FAKE_VIT") != "none":
        Path(os.environ["QSA_EVIDENCE_VIT_DIR"], f"rank-{rank}.jsonl").write_text("{}\\n")
(run_dir / "ready.json").write_text("{}")
class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps({"max_total_num_tokens": 1} if self.path == "/get_server_info" else {})
        self.send_response(200)
        self.end_headers()
        self.wfile.write(body.encode())
    def log_message(self, *a):
        pass
http.server.HTTPServer(("127.0.0.1", int(args[args.index("--port") + 1])), Handler).serve_forever()
"""
SESSIONS = (("vit-cache-on", None), ("vit-cache-off", "0"))


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def port_closed(port):
    with socket.socket() as probe:
        return probe.connect_ex(("127.0.0.1", port)) != 0


@pytest.fixture
def fake(evidence, tmp_path, monkeypatch):
    run = evidence("run_image")
    package = tmp_path / "plugin" / "src" / "sglang_qsa_hisparse"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "launch.py").write_text(FAKE_LAUNCHER)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "nvidia-smi").write_text("#!/bin/sh\n")
    (bin_dir / "nvidia-smi").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("SGLANG_VLM_CACHE_SIZE_MB", "77")  # Never reaches the cache-on session.
    for name, value in (
        ("PYTHON", Path(sys.executable)),
        ("PLUGIN_ROOT", tmp_path / "plugin"),
        ("FORK_ROOT", tmp_path / "fork"),
        ("PIN_ROOT", tmp_path / "pin"),
    ):
        monkeypatch.setattr(run.run_g2, name, value)
    calls, status = [], [0]

    def harness(argv):
        urllib.request.urlopen(argv[1] + "/health", timeout=5).read()  # Its server is up.
        calls.append(argv)
        return status[0]

    monkeypatch.setattr(run.image_prefix_harness, "main", harness)
    return run, calls, status


def arguments(tmp_path, port, reference_sha=None):
    profile = {
        "command": ["/old/python", "-m", "sglang.launch_server", "--model-path", "/private/model",
                    "--tp-size", "2", "--port", str(port), "--enable-deterministic-inference"],
        "environment": {"SGLANG_QSA_HISPARSE_V3": "p2-offload", "SGLANG_VLM_CACHE_SIZE_MB": "88"},
        "listen": {"host": "127.0.0.1", "port": port},
        "readiness": {"path": "/health", "timeout_seconds": 60, "poll_seconds": 0.2, "request_timeout_seconds": 2},
        "stop_timeout_seconds": 10,
    }  # fmt: skip
    (tmp_path / "base.json").write_text(json.dumps(profile))
    text = tmp_path / "text.json"
    text.write_text(json.dumps({"cases": [{"name": "prefix-64-copy", "prefix_length": 64}]}))
    sha = reference_sha or hashlib.sha256(text.read_bytes()).hexdigest()
    (tmp_path / "reference.json").write_text(
        json.dumps({"fixtures_sha256": sha, "cases": [{"name": "prefix-64-copy"}]})
    )
    (tmp_path / "images.json").write_text('{"cases": []}')
    return [
        "--base-profile", str(tmp_path / "base.json"),
        "--fixtures", str(tmp_path / "images.json"),
        "--text-fixtures", str(text),
        "--text-reference", str(tmp_path / "reference.json"),
        "--output", str(tmp_path / "out"),
    ]  # fmt: skip


def test_two_sessions_with_the_image_flags_then_observer_checks(fake, tmp_path):
    run, calls, _ = fake
    port = free_port()
    assert run.main(arguments(tmp_path, port)) == 0
    out = tmp_path / "out"
    summary = json.loads((out / "summary.json").read_text())
    assert summary == {name: {"harness_exit": 0, "observer_problems": []} for name, _ in SESSIONS}
    assert len(calls) == 2
    for (name, vlm_cache), argv in zip(SESSIONS, calls):
        session = out / name
        log = (session / "server.log").read_text().splitlines()
        server_args = json.loads(log[0].removeprefix("argv "))
        env = json.loads(log[1].removeprefix("env "))
        assert server_args[-2:] == ["--mm-preprocess-cache-size-mb", "512"]
        assert server_args[server_args.index("--model-path") + 1] == "/private/model"
        assert env.get("SGLANG_VLM_CACHE_SIZE_MB") == vlm_cache
        assert env["QSA_EVIDENCE_VIT_DIR"] == str(session / "vit")
        assert env["QSA_EVIDENCE_OBSERVER_DIR"] == str(session / "observer")
        assert env["SGLANG_QSA_MODEL_COMPAT"] == "1"
        assert env["SGLANG_CACHE_DIR"] == str(out / "cache")
        assert argv[:2] == ["--url", f"http://127.0.0.1:{port}"]
        assert argv[argv.index("--fixtures") + 1] == str(tmp_path / "images.json")
        assert argv[argv.index("--vit-log") + 1] == str(session / "vit")
        assert argv[argv.index("--observer-log") + 1] == str(session / "observer")
        assert argv[argv.index("--output") + 1] == str(session / "image-prefix.json")
        assert ("--vit-cache-off" in argv) == (vlm_cache == "0")
        assert json.loads((session / "run.json").read_text())["argv"][-1] == "512"
    assert port_closed(port)


@pytest.mark.parametrize(
    "variables, harness_status, problem",
    [
        ({"FAKE_OBSERVER": "bad"}, 0, "observer/rank-0.jsonl restore vs capture"),
        ({"FAKE_VIT": "none"}, 0, "vit/rank-1.jsonl is missing"),
        ({}, 1, None),
    ],
)
def test_observer_problems_or_a_failed_harness_fail_the_run(
    fake, tmp_path, monkeypatch, variables, harness_status, problem
):
    run, _, status = fake
    status[0] = harness_status
    for key, value in variables.items():
        monkeypatch.setenv(key, value)
    port = free_port()
    assert run.main(arguments(tmp_path, port)) == 1
    for result in json.loads((tmp_path / "out" / "summary.json").read_text()).values():
        assert result["harness_exit"] == harness_status
        assert (problem is None) == (not result["observer_problems"])
        assert problem is None or any(problem in p for p in result["observer_problems"])
    assert port_closed(port)


def test_a_reference_of_other_text_fixtures_stops_before_any_server(fake, tmp_path):
    run, calls, _ = fake
    with pytest.raises(SystemExit, match="not the fixtures"):
        run.main(arguments(tmp_path, free_port(), reference_sha="0" * 64))
    assert not (tmp_path / "out").exists() and not calls
