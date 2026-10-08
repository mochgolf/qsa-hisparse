"""run_g23.py (G2-3 native checks) on CPU: arm commands from a wrapped
production profile, a fake server and harnesses end to end, and the
cross-arm comparison."""

import hashlib
import json
import os
import socket
import sys
from pathlib import Path

import pytest

FAKE_SERVER = """
import http.server, json, os, sys
from pathlib import Path
args = sys.argv[1:]
if "--run-dir" in args:  # The plugin launcher.
    run_dir = Path(args[args.index("--run-dir") + 1])
    run_dir.mkdir()
    (run_dir / "ready.json").write_text("{}")
    args = args[args.index("--") + 1 :]
Path(os.environ["FAKE_SCRIPTS"], "sglang_temp_file_1.sh").write_text(
    '#!/bin/sh\\nexec numactl --cpunodebind=0 --interleave=all /x/python "$@"')
chats = []
class Handler(http.server.BaseHTTPRequestHandler):
    def reply(self, body):
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
    def do_GET(self):
        self.reply({"argv": args, "pythonpath": os.environ["PYTHONPATH"]})
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/generate":
            count = body["sampling_params"]["max_new_tokens"]
            cached = 0 if body["rid"].endswith("-seed") else 65536
            self.reply({"output_ids": [1] * count, "meta_info": {
                "finish_reason": {"type": "length"}, "cached_tokens": cached,
                "prompt_tokens": len(body["input_ids"])}})
        else:
            chats.append(1)
            details = None if len(chats) == 1 else {"cached_tokens": 2048}
            self.reply({"choices": [{"message": {"content": "42"}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 2089, "prompt_tokens_details": details}})
    def log_message(self, *a):
        pass
http.server.HTTPServer(("127.0.0.1", int(args[args.index("--port") + 1])), Handler).serve_forever()
"""
FAKE_HARNESS = """
import json, sys
args = sys.argv[1:]
out = args[args.index("--output") + 1]
if "latency" in sys.argv[0]:
    pair = {"cold": {"prompt_tokens": 70, "cached_tokens": 0, "wall_seconds": 2.0},
            "warm": {"prompt_tokens": 70, "cached_tokens": 64, "wall_seconds": 1.0}}
    report = {"passed": True, "cases": [{"prefix_length": 64, "pairs": [pair],
              "median_cold_seconds": 2.0, "median_warm_seconds": 1.0}]}
else:
    step = {"prompt_tokens": 70, "cached_tokens": 64}
    report = {"passed": True, **{k: step for k in ("seed", "warm", "after_flush",
              "after_reseed", "after_abort")},
              "logprob": [{"start": 0, "meta_info": {"prompt_tokens": 70, "cached_tokens": 0}}]}
json.dump(report, open(out, "w"))
"""


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def profile(tmp_path, **changes):
    model = tmp_path / "model"
    model.mkdir(exist_ok=True)
    (model / "tokenizer.json").write_text("{}")
    base = {
        "command": ["/usr/bin/env", "/old/python", "-m", "sglang.launch_server",
                    "--model-path", str(model), "--tp-size", "2", "--port", "8081",
                    "--enable-cache-report"],
        "environment": {"SGLANG_QSA_HISPARSE_V3": "p2-offload", "SGLANG_NUMA_INTERLEAVE": "1",
                        "SGLANG_PLUGINS": "other", "SGLANG_QSA_MODEL_COMPAT": "0"},
        "listen": {"host": "127.0.0.1", "port": 8081},
        "readiness": {"path": "/health", "model": "served-name", "timeout_seconds": 60,
                      "poll_seconds": 0.1, "request_timeout_seconds": 2},
        "stop_timeout_seconds": 10,
    }  # fmt: skip
    base.update(changes)
    return base


@pytest.fixture
def g23(evidence, tmp_path, monkeypatch):
    run = evidence("run_g23")
    for name, value in (
        ("PYTHON", Path(sys.executable)),
        ("PLUGIN_ROOT", tmp_path / "plugin"),
        ("FORK_ROOT", tmp_path / "fork"),
        ("PIN_ROOT", tmp_path / "pin"),
    ):
        monkeypatch.setattr(run.run_g2, name, value)
    return run


def test_arm_commands_keep_the_wrapper_and_change_only_port_and_arm(g23, tmp_path):
    base = profile(tmp_path)
    out = tmp_path / "out"
    argv, env, ready = g23.server("fork", base, out, 8090)
    tail = ["--model-path", str(tmp_path / "model"), "--tp-size", "2", "--port", "8090",
            "--enable-cache-report"]  # fmt: skip
    assert argv == ["/usr/bin/env", sys.executable, "-m", "sglang.launch_server", *tail]
    assert ready is None and env["PYTHONPATH"] == str(tmp_path / "fork" / "python")
    assert "SGLANG_PLUGINS" not in env and "SGLANG_QSA_MODEL_COMPAT" not in env
    assert env["SGLANG_NUMA_INTERLEAVE"] == "1" and env["SGLANG_CACHE_DIR"] == str(out / "cache")
    argv, env, ready = g23.server("plugin", base, out, 8090)
    assert argv == ["/usr/bin/env", sys.executable, "-m", "sglang_qsa_hisparse.launch",
                    "--run-dir", str(out / "launcher"), "--", *tail]  # fmt: skip
    assert ready == out / "launcher" / "ready.json"
    assert env["SGLANG_QSA_MODEL_COMPAT"] == "1"
    assert env["PYTHONPATH"] == os.pathsep.join(
        [str(tmp_path / "plugin" / "src"), str(tmp_path / "pin" / "python")]
    )
    assert base["command"][base["command"].index("--port") + 1] == "8081"  # Not mutated.


@pytest.mark.parametrize(
    "change",
    [
        {"environment": {}},
        {"command": ["/p", "-m", "sglang.launch_server", "--enable-deterministic-inference", "--port", "1"]},
        {"command": ["/p", "-m", "other", "--port", "1"]},
        {"command": ["/p", "-m", "sglang.launch_server"]},
    ],
)
def test_other_profiles_are_refused(g23, tmp_path, change):
    with pytest.raises(SystemExit):
        g23.server("fork", profile(tmp_path, **change), tmp_path / "out", 8090)


def write_tree(tmp_path):
    for module in ("fork/python/sglang/launch_server.py", "plugin/src/sglang_qsa_hisparse/launch.py"):
        (tmp_path / module).parent.mkdir(parents=True)
        (tmp_path / module).write_text(FAKE_SERVER)
    for name in ("sglang", "sglang_qsa_hisparse"):
        package = tmp_path / ("fork/python" if name == "sglang" else "plugin/src") / name
        (package / "__init__.py").write_text("")
    manual = tmp_path / "fork" / "test" / "manual"
    manual.mkdir(parents=True)
    for name in ("latency", "lifecycle"):
        (manual / f"qsa_hisparse_prefix_{name}.py").write_text(FAKE_HARNESS)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "nvidia-smi").write_text("#!/bin/sh\n")
    (bin_dir / "nvidia-smi").chmod(0o755)
    return bin_dir


@pytest.mark.parametrize("arm", ["fork", "plugin"])
def test_arm_runs_every_check_against_its_server_then_stops_it(g23, tmp_path, monkeypatch, arm):
    bin_dir = write_tree(tmp_path)
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "sglang_temp_file_0.sh").write_text("exec numactl --stale x \"$@\"")
    os.utime(scripts / "sglang_temp_file_0.sh", (0, 0))  # Before this run: ignored.
    monkeypatch.setattr(g23, "NUMACTL_SCRIPTS", str(scripts / "sglang_temp_file_*.sh"))
    monkeypatch.setenv("FAKE_SCRIPTS", str(scripts))
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    base = profile(tmp_path)
    (tmp_path / "base.json").write_text(json.dumps(base))
    cases = [{"name": "prefix-65536-copy", "prefix_ids": [1, 2], "input_ids": [1, 2, 3]}]
    tokenizer = hashlib.sha256(b"{}").hexdigest()
    (tmp_path / "fixtures.json").write_text(json.dumps({"tokenizer_sha256": tokenizer, "cases": cases}))
    port, out = free_port(), tmp_path / f"arm-{arm}"
    args = ["run", "--arm", arm, "--base-profile", str(tmp_path / "base.json"),
            "--fixtures", str(tmp_path / "fixtures.json"), "--output", str(out), "--port", str(port)]  # fmt: skip
    assert g23.main(args) == 0
    assert json.loads((out / "g23.json").read_text()) == dict.fromkeys(
        ("latency", "lifecycle", "concurrency", "smoke"), 0
    )
    info = json.loads((out / "server_info.json").read_text())
    assert info["argv"][info["argv"].index("--port") + 1] == str(port)
    assert json.loads((out / "numactl.json").read_text()) == ["--cpunodebind=0 --interleave=all"]
    smoke = json.loads((out / "smoke.json").read_text())
    assert smoke["passed"] and smoke["request"]["model"] == "served-name"
    assert len(json.loads((out / "concurrency.json").read_text())["responses"]) == 8
    with socket.socket() as probe:
        assert probe.connect_ex(("127.0.0.1", port)) != 0  # Stopped.


def test_fixtures_of_another_tokenizer_are_refused_before_any_server(g23, tmp_path):
    base = profile(tmp_path)
    (tmp_path / "base.json").write_text(json.dumps(base))
    (tmp_path / "fixtures.json").write_text(json.dumps({"tokenizer_sha256": "0", "cases": []}))
    args = ["run", "--arm", "fork", "--base-profile", str(tmp_path / "base.json"),
            "--fixtures", str(tmp_path / "fixtures.json"), "--output", str(tmp_path / "F")]  # fmt: skip
    with pytest.raises(SystemExit, match="tokenizer"):
        g23.main(args)
    assert not (tmp_path / "F").exists()


def arm_dir(root, *, warm_cached=64, numactl=("--cpunodebind=0 --interleave=all",), status=0):
    root.mkdir()
    pair = {"cold": {"prompt_tokens": 70, "cached_tokens": 0},
            "warm": {"prompt_tokens": 70, "cached_tokens": warm_cached}}  # fmt: skip
    step = {"prompt_tokens": 70, "cached_tokens": 64}
    files = {
        "g23.json": {"latency": status, "lifecycle": 0, "concurrency": 0, "smoke": 0},
        "latency.json": {"cases": [{"prefix_length": 64, "pairs": [pair],
                                    "median_cold_seconds": 2.0, "median_warm_seconds": 1.0}]},
        "lifecycle.json": {**{k: step for k in ("seed", "warm", "after_flush", "after_reseed", "after_abort")},
                           "logprob": [{"start": 0, "meta_info": {"prompt_tokens": 70, "cached_tokens": 0}}]},
        "concurrency.json": {"responses": [{"meta_info": {"prompt_tokens": 70, "cached_tokens": 65536}}] * 8},
        "smoke.json": {"responses": [
            {"label": "cold", "response": {"usage": {"prompt_tokens": 2089, "prompt_tokens_details": None}}},
            {"label": "warm", "response": {"usage": {"prompt_tokens": 2089,
                                                     "prompt_tokens_details": {"cached_tokens": 2048}}}}]},
        "numactl.json": list(numactl),
    }  # fmt: skip
    for name, value in files.items():
        (root / name).write_text(json.dumps(value))
    return root


@pytest.mark.parametrize(
    "plugin, passed",
    [
        ({}, True),
        ({"warm_cached": 0}, False),
        ({"numactl": ("--cpunodebind=0 --membind=0",)}, False),
        ({"status": 1}, False),
    ],
)
def test_compare_requires_passing_arms_equal_accounting_and_numactl(g23, tmp_path, capsys, plugin, passed):
    fork = arm_dir(tmp_path / "F")
    other = arm_dir(tmp_path / "P", **plugin)
    assert g23.main(["compare", str(fork), str(other)]) == (0 if passed else 1)
    printed = capsys.readouterr().out
    assert ("PASS" in printed) == passed
    if passed:
        assert "2.000  2.000  1.000  1.000" in printed  # Latency side by side.
