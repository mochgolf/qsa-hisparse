"""Reduced compat-only profile: its arithmetic and its run script, on CPU.

The end-to-end test runs the script against a fake fork tree whose
``sglang.launch_server`` is a small HTTP server and a fake ``nvidia-smi``.
"""

import hashlib
import json
import os
import socket
import sys
import textwrap
from pathlib import Path

import pytest

PROFILE = json.loads(
    (Path(__file__).resolve().parents[2] / "tools/evidence/compat_profile.json").read_text()
)
PRIVATE_MODEL = "/private/model/path"


def test_profile_fits_without_offload_and_admits_a_full_context():
    g = PROFILE["geometry"]
    raw = g["full_attention_layers"] * 2 * g["kv_heads_per_rank"] * g["head_dim"]
    cell = raw * g["kv_bytes_per_element"] + g["index_bytes_per_token"]
    budget = g["base_fixed_bytes"] + g["base_max_total_tokens"] * g["index_bytes_per_token"]
    tokens = int(PROFILE["set_flags"]["--max-total-tokens"])
    assert (raw, cell, budget) == (6144, 6912, 3222012672)
    assert g["base_max_total_tokens"] * cell > budget  # B8 needs offload.
    assert (tokens + g["page_size"]) * cell <= budget
    assert tokens == g["context_length"] + g["clip_max_new_tokens"] + g["page_size"]
    assert tokens % g["page_size"] == 0
    largest = PROFILE["fixtures"]["largest_request_tokens"]
    assert largest <= g["context_length"] and largest + g["page_size"] < tokens
    assert PROFILE["status"] == "unvalidated until G2"
    assert PROFILE["base_profile"]["requires"]["--context-length"] == str(g["context_length"])


def base_profile(port, **changes):
    args = [
        "--model-path", PRIVATE_MODEL,
        "--host", "127.0.0.1",
        "--tp-size", "2",
        "--context-length", "262144",
        "--page-size", "64",
        "--kv-cache-dtype", "fp8_e4m3",
        "--mem-fraction-static", "0.96",
        "--disable-radix-cache",
        "--max-running-requests", "8",
        "--max-total-tokens", "2097152",
        "--chunked-prefill-size", "2048",
        "--cuda-graph-bs-decode", "1", "2", "3", "4", "5", "6", "7", "8",
        "--port", str(port),
        "--enable-deterministic-inference",
    ]  # fmt: skip
    for flag, value in changes.items():
        flag = "--" + flag.replace("_", "-")
        if value is None:
            args.remove(flag)
        else:
            args[args.index(flag) + 1] = value
    return {
        "command": ["/old/python", "-m", "sglang.launch_server", *args],
        "environment": {
            "PATH": os.environ["PATH"],
            "PYTHONPATH": "/base/python",
            "SGLANG_QWEN38_GDN_QKVZ_WNA16": "0",
            "SGLANG_QSA_HISPARSE_V3": "p2-offload",
            "SGLANG_QSA_HISPARSE_V3_OBSERVE": "strict",
            "SGLANG_QSA_HISPARSE_PREFIX_CACHE_MB": "8192",
            "SGLANG_QSA_HISPARSE_V3_EVENTS": "/base/events",
            "SGLANG_LOGPROB_CHUNK_SIZE": "256",
        },
        "readiness": {"path": "/health", "timeout_seconds": 60, "poll_seconds": 0.2, "request_timeout_seconds": 2},
        "stop_timeout_seconds": 10,
    }  # fmt: skip


ROOTS = {"fork": Path("/f"), "plugin": Path("/p"), "pin": Path("/n")}


def test_server_command_changes_only_the_profile_and_arm(evidence, monkeypatch):
    run = evidence("run_compat")
    monkeypatch.setenv("SGLANG_QSA_MODEL_COMPAT", "1")  # Never leaks from the caller.
    monkeypatch.setenv("SGLANG_PLUGINS", "other")
    base = base_profile(8082)
    expected = [a if a != "2097152" else "266304" for a in base["command"][3:]]
    common = dict(python="py", roots=ROOTS, output=Path("/out"))

    argv, env = run.server_command("fork", base, PROFILE, **common)
    assert argv == ["py", "-m", "sglang.launch_server", *expected]
    assert not [k for k in env if k.startswith("SGLANG_QSA_") or k == "SGLANG_PLUGINS"]
    assert env["PYTHONPATH"] == "/f/python"
    assert env["SGLANG_QWEN38_GDN_QKVZ_WNA16"] == "0" and env["SGLANG_LOGPROB_CHUNK_SIZE"] == "256"
    assert env["SGLANG_CACHE_DIR"] == "/out/cache" and env["PYTHONDONTWRITEBYTECODE"] == "1"

    argv, env = run.server_command("plugin", base, PROFILE, **common)
    assert argv == ["py", "-m", "sglang_qsa_hisparse.launch", "--", *expected]
    assert {k: v for k, v in env.items() if k.startswith("SGLANG_QSA_") or k == "SGLANG_PLUGINS"} == {
        "SGLANG_QSA_MODEL_COMPAT": "1",
        "SGLANG_PLUGINS": "qsa_hisparse",
    }
    assert env["PYTHONPATH"] == os.pathsep.join(["/p/src", "/n/python"])


@pytest.mark.parametrize(
    "changes",
    [
        {"context_length": "131072"},
        {"max_total_tokens": "266304"},
        {"enable_deterministic_inference": None},
        {"disable_radix_cache": None},
    ],
)
def test_unexpected_base_profile_is_refused(evidence, changes):
    with pytest.raises(SystemExit, match="base profile must contain"):
        evidence("run_compat").edited_args(base_profile(8082, **changes)["command"], PROFILE)


def fixtures(tmp_path, lengths=(64, 128, 262016), name="fixtures.json"):
    cases = [
        {
            "name": f"prefix-{n}-{name}",
            "prefix_length": n,
            "input_ids": [7] * (n + extra),
            "max_new_tokens": 32,
        }
        for n in lengths
        for name, extra in (("copy", 12), ("arithmetic", 21))
    ]
    path = tmp_path / name
    path.write_text(json.dumps({"format": 1, "cases": cases}))
    return path


def profile_for(path, **fixture_changes):
    spec = dict(PROFILE["fixtures"], sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    spec.update(kept_cases=6, max_prefix=262016, largest_request_tokens=262069)
    spec.update(fixture_changes)
    return dict(PROFILE, fixtures=spec)


def test_fixture_rule_keeps_cases_that_pass_the_context_check(evidence, tmp_path):
    run = evidence("run_compat")
    path = fixtures(tmp_path)
    assert len(run.checked_fixtures(path, profile_for(path))) == 6
    longer = fixtures(tmp_path, (64, 128, 262016, 262144), "longer.json")  # 262144 + 12 + 32 exceeds it.
    kept = run.checked_fixtures(longer, profile_for(longer))
    assert [c["prefix_length"] for c in kept] == [64, 64, 128, 128, 262016, 262016]
    with pytest.raises(SystemExit, match="differ"):
        run.checked_fixtures(longer, profile_for(longer, max_prefix=262144))
    with pytest.raises(SystemExit, match="differ"):
        run.checked_fixtures(path, profile_for(path, largest_request_tokens=262068))
    with pytest.raises(SystemExit, match="not the frozen fixture file"):
        run.checked_fixtures(path, PROFILE)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def fake_tree(tmp_path, gpu_apps=""):
    fork = tmp_path / "fork"
    (fork / "python/sglang").mkdir(parents=True)
    (fork / "python/sglang/__init__.py").write_text("")
    (fork / "python/sglang/launch_server.py").write_text(
        textwrap.dedent(
            """
            import http.server, json, os, sys
            args = sys.argv[1:]
            print("argv", json.dumps(args), flush=True)
            print("env", json.dumps(dict(os.environ)), flush=True)
            class Handler(http.server.BaseHTTPRequestHandler):
                def do_GET(self):
                    info = {"max_total_num_tokens": int(args[args.index("--max-total-tokens") + 1])}
                    body = json.dumps(info if self.path == "/get_server_info" else {}).encode()
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(body)
                def log_message(self, *a):
                    pass
            port = int(args[args.index("--port") + 1])
            http.server.HTTPServer(("127.0.0.1", port), Handler).serve_forever()
            """
        )
    )
    (fork / "test/manual").mkdir(parents=True)
    (fork / "test/manual/qsa_hisparse_prefix_acceptance.py").write_text(
        textwrap.dedent(
            """
            import json, os, sys, urllib.request
            args = sys.argv[1:]
            urllib.request.urlopen(args[args.index("--url") + 1] + "/health").read()
            with open(args[args.index("--output") + 1], "w") as out:
                json.dump({"passed": True, "argv": args, "cwd": os.getcwd()}, out)
            """
        )
    )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    smi = bin_dir / "nvidia-smi"
    smi.write_text(f"#!/bin/sh\nprintf '{gpu_apps}'\n")
    smi.chmod(0o755)
    return fork, bin_dir


def arm_args(tmp_path, fork, port, path):
    base = tmp_path / "base.json"
    base.write_text(json.dumps(base_profile(port)))
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps(profile_for(path)))
    return [
        "--arm", "fork",
        "--base-profile", str(base),
        "--fixtures", str(path),
        "--output", str(tmp_path / "F"),
        "--python", sys.executable,
        "--fork-root", str(fork),
        "--profile", str(profile),
    ]  # fmt: skip


def test_fork_arm_runs_the_harness_and_stops_its_server(evidence, tmp_path, monkeypatch):
    run = evidence("run_compat")
    fork, bin_dir = fake_tree(tmp_path)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    port = free_port()
    path = fixtures(tmp_path)
    assert run.main(arm_args(tmp_path, fork, port, path)) == 0

    out = tmp_path / "F"
    log = (out / "server.log").read_text().splitlines()
    argv = json.loads(log[0].removeprefix("argv "))
    env = json.loads(log[1].removeprefix("env "))
    assert argv[argv.index("--max-total-tokens") + 1] == "266304"
    assert argv[argv.index("--model-path") + 1] == PRIVATE_MODEL
    assert not [k for k in env if k.startswith("SGLANG_QSA_")]
    assert env["PYTHONPATH"] == str(fork / "python")
    assert json.loads((out / "server_info.json").read_text()) == {"max_total_num_tokens": 266304}
    report = json.loads((out / "compat-cold.json").read_text())
    assert report["argv"][:1] == ["baseline"]
    assert report["argv"][report["argv"].index("--max-prefix") + 1] == "262016"
    assert report["cwd"] == str(fork / "test/manual")
    with socket.socket() as probe:
        assert probe.connect_ex(("127.0.0.1", port)) != 0  # Its server was stopped.


def test_busy_gpu_used_port_and_production_port_are_refused(evidence, tmp_path, monkeypatch):
    run = evidence("run_compat")
    fork, bin_dir = fake_tree(tmp_path, gpu_apps="4242, 45000 MiB")
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    path = fixtures(tmp_path)
    with pytest.raises(SystemExit, match="GPU compute processes are running"):
        run.main(arm_args(tmp_path, fork, free_port(), path))
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        with pytest.raises(SystemExit, match="is in use"):
            run.main(arm_args(tmp_path, fork, listener.getsockname()[1], path))
    with pytest.raises(SystemExit, match="production port"):
        run.main(arm_args(tmp_path, fork, 8081, path))
    assert not (tmp_path / "F").exists()
