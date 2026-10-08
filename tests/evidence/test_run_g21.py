"""run_g21.py (G2-1 arms) on CPU: the planned files exist in production's
commit, the pin and the plugin; the steps run in order and report failures."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from sglang_qsa_hisparse import REFERENCE_FORK_COMMIT

REPO = Path(__file__).resolve().parents[2]
FORK = Path("/production")


def pin_root() -> Path:
    import sglang

    return Path(sglang.__file__).resolve().parents[2]


def in_reference(path: str) -> bool:
    """``path`` exists in production's commit (the pin checkout is a worktree
    of the fork repository, which holds it)."""
    return not subprocess.run(
        ["git", "-C", str(pin_root()), "cat-file", "-e", f"{REFERENCE_FORK_COMMIT}:{path}"],
        capture_output=True,
    ).returncode


@pytest.fixture
def roots(evidence, monkeypatch):
    run = evidence("run_g21")
    for name, value in (
        ("PYTHON", Path(sys.executable)),
        ("PLUGIN_ROOT", REPO),
        ("FORK_ROOT", FORK),
        ("PIN_ROOT", pin_root()),
    ):
        monkeypatch.setattr(run.run_g2, name, value)
    return run


@pytest.mark.parametrize("arm", ["fork", "plugin"])
def test_every_planned_script_and_test_file_exists(roots, arm, tmp_path):
    steps, pythonpath = roots.plan(arm, tmp_path)
    assert [name for name, *_ in steps] == (
        ["step2", "step3", "step4", "step5", "step6", "step7"]
        + (["step8"] if arm == "fork" else ["step8a", "step8b"])
    )
    checked = 0
    for name, cwd, argv, _ in steps:
        for arg in argv[1:]:
            if not arg.endswith(".py"):
                continue
            path = Path(cwd) / arg
            if path.is_relative_to(FORK):
                assert in_reference(str(path.relative_to(FORK))), (name, arg)
            else:
                assert path.is_file(), (name, path)
            checked += 1
    assert checked == (10 if arm == "fork" else 17)  # plugin: plugin_probe.py too
    expected = (
        [str(FORK / "python")]
        if arm == "fork"
        else [str(REPO / "src"), str(pin_root() / "python")]
    )
    assert pythonpath.split(os.pathsep) == expected


FAKE_PYTHON = f"""#!{sys.executable}
import json, os, sys
NAMES = ("CUDA_VISIBLE_DEVICES", "PYTHONPATH", "TRITON_INTERPRET",
         "SGLANG_QSA_MODEL_COMPAT", "QSA_GPU_TESTS", "QSA_FORK_ROOT")
with open(os.environ["FAKE_CALLS"], "a") as log:
    env = {{name: os.environ.get(name) for name in NAMES}}
    log.write(json.dumps({{"argv": sys.argv[1:], "cwd": os.getcwd(), "env": env}}) + "\\n")
print("ran", " ".join(sys.argv[1:]))
sys.exit(1 if "--native" in sys.argv else 0)
"""


def test_plugin_arm_runs_every_step_and_fails_on_one_failure(roots, tmp_path, monkeypatch):
    fake = tmp_path / "python"
    fake.write_text(FAKE_PYTHON)
    fake.chmod(0o755)
    monkeypatch.setattr(roots.run_g2, "PYTHON", fake)
    monkeypatch.setenv("FAKE_CALLS", str(tmp_path / "calls.jsonl"))
    monkeypatch.setenv("TRITON_INTERPRET", "1")
    monkeypatch.setenv("SGLANG_QSA_MODEL_COMPAT", "1")
    output = tmp_path / "P"
    assert roots.main(["--arm", "plugin", "--output", str(output)]) == 1
    record = json.loads((output / "steps.json").read_text())
    assert [(r["step"], r["exit"]) for r in record] == [
        ("step2", 0), ("step3", 0), ("step4", 0), ("step5", 0),
        ("step6", 1), ("step7", 0), ("step8a", 0), ("step8b", 0),
    ]  # fmt: skip
    calls = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert len(calls) == 8
    for call in calls:
        env = call["env"]
        assert env["CUDA_VISIBLE_DEVICES"] == "0"
        assert env["PYTHONPATH"] == os.pathsep.join([str(REPO / "src"), str(pin_root() / "python")])
        assert env["TRITON_INTERPRET"] is None and env["SGLANG_QSA_MODEL_COMPAT"] is None
    assert calls[0]["env"]["QSA_FORK_ROOT"] == str(FORK)
    assert calls[-1]["cwd"] == str(pin_root() / "test")
    assert (output / "step6.log").read_text().startswith("ran ")
    with pytest.raises(FileExistsError):
        roots.main(["--arm", "plugin", "--output", str(output)])
