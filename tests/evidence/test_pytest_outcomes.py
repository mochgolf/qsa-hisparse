"""pytest_outcomes.py: per-test outcomes must match and failures be declared."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "evidence"))

import pytest_outcomes  # noqa: E402

FORK = """PASSED test/registered/kernel/qsa/test_qsa.py::test_ok[1]
FAILED test/registered/kernel/qsa/test_qsa.py::test_known
SKIPPED [1] test/registered/kernel/qsa/test_qsa.py:10: SM121-only kernel
"""
PLUGIN = """PASSED tests/qsa/test_qsa.py::test_ok[1]
XFAIL tests/qsa/test_qsa.py::test_known - Known fork failure
SKIPPED [1] tests/qsa/test_qsa.py:12: SM121-only kernel
"""


def run(tmp_path, fork, plugin, *known):
    (tmp_path / "f.log").write_text(fork)
    (tmp_path / "p.log").write_text(plugin)
    args = [str(tmp_path / "f.log"), str(tmp_path / "p.log")]
    return pytest_outcomes.main(args + (["--known", *known] if known else []))


def test_equal_outcomes_with_declared_failure_pass(tmp_path):
    assert run(tmp_path, FORK, PLUGIN, "test_qsa.py::test_known") == 0


def test_undeclared_failure_fails(tmp_path):
    assert run(tmp_path, FORK, PLUGIN) == 1


def test_differing_outcome_fails(tmp_path):
    plugin = PLUGIN.replace("PASSED tests/qsa/test_qsa.py::test_ok[1]", "FAILED tests/qsa/test_qsa.py::test_ok[1]")
    assert run(tmp_path, FORK, plugin, "test_qsa.py::test_known") == 1


def test_missing_test_fails(tmp_path):
    plugin = PLUGIN.replace("PASSED tests/qsa/test_qsa.py::test_ok[1]\n", "")
    assert run(tmp_path, FORK, plugin, "test_qsa.py::test_known") == 1
