"""pytest_outcomes.py: equal per-test outcomes, declared failures, complete evidence."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "evidence"))

import pytest_outcomes  # noqa: E402

FORK = """_________________ test_known _________________
E       AttributeError: 'Backend' object has no attribute 'qsa_hisparse'
PASSED test/registered/kernel/qsa/test_qsa.py::test_ok[1]
FAILED test/registered/kernel/qsa/test_qsa.py::test_known
SKIPPED [1] test/registered/kernel/qsa/test_qsa.py:10: SM121-only kernel
1 failed, 1 passed, 1 skipped, 3 warnings in 1.00s
"""
PLUGIN = """PASSED tests/qsa/test_qsa.py::test_ok[1]
XFAIL tests/qsa/test_qsa.py::test_known - Known fork failure
SKIPPED [1] tests/qsa/test_qsa.py:12: SM121-only kernel
1 passed, 1 skipped, 1 xfailed in 1.00s
"""
KNOWN = ["--known", "test_qsa.py::test_known=AttributeError"]


def run(tmp_path, fork=FORK, plugin=PLUGIN, expect=3, known=KNOWN):
    (tmp_path / "f.log").write_text(fork)
    (tmp_path / "p.log").write_text(plugin)
    return pytest_outcomes.main(
        ["--fork", str(tmp_path / "f.log"), "--plugin", str(tmp_path / "p.log"),
         "--expect-tests", str(expect), *known]
    )


def test_equal_outcomes_with_declared_failure_pass(tmp_path):
    assert run(tmp_path) == 0


@pytest.mark.parametrize(
    "fork, plugin, expect, known",
    [
        (FORK, PLUGIN, 3, []),  # undeclared failure
        (FORK, PLUGIN, 3, ["--known", "test_qsa.py::test_known=RuntimeError"]),  # other exception
        (FORK, PLUGIN.replace("PASSED tests/qsa/test_qsa.py::test_ok[1]", "FAILED tests/qsa/test_qsa.py::test_ok[1]").replace("1 passed", "1 failed"), 3, KNOWN),
        (FORK, PLUGIN.replace("XFAIL tests/qsa/test_qsa.py::test_known - Known fork failure", "ERROR tests/qsa/test_qsa.py::test_known").replace("1 xfailed", "1 error"), 3, KNOWN),
        (FORK, PLUGIN.replace("XFAIL tests/qsa/test_qsa.py::test_known - Known fork failure", "FAILED tests/qsa/test_qsa.py::test_known - [XPASS(strict)] Known fork failure").replace("1 xfailed", "1 failed"), 3, KNOWN),
        (FORK, PLUGIN + "PASSED tests/qsa/test_qsa.py::test_known\n", 3, KNOWN),  # conflicting duplicate
        ("ImportError while loading conftest\n", "ImportError while loading conftest\n", 0, KNOWN),  # no summary
        (FORK.replace("PASSED test/registered/kernel/qsa/test_qsa.py::test_ok[1]\n", "").replace("1 passed, ", ""),
         PLUGIN.replace("PASSED tests/qsa/test_qsa.py::test_ok[1]\n", "").replace("1 passed, ", ""), 3, KNOWN),  # removed from both
        (FORK, PLUGIN.replace("1 passed, 1 skipped, 1 xfailed", "2 passed, 1 skipped, 1 xfailed"), 3, KNOWN),  # summary mismatch
    ],
    ids=["undeclared", "other-exception", "differing", "error", "strict-xpass",
         "duplicate", "no-summary", "removed-from-both", "summary-mismatch"],
)
def test_false_pass_cases_fail(tmp_path, fork, plugin, expect, known):
    assert run(tmp_path, fork, plugin, expect, known) == 1
