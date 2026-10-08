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
SIGNATURE = "AttributeError: 'Backend' object has no attribute 'qsa_hisparse'"
KNOWN = ["--known", f"test_qsa.py::test_known={SIGNATURE}"]
INVENTORY = "test_qsa.py::test_ok[1]\ntest_qsa.py::test_known\nskip test_qsa.py SM121-only kernel\n"
SUPPLEMENT = f"""_________________ test_known _________________
E       {SIGNATURE}
FAILED tests/qsa/test_qsa.py::test_known
1 failed in 1.00s
"""


def run(tmp_path, fork=FORK, plugin=PLUGIN, inventory=INVENTORY, known=KNOWN, supplement=SUPPLEMENT):
    (tmp_path / "f.log").write_text(fork)
    (tmp_path / "p.log").write_text(plugin)
    (tmp_path / "inventory.txt").write_text(inventory)
    (tmp_path / "s.log").write_text(supplement)
    return pytest_outcomes.main(
        ["--fork", str(tmp_path / "f.log"), "--plugin", str(tmp_path / "p.log"),
         "--inventory", str(tmp_path / "inventory.txt"), "--supplement", str(tmp_path / "s.log"), *known]
    )


def test_equal_outcomes_with_declared_failure_pass(tmp_path):
    assert run(tmp_path) == 0


OTHER = SUPPLEMENT.replace("qsa_hisparse", "unrelated_regression")


@pytest.mark.parametrize(
    "fork, plugin, inventory, known, supplement",
    [
        (FORK, PLUGIN, INVENTORY, [], SUPPLEMENT),  # undeclared failure
        (FORK, PLUGIN, INVENTORY, ["--known", "test_qsa.py::test_known=RuntimeError"], SUPPLEMENT),
        (FORK.replace("qsa_hisparse", "unrelated_regression"), PLUGIN, INVENTORY, KNOWN, SUPPLEMENT),  # other AttributeError (F)
        (FORK, PLUGIN, INVENTORY, KNOWN, OTHER),  # other AttributeError behind P's XFAIL
        (FORK, PLUGIN, INVENTORY, KNOWN, "1 passed in 1.00s\n"),  # XFAIL without --runxfail evidence
        (FORK, PLUGIN.replace("PASSED tests/qsa/test_qsa.py::test_ok[1]", "FAILED tests/qsa/test_qsa.py::test_ok[1]").replace("1 passed", "1 failed"), INVENTORY, KNOWN, SUPPLEMENT),
        (FORK, PLUGIN.replace("XFAIL tests/qsa/test_qsa.py::test_known - Known fork failure", "ERROR tests/qsa/test_qsa.py::test_known").replace("1 xfailed", "1 error"), INVENTORY, KNOWN, SUPPLEMENT),
        (FORK, PLUGIN.replace("XFAIL tests/qsa/test_qsa.py::test_known - Known fork failure", "FAILED tests/qsa/test_qsa.py::test_known - [XPASS(strict)] Known fork failure").replace("1 xfailed", "1 failed"), INVENTORY, KNOWN, SUPPLEMENT),
        (FORK, PLUGIN + "PASSED tests/qsa/test_qsa.py::test_known\n", INVENTORY, KNOWN, SUPPLEMENT),  # conflicting duplicate
        ("ImportError while loading conftest\n", "ImportError while loading conftest\n", INVENTORY, KNOWN, SUPPLEMENT),
        (FORK.replace("test_ok[1]", "test_other"), PLUGIN.replace("test_ok[1]", "test_other"), INVENTORY, KNOWN, SUPPLEMENT),  # same swap in both
        (FORK, PLUGIN.replace("1 passed, 1 skipped, 1 xfailed", "2 passed, 1 skipped, 1 xfailed"), INVENTORY, KNOWN, SUPPLEMENT),
        (FORK.replace("SM121-only kernel", "other reason"), PLUGIN.replace("SM121-only kernel", "other reason"), INVENTORY, KNOWN, SUPPLEMENT),
    ],
    ids=["undeclared", "other-exception-type", "other-attribute-fork", "other-attribute-xfail",
         "xfail-without-evidence", "differing", "error", "strict-xpass", "duplicate",
         "no-summary", "swapped-in-both", "summary-mismatch", "skip-reason-changed"],
)
def test_false_pass_cases_fail(tmp_path, fork, plugin, inventory, known, supplement):
    assert run(tmp_path, fork, plugin, inventory, known, supplement) == 1


def test_signature_must_match_the_whole_exception_line(tmp_path):
    wrapped = FORK.replace(f"E       {SIGNATURE}", f"E       RuntimeError: {SIGNATURE}")
    assert run(tmp_path, fork=wrapped) == 1


VERBOSE_FORK = FORK + "test/registered/kernel/qsa/test_qsa.py::test_sm121 SKIPPED (SM121-only kernel) [ 66%]\n"
VERBOSE_PLUGIN = PLUGIN + "tests/qsa/test_qsa.py::test_sm121 SKIPPED (SM121-only kernel) [ 66%]\n"
VERBOSE_INVENTORY = "test_qsa.py::test_ok[1]\ntest_qsa.py::test_known\nskip test_qsa.py::test_sm121 SM121-only kernel\n"


def test_verbose_skip_ids_match_the_inventory(tmp_path):
    assert run(tmp_path, VERBOSE_FORK, VERBOSE_PLUGIN, VERBOSE_INVENTORY) == 0


def test_a_different_skipped_test_fails(tmp_path):
    plugin = VERBOSE_PLUGIN.replace("::test_sm121 SKIPPED", "::test_unrelated SKIPPED")
    assert run(tmp_path, VERBOSE_FORK, plugin, VERBOSE_INVENTORY) == 1


def test_verbose_logs_require_named_skips_in_the_inventory(tmp_path):
    assert run(tmp_path, VERBOSE_FORK, VERBOSE_PLUGIN, INVENTORY) == 1


def test_v0_5_21_renamed_hc_mix_file_keeps_its_reference_ids(tmp_path):
    fork = "PASSED test/registered/kernel/hyperconnection/test_hc_mix_triton.py::test_mix\n1 passed in 1.00s\n"
    plugin = "PASSED registered/kernels/ops/gemm/test_hc_mix.py::test_mix\n1 passed in 1.00s\n"
    assert run(tmp_path, fork, plugin, "test_hc_mix_triton.py::test_mix\n", []) == 0
