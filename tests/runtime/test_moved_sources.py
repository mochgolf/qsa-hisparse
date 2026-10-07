"""Rows N01-N04: the runtime package and the ported tests are the fork's code.

Each moved module must equal its fork file at the reference commit after
exactly the import rewrites of PLAN.md ("Import mapping"); ``depends.py`` is
the only plugin-owned module in the package. The ported fork tests in this
directory may differ only by the same rewrites plus an added ``pytest``
import and ``integration`` marks. The fork is read with ``git show`` from
``$QSA_FORK_ROOT`` (default ``../qsa-hisparse`` next to this repository).
"""

import difflib
import os
import re
import subprocess
from pathlib import Path

import pytest

from sglang_qsa_hisparse import REFERENCE_FORK_COMMIT

REPO = Path(__file__).resolve().parents[2]
FORK = Path(os.environ.get("QSA_FORK_ROOT", REPO.parent / "qsa-hisparse"))
PACKAGE = REPO / "src" / "sglang_qsa_hisparse" / "hisparse"
FORK_PACKAGE = "python/sglang/srt/mem_cache/qsa_hisparse"
IMPORT_MAP = (
    (b"sglang.srt.mem_cache.qsa_hisparse", b"sglang_qsa_hisparse.hisparse"),
    (
        b"sglang.srt.layers.attention.qsa.hisparse_graph",
        b"sglang_qsa_hisparse.hisparse.graph",
    ),
)
MOVED = {  # plugin module -> fork file
    **{
        f"{name}.py": f"{FORK_PACKAGE}/{name}.py"
        for name in (
            "__init__",
            "config",
            "coordinator",
            "layout",
            "runtime",
            "single_request",
            "slots",  # N01
            "prefix",
            "prefix_cache",  # N02
        )
    },
    "graph.py": "python/sglang/srt/layers/attention/qsa/hisparse_graph.py",  # N03
}
PORTED = ("test_gather.py", "test_runtime.py", "test_single_request.py", "test_slots.py")
ALIASES = {  # N04, dropped
    f"python/sglang/srt/mem_cache/qsa_hisparse_{name}.py"
    for name in ("p2", "slots", "v3")
}
ADDED_TEST_LINE = re.compile(
    rb"import pytest\n|( {4})?@pytest\.mark\.integration\(rows=\(.+\)\)\n"
)


def git(*args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(FORK), *args], capture_output=True, check=True
    ).stdout


def fork_file(path: str) -> bytes:
    return git("show", f"{REFERENCE_FORK_COMMIT}:{path}")


def rewrite(text: bytes) -> bytes:
    for old, new in IMPORT_MAP:
        text = re.sub(re.escape(old) + rb"(?!\w)", new, text)
    return text


@pytest.fixture(scope="module", autouse=True)
def reference_fork():
    found = subprocess.run(
        ["git", "-C", str(FORK), "cat-file", "-e", f"{REFERENCE_FORK_COMMIT}^{{commit}}"],
        capture_output=True,
    )
    if found.returncode:
        pytest.fail(f"{REFERENCE_FORK_COMMIT} not found in {FORK}; set QSA_FORK_ROOT")


def test_package_holds_exactly_the_moved_modules():
    assert {p.name for p in PACKAGE.glob("*.py")} == set(MOVED) | {"depends.py"}
    listed = git("ls-tree", "--name-only", REFERENCE_FORK_COMMIT, f"{FORK_PACKAGE}/")
    assert {Path(p).name for p in listed.decode().split()} == set(MOVED) - {"graph.py"}


@pytest.mark.parametrize("module", sorted(MOVED))
def test_moved_module_is_the_fork_file(module):
    assert (PACKAGE / module).read_bytes() == rewrite(fork_file(MOVED[module]))


@pytest.mark.parametrize("name", PORTED)
def test_ported_test_differs_only_by_imports_and_marks(name):
    fork = rewrite(fork_file(f"test/qsa_hisparse/{name}")).splitlines(keepends=True)
    ours = (Path(__file__).parent / name).read_bytes().splitlines(keepends=True)
    added = []
    matcher = difflib.SequenceMatcher(None, fork, ours, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            assert tag == "insert", (fork[i1:i2], ours[j1:j2])
            added += ours[j1:j2]
    assert all(ADDED_TEST_LINE.fullmatch(line) for line in added), added


def test_dropped_aliases_have_no_importers():
    found = subprocess.run(
        ["git", "-C", str(FORK), "grep", "-l", "-E",
         "qsa_hisparse_(p2|slots|v3)|QSAHiSparse(P2|V3)",
         REFERENCE_FORK_COMMIT, "--", "python", "test", "scripts"],
        capture_output=True,
    )
    assert found.returncode in (0, 1), found.stderr
    users = {line.split(":", 1)[1] for line in found.stdout.decode().splitlines()}
    assert users <= ALIASES
    for alias in ALIASES:
        git("cat-file", "-e", f"{REFERENCE_FORK_COMMIT}:{alias}")
