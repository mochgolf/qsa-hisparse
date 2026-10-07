"""Inventory completeness: every fork hunk maps to exactly one inventory row.

The fork diff is read (read-only) from the pinned SGLang checkout, which is
a worktree of the fork repository; the pinned checkout is the one the tests
import SGLang from.
"""

import re
import subprocess
from collections import Counter
from pathlib import Path

import pytest

from sglang_qsa_hisparse import PINNED_SGLANG_COMMIT, REFERENCE_FORK_COMMIT

REPO = Path(__file__).resolve().parents[2]
INVENTORY = REPO / "docs" / "patch-inventory.md"
HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


def pin_root() -> Path:
    import sglang

    return Path(sglang.__file__).resolve().parents[2]


def fork_hunks() -> Counter:
    """(file under python/sglang, '+start,count' or 'new') for the fork diff."""
    diff = subprocess.run(
        ["git", "--no-optional-locks", "-C", str(pin_root()), "diff", "-U0",
         "--no-color", "--no-ext-diff", "--no-renames", "--diff-algorithm=myers",
         PINNED_SGLANG_COMMIT, REFERENCE_FORK_COMMIT, "--", "python/sglang"],
        capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    hunks: Counter = Counter()
    path = None
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            path = line.split(" b/python/sglang/", 1)[1]
        elif line.startswith("new file mode"):
            hunks[(path, "new")] += 1
        elif line.startswith("deleted file mode"):
            hunks[(path, "deleted")] += 1
        elif match := HUNK.match(line):
            if (path, "new") not in hunks:
                hunks[(path, f"+{match[1]},{match[2] or 1}")] += 1
    return hunks


def section(text: str, heading: str) -> str:
    start = text.index(heading)
    end = text.find("\n## ", start + len(heading))
    return text[start : end if end != -1 else len(text)]


def appendix_a() -> list[tuple[str, str, str]]:
    entries = []
    for line in section(INVENTORY.read_text(), "## Appendix A.").splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 3 or not cells[1].startswith(("+", "(new file)")):
            continue
        file, hunk, row = cells
        hunk = "new" if hunk == "(new file)" else hunk.split(" ", 1)[0]
        entries.append((file, hunk, row))
    return entries


def inventory_rows() -> set[str]:
    rows = section(INVENTORY.read_text(), "## 1. Inventory")
    return set(re.findall(r"^\| ([A-Z]+\d+) \|", rows, flags=re.MULTILINE))


def test_every_fork_hunk_is_mapped_exactly_once():
    hunks = fork_hunks()
    mapped = Counter((file, hunk) for file, hunk, _ in appendix_a())
    stated = re.search(
        r"(\d+) modified files, (\d+) hunks, (\d+) new files",
        section(INVENTORY.read_text(), "## Appendix A."),
    )
    files = {file for file, hunk in hunks if hunk != "new"}
    new = [file for file, hunk in hunks if hunk == "new"]
    assert (len(files), sum(hunks.values()) - len(new), len(new)) == tuple(
        map(int, stated.groups())
    )
    unmapped = sorted((hunks - mapped).elements())
    stale = sorted((mapped - hunks).elements())
    repeated = sorted(key for key, count in mapped.items() if count > 1)
    assert not unmapped, f"fork hunks missing from Appendix A: {unmapped}"
    assert not stale, f"Appendix A entries with no fork hunk: {stale}"
    assert not repeated, f"hunks mapped more than once: {repeated}"


def test_appendix_rows_are_inventory_rows():
    defined = inventory_rows()
    used = {row for _, _, row in appendix_a()}
    assert used, "Appendix A parsed empty"
    assert sorted(used - defined) == [], "Appendix A rows not defined in section 1"
    assert sorted(defined - used) == [], "Section 1 rows that map no fork hunk"


@pytest.mark.parametrize(
    "line, key",
    [
        ("@@ -40,0 +41,3 @@ def f():", "+41,3"),
        ("@@ -5 +5 @@", "+5,1"),
        ("@@ -6,2 +5,0 @@", "+5,0"),
    ],
)
def test_hunk_header_parsing(line, key):
    match = HUNK.match(line)
    assert f"+{match[1]},{match[2] or 1}" == key
