"""Every REPLACE copy makes the fork's change to the pinned definition.

The reference fork changed each REPLACE target relative to its base commit
(FORK_BASE_COMMIT). At any pin, the plugin's copy, with its mechanical edits
reverted, must differ from the pinned definition by exactly the edit script
that takes the fork base to the fork. At the fork base itself this means
"verbatim fork copy"; after a pin upgrade it means the fork's change carried
onto the new upstream body. A row whose fork change overlaps an upstream edit
lists its hand resolution in RESOLVED; the review checks those.

Sources are read as text (fork base and fork with ``git show`` in the pinned
checkout, which is a worktree of the fork repository); nothing is imported.
"""

import ast
import difflib
import json
import subprocess
import textwrap
from pathlib import Path

import pytest

from sglang_qsa_hisparse import FORK_BASE_COMMIT, REFERENCE_FORK_COMMIT, fingerprint

REPO = Path(__file__).resolve().parents[2]
PATCHES = REPO / "src" / "sglang_qsa_hisparse" / "patches"
MANIFEST = REPO / "src" / "sglang_qsa_hisparse" / "manifest.json"

# row: (patch module under patches/, qualname of the copy in it, mechanical
#       edits as (fork text, plugin text) including any rename of the copy
#       [, the target's module file at the fork if upstream moved it])
COPIES: dict[str, tuple] = {
    "B02": ("hisparse/scheduler.py", "process_batch_result_prebuilt", ()),
    "B03": ("hisparse/scheduler.py", "process_batch_result_prefill", ()),
    "B04": ("hisparse/scheduler.py", "_handle_sampling_mask_abort", ()),
    "B05": ("hisparse/scheduler.py", "_handle_finish_state_updated_req", ()),
    "M01": ("hisparse/lifecycle.py", "alloc_for_extend", ()),
    "M03": ("hisparse/lifecycle.py", "release_kv_cache", ()),
    "P02": ("hisparse/scheduler.py", "add_one_req", ()),
    "S04": ("hisparse/scheduler.py", "get_next_batch_to_run", ()),
    "S06": ("hisparse/scheduler.py", "_get_new_batch_prefill_raw", ()),
    "S07": ("hisparse/scheduler.py", "on_idle", ()),
}

# row: why the copy differs from "pin + fork change" (an upstream edit overlaps
# the fork's change and was merged by hand).
RESOLVED: dict[str, str] = {
    "P02": (
        "The fork split add_one_req into a new wrapper and the renamed original "
        "body _add_one_req, so on the target qualname its change replaces the "
        "whole body; the deleted side is the pinned body, which v0.5.21 edited "
        "(per_req_token_overhead, has_chunked_req, KV-shard scratch). The "
        "wrapper is the fork's, unchanged; _add_one_req is the pinned "
        "add_one_req with the fork's ignore_eos edit, checked mechanically by "
        "tests/lifecycle/test_renamed_copies.py."
    ),
}


def replace_rows() -> dict[str, str]:
    rows = json.loads(MANIFEST.read_text())["rows"]
    return {
        row: patch["target"]
        for row, spec in rows.items()
        for patch in spec["patches"]
        if patch["hook_type"] == "replace"
    }


def pin_root() -> Path:
    import sglang

    return Path(sglang.__file__).resolve().parents[2]


def _plugin_decorator(node: ast.expr) -> bool:
    func = node.func if isinstance(node, ast.Call) else node
    return isinstance(func, ast.Name) and func.id in {"patch", "attach"}


def segment(source: str, qualname: str) -> list[str]:
    """The definition with its decorators (except the plugin's own
    ``@patch``/``@attach`` declarations), dedented, as lines."""
    node, _ = fingerprint._locate(ast.parse(source), qualname)
    kept = [d for d in node.decorator_list if not _plugin_decorator(d)]
    first = min([node.lineno] + [d.lineno for d in kept])
    lines = source.splitlines(keepends=True)[first - 1 : node.end_lineno]
    return textwrap.dedent("".join(lines)).splitlines()


def git_show(commit: str, path: str) -> str:
    return subprocess.run(
        ["git", "--no-optional-locks", "-C", str(pin_root()), "show", f"{commit}:python/{path}"],
        capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip


def changes(a: list[str], b: list[str]) -> list[tuple[list[str], list[str]]]:
    matcher = difflib.SequenceMatcher(None, a, b, autojunk=False)
    return [
        (a[i1:i2], b[j1:j2])
        for tag, i1, i2, j1, j2 in matcher.get_opcodes()
        if tag != "equal"
    ]


def revert(text: str, edits) -> str:
    for fork_text, plugin_text in edits:
        assert text.count(plugin_text) == 1, plugin_text
        text = text.replace(plugin_text, fork_text)
    return text


def deltas(row: str) -> tuple[list, list]:
    """(fork base -> fork, pin -> plugin copy) edit scripts for a REPLACE row."""
    module, copy_qualname, edits, *moved = COPIES[row]
    path, qualname = fingerprint.resolve_file(pin_root() / "python", replace_rows()[row])
    fork_path = moved[0] if moved else path
    base = segment(git_show(FORK_BASE_COMMIT, fork_path), qualname)
    fork = segment(git_show(REFERENCE_FORK_COMMIT, fork_path), qualname)
    pinned = segment((pin_root() / "python" / path).read_text(), qualname)
    plugin_source = revert((PATCHES / module).read_text(), edits)
    plugin = segment(plugin_source, copy_qualname)
    return changes(base, fork), changes(pinned, plugin)


def test_every_replace_row_is_registered():
    assert sorted(COPIES) == sorted(replace_rows())
    assert set(RESOLVED) <= set(COPIES)


@pytest.mark.parametrize("row", sorted(set(COPIES) - set(RESOLVED)))
def test_copy_makes_the_fork_change_to_the_pin(row):
    fork_change, plugin_change = deltas(row)
    assert fork_change, f"{row}: the fork does not change its target"
    assert plugin_change == fork_change


@pytest.mark.parametrize("row", sorted(RESOLVED))
def test_resolved_rows_really_differ(row):
    """A RESOLVED entry is needed only while the mechanical check fails."""
    fork_change, plugin_change = deltas(row)
    assert plugin_change != fork_change, f"{row}: matches; drop its RESOLVED entry"


def test_changes_reports_an_edit_script():
    base = ["def f(x):", "    a = 1", "    return a"]
    fork = ["def f(x):", "    a = 1", "    hook(a)", "    return a"]
    pinned = ["def f(x):", "    a = 2", "    return a"]
    carried = ["def f(x):", "    a = 2", "    hook(a)", "    return a"]
    dropped = ["def f(x):", "    a = 1", "    hook(a)", "    return a"]
    assert changes(pinned, carried) == changes(base, fork) == [([], ["    hook(a)"])]
    assert changes(pinned, dropped) != changes(base, fork)
