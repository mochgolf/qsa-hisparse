"""Renamed fork bodies make the fork's change to the pinned target.

The fork split two REPLACE targets into a new wrapper and the renamed original
body: P02 ``PrefillAdder.add_one_req`` -> ``_add_one_req`` and M01
``alloc_for_extend`` -> ``_alloc_for_extend``. ``test_replace_deltas.py``
compares the target's qualname, i.e. the wrapper; this applies the same check
to the renamed body: the plugin copy equals the fork's change (target at the
fork base -> renamed body) merged into the pinned target.
"""

import importlib.util
from pathlib import Path

import pytest

from sglang_qsa_hisparse import FORK_BASE_COMMIT, REFERENCE_FORK_COMMIT, fingerprint

_spec = importlib.util.spec_from_file_location(
    "replace_deltas",
    Path(__file__).resolve().parents[1] / "regression" / "test_replace_deltas.py",
)
deltas = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(deltas)

# row: (target, renamed qualname in the fork, patch module, plugin copy name)
RENAMED = {
    "M01": (
        "sglang.srt.mem_cache.allocation.alloc_for_extend",
        "_alloc_for_extend",
        "hisparse/lifecycle.py",
        "_alloc_for_extend",
    ),
    "P02": (
        "sglang.srt.managers.schedule_policy.PrefillAdder.add_one_req",
        "PrefillAdder._add_one_req",
        "hisparse/scheduler.py",
        "_add_one_req",
    ),
}


@pytest.mark.parametrize("row", sorted(RENAMED))
def test_renamed_body_makes_the_fork_change_to_the_pin(row):
    target, fork_qualname, module, copy = RENAMED[row]
    root = deltas.pin_root() / "python"
    path, qualname = fingerprint.resolve_file(root, target)
    base = deltas.segment(deltas.git_show(FORK_BASE_COMMIT, path), qualname)
    fork = deltas.segment(deltas.git_show(REFERENCE_FORK_COMMIT, path), fork_qualname)
    pinned = deltas.segment((root / path).read_text(), qualname)
    plugin = deltas.segment((deltas.PATCHES / module).read_text(), copy)
    assert base != fork, f"{row}: the fork does not change its renamed body"
    assert plugin == deltas.carried(base, fork, pinned)
