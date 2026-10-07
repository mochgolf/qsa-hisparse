"""RUNTIME_DEPENDS: every entry is a pinned, resolvable function or class."""

import json
from importlib import resources

from sglang_qsa_hisparse import fingerprint, patching
from sglang_qsa_hisparse.hisparse.depends import RUNTIME_DEPENDS


def test_runtime_depends_are_pinned_definitions():
    assert len(set(RUNTIME_DEPENDS)) == len(RUNTIME_DEPENDS)
    path = resources.files("sglang_qsa_hisparse").joinpath("fingerprints/runtime.json")
    assert set(json.loads(path.read_text())) == set(RUNTIME_DEPENDS)
    # Checks the module bytes of the SGLang on sys.path (the pin).
    records = fingerprint.verify(list(RUNTIME_DEPENDS))
    assert {record["kind"] for record in records.values()} <= {"function", "class"}
    for name in RUNTIME_DEPENDS:
        assert not isinstance(patching._raw_attribute(name), property), name
