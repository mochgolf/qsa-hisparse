"""All W3 rows activate together against manifest.json (plus pending entries)."""

from pools.activation import ROWS, activated, manifest_rows
from sglang_qsa_hisparse import patching


def test_w3_rows_activate_together():
    manifest = manifest_rows(ROWS)
    assert set(manifest) == set(ROWS)  # C03 (a docstring) is a drop.
    expected = {p["target"] for entry in manifest.values() for p in entry["patches"]}
    with activated(*ROWS):
        assert set(patching._applied) == expected
        assert {spec.row for spec in patching._declared} == set(ROWS)
