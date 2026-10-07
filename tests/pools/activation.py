"""Activate W3's hisparse rows against a manifest restricted to them.

Real activation of ``hisparse`` needs every row of the feature (other
workstreams), so tests activate only the rows they exercise: the declarations
and ``manifest.json`` entries of those rows, pinned fingerprints included.
Patched attributes and the registry are restored on exit.
"""

import contextlib
import importlib
import pkgutil

from sglang_qsa_hisparse import patching
from sglang_qsa_hisparse.features import Features

MODULES = (
    "sglang_qsa_hisparse.patches.hisparse.pools",
    "sglang_qsa_hisparse.patches.hisparse.graph",
)
ROWS = ("K01", "K02", "K03", "F01", "R01", "R02", "C01", "C02", "G01", "G02", "G03")
# Hooks W3 added beyond the inventory, until the orchestrator updates
# docs/patch-inventory.md and manifest.json: under deviation D3 the F01
# instance attribute must also survive EagerRunner.load_batch's copy.
PENDING_MANIFEST: dict = {}  # F01's EagerRunner hook is in manifest.json now.


def manifest_rows(rows):
    """``manifest.json`` entries of ``rows``, plus pending entries not in it yet."""
    manifest = {}
    for row, entry in patching.load_manifest().items():
        if row in rows:
            manifest[row] = {**entry, "patches": list(entry["patches"])}
    for row, patches in PENDING_MANIFEST.items():
        if row in manifest:
            manifest[row]["patches"] += [p for p in patches if p not in manifest[row]["patches"]]
    return manifest


@contextlib.contextmanager
def activated(*rows):
    from sglang.srt.plugins.hook_registry import HookRegistry

    assert set(rows) <= set(ROWS), rows
    for module in MODULES:
        importlib.import_module(module)
    manifest = manifest_rows(rows)
    specs = [spec for spec in patching._declared if spec.row in rows]
    originals = {spec.target: patching._raw_attribute(spec.target) for spec in specs}
    saved = (
        list(patching._declared),
        list(patching._attached),
        patching.load_manifest,
        patching._import_feature_modules,
        patching._import_framework_hooks,
    )
    patching._declared[:] = specs
    patching._attached[:] = []
    patching.load_manifest = lambda: manifest
    patching._import_feature_modules = lambda feature: None
    patching._import_framework_hooks = lambda: None
    try:
        patching.activate(Features(hisparse_mode="p2-offload"))
        yield
    finally:
        for target, original in originals.items():
            owner, name = target.rsplit(".", 1)
            setattr(pkgutil.resolve_name(owner), name, original)
        HookRegistry.reset()
        patching._activated = None
        patching._applied.clear()
        patching._frozen_hooks.clear()
        patching._frozen_depends.clear()
        patching._attached_live.clear()
        (
            patching._declared[:],
            patching._attached[:],
            patching.load_manifest,
            patching._import_feature_modules,
            patching._import_framework_hooks,
        ) = saved
