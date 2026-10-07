"""W5 fixtures: activate W5's model_compat rows on the pinned SGLang, then undo.

Until every model_compat row is implemented, real activation fails by design,
so ``compat`` activates against the manifest restricted to W5's rows (the
pattern of tests/test_framework.py) and restores every patched attribute,
including the bindings HookRegistry propagated into other modules.
"""

import importlib
import pkgutil
import sys
from types import SimpleNamespace

import pytest

from sglang_qsa_hisparse import patching, scope
from sglang_qsa_hisparse.features import Features

W5_ROWS = frozenset(
    {"J03", "Z01", "Z02", "Z03", "Z04", "H05", "H06", "H08", "E03", "E06", "E07", "E08"}
)
PATCH_MODULES = tuple(
    f"sglang_qsa_hisparse.patches.model_compat.{name}"
    for name in ("hyperconnection", "marlin", "quantization", "qwen4_exp")
)


def _restore(originals):
    for target, original in originals.items():
        live = patching._raw_attribute(target)
        owner, name = target.rsplit(".", 1)
        setattr(pkgutil.resolve_name(owner), name, original)
        for module in list(sys.modules.values()):
            try:
                bindings = vars(module)
            except TypeError:
                continue
            for attr, value in list(bindings.items()):
                if value is live:
                    setattr(module, attr, original)
        assert patching._raw_attribute(target) is original, target


@pytest.fixture(scope="module")
def compat():
    """``compat.specs`` activated; ``compat.originals``: target -> pinned object."""
    from sglang.srt.plugins.hook_registry import HookRegistry

    for name in PATCH_MODULES:
        importlib.import_module(name)
    saved = (
        list(patching._declared),
        list(patching._attached),
        patching._activated,
    )
    manifest = {
        row: entry for row, entry in patching.load_manifest().items() if row in W5_ROWS
    }
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(patching, "load_manifest", lambda: manifest)
        mp.setattr(patching, "_import_feature_modules", lambda feature: None)
        mp.setattr(patching, "_import_framework_hooks", lambda: None)
        patching._declared[:] = [s for s in saved[0] if s.row in W5_ROWS]
        patching._attached[:] = []
        patching._activated = None
        HookRegistry.reset()
        originals = {
            spec.target: patching._raw_attribute(spec.target)
            for spec in patching._declared
        }
        try:
            specs = patching.activate(Features(model_compat=True))
            yield SimpleNamespace(specs=specs, originals=originals)
        finally:
            _restore(originals)
            HookRegistry.reset()
            patching._declared[:] = saved[0]
            patching._attached[:] = saved[1]
            patching._activated = saved[2]
            patching._applied.clear()
            patching._frozen_hooks.clear()
            patching._frozen_depends.clear()
            patching._attached_live.clear()


@pytest.fixture
def target(monkeypatch):
    """Serve the target model (rule 9)."""
    monkeypatch.setattr(scope, "target_model_active", lambda: True)
