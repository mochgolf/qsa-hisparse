"""Activate rows U02-U04 (NUMA interleave) on the pinned SGLang, then undo.

Real ``model_compat`` activation needs every row of the feature, so the rows
activate against the manifest restricted to them (the pattern of
tests/model_compat/conftest.py). Patched functions, the bindings HookRegistry
propagated into other modules (the ported test imports the functions by
name) and the attached ``Envs`` field are restored on exit.
"""

import importlib
import pkgutil
import sys

import pytest

from sglang_qsa_hisparse import patching
from sglang_qsa_hisparse.features import Features

ROWS = frozenset({"U02", "U03", "U04"})


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


@pytest.fixture(scope="module")
def numa_rows():
    from sglang.srt.plugins.hook_registry import HookRegistry

    importlib.import_module("sglang_qsa_hisparse.patches.model_compat.numa")
    saved = (list(patching._declared), list(patching._attached), patching._activated)
    manifest = {
        row: entry for row, entry in patching.load_manifest().items() if row in ROWS
    }
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(patching, "load_manifest", lambda: manifest)
        mp.setattr(patching, "_import_feature_modules", lambda feature: None)
        mp.setattr(patching, "_import_framework_hooks", lambda: None)
        patching._declared[:] = [s for s in saved[0] if s.row in ROWS]
        patching._attached[:] = [s for s in saved[1] if s.row in ROWS]
        patching._activated = None
        HookRegistry.reset()
        originals = {
            spec.target: patching._raw_attribute(spec.target)
            for spec in patching._declared
        }
        attached = list(patching._attached)
        try:
            yield patching.activate(Features(model_compat=True))
        finally:
            _restore(originals)
            for spec in attached:
                owner = pkgutil.resolve_name(spec.owner)
                if spec.name in vars(owner):
                    delattr(owner, spec.name)
            HookRegistry.reset()
            patching._declared[:] = saved[0]
            patching._attached[:] = saved[1]
            patching._activated = saved[2]
            patching._applied.clear()
            patching._frozen_hooks.clear()
            patching._frozen_depends.clear()
            patching._attached_live.clear()
