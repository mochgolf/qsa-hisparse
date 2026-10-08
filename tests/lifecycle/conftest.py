"""Activate only W2's inventory rows in-process, then restore pinned SGLang.

Real activation of a feature fails until every workstream's rows exist, so
these tests activate both features against ``manifest.json`` restricted to
W2's rows (README of docs/tasks). Teardown puts every patched attribute and
every propagated module binding back, so later tests see pinned SGLang.
"""

import contextlib
import importlib
import pkgutil
from types import SimpleNamespace

import pytest

from sglang_qsa_hisparse import patching
from sglang_qsa_hisparse.features import Features

W2_ROWS = frozenset(
    "S01 S02 S03 S04 S05 S06 S07 S08 S09 B02 B03 B04 B05 B06 "
    "P01 P02 M01 M02 M03 M04 M05 M06".split()
)
W2_MODULES = (
    "hisparse.scheduler",
    "hisparse.lifecycle",
    "model_compat.scheduler",
    "model_compat.lifecycle",
)
BOTH = Features(model_compat=True, hisparse_mode="p2-offload")


def w2_manifest() -> dict:
    return {row: e for row, e in patching.load_manifest().items() if row in W2_ROWS}


def w2_targets() -> list[str]:
    return sorted({p["target"] for e in w2_manifest().values() for p in e["patches"]})


def import_w2_modules() -> None:
    for name in W2_MODULES:
        importlib.import_module(f"sglang_qsa_hisparse.patches.{name}")


@contextlib.contextmanager
def w2_active():
    from sglang.srt.plugins.hook_registry import HookRegistry, _propagate_patch

    import_w2_modules()
    manifest = w2_manifest()
    originals = {t: patching._raw_attribute(t) for t in w2_targets()}
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(patching, "load_manifest", lambda: manifest)
        mp.setattr(patching, "_import_feature_modules", lambda feature: None)
        mp.setattr(patching, "_import_framework_hooks", lambda: None)
        mp.setattr(
            patching, "_declared", [s for s in patching._declared if s.row in W2_ROWS]
        )
        mp.setattr(
            patching, "_attached", [a for a in patching._attached if a.row in W2_ROWS]
        )
        try:
            patching.activate(BOTH)
            yield
        finally:
            for spec in patching._attached_live:
                owner = pkgutil.resolve_name(spec.owner)
                if owner.__dict__.get(spec.name) is spec.value:
                    delattr(owner, spec.name)
            for target, original in originals.items():
                owner_path, name = target.rsplit(".", 1)
                owner = pkgutil.resolve_name(owner_path)
                wrapped = patching._raw_attribute(target)
                setattr(owner, name, original)
                if wrapped is not original:
                    _propagate_patch(wrapped, original, owner)
            HookRegistry.reset()
            patching._activated = None
            patching._applied.clear()
            patching._frozen_hooks.clear()
            patching._frozen_depends.clear()
            patching._attached_live.clear()


@pytest.fixture(scope="module")
def w2_patches():
    with w2_active():
        yield


@pytest.fixture
def w2():
    import_w2_modules()
    return SimpleNamespace(rows=W2_ROWS, targets=w2_targets(), active=w2_active)


@pytest.fixture
def published_context():
    """A published SGLang config context, as upstream unit tests install it."""
    from sglang.srt.runtime_context import get_context

    with get_context().override_server_args(attention_backend="torch_native", dcp_size=1):
        yield


@pytest.fixture
def target_model(monkeypatch):
    """Answer the rule-9 scope question for scoped model_compat hooks."""
    from sglang_qsa_hisparse import scope

    def set_active(active: bool) -> None:
        monkeypatch.setattr(scope, "target_model_active", lambda: active)

    set_active(True)
    return set_active
