"""Run tests/qsa with the W4 rows activated on the pinned SGLang.

Activation uses the real framework (manifest check, fingerprints, registry,
attach) against ``manifest.json`` restricted to this workstream's rows, with
both features on, as in the fork. Each test module activates once and the
patches are removed afterwards, so other test directories see pristine SGLang.
Target-model scope is in effect by default; tests switch it off explicitly.
"""

import copy
import pkgutil

import pytest

from sglang_qsa_hisparse import patching, scope
from sglang_qsa_hisparse.features import Features

W4_ROWS = frozenset(
    {f"Q{i:02d}" for i in range(1, 14)}
    | {"T01", "T02", "T03", "T04", "T05"}
    | {f"A{i:02d}" for i in range(1, 13)}
)
BOTH = Features(model_compat=True, hisparse_mode="p2-offload")
BACKEND_MODULE = "sglang.srt.layers.attention.qwen_sparse_attn_backend"
# Declared by this workstream but not yet in manifest.json (inventory section 6,
# G5); the orchestrator adds it to the inventory and manifest at merge.
PENDING_MANIFEST_ATTACH: dict = {}  # Q01's cache_clear attach is in manifest.json now.

_load_real_manifest = patching.load_manifest


def w4_manifest():
    rows = {
        row: copy.deepcopy(entry)
        for row, entry in _load_real_manifest().items()
        if row in W4_ROWS
    }
    for row, attaches in PENDING_MANIFEST_ATTACH.items():
        rows[row]["attach"].extend(attaches)
    return rows


def w4_declarations():
    import sglang_qsa_hisparse.patches.hisparse.qsa_backend  # noqa: F401
    import sglang_qsa_hisparse.patches.model_compat.qsa_attention  # noqa: F401

    return (
        [s for s in patching._declared if s.row in W4_ROWS],
        [a for a in patching._attached if a.row in W4_ROWS],
    )


def _reset_activation_state():
    from sglang.srt.plugins.hook_registry import HookRegistry

    HookRegistry.reset()
    patching._activated = None
    patching._applied.clear()
    patching._frozen_hooks.clear()
    patching._frozen_depends.clear()
    patching._attached_live.clear()


def _deactivate(originals):
    from sglang.srt.plugins.hook_registry import _propagate_patch

    for spec in patching._attached_live:
        delattr(pkgutil.resolve_name(spec.owner), spec.name)
    for target, original in originals.items():
        owner_path, name = target.rsplit(".", 1)
        owner = pkgutil.resolve_name(owner_path)
        live = patching._raw_attribute(target)
        setattr(owner, name, original)
        if live is not original:
            _propagate_patch(live, original, owner)


@pytest.fixture(scope="module", autouse=True)
def qsa_patches():
    specs, attaches = w4_declarations()
    saved = list(patching._declared), list(patching._attached)
    originals = {s.target: patching._raw_attribute(s.target) for s in specs}
    _reset_activation_state()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(patching, "load_manifest", w4_manifest)
        mp.setattr(patching, "_import_feature_modules", lambda feature: None)
        mp.setattr(patching, "_import_framework_hooks", lambda: None)
        patching._declared[:], patching._attached[:] = specs, attaches
        try:
            patching.activate(BOTH)
            yield
        finally:
            _deactivate(originals)
            _reset_activation_state()
            patching._declared[:], patching._attached[:] = saved
    assert all(patching._raw_attribute(t) is o for t, o in originals.items())
    assert all(
        getattr(pkgutil.resolve_name(a.owner), a.name, None) is not a.value
        for a in attaches
    )


@pytest.fixture(autouse=True)
def target_model(monkeypatch):
    """Serve the target model unless a test switches scope off."""
    monkeypatch.setattr(scope, "target_model_active", lambda: True)

    def set_active(active: bool):
        monkeypatch.setattr(scope, "target_model_active", lambda: active)

    return set_active
