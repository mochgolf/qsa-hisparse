"""Patch declarations and fail-closed activation through SGLang's HookRegistry.

Patch modules live in ``patches/model_compat/`` and ``patches/hisparse/`` and
are discovered automatically, so parallel work adds files instead of editing
a shared list. A module declares patches with ``@patch(...)`` at import time;
modules are imported only when their feature is active.

Every target, and every definition named in ``depends`` (code a hook copies or
whose behavior it assumes), must have a pinned fingerprint. Activation
verifies all fingerprints before registering anything, then applies the hooks
and checks that each target was actually replaced.
"""

import importlib
import logging
import pkgutil
from collections.abc import Callable
from dataclasses import dataclass

from sglang_qsa_hisparse import fingerprint
from sglang_qsa_hisparse.errors import PluginActivationError
from sglang_qsa_hisparse.features import FEATURES, Features

logger = logging.getLogger(__name__)

HOOK_TYPES = ("before", "after", "around", "replace")
PLUGIN_NAME = "qsa_hisparse"
DIST_NAME = "sglang-qsa-hisparse"


@dataclass(frozen=True)
class PatchSpec:
    target: str
    hook_type: str
    hook: Callable
    feature: str
    depends: tuple[str, ...] = ()
    reason: str = ""


_declared: list[PatchSpec] = []
_activated: Features | None = None


def patch(
    target: str,
    hook_type: str,
    *,
    feature: str,
    depends: tuple[str, ...] = (),
    reason: str = "",
) -> Callable:
    """Declare a hook on a pinned SGLang definition (HookRegistry signatures)."""
    if hook_type not in HOOK_TYPES:
        raise ValueError(f"Unknown hook type {hook_type!r}")
    if feature not in FEATURES:
        raise ValueError(f"Unknown feature {feature!r}")

    def decorator(hook: Callable) -> Callable:
        _declared.append(
            PatchSpec(target, hook_type, hook, feature, tuple(depends), reason)
        )
        return hook

    return decorator


def _import_feature_modules(feature: str) -> None:
    package = importlib.import_module(f"sglang_qsa_hisparse.patches.{feature}")
    for info in sorted(pkgutil.iter_modules(package.__path__), key=lambda i: i.name):
        importlib.import_module(f"{package.__name__}.{info.name}")


def collect(features: Features) -> list[PatchSpec]:
    for feature in features.active:
        _import_feature_modules(feature)
    specs = [spec for spec in _declared if spec.feature in features.active]
    replaced: dict[str, PatchSpec] = {}
    for spec in specs:
        if spec.hook_type != "replace":
            continue
        if spec.target in replaced:
            raise PluginActivationError(f"Two REPLACE patches on {spec.target}")
        replaced[spec.target] = spec
    return specs


def _raw_attribute(target: str):
    owner_path, name = target.rsplit(".", 1)
    owner = pkgutil.resolve_name(owner_path)
    if isinstance(owner, type) and name in owner.__dict__:
        return owner.__dict__[name]
    return getattr(owner, name)


def activate(features: Features) -> list[PatchSpec]:
    """Apply all patches for ``features`` exactly once per process."""
    global _activated
    if _activated is not None:
        if _activated != features:
            raise PluginActivationError(
                f"Already activated with {_activated}, requested {features}"
            )
        return [spec for spec in _declared if spec.feature in features.active]

    specs = collect(features)
    names = sorted({s.target for s in specs} | {d for s in specs for d in s.depends})
    fingerprint.verify(names)

    from sglang.srt.plugins.hook_registry import HookRegistry, HookSource, HookType

    targets = sorted({spec.target for spec in specs})
    foreign = [t for t in targets if t in HookRegistry._patched]
    if foreign:
        raise PluginActivationError(f"Targets already patched elsewhere: {foreign}")
    originals = {target: _raw_attribute(target) for target in targets}
    source = HookSource(plugin_name=PLUGIN_NAME, dist_name=DIST_NAME)
    for spec in specs:
        HookRegistry.register(
            spec.target, spec.hook, HookType(spec.hook_type), source=source
        )
    # HookRegistry logs and skips failures; verify every target instead.
    HookRegistry.apply_hooks()
    failed = [
        target
        for target in targets
        if target not in HookRegistry._patched
        or _raw_attribute(target) is originals[target]
    ]
    if failed:
        raise PluginActivationError(f"Hooks were not applied to: {failed}")
    _activated = features
    logger.info(
        "QSA HiSparse plugin activated %s with %d patches", features.active, len(specs)
    )
    return specs
