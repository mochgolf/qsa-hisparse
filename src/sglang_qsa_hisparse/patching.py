"""Patch declarations and fail-closed activation through SGLang's HookRegistry.

Patch modules live in ``patches/model_compat/`` and ``patches/hisparse/`` and
are discovered automatically, so parallel work adds files instead of editing
a shared list. A module declares patches with ``@patch(...)`` and new members
with ``@attach(...)`` at import time; modules are imported only when their
feature is active. ``patches/framework.py`` is imported whenever any feature
is active.

Activation, in order:
1. every target and ``depends`` name must match its pinned module bytes, and
   the live object must be the fingerprinted definition;
2. no hook from another plugin may overlap a target (same path, an ancestor
   class, or a member of a class we replace);
3. hooks are applied and each target is checked to have been replaced;
4. members are attached to the final (possibly replaced) owners.
``verify_final`` repeats the overlap and identity checks once every plugin
has registered (from the framework hook on ``Scheduler.__init__``) and
writes an activation record when ``SGLANG_QSA_ACTIVATION_DIR`` is set.
"""

import importlib
import inspect
import json
import logging
import os
import pkgutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from sglang_qsa_hisparse import PINNED_SGLANG_COMMIT, fingerprint
from sglang_qsa_hisparse.errors import PluginActivationError
from sglang_qsa_hisparse.features import FEATURES, Features

logger = logging.getLogger(__name__)

HOOK_TYPES = ("before", "after", "around", "replace")
FRAMEWORK = "framework"
PLUGIN_NAME = "qsa_hisparse"
DIST_NAME = "sglang-qsa-hisparse"
ACTIVATION_DIR_ENV = "SGLANG_QSA_ACTIVATION_DIR"


@dataclass(frozen=True)
class PatchSpec:
    target: str
    hook_type: str
    hook: Callable
    feature: str
    depends: tuple[str, ...] = ()
    reason: str = ""
    # Names (target or depends) whose live object cannot be traced back to
    # source (e.g. custom-op objects); their module bytes are still pinned.
    unchecked_bindings: tuple[str, ...] = ()


@dataclass(frozen=True)
class AttachSpec:
    """A member the fork added to a pinned class or module.

    HookRegistry can only wrap existing attributes. Attaching requires the
    name to be absent from the owner (and its MRO): if upstream later defines
    the same name, activation fails instead of shadowing or replacing it.
    """

    owner: str
    name: str
    value: object
    feature: str
    depends: tuple[str, ...] = ()
    reason: str = ""

    @property
    def target(self) -> str:
        return f"{self.owner}.{self.name}"


_declared: list[PatchSpec] = []
_attached: list[AttachSpec] = []
_activated: Features | None = None
_applied: dict[str, object] = {}
_attached_live: list[AttachSpec] = []


def _check_feature(feature: str) -> None:
    if feature not in FEATURES + (FRAMEWORK,):
        raise ValueError(f"Unknown feature {feature!r}")


def patch(
    target: str,
    hook_type: str,
    *,
    feature: str,
    depends: tuple[str, ...] = (),
    reason: str = "",
    unchecked_bindings: tuple[str, ...] = (),
) -> Callable:
    """Declare a hook on a pinned SGLang definition (HookRegistry signatures)."""
    if hook_type not in HOOK_TYPES:
        raise ValueError(f"Unknown hook type {hook_type!r}")
    _check_feature(feature)

    def decorator(hook: Callable) -> Callable:
        _declared.append(
            PatchSpec(
                target,
                hook_type,
                hook,
                feature,
                tuple(depends),
                reason,
                tuple(unchecked_bindings),
            )
        )
        return hook

    return decorator


def attach_value(
    owner: str,
    name: str,
    value: object,
    *,
    feature: str,
    depends: tuple[str, ...] = (),
    reason: str = "",
) -> object:
    """Declare a new member ``owner.name = value`` absent at the pin."""
    _check_feature(feature)
    _attached.append(AttachSpec(owner, name, value, feature, tuple(depends), reason))
    return value


def attach(
    owner: str,
    name: str | None = None,
    *,
    feature: str,
    depends: tuple[str, ...] = (),
    reason: str = "",
) -> Callable:
    """Decorator form of ``attach_value``; the name defaults to ``__name__``."""

    def decorator(value: Callable) -> Callable:
        member = name or getattr(value, "__name__", None)
        if not member:
            raise ValueError(f"attach on {owner} needs an explicit name")
        attach_value(
            owner, member, value, feature=feature, depends=depends, reason=reason
        )
        return value

    return decorator


def _import_feature_modules(feature: str) -> None:
    package = importlib.import_module(f"sglang_qsa_hisparse.patches.{feature}")
    for info in sorted(pkgutil.iter_modules(package.__path__), key=lambda i: i.name):
        importlib.import_module(f"{package.__name__}.{info.name}")


def _import_framework_hooks() -> None:
    importlib.import_module("sglang_qsa_hisparse.patches.framework")


def _overlaps(a: str, b: str) -> bool:
    return a == b or a.startswith(b + ".") or b.startswith(a + ".")


def collect(features: Features) -> tuple[list[PatchSpec], list[AttachSpec]]:
    _import_framework_hooks()
    for feature in features.active:
        _import_feature_modules(feature)
    wanted = set(features.active) | {FRAMEWORK}
    specs = [spec for spec in _declared if spec.feature in wanted]
    attaches = [spec for spec in _attached if spec.feature in wanted]

    replaced: set[str] = set()
    for spec in specs:
        if spec.hook_type != "replace":
            continue
        if spec.target in replaced:
            raise PluginActivationError(f"Two REPLACE patches on {spec.target}")
        replaced.add(spec.target)
    class_replaced = {s.target for s in specs if s.hook_type == "replace" and isinstance(s.hook, type)}
    nested = sorted(
        s.target
        for s in specs
        for owner in class_replaced
        if s.target.startswith(owner + ".")
    )
    if nested:
        raise PluginActivationError(
            f"Hooks on members of a replaced class (implement them in the "
            f"replacement class instead): {nested}"
        )
    attached = [spec.target for spec in attaches]
    duplicates = sorted({t for t in attached if attached.count(t) > 1})
    if duplicates:
        raise PluginActivationError(f"Members attached twice: {duplicates}")
    hooked = sorted(set(attached) & {spec.target for spec in specs})
    if hooked:
        raise PluginActivationError(f"Hooks on attached members: {hooked}")
    return specs, attaches


def _split(target: str) -> tuple[object, str]:
    owner_path, name = target.rsplit(".", 1)
    return pkgutil.resolve_name(owner_path), name


def _raw_attribute(target: str):
    owner, name = _split(target)
    if isinstance(owner, type) and name in owner.__dict__:
        return owner.__dict__[name]
    return getattr(owner, name)


def _source_object(value: object) -> object:
    if isinstance(value, (staticmethod, classmethod)):
        value = value.__func__
    elif isinstance(value, property):
        value = value.fget
    return inspect.unwrap(value)


def _check_binding(name: str, record: dict) -> str | None:
    """Return a problem if the live object is not the fingerprinted source."""
    owner, member = _split(name)
    if isinstance(owner, type) and member not in owner.__dict__:
        return f"{name}: not defined directly on {owner.__qualname__}"
    live = _source_object(_raw_attribute(name))
    path = Path(record["path"]).resolve()
    if record["kind"] == "class":
        try:
            source = Path(inspect.getsourcefile(live)).resolve()
        except TypeError:
            return f"{name}: live object is not a class"
        if live.__qualname__ != record["qualname"] or source != path:
            return f"{name}: live class is not the pinned definition"
        return None
    code = getattr(live, "__code__", None)
    if (
        code is None
        or Path(code.co_filename).resolve() != path
        or code.co_firstlineno not in (record["first_line"], record["def_line"])
    ):
        return f"{name}: live function is not the pinned definition"
    return None


def _foreign_overlaps(registry, targets: set[str]) -> list[str]:
    found = []
    for key, hooks in registry._hooks.items():
        foreign = [h for h in hooks if h[2] is None or h[2].plugin_name != PLUGIN_NAME]
        if foreign and any(_overlaps(key, target) for target in targets):
            found.append(key)
    return sorted(found)


def _defines(owner: object, name: str) -> bool:
    if isinstance(owner, type):
        return any(name in klass.__dict__ for klass in owner.__mro__)
    return hasattr(owner, name)


def _attach_all(attaches: list[AttachSpec]) -> None:
    owners = {spec.owner: pkgutil.resolve_name(spec.owner) for spec in attaches}
    present = [spec.target for spec in attaches if _defines(owners[spec.owner], spec.name)]
    if present:
        raise PluginActivationError(
            f"Attach targets already defined upstream (pin drift): {present}"
        )
    for spec in attaches:
        setattr(owners[spec.owner], spec.name, spec.value)
    _verify_attached(attaches)


def _verify_attached(attaches: list[AttachSpec]) -> None:
    missing = [
        spec.target for spec in attaches if _raw_attribute(spec.target) is not spec.value
    ]
    if missing:
        raise PluginActivationError(f"Members were not attached: {missing}")


def activate(features: Features) -> list[PatchSpec]:
    """Apply all patches for ``features`` exactly once per process."""
    global _activated
    if _activated is not None:
        if _activated != features:
            raise PluginActivationError(
                f"Already activated with {_activated}, requested {features}"
            )
        return [s for s in _declared if s.feature in set(features.active) | {FRAMEWORK}]

    specs, attaches = collect(features)
    names = {s.target for s in specs}
    names |= {d for s in specs for d in s.depends}
    names |= {d for s in attaches for d in s.depends}
    records = fingerprint.verify(sorted(names))
    unchecked = {n for s in specs for n in s.unchecked_bindings}
    problems = [
        problem
        for name in sorted(names - unchecked)
        if (problem := _check_binding(name, records[name])) is not None
    ]
    if problems:
        raise PluginActivationError("; ".join(problems))

    from sglang.srt.plugins.hook_registry import HookRegistry, HookSource, HookType

    targets = {spec.target for spec in specs}
    overlap = _foreign_overlaps(HookRegistry, targets | {a.target for a in attaches})
    if overlap:
        raise PluginActivationError(f"Other plugins hook overlapping targets: {overlap}")
    originals = {target: _raw_attribute(target) for target in targets}
    source = HookSource(plugin_name=PLUGIN_NAME, dist_name=DIST_NAME)
    for spec in specs:
        HookRegistry.register(
            spec.target, spec.hook, HookType(spec.hook_type), source=source
        )
    # HookRegistry logs and skips failures; verify every target instead.
    HookRegistry.apply_hooks()
    failed = sorted(
        target
        for target in targets
        if target not in HookRegistry._patched
        or _raw_attribute(target) is originals[target]
    )
    if failed:
        raise PluginActivationError(f"Hooks were not applied to: {failed}")
    _applied.update({target: _raw_attribute(target) for target in targets})
    # Attach last, so owners resolve to any class we replaced.
    _attach_all(attaches)
    _attached_live[:] = attaches
    _activated = features
    logger.info(
        "QSA HiSparse plugin activated %s with %d patches and %d attached members",
        features.active,
        len(specs),
        len(attaches),
    )
    return specs


def verify_final(role: str, **details: object) -> None:
    """Re-check activation after every plugin registered; record the result."""
    if _activated is None:
        raise PluginActivationError("verify_final called before activation")
    from sglang.srt.plugins.hook_registry import HookRegistry

    targets = set(_applied) | {a.target for a in _attached_live}
    overlap = _foreign_overlaps(HookRegistry, targets)
    if overlap:
        raise PluginActivationError(
            f"Other plugins registered overlapping hooks after activation: {overlap}"
        )
    changed = sorted(t for t, live in _applied.items() if _raw_attribute(t) is not live)
    if changed:
        raise PluginActivationError(f"Patched targets were replaced afterwards: {changed}")
    _verify_attached(_attached_live)

    directory = os.environ.get(ACTIVATION_DIR_ENV)
    if directory:
        record = {
            "pid": os.getpid(),
            "role": role,
            "features": list(_activated.active),
            "hisparse_mode": _activated.hisparse_mode,
            "patches": sorted(_applied),
            "attached": sorted(a.target for a in _attached_live),
            "pinned_sglang_commit": PINNED_SGLANG_COMMIT,
            **details,
        }
        path = Path(directory) / f"{role}-{os.getpid()}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
