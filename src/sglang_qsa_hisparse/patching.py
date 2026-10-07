"""Patch declarations and fail-closed activation through SGLang's HookRegistry.

Patch modules live in ``patches/model_compat/`` and ``patches/hisparse/`` and
are discovered automatically, so parallel work adds files instead of editing
a shared list. A module declares hooks with ``@patch(...)`` and new members
with ``@attach(...)`` at import time, each tagged with its inventory row;
modules are imported only when their feature is active.
``patches/framework.py`` is imported whenever any feature is active.

Activation, in order:
1. the declarations of every requested feature (and the framework) must
   equal ``manifest.json`` exactly (rows, targets, hook types, members);
2. every target and ``depends`` name must match its pinned module bytes and
   pinned live binding chain; properties cannot be hook targets;
3. no hook from another plugin may overlap a target (same path, an ancestor
   class, or a member of a replaced class);
4. only this plugin's targets are applied (other plugins' hooks are left for
   SGLang's loader), and each is checked to have been replaced;
5. members are attached to the final (possibly replaced) owners.
``verify_final`` runs from the framework hook on
``configure_scheduler_process`` in every scheduler/TP process, after
``load_plugins()`` returned: it requires the registry entries of every
patched target to be exactly the frozen set, no late hooks from this plugin
or overlapping hooks from others, unchanged patched attributes and
attachments, and writes an activation record when
``SGLANG_QSA_ACTIVATION_DIR`` is set.
"""

import importlib
import json
import logging
import os
import pkgutil
from collections.abc import Callable
from dataclasses import dataclass
from importlib import resources
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
    row: str
    depends: tuple[str, ...] = ()
    reason: str = ""
    # Names (target or depends) whose live object cannot be described by a
    # binding chain; their module bytes are still pinned.
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
    row: str
    depends: tuple[str, ...] = ()
    reason: str = ""

    @property
    def target(self) -> str:
        return f"{self.owner}.{self.name}"


_declared: list[PatchSpec] = []
_attached: list[AttachSpec] = []
_activated: Features | None = None
_applied: dict[str, object] = {}
_frozen_hooks: dict[str, tuple] = {}
_attached_live: list[AttachSpec] = []


def _check_feature(feature: str) -> None:
    if feature not in FEATURES + (FRAMEWORK,):
        raise ValueError(f"Unknown feature {feature!r}")


def patch(
    target: str,
    hook_type: str,
    *,
    feature: str,
    row: str,
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
                row,
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
    row: str,
    depends: tuple[str, ...] = (),
    reason: str = "",
) -> object:
    """Declare a new member ``owner.name = value`` absent at the pin."""
    _check_feature(feature)
    _attached.append(
        AttachSpec(owner, name, value, feature, row, tuple(depends), reason)
    )
    return value


def attach(
    owner: str,
    name: str | None = None,
    *,
    feature: str,
    row: str,
    depends: tuple[str, ...] = (),
    reason: str = "",
) -> Callable:
    """Decorator form of ``attach_value``; the name defaults to ``__name__``."""

    def decorator(value: Callable) -> Callable:
        member = name or getattr(value, "__name__", None)
        if not member:
            raise ValueError(f"attach on {owner} needs an explicit name")
        attach_value(
            owner, member, value, feature=feature, row=row, depends=depends, reason=reason
        )
        return value

    return decorator


def load_manifest() -> dict[str, dict]:
    path = resources.files("sglang_qsa_hisparse").joinpath("manifest.json")
    return json.loads(path.read_text(encoding="utf-8"))["rows"]


def _import_feature_modules(feature: str) -> None:
    package = importlib.import_module(f"sglang_qsa_hisparse.patches.{feature}")
    for info in sorted(pkgutil.iter_modules(package.__path__), key=lambda i: i.name):
        importlib.import_module(f"{package.__name__}.{info.name}")


def _import_framework_hooks() -> None:
    importlib.import_module("sglang_qsa_hisparse.patches.framework")


def _overlaps(a: str, b: str) -> bool:
    return a == b or a.startswith(b + ".") or b.startswith(a + ".")


def _check_manifest(
    wanted: set[str], specs: list[PatchSpec], attaches: list[AttachSpec]
) -> None:
    manifest = load_manifest()
    for feature in sorted(wanted - {FRAMEWORK}):
        if not any(entry["feature"] == feature for entry in manifest.values()):
            raise PluginActivationError(f"Feature {feature} has no manifest rows")
    expected = set()
    for row, entry in manifest.items():
        if entry["feature"] not in wanted:
            continue
        expected |= {("patch", row, p["target"], p["hook_type"]) for p in entry["patches"]}
        expected |= {("attach", row, f"{a['owner']}.{a['name']}", "") for a in entry["attach"]}
    declared = {("patch", s.row, s.target, s.hook_type) for s in specs}
    declared |= {("attach", a.row, a.target, "") for a in attaches}
    missing, extra = sorted(expected - declared), sorted(declared - expected)
    if missing or extra:
        raise PluginActivationError(
            f"Declarations differ from manifest.json; missing {missing}; "
            f"undeclared in manifest {extra}"
        )


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
    class_replaced = {
        s.target for s in specs if s.hook_type == "replace" and isinstance(s.hook, type)
    }
    nested = sorted(
        s.target for s in specs for owner in class_replaced if s.target.startswith(owner + ".")
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
    _check_manifest(wanted, specs, attaches)
    return specs, attaches


def _raw_attribute(target: str):
    return fingerprint.raw_attribute(target)


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
        wanted = set(features.active) | {FRAMEWORK}
        return [s for s in _declared if s.feature in wanted]

    specs, attaches = collect(features)
    names = {s.target for s in specs}
    names |= {d for s in specs for d in s.depends}
    names |= {d for s in attaches for d in s.depends}
    unchecked = {n for s in specs for n in s.unchecked_bindings}
    fingerprint.verify(sorted(names - unchecked))
    fingerprint.verify(sorted(names & unchecked), check_bindings=False)
    properties = sorted(s.target for s in specs if isinstance(_raw_attribute(s.target), property))
    if properties:
        raise PluginActivationError(f"Properties cannot be hook targets: {properties}")

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
    # Apply only this plugin's targets (class REPLACE first, as the registry
    # does); other plugins' entries are left for load_plugins()'s final pass.
    # The registry's own apply loop logs and skips failures; this does not.
    ours = sorted(
        ((t, HookRegistry._hooks[t]) for t in targets), key=HookRegistry._target_sort_key
    )
    for target, hooks in ours:
        HookRegistry._apply_target(target, hooks)
        HookRegistry._patched.add(target)
    failed = sorted(t for t in targets if _raw_attribute(t) is originals[t])
    if failed:
        raise PluginActivationError(f"Hooks were not applied to: {failed}")
    _applied.update({target: _raw_attribute(target) for target in targets})
    _frozen_hooks.update({t: tuple(HookRegistry._hooks[t]) for t in targets})
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
    late = sorted(
        key
        for key, hooks in HookRegistry._hooks.items()
        if tuple(hooks) != _frozen_hooks.get(key, tuple(hooks))
        or (
            key not in _frozen_hooks
            and any(h[2] is not None and h[2].plugin_name == PLUGIN_NAME for h in hooks)
        )
    )
    if late:
        raise PluginActivationError(f"Hooks registered after activation: {late}")
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
