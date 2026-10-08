"""Subprocess probe for ``test_scope_delegation.py`` (not collected by pytest).

    python scope_delegation_probe.py <model_dir> <ROW,ROW,...> [--selftest]

Publishes ``<model_dir>`` as the served model, activates the requested
features through ``plugin.load()``, and checks every declared model_compat
hook of the given rows against the pinned original with a sentinel call
(``object()`` for each required positional parameter of the original):

- before / around / replace: the patched attribute and the original give the
  same outcome (repr of the returned value, or exception type, message and
  innermost frame), i.e. the hook handed the call to the original untouched;
- after: the hook returns None or the original's result unchanged.

``--selftest`` replaces the manifest with stand-in model_compat rows on real
pinned targets (delegating hooks of every type, a REPLACE that captured the
original, a custom-op target, and a REPLACE and an AFTER hook that ignore the
scope) to test the probe itself.

Prints one line: ``QSA_SCOPE_PROBE {"active": ..., "checked": [...],
"failures": {target: reason}}``.
"""

import inspect
import json
import pkgutil
import sys
import traceback

MARKER = "QSA_SCOPE_PROBE "
SELFTEST = {
    "ST1": ("sglang.srt.mem_cache.memory_pool.ReqToTokenPool.clear", "around"),
    "ST2": (
        "sglang.srt.layers.quantization.gptq.schemes.gptq_moe.GPTQMarlinMoEScheme.create_weights",
        "replace",
    ),
    "ST3": ("sglang.srt.layers.moe.fused_moe_triton.layer.FusedMoE.weight_loader", "before"),
    "ST4": (
        "sglang.srt.layers.quantization.auto_round.AutoRoundConfig.apply_gptq_quant_layer",
        "replace",
    ),
    "ST5": ("sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe.fused_marlin_moe", "around"),
    "ST6": ("sglang.srt.layers.hyperconnection.GatedResidual.mix", "after"),
    "ST7": ("sglang.kernels.ops.gemm.hc_mix.fused_hc_mix_supported", "after"),
}
# Stand-ins that must be reported (they ignore the scope).
SELFTEST_NON_DELEGATING = ("ST2", "ST7")


def sentinel_args(original) -> tuple:
    try:
        parameters = inspect.signature(original).parameters.values()
    except (TypeError, ValueError):  # No Python signature (custom-op packets).
        return ()
    return tuple(
        object()
        for p in parameters
        if p.default is p.empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    )


def outcome(function, args) -> list:
    try:
        value = function(*args)
    except Exception as error:
        frame = traceback.extract_tb(error.__traceback__)[-1]
        return ["raised", type(error).__name__, str(error), frame.filename, frame.lineno]
    return ["returned", repr(value)]


def check(spec, live) -> str | None:
    original = live.__wrapped__
    args = sentinel_args(original)
    if spec.hook_type == "after":
        result = object()
        returned = spec.hook(result, *args)
        if returned is None or returned is result:
            return None
        return f"after hook returned {returned!r} instead of the original result"
    expected, actual = outcome(original, args), outcome(live, args)
    return None if expected == actual else f"original {expected}, patched {actual}"


def install_selftest(patching) -> None:
    from sglang_qsa_hisparse import fingerprint
    from sglang_qsa_hisparse.scope import target_model_active

    manifest = {
        row: entry
        for row, entry in patching.load_manifest().items()
        if entry["feature"] == "framework"
    }
    for row, (target, hook_type) in SELFTEST.items():
        manifest[row] = {
            "feature": "model_compat",
            "attach": [],
            "patches": [{"target": target, "hook_type": hook_type}],
        }
    patching.load_manifest = lambda: manifest
    pinned = fingerprint.load_pinned()
    root = fingerprint.installed_source_root()
    pinned.update({target: fingerprint.record(root, target) for target, _ in SELFTEST.values()})
    fingerprint.load_pinned = lambda: pinned

    owner, name = SELFTEST["ST4"][0].rsplit(".", 1)
    captured = pkgutil.resolve_name(owner).__dict__[name]

    def delegating_around(original, *args, **kwargs):
        if not target_model_active():
            return original(*args, **kwargs)
        return "fork"

    def non_delegating_replace(*args, **kwargs):
        return "fork"

    def delegating_before(*args, **kwargs):
        if target_model_active():
            return args, kwargs
        return None

    def delegating_replace(*args, **kwargs):
        if not target_model_active():
            return captured(*args, **kwargs)
        return "fork"

    def delegating_after(result, *args, **kwargs):
        return result if not target_model_active() else "fork"

    def non_delegating_after(result, *args, **kwargs):
        return "fork"

    hooks = {
        "ST1": delegating_around,
        "ST2": non_delegating_replace,
        "ST3": delegating_before,
        "ST4": delegating_replace,
        "ST5": delegating_around,
        "ST6": delegating_after,
        "ST7": non_delegating_after,
    }

    def import_features(feature):
        for row, (target, hook_type) in SELFTEST.items():
            patching.patch(target, hook_type, feature="model_compat", row=row)(hooks[row])

    patching._import_feature_modules = import_features


def main() -> None:
    model_dir, rows = sys.argv[1], set(sys.argv[2].split(","))
    from sglang.srt.runtime_context import get_context

    get_context().override_server_args(model_path=model_dir).install()
    from sglang_qsa_hisparse import patching, plugin, scope

    if "--selftest" in sys.argv:
        install_selftest(patching)
    plugin.load()
    result = {"active": scope.target_model_active(), "checked": [], "failures": {}}
    for spec in patching._declared:
        if spec.feature != "model_compat" or spec.row not in rows:
            continue
        result["checked"].append(spec.target)
        try:
            problem = check(spec, patching._raw_attribute(spec.target))
        except Exception as error:
            problem = f"check failed: {error!r}"
        if problem:
            result["failures"][spec.target] = problem
    print(MARKER + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
