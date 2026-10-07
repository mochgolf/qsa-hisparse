"""Plugin framework contracts: no-op when off, fail-closed when on."""

import functools
import json
import os
import subprocess
import sys
import textwrap

import pytest

from sglang_qsa_hisparse import fingerprint, patching
from sglang_qsa_hisparse.errors import (
    FingerprintMismatch,
    PluginActivationError,
    PluginConfigError,
)
from sglang_qsa_hisparse.features import Features, read_features

MODULE = textwrap.dedent(
    """
    import functools


    def double(x):
        return 2 * x


    def configure():
        return "configured"


    @functools.lru_cache(maxsize=None)
    def cached(x):
        return x


    class Box:
        def value(self):
            return 1

        @property
        def size(self):
            return 3


    class Child(Box):
        pass
    """
)
COMPAT = Features(model_compat=True)
BOTH = Features(model_compat=True, hisparse_mode="p2-offload")


@pytest.fixture
def registry():
    from sglang.srt.plugins.hook_registry import HookRegistry

    saved = list(patching._declared), list(patching._attached)

    def clear():
        HookRegistry.reset()
        patching._declared.clear()
        patching._attached.clear()
        patching._activated = None
        patching._applied.clear()
        patching._frozen_hooks.clear()
        patching._attached_live.clear()

    clear()
    yield HookRegistry
    clear()
    patching._declared[:], patching._attached[:] = saved


@pytest.fixture
def fake(tmp_path, monkeypatch):
    """A throwaway module tree standing in for the installed SGLang sources.

    The manifest is derived from whatever the test declares unless the test
    sets ``fake.manifest`` explicitly.
    """
    package = tmp_path / "qsa_fake"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "mod.py").write_text(MODULE)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(fingerprint, "installed_source_root", lambda: tmp_path)
    monkeypatch.setattr(patching, "_import_feature_modules", lambda feature: None)
    monkeypatch.setattr(patching, "_import_framework_hooks", lambda: None)
    for name in [m for m in sys.modules if m.startswith("qsa_fake")]:
        monkeypatch.delitem(sys.modules, name)

    class Fake:
        root = tmp_path
        path = package / "mod.py"
        manifest = None

        def pin(self, *targets):
            pinned = {t: fingerprint.record(tmp_path, t) for t in targets}
            monkeypatch.setattr(fingerprint, "load_pinned", lambda: pinned)
            return pinned

    state = Fake()

    def manifest():
        if state.manifest is not None:
            return state.manifest
        rows = {}
        for spec in patching._declared:
            entry = rows.setdefault(spec.row, {"feature": spec.feature, "patches": [], "attach": []})
            entry["patches"].append({"target": spec.target, "hook_type": spec.hook_type})
        for spec in patching._attached:
            entry = rows.setdefault(spec.row, {"feature": spec.feature, "patches": [], "attach": []})
            entry["attach"].append({"owner": spec.owner, "name": spec.name})
        return rows

    monkeypatch.setattr(patching, "load_manifest", manifest)
    return state


def _mod():
    import qsa_fake.mod as mod

    return mod


def _foreign():
    from sglang.srt.plugins.hook_registry import HookSource

    return HookSource(plugin_name="other", dist_name="other-dist")


# Feature switches -----------------------------------------------------------


@pytest.mark.parametrize(
    "environ, expected",
    [
        ({}, Features()),
        ({"SGLANG_QSA_MODEL_COMPAT": "0"}, Features()),
        ({"SGLANG_QSA_MODEL_COMPAT": "1"}, COMPAT),
        ({"SGLANG_QSA_MODEL_COMPAT": "1", "SGLANG_QSA_HISPARSE_V3": "p2-offload"}, BOTH),
    ],
)
def test_feature_switches(environ, expected):
    assert read_features(environ) == expected


@pytest.mark.parametrize(
    "environ",
    [
        {"SGLANG_QSA_MODEL_COMPAT": "yes"},
        {"SGLANG_QSA_HISPARSE_V3": "p2-offload"},
        {"SGLANG_QSA_MODEL_COMPAT": "1", "SGLANG_QSA_HISPARSE_V3": "p3"},
    ],
)
def test_invalid_switches_fail(environ):
    with pytest.raises(PluginConfigError):
        read_features(environ)


def test_activation_errors_escape_exception_handlers():
    assert not issubclass(PluginActivationError, Exception)
    with pytest.raises(PluginActivationError):
        try:
            raise FingerprintMismatch("pinned source changed")
        except Exception:  # The SGLang loader's handler shape.
            pytest.fail("activation error was swallowed")


# Fingerprints ---------------------------------------------------------------


def test_record_locates_definitions(fake):
    record = fingerprint.record(fake.root, "qsa_fake.mod.Box.value")
    assert record["file"] == "qsa_fake/mod.py"
    assert record["qualname"] == "Box.value"
    assert record["kind"] == "function"
    assert record["chains"] == {
        "cpu": [{"type": "builtins.function", "code": ["qsa_fake/mod.py", record["def_line"]]}]
    }
    cached = fingerprint.record(fake.root, "qsa_fake.mod.cached")
    assert [level["type"] for level in cached["chains"]["cpu"]] == [
        "functools._lru_cache_wrapper",
        "builtins.function",
    ]
    assert fingerprint.record(fake.root, "qsa_fake.mod.Box")["kind"] == "class"


@pytest.mark.parametrize(
    "edit",
    [
        lambda s: s.replace("return 1", "return 2"),
        lambda s: s + "\nif True:\n    def double(x):\n        return 3 * x\n",
        lambda s: s.replace("\n", "\r\n"),
        lambda s: "if True:\n" + textwrap.indent(s, "    "),
        lambda s: s + "\nLIMIT = 4\n",
    ],
    ids=["body", "conditional-redefinition", "crlf", "indent-shift", "module-constant"],
)
def test_any_module_edit_fails_verification(fake, edit):
    pinned = fake.pin("qsa_fake.mod.Box.value")
    fake.path.write_bytes(edit(fake.path.read_text()).encode())
    with pytest.raises(FingerprintMismatch):
        fingerprint.verify(["qsa_fake.mod.Box.value"], pinned=pinned, check_bindings=False)


def test_method_record_covers_class_header(fake):
    before = fingerprint.definition_record(fake.path, "Box.value")["sha256"]
    fake.path.write_text(fake.path.read_text().replace("class Box:", "@dataclass\nclass Box(Base):"))
    assert fingerprint.definition_record(fake.path, "Box.value")["sha256"] != before


@pytest.mark.parametrize(
    "suffix, qualname",
    [
        ("\nif True:\n    def double(x):\n        return 3 * x\n", "double"),
        ("\ndouble = staticmethod(double)\n", "double"),
        ("\nfrom os import path as double\n", "double"),
        ("\ntry:\n    pass\nexcept Exception as double:\n    pass\n", "double"),
        ("\nmatch 1:\n    case double:\n        pass\n", "double"),
        ("\nif (double := 3):\n    pass\n", "double"),
        ("\nfor double in ():\n    pass\n", "double"),
        ("\nwith open(__file__) as double:\n    pass\n", "double"),
        ("\ndel double\n", "double"),
        ("", "Child.value"),
    ],
    ids=["conditional", "reassigned", "imported", "except-alias", "match-capture",
         "walrus", "for-target", "with-target", "del", "inherited"],
)
def test_ambiguous_or_inherited_bindings_cannot_be_pinned(fake, suffix, qualname):
    fake.path.write_text(MODULE + suffix)
    with pytest.raises(LookupError):
        fingerprint.definition_record(fake.path, qualname)


@pytest.mark.parametrize(
    "suffix",
    [
        "\ndouble.extra = 1\n",
        "\ndouble: object\n",
        "\nvalues = [double for double in ()]\n",
        "\ndef other():\n    double = 1\n    return double\n",
        "\nclass Other:\n    double = 1\n",
    ],
    ids=["attribute", "annotation-only", "comprehension", "nested-function", "nested-class"],
)
def test_non_bindings_do_not_count(fake, suffix):
    fake.path.write_text(MODULE + suffix)
    assert fingerprint.definition_record(fake.path, "double")["qualname"] == "double"


# Activation -----------------------------------------------------------------


def test_activate_applies_and_verifies(registry, fake):
    fake.pin("qsa_fake.mod.double", "qsa_fake.mod.Box.value")

    @patching.patch("qsa_fake.mod.double", "around", feature="model_compat", row="R1")
    def triple(original, x):
        return original(x) + x

    @patching.patch(
        "qsa_fake.mod.Box.value",
        "replace",
        feature="hisparse",
        row="R2",
        depends=("qsa_fake.mod.double",),
    )
    def value(self):
        return 7

    specs = patching.activate(COMPAT)
    mod = _mod()
    assert [spec.target for spec in specs] == ["qsa_fake.mod.double"]
    assert mod.double(2) == 6
    assert mod.Box().value() == 1  # hisparse was not requested.
    assert patching.activate(COMPAT) == specs
    with pytest.raises(PluginActivationError):
        patching.activate(BOTH)


def test_manifest_must_match_declarations(registry, fake):
    fake.pin("qsa_fake.mod.double", "qsa_fake.mod.Box.value")
    patching.patch("qsa_fake.mod.double", "after", feature="model_compat", row="R1")(
        lambda result, x: result
    )
    hook_entry = {"target": "qsa_fake.mod.Box.value", "hook_type": "after"}
    fake.manifest = {
        "R1": {"feature": "model_compat", "patches": [{"target": "qsa_fake.mod.double", "hook_type": "after"}], "attach": []},
        "R2": {"feature": "model_compat", "patches": [hook_entry], "attach": []},
    }
    with pytest.raises(PluginActivationError, match="missing"):
        patching.activate(COMPAT)
    fake.manifest = {}
    with pytest.raises(PluginActivationError, match="no manifest rows"):
        patching.activate(COMPAT)
    fake.manifest = {
        "R1": {"feature": "model_compat", "patches": [{"target": "qsa_fake.mod.double", "hook_type": "around"}], "attach": []},
    }
    with pytest.raises(PluginActivationError, match="undeclared in manifest"):
        patching.activate(COMPAT)
    assert not registry._hooks


def test_fingerprint_mismatch_registers_nothing(registry, fake):
    fake.pin("qsa_fake.mod.double")
    fake.path.write_text(fake.path.read_text().replace("2 * x", "x + x"))
    patching.patch("qsa_fake.mod.double", "around", feature="model_compat", row="R1")(
        lambda original, x: original(x)
    )
    with pytest.raises(FingerprintMismatch):
        patching.activate(COMPAT)
    assert not registry._hooks
    assert patching._activated is None


def test_unpinned_dependency_fails(registry, fake):
    fake.pin("qsa_fake.mod.double")
    patching.patch(
        "qsa_fake.mod.double",
        "around",
        feature="model_compat",
        row="R1",
        depends=("qsa_fake.mod.Box.value",),
    )(lambda original, x: original(x))
    with pytest.raises(FingerprintMismatch, match="no pinned fingerprint"):
        patching.activate(COMPAT)


def test_missing_mode_chain_fails(registry, fake, monkeypatch):
    fake.pin("qsa_fake.mod.double")
    monkeypatch.setattr(fingerprint, "binding_mode", lambda: "cuda")
    patching.patch("qsa_fake.mod.double", "after", feature="model_compat", row="R1")(
        lambda result, x: result
    )
    with pytest.raises(FingerprintMismatch, match="no pinned binding chain for mode cuda"):
        patching.activate(COMPAT)


@pytest.mark.parametrize("kind", ["raw", "wraps"])
def test_live_object_must_be_the_pinned_definition(registry, fake, monkeypatch, kind):
    fake.pin("qsa_fake.mod.double")
    mod = _mod()
    original = mod.double
    if kind == "raw":
        replacement = lambda x: 50 * x  # noqa: E731
    else:

        @functools.wraps(original)
        def replacement(x):
            return 50 * original(x)

    monkeypatch.setattr(mod, "double", replacement)
    patching.patch("qsa_fake.mod.double", "after", feature="model_compat", row="R1")(
        lambda result, x: result + 1
    )
    with pytest.raises(FingerprintMismatch, match="not the pinned definition"):
        patching.activate(COMPAT)
    assert not registry._hooks


def test_decorated_definition_passes_with_its_pinned_chain(registry, fake):
    fake.pin("qsa_fake.mod.cached")
    patching.patch("qsa_fake.mod.cached", "after", feature="model_compat", row="R1")(
        lambda result, x: result + 1
    )
    patching.activate(COMPAT)
    assert _mod().cached(1) == 2


def test_inherited_member_is_rejected(registry, fake, monkeypatch):
    pinned = fake.pin("qsa_fake.mod.Box.value")
    pinned["qsa_fake.mod.Child.value"] = pinned["qsa_fake.mod.Box.value"]
    patching.patch("qsa_fake.mod.Child.value", "after", feature="model_compat", row="R1")(
        lambda result, self: result
    )
    with pytest.raises(FingerprintMismatch, match="not the pinned definition"):
        patching.activate(COMPAT)


def test_property_targets_are_rejected(registry, fake):
    fake.pin("qsa_fake.mod.Box.size")
    patching.patch("qsa_fake.mod.Box.size", "after", feature="model_compat", row="R1")(
        lambda result, self: result
    )
    with pytest.raises(PluginActivationError, match="Properties cannot be hook targets"):
        patching.activate(COMPAT)
    assert _mod().Box().size == 3


def test_duplicate_replace_fails(registry, fake):
    fake.pin("qsa_fake.mod.double")
    for _ in range(2):
        patching.patch("qsa_fake.mod.double", "replace", feature="model_compat", row="R1")(
            lambda x: x
        )
    with pytest.raises(PluginActivationError, match="Two REPLACE"):
        patching.activate(COMPAT)


@pytest.mark.parametrize("foreign_target", ["qsa_fake.mod.Box.value", "qsa_fake.mod.Box"])
def test_earlier_foreign_hooks_fail(registry, fake, foreign_target):
    from sglang.srt.plugins.hook_registry import HookType

    fake.pin("qsa_fake.mod.Box.value")
    mod = _mod()
    if foreign_target.endswith("Box"):
        hook, hook_type = type("Other", (mod.Box,), {}), HookType.REPLACE
    else:
        hook, hook_type = (lambda original, self: 0), HookType.AROUND
    registry.register(foreign_target, hook, hook_type, source=_foreign())
    patching.patch("qsa_fake.mod.Box.value", "after", feature="model_compat", row="R1")(
        lambda result, self: result + 1
    )
    with pytest.raises(PluginActivationError, match="overlapping"):
        patching.activate(COMPAT)


def test_activation_leaves_other_plugins_hooks_to_the_loader(registry, fake):
    from sglang.srt.plugins.hook_registry import HookType

    fake.pin("qsa_fake.mod.Box.value")
    registry.register(
        "qsa_fake.mod.double", lambda r, x: r + 1, HookType.AFTER, source=_foreign()
    )
    patching.patch("qsa_fake.mod.Box.value", "after", feature="model_compat", row="R1")(
        lambda result, self: result + 1
    )
    patching.activate(COMPAT)
    assert "qsa_fake.mod.double" not in registry._patched
    registry.register(
        "qsa_fake.mod.double", lambda r, x: r + 10, HookType.AFTER, source=_foreign()
    )
    registry.apply_hooks()  # load_plugins()'s final pass after every plugin ran.
    mod = _mod()
    assert mod.double(1) == 13
    assert mod.Box().value() == 2
    patching.verify_final("scheduler")


def test_late_hooks_fail_final_verification(registry, fake, tmp_path, monkeypatch):
    from sglang.srt.plugins.hook_registry import HookSource, HookType

    fake.pin("qsa_fake.mod.Box.value", "qsa_fake.mod.configure")
    patching.patch("qsa_fake.mod.Box.value", "after", feature="model_compat", row="R1")(
        lambda result, self: result + 1
    )
    calls = []

    @patching.patch("qsa_fake.mod.configure", "before", feature="framework", row="FW")
    def verifier(*args, **kwargs):
        calls.append(1)
        patching.verify_final("scheduler", tp_rank=0)

    patching.activate(COMPAT)
    monkeypatch.setenv(patching.ACTIVATION_DIR_ENV, str(tmp_path / "records"))
    mod = _mod()

    # A later plugin replacing an unrelated class cannot remove the verifier.
    registry.register("qsa_fake.mod.Child", type("Other", (mod.Box,), {}), HookType.REPLACE, source=_foreign())
    registry.apply_hooks()
    assert mod.configure() == "configured" and calls == [1]
    record = json.loads(next((tmp_path / "records").glob("scheduler-*.json")).read_text())
    assert record["patches"] == ["qsa_fake.mod.Box.value", "qsa_fake.mod.configure"]

    # A later hook on the verifier's own target is skipped by the registry,
    # which final verification reports.
    registry.register("qsa_fake.mod.configure", lambda *a, **k: None, HookType.BEFORE, source=_foreign())
    registry.apply_hooks()
    with pytest.raises(PluginActivationError, match="overlapping"):
        mod.configure()

    registry._hooks["qsa_fake.mod.configure"].pop()
    own = HookSource(plugin_name=patching.PLUGIN_NAME, dist_name=patching.DIST_NAME)
    registry.register("qsa_fake.mod.double", lambda r, x: r, HookType.AFTER, source=own)
    with pytest.raises(PluginActivationError, match="registered after activation"):
        mod.configure()


def test_hooks_inside_a_replaced_class_are_rejected(registry, fake):
    fake.pin("qsa_fake.mod.Box", "qsa_fake.mod.Box.value")
    mod = _mod()
    patching.patch("qsa_fake.mod.Box", "replace", feature="model_compat", row="R1")(
        type("Replacement", (mod.Box,), {})
    )
    patching.patch("qsa_fake.mod.Box.value", "after", feature="model_compat", row="R2")(
        lambda result, self: result
    )
    with pytest.raises(PluginActivationError, match="replaced class"):
        patching.activate(COMPAT)


# Attach ---------------------------------------------------------------------


def test_attach_adds_absent_member(registry, fake):
    fake.pin("qsa_fake.mod.Box.value")

    @patching.attach(
        "qsa_fake.mod.Box", feature="model_compat", row="R1", depends=("qsa_fake.mod.Box.value",)
    )
    def doubled(self):
        return 2 * self.value()

    patching.attach_value("qsa_fake.mod", "LIMIT", 3, feature="model_compat", row="R1")
    patching.activate(COMPAT)
    mod = _mod()
    assert mod.Box().doubled() == 2
    assert mod.LIMIT == 3


def test_attach_follows_a_replaced_class(registry, fake):
    fake.pin("qsa_fake.mod.Box")
    mod = _mod()
    replacement = type("Replacement", (mod.Box,), {})
    patching.patch("qsa_fake.mod.Box", "replace", feature="model_compat", row="R1")(replacement)
    patching.attach_value("qsa_fake.mod.Box", "extra", 5, feature="model_compat", row="R1")
    patching.activate(COMPAT)
    assert mod.Box is replacement
    assert replacement.__dict__["extra"] == 5


@pytest.mark.parametrize("name", ["value", "__init__"])
def test_attach_fails_when_upstream_defines_the_name(registry, fake, name):
    fake.pin()
    patching.attach_value("qsa_fake.mod.Box", name, lambda self: 0, feature="model_compat", row="R1")
    with pytest.raises(PluginActivationError, match="already defined"):
        patching.activate(COMPAT)
    assert _mod().Box().value() == 1
    assert patching._activated is None


def test_attach_conflicts_fail(registry, fake):
    fake.pin("qsa_fake.mod.double")
    for _ in range(2):
        patching.attach_value("qsa_fake.mod.Box", "extra", 1, feature="model_compat", row="R1")
    with pytest.raises(PluginActivationError, match="attached twice"):
        patching.activate(COMPAT)
    patching._attached.clear()
    patching.attach_value("qsa_fake.mod.Box", "extra", 1, feature="model_compat", row="R1")
    patching.patch("qsa_fake.mod.Box.extra", "after", feature="hisparse", row="R2")(
        lambda result, self: result
    )
    with pytest.raises(PluginActivationError, match="Hooks on attached"):
        patching.activate(BOTH)


# Real SGLang loader in a fresh process --------------------------------------


def _run_loader(tmp_path, environ, prelude="", epilogue=""):
    dist = tmp_path / "sglang_qsa_hisparse-0.1.0.dev0.dist-info"
    dist.mkdir(exist_ok=True)
    (dist / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: sglang-qsa-hisparse\nVersion: 0.1.0.dev0\n"
    )
    (dist / "entry_points.txt").write_text(
        "[sglang.srt.plugins]\nqsa_hisparse = sglang_qsa_hisparse.plugin:load\n"
    )
    code = prelude + textwrap.dedent(
        """
        import sys
        from sglang.srt.plugins import load_plugins
        from sglang.srt.plugins.hook_registry import HookRegistry
        load_plugins()
        print("hooks", sorted(HookRegistry._hooks))
        print("patched", sorted(HookRegistry._patched))
        print("patching", "sglang_qsa_hisparse.patching" in sys.modules)
        """
    ) + epilogue
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("SGLANG_QSA_") and key != "SGLANG_PLUGINS"
    }
    env.update(environ)
    env["PYTHONPATH"] = f"{tmp_path}{os.pathsep}{env.get('PYTHONPATH', '')}"
    return subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )


def test_entry_point_is_noop_when_off(tmp_path):
    result = _run_loader(tmp_path, {})
    assert result.returncode == 0, result.stderr
    assert "hooks []" in result.stdout
    assert "patching False" in result.stdout


def test_entry_point_rejects_unimplemented_features(tmp_path):
    result = _run_loader(tmp_path, {"SGLANG_QSA_MODEL_COMPAT": "1"})
    assert result.returncode != 0
    assert "PluginActivationError" in result.stderr
    assert "differ from manifest.json" in result.stderr
    assert "hooks" not in result.stdout


def test_entry_point_activates_on_the_pin(tmp_path):
    # Stand-in for a finished feature: one pinned model_compat row.
    prelude = textwrap.dedent(
        """
        import sglang_qsa_hisparse.patching as patching
        manifest = patching.load_manifest()
        manifest = {k: v for k, v in manifest.items() if v["feature"] == "framework"}
        target = "sglang.srt.managers.scheduler.run_scheduler_process"
        manifest["T1"] = {"feature": "model_compat", "attach": [],
                          "patches": [{"target": target, "hook_type": "around"}]}
        patching.load_manifest = lambda: manifest
        real_import = patching._import_feature_modules
        def import_features(feature):
            patching.patch(target, "around", feature="model_compat", row="T1")(
                lambda original, *a, **k: original(*a, **k))
        patching._import_feature_modules = import_features
        """
    )
    epilogue = textwrap.dedent(
        """
        patching.verify_final("test")
        print("verified")
        """
    )
    result = _run_loader(tmp_path, {"SGLANG_QSA_MODEL_COMPAT": "1"}, prelude, epilogue)
    assert result.returncode == 0, result.stderr
    assert "sglang.srt.managers.scheduler.configure_scheduler_process" in result.stdout
    assert "sglang.srt.managers.scheduler.run_scheduler_process" in result.stdout
    assert "verified" in result.stdout


def test_entry_point_config_failure_stops_the_process(tmp_path):
    result = _run_loader(tmp_path, {"SGLANG_QSA_HISPARSE_V3": "p2-offload"})
    assert result.returncode != 0
    assert "PluginConfigError" in result.stderr
    assert "hooks" not in result.stdout


def test_entry_point_import_failure_stops_the_process(tmp_path):
    prelude = textwrap.dedent(
        """
        import sglang_qsa_hisparse.patching as patching
        def broken(feature):
            raise ModuleNotFoundError("injected patch module import failure")
        patching._import_feature_modules = broken
        """
    )
    result = _run_loader(tmp_path, {"SGLANG_QSA_MODEL_COMPAT": "1"}, prelude)
    assert result.returncode != 0
    assert "PluginActivationError" in result.stderr
    assert "injected patch module import failure" in result.stderr
    assert "hooks" not in result.stdout
