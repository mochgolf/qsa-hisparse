"""Plugin framework contracts: no-op when off, fail-closed when on."""

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
    def double(x):
        return 2 * x


    class Box:
        def value(self):
            return 1


    class Child(Box):
        pass
    """
)


@pytest.fixture
def registry():
    from sglang.srt.plugins.hook_registry import HookRegistry

    saved = list(patching._declared), list(patching._attached)
    HookRegistry.reset()
    patching._declared.clear()
    patching._attached.clear()
    patching._activated = None
    patching._applied.clear()
    patching._attached_live.clear()
    yield HookRegistry
    HookRegistry.reset()
    patching._declared[:], patching._attached[:] = saved
    patching._activated = None
    patching._applied.clear()
    patching._attached_live.clear()


@pytest.fixture
def fake_sglang(tmp_path, monkeypatch):
    """A throwaway module tree standing in for the installed SGLang sources."""
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
    return tmp_path


def _pin(root, *targets):
    return {target: fingerprint.record(root, target) for target in targets}


def _use_pin(monkeypatch, root, *targets):
    pinned = _pin(root, *targets)
    monkeypatch.setattr(fingerprint, "load_pinned", lambda: pinned)
    return pinned


def _foreign():
    from sglang.srt.plugins.hook_registry import HookSource

    return HookSource(plugin_name="other", dist_name="other-dist")


# Feature switches -----------------------------------------------------------


@pytest.mark.parametrize(
    "environ, expected",
    [
        ({}, Features()),
        ({"SGLANG_QSA_MODEL_COMPAT": "0"}, Features()),
        ({"SGLANG_QSA_MODEL_COMPAT": "1"}, Features(model_compat=True)),
        (
            {"SGLANG_QSA_MODEL_COMPAT": "1", "SGLANG_QSA_HISPARSE_V3": "p2-offload"},
            Features(model_compat=True, hisparse_mode="p2-offload"),
        ),
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


def test_record_locates_definitions(fake_sglang):
    record = fingerprint.record(fake_sglang, "qsa_fake.mod.Box.value")
    assert record["file"] == "qsa_fake/mod.py"
    assert record["qualname"] == "Box.value"
    assert record["kind"] == "function"
    assert record["first_line"] == record["def_line"] == 7
    assert fingerprint.record(fake_sglang, "qsa_fake.mod.Box")["kind"] == "class"


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
def test_any_module_edit_fails_verification(fake_sglang, edit):
    pinned = _pin(fake_sglang, "qsa_fake.mod.Box.value")
    path = fake_sglang / "qsa_fake" / "mod.py"
    path.write_bytes(edit(path.read_text()).encode())
    with pytest.raises(FingerprintMismatch):
        fingerprint.verify(["qsa_fake.mod.Box.value"], pinned=pinned)


def test_method_record_covers_class_header(fake_sglang):
    before = fingerprint.record(fake_sglang, "qsa_fake.mod.Box.value")["sha256"]
    path = fake_sglang / "qsa_fake" / "mod.py"
    path.write_text(path.read_text().replace("class Box:", "@dataclass\nclass Box(Base):"))
    assert fingerprint.record(fake_sglang, "qsa_fake.mod.Box.value")["sha256"] != before


@pytest.mark.parametrize(
    "source, target",
    [
        (MODULE + "\nif True:\n    def double(x):\n        return 3 * x\n", "qsa_fake.mod.double"),
        (MODULE + "\ndouble = staticmethod(double)\n", "qsa_fake.mod.double"),
        (MODULE + "\nfrom os import path as double\n", "qsa_fake.mod.double"),
        (MODULE, "qsa_fake.mod.Child.value"),
    ],
    ids=["conditional", "reassigned", "imported", "inherited"],
)
def test_ambiguous_or_inherited_bindings_cannot_be_pinned(fake_sglang, source, target):
    (fake_sglang / "qsa_fake" / "mod.py").write_text(source)
    with pytest.raises(LookupError):
        fingerprint.record(fake_sglang, target)


# Activation -----------------------------------------------------------------


def test_activate_applies_and_verifies(registry, fake_sglang, monkeypatch):
    _use_pin(monkeypatch, fake_sglang, "qsa_fake.mod.double", "qsa_fake.mod.Box.value")

    @patching.patch("qsa_fake.mod.double", "around", feature="model_compat")
    def triple(original, x):
        return original(x) + x

    @patching.patch(
        "qsa_fake.mod.Box.value",
        "replace",
        feature="hisparse",
        depends=("qsa_fake.mod.double",),
    )
    def value(self):
        return 7

    specs = patching.activate(Features(model_compat=True))
    import qsa_fake.mod as mod

    assert [spec.target for spec in specs] == ["qsa_fake.mod.double"]
    assert mod.double(2) == 6
    assert mod.Box().value() == 1  # hisparse was not requested.
    assert patching.activate(Features(model_compat=True)) == specs
    with pytest.raises(PluginActivationError):
        patching.activate(Features(model_compat=True, hisparse_mode="p2-offload"))


def test_fingerprint_mismatch_registers_nothing(registry, fake_sglang, monkeypatch):
    _use_pin(monkeypatch, fake_sglang, "qsa_fake.mod.double")
    path = fake_sglang / "qsa_fake" / "mod.py"
    path.write_text(path.read_text().replace("2 * x", "x + x"))

    @patching.patch("qsa_fake.mod.double", "around", feature="model_compat")
    def hook(original, x):
        return original(x)

    with pytest.raises(FingerprintMismatch):
        patching.activate(Features(model_compat=True))
    assert not registry._hooks
    assert patching._activated is None


def test_unpinned_dependency_fails(registry, fake_sglang, monkeypatch):
    _use_pin(monkeypatch, fake_sglang, "qsa_fake.mod.double")

    @patching.patch(
        "qsa_fake.mod.double",
        "around",
        feature="model_compat",
        depends=("qsa_fake.mod.Box.value",),
    )
    def hook(original, x):
        return original(x)

    with pytest.raises(FingerprintMismatch, match="no pinned fingerprint"):
        patching.activate(Features(model_compat=True))


def test_live_object_must_be_the_pinned_definition(registry, fake_sglang, monkeypatch):
    _use_pin(monkeypatch, fake_sglang, "qsa_fake.mod.double")
    import qsa_fake.mod as mod

    monkeypatch.setattr(mod, "double", lambda x: 3 * x)  # e.g. a raw monkeypatch
    patching.patch("qsa_fake.mod.double", "around", feature="model_compat")(
        lambda original, x: original(x)
    )
    with pytest.raises(PluginActivationError, match="not the pinned definition"):
        patching.activate(Features(model_compat=True))
    assert not registry._hooks


def test_inherited_member_is_rejected(registry, fake_sglang, monkeypatch):
    pinned = _pin(fake_sglang, "qsa_fake.mod.Box.value")
    pinned["qsa_fake.mod.Child.value"] = pinned["qsa_fake.mod.Box.value"]
    monkeypatch.setattr(fingerprint, "load_pinned", lambda: pinned)
    patching.patch("qsa_fake.mod.Child.value", "after", feature="model_compat")(
        lambda result, self: result
    )
    with pytest.raises(PluginActivationError, match="not defined directly"):
        patching.activate(Features(model_compat=True))


def test_duplicate_replace_fails(registry, fake_sglang, monkeypatch):
    _use_pin(monkeypatch, fake_sglang, "qsa_fake.mod.double")
    for _ in range(2):
        patching.patch("qsa_fake.mod.double", "replace", feature="model_compat")(
            lambda x: x
        )
    with pytest.raises(PluginActivationError, match="Two REPLACE"):
        patching.activate(Features(model_compat=True))


@pytest.mark.parametrize(
    "foreign_target, hook",
    [
        ("qsa_fake.mod.Box.value", lambda original, self: 0),
        ("qsa_fake.mod.Box", "class"),
    ],
    ids=["same-target", "ancestor-class"],
)
def test_earlier_foreign_hooks_fail(registry, fake_sglang, monkeypatch, foreign_target, hook):
    from sglang.srt.plugins.hook_registry import HookType

    _use_pin(monkeypatch, fake_sglang, "qsa_fake.mod.Box.value")
    import qsa_fake.mod as mod

    if hook == "class":
        hook = type("Other", (mod.Box,), {})
        hook_type = HookType.REPLACE
    else:
        hook_type = HookType.AROUND
    registry.register(foreign_target, hook, hook_type, source=_foreign())
    patching.patch("qsa_fake.mod.Box.value", "after", feature="model_compat")(
        lambda result, self: result + 1
    )
    with pytest.raises(PluginActivationError, match="overlapping"):
        patching.activate(Features(model_compat=True))


def test_later_foreign_class_replace_fails_final_check(
    registry, fake_sglang, monkeypatch, tmp_path
):
    from sglang.srt.plugins.hook_registry import HookType

    _use_pin(monkeypatch, fake_sglang, "qsa_fake.mod.Box.value")
    patching.patch("qsa_fake.mod.Box.value", "after", feature="model_compat")(
        lambda result, self: result + 1
    )
    patching.activate(Features(model_compat=True))
    monkeypatch.setenv(patching.ACTIVATION_DIR_ENV, str(tmp_path / "records"))
    patching.verify_final("scheduler", tp_rank=0)
    record = json.loads(next((tmp_path / "records").glob("scheduler-*.json")).read_text())
    assert record["patches"] == ["qsa_fake.mod.Box.value"]
    assert record["tp_rank"] == 0

    import qsa_fake.mod as mod

    replacement = type("Other", (mod.Box,), {"value": lambda self: 0})
    registry.register("qsa_fake.mod.Box", replacement, HookType.REPLACE, source=_foreign())
    registry.apply_hooks()  # load_plugins() does this after every plugin ran.
    with pytest.raises(PluginActivationError, match="overlapping"):
        patching.verify_final("scheduler", tp_rank=0)


def test_hooks_inside_a_replaced_class_are_rejected(registry, fake_sglang, monkeypatch):
    _use_pin(monkeypatch, fake_sglang, "qsa_fake.mod.Box", "qsa_fake.mod.Box.value")
    import qsa_fake.mod as mod

    patching.patch("qsa_fake.mod.Box", "replace", feature="model_compat")(
        type("Replacement", (mod.Box,), {})
    )
    patching.patch("qsa_fake.mod.Box.value", "after", feature="model_compat")(
        lambda result, self: result
    )
    with pytest.raises(PluginActivationError, match="replaced class"):
        patching.activate(Features(model_compat=True))


# Attach ---------------------------------------------------------------------


def test_attach_adds_absent_member(registry, fake_sglang, monkeypatch):
    _use_pin(monkeypatch, fake_sglang, "qsa_fake.mod.Box.value")

    @patching.attach(
        "qsa_fake.mod.Box", feature="model_compat", depends=("qsa_fake.mod.Box.value",)
    )
    def doubled(self):
        return 2 * self.value()

    patching.attach_value("qsa_fake.mod", "LIMIT", 3, feature="model_compat")
    patching.activate(Features(model_compat=True))
    import qsa_fake.mod as mod

    assert mod.Box().doubled() == 2
    assert mod.LIMIT == 3


def test_attach_follows_a_replaced_class(registry, fake_sglang, monkeypatch):
    _use_pin(monkeypatch, fake_sglang, "qsa_fake.mod.Box")
    import qsa_fake.mod as mod

    replacement = type("Replacement", (mod.Box,), {})
    patching.patch("qsa_fake.mod.Box", "replace", feature="model_compat")(replacement)
    patching.attach_value("qsa_fake.mod.Box", "extra", 5, feature="model_compat")
    patching.activate(Features(model_compat=True))
    assert mod.Box is replacement
    assert replacement.__dict__["extra"] == 5


@pytest.mark.parametrize("name", ["value", "__init__"])
def test_attach_fails_when_upstream_defines_the_name(
    registry, fake_sglang, monkeypatch, name
):
    monkeypatch.setattr(fingerprint, "load_pinned", lambda: {})
    patching.attach_value("qsa_fake.mod.Box", name, lambda self: 0, feature="model_compat")
    with pytest.raises(PluginActivationError, match="already defined"):
        patching.activate(Features(model_compat=True))
    import qsa_fake.mod as mod

    assert mod.Box().value() == 1
    assert patching._activated is None


def test_attach_conflicts_fail(registry, fake_sglang, monkeypatch):
    _use_pin(monkeypatch, fake_sglang, "qsa_fake.mod.double")
    for _ in range(2):
        patching.attach_value("qsa_fake.mod.Box", "extra", 1, feature="model_compat")
    with pytest.raises(PluginActivationError, match="attached twice"):
        patching.activate(Features(model_compat=True))
    patching._attached.clear()
    patching.attach_value("qsa_fake.mod.Box", "extra", 1, feature="model_compat")
    patching.patch("qsa_fake.mod.Box.extra", "after", feature="hisparse")(
        lambda result, self: result
    )
    with pytest.raises(PluginActivationError, match="Hooks on attached"):
        patching.activate(Features(model_compat=True, hisparse_mode="p2-offload"))


# Real SGLang loader in a fresh process --------------------------------------


def _run_loader(tmp_path, environ, prelude=""):
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
    )
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


def test_entry_point_activates_on_the_pin(tmp_path):
    result = _run_loader(tmp_path, {"SGLANG_QSA_MODEL_COMPAT": "1"})
    assert result.returncode == 0, result.stderr
    assert "sglang.srt.managers.scheduler.Scheduler.__init__" in result.stdout
    assert "patching True" in result.stdout


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
