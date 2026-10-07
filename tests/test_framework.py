"""Plugin framework contracts: no-op when off, fail-closed when on."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from sglang_qsa_hisparse import fingerprint, patching
from sglang_qsa_hisparse.errors import (
    FingerprintMismatch,
    PluginActivationError,
    PluginConfigError,
)
from sglang_qsa_hisparse.features import Features, read_features


@pytest.fixture
def registry():
    from sglang.srt.plugins.hook_registry import HookRegistry

    saved = list(patching._declared), list(patching._attached)
    HookRegistry.reset()
    patching._declared.clear()
    patching._attached.clear()
    patching._activated = None
    yield HookRegistry
    HookRegistry.reset()
    patching._declared[:], patching._attached[:] = saved
    patching._activated = None


@pytest.fixture
def fake_sglang(tmp_path, monkeypatch):
    """A throwaway module tree standing in for the installed SGLang sources."""
    package = tmp_path / "qsa_fake"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "mod.py").write_text(
        textwrap.dedent(
            """
            def double(x):
                return 2 * x


            class Box:
                def value(self):
                    return 1
            """
        )
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(fingerprint, "installed_source_root", lambda: tmp_path)
    monkeypatch.setattr(patching, "_import_feature_modules", lambda feature: None)
    for name in [m for m in sys.modules if m.startswith("qsa_fake")]:
        monkeypatch.delitem(sys.modules, name)
    return tmp_path


def _pin(root, *targets):
    return {target: fingerprint.record(root, target) for target in targets}


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


def test_fingerprint_tracks_definition_text(fake_sglang):
    before = fingerprint.record(fake_sglang, "qsa_fake.mod.Box.value")
    assert before["file"] == "qsa_fake/mod.py"
    assert before["qualname"] == "Box.value"
    path = fake_sglang / "qsa_fake" / "mod.py"
    path.write_text(path.read_text().replace("return 1", "return 2"))
    after = fingerprint.record(fake_sglang, "qsa_fake.mod.Box.value")
    assert after["sha256"] != before["sha256"]
    assert fingerprint.record(fake_sglang, "qsa_fake.mod.double") == _pin(
        fake_sglang, "qsa_fake.mod.double"
    )["qsa_fake.mod.double"]


def test_activate_applies_and_verifies(registry, fake_sglang, monkeypatch):
    pinned = _pin(fake_sglang, "qsa_fake.mod.double", "qsa_fake.mod.Box.value")
    monkeypatch.setattr(fingerprint, "load_pinned", lambda: pinned)

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
    pinned = _pin(fake_sglang, "qsa_fake.mod.double")
    monkeypatch.setattr(fingerprint, "load_pinned", lambda: pinned)
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
    pinned = _pin(fake_sglang, "qsa_fake.mod.double")
    monkeypatch.setattr(fingerprint, "load_pinned", lambda: pinned)

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


def test_duplicate_replace_fails(registry, fake_sglang, monkeypatch):
    pinned = _pin(fake_sglang, "qsa_fake.mod.double")
    monkeypatch.setattr(fingerprint, "load_pinned", lambda: pinned)
    for _ in range(2):
        patching.patch("qsa_fake.mod.double", "replace", feature="model_compat")(
            lambda x: x
        )
    with pytest.raises(PluginActivationError, match="Two REPLACE"):
        patching.activate(Features(model_compat=True))


def test_attach_adds_absent_member(registry, fake_sglang, monkeypatch):
    pinned = _pin(fake_sglang, "qsa_fake.mod.Box.value")
    monkeypatch.setattr(fingerprint, "load_pinned", lambda: pinned)

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
    pinned = _pin(fake_sglang, "qsa_fake.mod.double")
    monkeypatch.setattr(fingerprint, "load_pinned", lambda: pinned)
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


def _run_loader(tmp_path, environ):
    """Run SGLang's real entry-point loader in a fresh process."""
    dist = tmp_path / "sglang_qsa_hisparse-0.1.0.dev0.dist-info"
    dist.mkdir(exist_ok=True)
    (dist / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: sglang-qsa-hisparse\nVersion: 0.1.0.dev0\n"
    )
    (dist / "entry_points.txt").write_text(
        "[sglang.srt.plugins]\nqsa_hisparse = sglang_qsa_hisparse.plugin:load\n"
    )
    code = textwrap.dedent(
        """
        import sys
        from sglang.srt.plugins import load_plugins
        from sglang.srt.plugins.hook_registry import HookRegistry
        load_plugins()
        print("hooks", len(HookRegistry._hooks))
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
    assert "hooks 0" in result.stdout
    assert "patching False" in result.stdout


def test_entry_point_failure_stops_the_process(tmp_path):
    result = _run_loader(tmp_path, {"SGLANG_QSA_HISPARSE_V3": "p2-offload"})
    assert result.returncode != 0
    assert "PluginConfigError" in result.stderr
    assert "hooks" not in result.stdout
