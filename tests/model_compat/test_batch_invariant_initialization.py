"""No GPU: exercise real initialization/disable with a recording dispatcher."""
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.batch_invariant_ops import batch_invariant_ops as native
from sglang_qsa_hisparse.patches.model_compat import batch_invariant as plugin

# Frozen main registration contract; expected rows do not come from the adapter.
CUDA_OPERATORS = (
    "aten::mm", "aten::mm.out", "aten::addmm", "aten::_log_softmax",
    "aten::mean.dim", "aten::rms_norm", "aten::mm.dtype", "aten::bmm",
)


@pytest.fixture
def dispatcher(monkeypatch):
    class Library:
        def __init__(self, namespace, kind):
            assert (namespace, kind) == ("aten", "IMPL")
            self.calls = []
            self.destroyed = False
            libraries.append(self)

        def impl(self, name, function, device):
            self.calls.append((name, function, device))

        def _destroy(self):
            self.destroyed = True

    libraries = []
    original_bmm = torch.bmm
    monkeypatch.setattr(torch.library, "Library", Library)
    monkeypatch.setattr(torch, "bmm", original_bmm)
    monkeypatch.setattr(native, "_batch_invariant_MODE", False)
    monkeypatch.setattr(native, "_batch_invariant_LIB", None)
    monkeypatch.setattr(native, "_original_torch_bmm", None)
    monkeypatch.setattr(plugin, "is_cuda", lambda: True)
    monkeypatch.setattr(native, "_is_npu", False)
    monkeypatch.setattr(native, "get_dispatch_device_backend", lambda: "CUDA")
    monkeypatch.setattr(plugin.scope, "target_model_active", lambda: True)
    return SimpleNamespace(libraries=libraries, original_bmm=original_bmm)


@pytest.mark.parametrize("enable_bmm", [False, True])
def test_target_keeps_native_callbacks_except_mm_out_and_native_lifecycle(dispatcher, enable_bmm):
    original = native.enable_batch_invariant_mode
    # The unadapted main initializer installs the numerically changed route.
    original(enable_bmm=enable_bmm)
    baseline = list(dispatcher.libraries[-1].calls)
    assert tuple(row[0] for row in baseline) == CUDA_OPERATORS[:7 + enable_bmm]
    native.disable_batch_invariant_mode()
    plugin._enable_batch_invariant_mode(original, enable_bmm=enable_bmm)
    library = native._batch_invariant_LIB
    assert library.calls == [row for row in baseline if row[0] != "aten::mm.out"]
    assert native._batch_invariant_MODE is True
    assert torch.bmm is (native.bmm_batch_invariant if enable_bmm else dispatcher.original_bmm)
    assert native._original_torch_bmm is (dispatcher.original_bmm if enable_bmm else None)
    # An already enabled native namespace must keep the same library/bmm owner.
    plugin._enable_batch_invariant_mode(original, enable_bmm=not enable_bmm)
    assert native._batch_invariant_LIB is library
    assert len(dispatcher.libraries) == 2
    native.disable_batch_invariant_mode()
    assert library.destroyed
    assert native._batch_invariant_LIB is None
    assert native._batch_invariant_MODE is False
    assert native._original_torch_bmm is None
    assert torch.bmm is dispatcher.original_bmm
    plugin._enable_batch_invariant_mode(original, enable_bmm=enable_bmm)
    assert native._batch_invariant_LIB is not library
    native.disable_batch_invariant_mode()


@pytest.mark.parametrize("target,device,npu", [(False, "CUDA", False), (True, "CPU", False),
    (True, "XPU", False), (True, "NPU", True)])
@pytest.mark.parametrize("enable_bmm", [False, True])
def test_other_models_and_devices_delegate(dispatcher, monkeypatch, target, device, npu, enable_bmm):
    monkeypatch.setattr(plugin.scope, "target_model_active", lambda: target)
    monkeypatch.setattr(native, "get_dispatch_device_backend", lambda: device)
    monkeypatch.setattr(native, "_is_npu", npu)
    calls = []
    def original(**kwargs):
        calls.append(kwargs)
        return "native-result"
    assert plugin._enable_batch_invariant_mode(original, enable_bmm=enable_bmm) == "native-result"
    assert calls == [{"enable_bmm": enable_bmm}]
    assert dispatcher.libraries == []


def test_hip_cuda_dispatch_delegates_before_scope(dispatcher, monkeypatch):
    # HIP uses the CUDA dispatcher key but must retain main registrations.
    monkeypatch.setattr(plugin, "is_cuda", lambda: False)
    def unavailable_scope():
        raise AssertionError("non-NVIDIA path must not inspect model scope")
    monkeypatch.setattr(plugin.scope, "target_model_active", unavailable_scope)
    calls = []
    assert plugin._enable_batch_invariant_mode(lambda **kw: calls.append(kw), False) is None
    assert calls == [{"enable_bmm": False}]
    assert dispatcher.libraries == []
