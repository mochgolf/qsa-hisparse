"""CPU launch spies preserve frozen reference reduction geometry under global ON."""
import ast
import hashlib
from pathlib import Path

import pytest

from sglang_qsa_hisparse.patches.model_compat import shared_gate as plugin


class Tensor:
    def __init__(self, shape): self.shape = shape
    def is_contiguous(self): return True


@pytest.fixture
def launch(monkeypatch):
    calls = []
    class Kernel:
        def __getitem__(self, grid):
            def call(*args, **kwargs): calls.append((grid, args, kwargs))
            return call
    monkeypatch.setattr(plugin, "_fused_gate_sigmoid_mul_add_kernel", Kernel())
    monkeypatch.setattr(plugin, "_is_hip", False)
    monkeypatch.setattr(plugin, "is_arch_support_pdl", lambda: False)
    monkeypatch.setattr(plugin, "is_cuda", lambda: True)
    monkeypatch.setattr(plugin, "is_batch_invariant_mode_enabled", lambda: True)
    monkeypatch.setattr(plugin.scope, "target_model_active", lambda: True)
    return calls


@pytest.mark.parametrize("rows,warps", [(1,16), (12,16), (1023,16), (1024,8), (2048,8)])
@pytest.mark.parametrize("do_add", [False, True])
def test_global_on_uses_reference_launch_geometry(launch, rows, warps, do_add):
    inputs = (Tensor((rows,2560)), Tensor((2560,)), Tensor((rows,2560)), Tensor((rows,2560)))
    def unexpected_native(*args, **kwargs): raise AssertionError("target global-ON must use reference launcher")
    plugin._shared_gate_launch(unexpected_native, *inputs, do_add=do_add)
    assert len(launch) == 1
    grid, actual_inputs, config = launch[0]
    assert grid == (rows,)
    assert actual_inputs == inputs
    # Frozen reference launch literals; expected values never read native config.
    assert config.get("GATE_ONLY", False) is False
    config = {key:value for key,value in config.items() if key != "GATE_ONLY"}
    assert config == {"hidden_dim":2560, "DO_ADD":do_add, "USE_PDL":False,
        "BLOCK_SIZE":4096, "num_warps":warps}


@pytest.mark.parametrize("platform,global_on,target", [
    ("cuda",True,False), ("cuda",False,True), ("cpu",True,True), ("hip",True,True)])
def test_non_target_cpu_hip_and_global_off_delegate(launch, monkeypatch, platform, global_on, target):
    monkeypatch.setattr(plugin, "is_cuda", lambda: platform == "cuda")
    monkeypatch.setattr(plugin, "is_batch_invariant_mode_enabled", lambda: global_on)
    def model_scope():
        if platform != "cuda" or not global_on: raise AssertionError("scope must not be read")
        return target
    monkeypatch.setattr(plugin.scope, "target_model_active", model_scope)
    inputs = (Tensor((12,2560)), Tensor((2560,)), Tensor((12,2560)), Tensor((12,2560)))
    calls = []
    def original(*args, **kwargs): calls.append((args,kwargs)); return "native-result"
    assert plugin._shared_gate_launch(original, *inputs, do_add=True) == "native-result"
    assert calls == [(inputs,{"do_add":True})]
    assert launch == []


def test_reference_helper_and_launch_constants_are_frozen():
    source = Path(plugin.__file__).read_text()
    node = next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)
        and n.name == "_launch_fused_gate_sigmoid_mul")
    segment = "".join(source.splitlines(keepends=True)[node.lineno-1:node.end_lineno])
    # production 897286b12a, kernels/ops/elementwise/elementwise.py helper
    assert hashlib.sha256(segment.encode()).hexdigest() == "a8d03ff8575ec2f543b683af38338518640e106c438f0d860f66c4df926a031c"
