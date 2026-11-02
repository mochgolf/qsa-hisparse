"""Main-branch maintenance regressions (2026-11-02)."""

import gc
import weakref
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sglang.srt.layers.quantization.gptq.schemes.gptq_moe import GPTQMarlinMoEScheme
from sglang_qsa_hisparse.patches.model_compat import qwen4_exp


@pytest.mark.parametrize("desc_act, expected_groups", [(False, 2), (True, 4)])
def test_upstream_gptq_scale_contract_replaces_z02(desc_act, expected_groups):
    """#35955: each rank owns local groups, or the full table with act-order."""
    scheme = SimpleNamespace(
        kernel=SimpleNamespace(),
        quant_config=SimpleNamespace(desc_act=desc_act, group_size=32, pack_factor=8),
    )
    layer = nn.Module()
    layer.moe_tp_size = 2
    GPTQMarlinMoEScheme.create_weights(
        scheme,
        layer=layer,
        num_experts=2,
        hidden_size=128,
        intermediate_size_per_partition=64,
        params_dtype=torch.bfloat16,
    )
    assert layer.w13_scales.dtype == layer.w2_scales.dtype == torch.bfloat16
    assert layer.w13_scales.shape == (2, 4, 128)
    assert layer.w2_scales.shape == (2, expected_groups, 128)
    assert layer.w2_qzeros.shape == (2, expected_groups, 16)
    assert layer.w2_scales.load_full_w2 is desc_act


def test_loader_parameter_snapshot_dies_before_cyclic_gc():
    """#38900: repacked parameters must not await a cyclic-GC collection."""
    refs = []

    class Snapshot:
        pass

    def parameters(**kwargs):
        snapshot = Snapshot()
        refs.append(weakref.ref(snapshot))
        return [("old.parameter", snapshot)]

    model = SimpleNamespace(
        config=SimpleNamespace(),
        named_parameters=parameters,
        named_buffers=lambda: [],
        named_modules=lambda: [],
        modules=lambda: [],
    )
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        loaded = qwen4_exp.ForkQwen4ExpForConditionalGeneration.load_weights(model, [])
        assert loaded == set()
        assert len(refs) == 1
        assert refs[0]() is None
    finally:
        if was_enabled:
            gc.enable()


def test_marlin_no_combine_returns_router_weighted_expert_rows(monkeypatch):
    """Main's runner owns the combine when it asks for expert rows."""
    from sglang.srt.layers.moe import fused_moe_triton
    from sglang_qsa_hisparse.patches.model_compat import marlin

    monkeypatch.setattr(
        marlin, "get_context",
        lambda: SimpleNamespace(is_config_namespace_published=lambda name: False),
    )
    monkeypatch.setattr(marlin, "get_scalar_type", lambda *args: 0)
    monkeypatch.setattr(
        fused_moe_triton, "moe_align_block_size",
        lambda *args: (torch.zeros(8, dtype=torch.int32), torch.zeros(1), torch.zeros(1)),
    )
    calls = []
    expert_rows = torch.arange(64, dtype=torch.float16).reshape(4, 16)

    def gemm(input, output, *args, **kwargs):
        calls.append(kwargs)
        if kwargs["mul_topk_weights"]:
            return expert_rows
        return output.fill_(1)

    monkeypatch.setattr(marlin, "moe_wna16_marlin_gemm", gemm, raising=False)
    result = marlin.fused_marlin_moe(
        hidden_states=torch.ones(2, 16, dtype=torch.float16),
        w1=torch.zeros(1, 1, 32, dtype=torch.int32),
        w2=torch.zeros(1, 1, 64, dtype=torch.int32),
        w1_scale=torch.ones(1, dtype=torch.float16),
        w2_scale=torch.ones(1, dtype=torch.float16),
        gating_output=torch.ones(2, 1),
        topk_weights=torch.ones(2, 2),
        topk_ids=torch.zeros(2, 2, dtype=torch.int64),
        workspace=torch.zeros(1, dtype=torch.int32),
        activation="relu2",
        is_gated=False,
        no_combine=True,
    )
    assert len(calls) == 2 and calls[-1]["mul_topk_weights"] is True
    assert result.shape == (2, 2, 16)
    assert torch.equal(result, expert_rows.reshape(2, 2, 16))


def test_compat_only_activation_rejects_changed_typed_router(tmp_path, monkeypatch):
    """A plausible K/V routing defect must fail before installing any patch."""
    import shutil

    from sglang.srt.plugins.hook_registry import HookRegistry
    from sglang_qsa_hisparse import fingerprint, patching
    from sglang_qsa_hisparse.errors import FingerprintMismatch
    from sglang_qsa_hisparse.features import Features
    from sglang_qsa_hisparse.patches.model_compat import qsa_attention

    features = Features(model_compat=True)
    specs, attaches = patching.collect(features)
    for row in ("Q08", "Q12"):
        spec = next(spec for spec in specs if spec.row == row)
        assert set(qsa_attention._QSA_READ_DEPENDS) <= set(spec.depends)
    names = {spec.target for spec in specs}
    names.update(dep for spec in (*specs, *attaches) for dep in spec.depends)
    pinned = fingerprint.load_pinned()
    source_root = fingerprint.installed_source_root()
    # Copy precisely the modules activation verifies, keeping the checkout intact.
    for file in {pinned[name]["file"] for name in names}:
        destination = tmp_path / file
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_root / file, destination)
    fingerprint.verify(sorted(names), source_root=tmp_path, pinned=pinned)
    router = tmp_path / pinned["sglang.srt.mem_cache.kv.router.KVLayer.key"]["file"]
    source = router.read_text()
    original = "return self.view.get_key_buffer(self.view_layer_id)"
    assert source.count(original) == 1
    router.write_text(source.replace(original, "return self.view.get_value_buffer(self.view_layer_id)"))

    before_hooks = {name: list(hooks) for name, hooks in HookRegistry._hooks.items()}
    monkeypatch.setattr(patching, "_activated", None)
    monkeypatch.setattr(fingerprint, "installed_source_root", lambda: tmp_path)
    with pytest.raises(FingerprintMismatch, match=r"kv\.router\..*differs from the pinned revision"):
        patching.activate(features)
    assert HookRegistry._hooks == before_hooks
    assert patching._activated is None
