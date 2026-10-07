"""W5 hooks on the pinned SGLang: activation, target-model scope, routing.

Rule 9: generic hooks run the fork behavior only when
``scope.target_model_active()``; tests fix its answer (``target`` fixture or
monkeypatch).
"""

import inspect
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sglang.srt.layers import hc_mix_triton
from sglang.srt.layers import hyperconnection as pinned_hyperconnection
from sglang.srt.layers.hyperconnection import GatedResidual
from sglang.srt.models import qwen4_exp as pinned_qwen4_exp
from sglang_qsa_hisparse import scope
from sglang_qsa_hisparse.kernels import hc_mix
from sglang_qsa_hisparse.patches.model_compat import (
    hyperconnection,
    marlin,
    quantization,
    qwen4_exp,
)

FUSED_MARLIN_MOE = "sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe.fused_marlin_moe"
REPLACE_COPIES = {
    FUSED_MARLIN_MOE: marlin.fused_marlin_moe_op,
    "sglang.srt.hardware_backend.gpu.quantization.gptq_kernels.GPTQMarlinMoEKernel"
    ".process_weights_after_loading": (
        quantization.ForkGPTQMarlinMoEKernel.process_weights_after_loading
    ),
    "sglang.srt.layers.quantization.gptq.schemes.gptq_moe.GPTQMarlinMoEScheme"
    ".create_weights": quantization.ForkGPTQMarlinMoEScheme.create_weights,
    "sglang.srt.layers.quantization.auto_round.AutoRoundConfig.apply_gptq_quant_layer": (
        quantization.ForkAutoRoundConfig.apply_gptq_quant_layer
    ),
    "sglang.srt.layers.hc_mix_triton.fused_hc_mix_supported": hc_mix.fused_hc_mix_supported,
    "sglang.srt.models.qwen4_exp.Qwen4ExpNGramEmbedding.__init__": (
        qwen4_exp.ForkQwen4ExpNGramEmbedding.__init__
    ),
    "sglang.srt.models.qwen4_exp.Qwen4ExpPinnedHostEmbedding.__init__": (
        qwen4_exp.ForkQwen4ExpPinnedHostEmbedding.__init__
    ),
    "sglang.srt.models.qwen4_exp.Qwen4ExpPinnedHostEmbedding.gather": (
        qwen4_exp.ForkQwen4ExpPinnedHostEmbedding.gather
    ),
    "sglang.srt.models.qwen4_exp.Qwen4ExpForConditionalGeneration.load_weights": (
        qwen4_exp.ForkQwen4ExpForConditionalGeneration.load_weights
    ),
}


def test_w5_rows_activate_on_the_pin(compat):
    assert {spec.row for spec in compat.specs} == {
        "J03", "Z01", "Z02", "Z03", "Z04", "H05", "H06", "H08", "E03", "E06", "E07", "E08"
    }
    # The pinned op itself is untouched; only module bindings are patched.
    assert torch.ops.sglang.fused_marlin_moe is compat.originals[FUSED_MARLIN_MOE]


def test_every_replace_is_scoped_between_the_pinned_original_and_its_copy(compat):
    replaces = {s.target: s.hook for s in compat.specs if s.hook_type == "replace"}
    assert replaces.keys() == REPLACE_COPIES.keys()
    for target, hook in replaces.items():
        assert inspect.getclosurevars(hook).nonlocals == {
            "original": compat.originals[target],
            "copy": REPLACE_COPIES[target],
        }, target


def test_scoped_never_guesses(monkeypatch):
    calls = []
    dispatch = quantization.scoped(
        lambda *a, **k: calls.append(("original", a, k)),
        lambda *a, **k: calls.append(("copy", a, k)),
    )
    for active in (False, True):
        monkeypatch.setattr(scope, "target_model_active", lambda: active)
        dispatch(1, key=2)
    assert calls == [("original", (1,), {"key": 2}), ("copy", (1,), {"key": 2})]

    def undecidable():
        raise scope.ScopeUndecidable("no published model configuration")

    monkeypatch.setattr(scope, "target_model_active", undecidable)
    with pytest.raises(scope.ScopeUndecidable):
        dispatch(1)
    assert len(calls) == 2


# Z04 -------------------------------------------------------------------------


def _moe_layer(flagged):
    layer = SimpleNamespace()
    if flagged:
        layer._marlin_g64_expand_scales = True
    return layer


def _doubled_rows(weight):
    # Each g128 group row becomes two identical, adjacent g64 group rows.
    return torch.stack([weight, weight], dim=1).reshape(-1, *weight.shape[1:])


@pytest.mark.parametrize("suffix", ["w13_scales", "w2_qzeros"])
def test_g64_layers_expand_scales_and_zeros_positionally_or_by_keyword(target, suffix):
    weight = torch.arange(12.0).reshape(3, 4)
    layer, param = _moe_layer(True), object()
    name = f"experts.{suffix}"
    args, kwargs = quantization._expand_g64_marlin_scales(
        layer, param, weight, name, "w1", 3
    )
    assert args[:2] == (layer, param) and args[3:] == (name, "w1", 3) and not kwargs
    assert torch.equal(args[2], _doubled_rows(weight))

    args, kwargs = quantization._expand_g64_marlin_scales(
        layer, param, loaded_weight=weight, weight_name=name, shard_id="w1", expert_id=3
    )
    assert args == (layer, param)
    assert kwargs.keys() == {"loaded_weight", "weight_name", "shard_id", "expert_id"}
    assert torch.equal(kwargs["loaded_weight"], _doubled_rows(weight))


def test_g64_expansion_leaves_other_loads_alone(target, monkeypatch):
    weight = torch.ones(2, 2)
    expand = quantization._expand_g64_marlin_scales
    assert expand(_moe_layer(False), None, weight, "experts.w13_scales", "w1", 0) is None
    assert expand(_moe_layer(True), None, weight, "experts.w13_qweight", "w1", 0) is None
    monkeypatch.setattr(scope, "target_model_active", lambda: False)
    assert expand(_moe_layer(True), None, weight, "experts.w13_scales", "w1", 0) is None


# H06 / H08 -------------------------------------------------------------------


def _recorder(name, calls, result=None):
    def record(*args, **kwargs):
        calls.append((name, args, kwargs))
        return result

    return record


def test_stable_hc_mix_launch_runs_the_fork_copy_only_for_the_target(monkeypatch):
    calls = []
    monkeypatch.setattr(hc_mix, "fused_hc_mix", _recorder("fork", calls))
    original = _recorder("pinned", calls)
    for active in (True, False):
        monkeypatch.setattr(scope, "target_model_active", lambda: active)
        hyperconnection._fused_hc_mix(original, "x", "down", "up", 4, 8, stable=True)
        hyperconnection._fused_hc_mix(original, "x", "down", "up", 4, 8)
    stable, plain = ("x", "down", "up", 4, 8), {}
    assert calls == [
        ("fork", stable, {"stable": True}),
        ("pinned", stable, plain),
        ("pinned", stable, {"stable": True}),
        ("pinned", stable, plain),
    ]


def test_copied_mix_reaches_the_patched_hc_mix_functions(compat):
    # Inventory 6, G3: the copy's globals are rebound to the H05/H06 hooks.
    assert hyperconnection.fused_hc_mix is hc_mix_triton.fused_hc_mix
    assert hyperconnection.fused_hc_mix_supported is hc_mix_triton.fused_hc_mix_supported
    assert hyperconnection.fused_hc_mix_supported is not hc_mix.fused_hc_mix_supported
    assert pinned_hyperconnection.fused_hc_mix_supported is hc_mix_triton.fused_hc_mix_supported


def _mix_reference(x, w_down, w_up, hc, hs):
    # The eager chain of GatedResidual._mix_compute.
    t = torch.nn.functional.silu(x @ w_down.t() / hc)
    gate = torch.sigmoid(t @ w_up.t()).unflatten(-1, (hc, hs))
    return (gate * x.unflatten(-1, (hc, hs))).mean(dim=-2)


def _gated_residual():
    layer = GatedResidual.__new__(GatedResidual)
    nn.Module.__init__(layer)
    layer.config = SimpleNamespace(hc_per_branch_norm=True)
    layer.hc_count, layer.hidden_size, layer.params_dtype = 2, 4, torch.float32
    layer.hc_norm = lambda x: 2 * x
    gen = torch.Generator().manual_seed(0)
    layer.input_mix_weight_down = SimpleNamespace(weight=torch.randn(3, 8, generator=gen))
    layer.input_mix_weight_up = SimpleNamespace(weight=torch.randn(8, 3, generator=gen))
    layer._jit_mix_ok = False
    layer._mix_compute = _mix_reference
    return layer


@pytest.mark.parametrize(
    "active, deterministic, stable, fork_path",
    [
        (False, True, None, False),  # Rule 9: other models keep upstream mix.
        (True, False, None, False),
        (True, True, None, True),  # E02: stable defaults to _stable_hc().
        (True, True, False, False),
        (True, False, True, True),
    ],
)
def test_gated_residual_mix_routes_stable_calls_to_the_fork_copy(
    compat, monkeypatch, active, deterministic, stable, fork_path
):
    monkeypatch.setattr(scope, "target_model_active", lambda: active)
    config = SimpleNamespace(
        deterministic=SimpleNamespace(enable_deterministic_inference=deterministic)
    )
    monkeypatch.setattr(qwen4_exp, "get_exec", lambda: config)
    calls = []
    # The pinned mix and the fork copy look these up in their own modules.
    monkeypatch.setattr(
        pinned_hyperconnection, "fused_hc_mix_supported", _recorder("pinned", calls, False)
    )
    monkeypatch.setattr(
        hyperconnection, "fused_hc_mix_supported", _recorder("fork", calls, False)
    )
    layer = _gated_residual()
    x = torch.randn(3, 8, generator=torch.Generator().manual_seed(1))
    kwargs = {} if stable is None else {"stable": stable}
    mixed, (residual, normed) = layer.mix(x, **kwargs)

    assert [(name, kw) for name, _, kw in calls] == [
        ("fork", {"stable": True}) if fork_path else ("pinned", {})
    ]
    assert residual is x and torch.equal(normed, 2 * x)
    w_down, w_up = layer.input_mix_weight_down.weight, layer.input_mix_weight_up.weight
    assert torch.equal(mixed, _mix_reference(2 * x, w_down, w_up, 2, 4))


# E08 -------------------------------------------------------------------------


def _int8_row_ple_model():
    """A Qwen4Exp model holding one offloaded int8_row PLE table of 8 rows."""
    Model = pinned_qwen4_exp.Qwen4ExpForConditionalGeneration
    table = pinned_qwen4_exp.Qwen4ExpPinnedHostEmbedding.__new__(
        pinned_qwen4_exp.Qwen4ExpPinnedHostEmbedding
    )
    nn.Module.__init__(table)
    table.org_vocab_size = table.num_org_embeddings_per_partition = 8
    table.shard_indices = SimpleNamespace(org_vocab_start_index=0, org_vocab_end_index=8)
    table.ple_row_scale_mode = True
    table.weight = nn.Parameter(torch.zeros(8, 4, dtype=torch.int8), requires_grad=False)
    nan = torch.full((8,), float("nan"), dtype=torch.bfloat16)
    table.register_buffer("row_scale", nan, persistent=False)
    ple = pinned_qwen4_exp.Qwen4ExpNGramEmbedding.__new__(
        pinned_qwen4_exp.Qwen4ExpNGramEmbedding
    )
    nn.Module.__init__(ple)
    ple.ngram_embedding = table
    model = Model.__new__(Model)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        split_ngram_parts=2, tie_word_embeddings=False, encoder_only=False
    )
    model.language_model_only = False
    model.ple = ple
    return model, table


def _shards(rows, scales, *, dtype=torch.int8):
    weights = []
    for shard in range(2):  # split_ngram_parts=2: four rows per shard.
        part = slice(4 * shard, 4 * shard + 4)
        weights.append((f"ple.ngram_embedding.shard_{shard}.weight", rows[part].to(dtype)))
        if scales is not None and shard < len(scales) // 4:
            weights.append((f"ple.ngram_embedding.shard_{shard}.row_scale", scales[part]))
    return weights


def test_int8_row_ple_shards_load_rows_and_row_scales(compat, target):
    model, table = _int8_row_ple_model()
    rows = torch.arange(32, dtype=torch.int8).reshape(8, 4) - 16
    scales = torch.linspace(0.5, 2.0, 8)
    loaded = model.load_weights(_shards(rows, scales))
    assert torch.equal(table.weight, rows)
    assert torch.equal(table.row_scale, scales.bfloat16())
    assert {"ple.ngram_embedding.weight", "ple.ngram_embedding.row_scale"} <= loaded


def test_int8_row_ple_load_rejects_unloaded_row_scales(compat, target):
    model, _ = _int8_row_ple_model()
    rows = torch.zeros(8, 4, dtype=torch.int8)
    with pytest.raises(ValueError, match="row_scale shards missing for ple: 4 of 8"):
        model.load_weights(_shards(rows, torch.ones(4)))


def test_int8_ple_storage_mismatch_fails_only_for_the_target(compat, monkeypatch):
    rows = torch.ones(8, 4)
    monkeypatch.setattr(scope, "target_model_active", lambda: True)
    model, _ = _int8_row_ple_model()
    with pytest.raises(ValueError, match="storage is int8 but checkpoint shards"):
        model.load_weights(_shards(rows, torch.ones(8), dtype=torch.bfloat16))
    # Other models keep upstream loading, which has no int8 consistency checks.
    monkeypatch.setattr(scope, "target_model_active", lambda: False)
    model, table = _int8_row_ple_model()
    model.load_weights(_shards(rows, None, dtype=torch.bfloat16))
    assert torch.equal(table.weight, rows.to(torch.int8))


# E07 + E04 (GPU) -------------------------------------------------------------


@pytest.mark.gpu
def test_int8_row_scale_gather_matches_an_independent_reference(compat, target):
    from sglang.srt.layers.quantization.unquant import UnquantizedEmbeddingMethod

    rows, dim, start = 6, 8, 10
    gen = torch.Generator().manual_seed(0)
    table = torch.randint(-127, 128, (rows, dim), dtype=torch.int8, generator=gen)
    scale = (torch.rand(rows, generator=gen) + 0.5).bfloat16()
    embedding = SimpleNamespace(
        quant_method=UnquantizedEmbeddingMethod(),
        weight=nn.Parameter(table.clone(), requires_grad=False),
        weight_scale=torch.ones(1, dtype=torch.bfloat16),
        ple_row_scale_mode=True,
        num_added_embeddings=0,
        quant_config=None,
        enable_tp=False,
        use_attn_tp_group=False,
        tp_size=1,
        num_embeddings=rows,
        org_vocab_size=rows,
        padding_size=1,
        use_presharded_weights=False,
        org_vocab_size_padded=rows,
        num_embeddings_padded=rows,
        shard_indices=SimpleNamespace(
            org_vocab_start_index=start, org_vocab_end_index=start + rows
        ),
        embedding_dim=dim,
        num_embeddings_per_partition=rows,
        num_org_embeddings_per_partition=rows,
        num_added_embeddings_per_partition=0,
    )
    host = pinned_qwen4_exp.Qwen4ExpPinnedHostEmbedding(embedding)
    assert host.weight.is_pinned() and host.row_scale.is_pinned()
    assert bool(torch.isnan(host.row_scale).all())
    host.weight.data.copy_(table)
    host.row_scale.copy_(scale)
    ids = torch.tensor([[10, 15, 3], [12, 16, 11]])
    out = host.gather(ids.cuda())

    local = (ids - start).clamp(0, rows - 1)
    expected = (table[local].float() * scale[local].float()[..., None]).bfloat16()
    expected[(ids < start) | (ids >= start + rows)] = 0
    assert torch.equal(out.cpu(), expected)
