"""Ported from the reference's test/qsa_hisparse/test_model_compatibility.py
(production 897286b12a).

Assertions unchanged. ``_stable_hc`` (E01) and the copied
``Qwen4ExpNGramEmbedding.__init__`` (E03) resolve their globals in the plugin
module, so the fork's monkeypatches of ``qwen4_exp.get_exec`` and
``qwen4_exp.VocabParallelEmbedding`` target that module; the embedding is
still constructed through the pinned class with W5's rows activated for the
target model. The constructor also wraps an offloaded table in
``Qwen4ExpPinnedHostEmbedding``, which the reference stubs with a
pass-through; the stub targets the plugin module too.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sglang.srt.models import qwen4_exp
from sglang_qsa_hisparse.patches.model_compat import qwen4_exp as plugin_qwen4_exp


def test_deterministic_inference_uses_stable_hc(monkeypatch):
    config = SimpleNamespace(
        deterministic=SimpleNamespace(enable_deterministic_inference=True)
    )
    monkeypatch.setattr(plugin_qwen4_exp, "get_exec", lambda: config)
    assert plugin_qwen4_exp._stable_hc()


@pytest.mark.parametrize(
    "storage_dtype, expected_dtype",
    [
        ("int8_row", torch.int8),
        ("int8", torch.int8),
        ("float8_e4m3fn", torch.float8_e4m3fn),
        ("bfloat16", torch.bfloat16),
    ],
)
def test_offloaded_ple_preserves_storage_dtype_on_meta(
    compat, target, monkeypatch, storage_dtype, expected_dtype
):
    devices = []

    class FakeEmbedding(nn.Module):
        def __init__(self, *_args, params_dtype, **_kwargs):
            super().__init__()
            devices.append(torch.empty(0).device.type)
            self.weight = nn.Parameter(
                torch.empty(1, dtype=params_dtype), requires_grad=False
            )

    monkeypatch.setattr(plugin_qwen4_exp, "VocabParallelEmbedding", FakeEmbedding)
    # This case checks the construction dtype and device; the real offload
    # wrapper's storage and gather behavior has separate tests.
    monkeypatch.setattr(
        plugin_qwen4_exp, "Qwen4ExpPinnedHostEmbedding", lambda table, **_: table
    )
    config = SimpleNamespace(
        ngram_size=2,
        heads_per_ngram=1,
        vocab_size=32,
        ngram_vocab_size_base=31,
        make_ngram_vocab_size_divisible_by=8,
        eos_token_id=2,
        seed=1234,
        ple_embedding_dtype=storage_dtype,
        ple_offload_embedding=True,
    )
    embedding = qwen4_exp.Qwen4ExpNGramEmbedding(config, embedding_dim=4)

    assert devices == ["meta"]
    assert embedding.ngram_embedding.weight.dtype == expected_dtype
    assert embedding.ple_row_scale_mode == (storage_dtype == "int8_row")
