from types import SimpleNamespace

import torch
from torch import nn

from sglang.srt.models import qwen4_exp


def test_deterministic_inference_uses_stable_hc(monkeypatch):
    config = SimpleNamespace(
        deterministic=SimpleNamespace(enable_deterministic_inference=True)
    )
    monkeypatch.setattr(qwen4_exp, "get_exec", lambda: config)
    assert qwen4_exp._stable_hc()


def test_offloaded_int8_row_ple_constructs_table_on_meta(monkeypatch):
    devices = []

    class FakeEmbedding(nn.Module):
        def __init__(self, *_args, params_dtype, **_kwargs):
            super().__init__()
            devices.append(torch.empty(0).device.type)
            self.weight = nn.Parameter(
                torch.empty(1, dtype=params_dtype), requires_grad=False
            )

    monkeypatch.setattr(qwen4_exp, "VocabParallelEmbedding", FakeEmbedding)
    config = SimpleNamespace(
        ngram_size=2,
        heads_per_ngram=1,
        vocab_size=32,
        ngram_vocab_size_base=31,
        make_ngram_vocab_size_divisible_by=8,
        eos_token_id=2,
        seed=1234,
        ple_embedding_dtype="int8_row",
        ple_offload_embedding=True,
    )
    embedding = qwen4_exp.Qwen4ExpNGramEmbedding(config, embedding_dim=4)

    assert devices == ["meta"]
    assert embedding.ngram_embedding.weight.dtype == torch.int8
    assert embedding.ple_row_scale_mode
