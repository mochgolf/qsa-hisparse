"""P2 offload pool sizing for the frozen B2/B4/B8 x 262,144 geometry.

Expected numbers come from the geometry alone (target model config, TP2, FP8
KV, QSA C4 index in BF16, page 64, five raw tail rows per lease), not from
the patched code, and are compared with the patched configurators: C01/C02 for
the fixed bias, logical bytes per token and capacity, K01/K02 for the raw
staging pool, K03 for passing it to ``QSATokenToKVPool``.
"""

import os
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from pools.activation import activated
from pools.pinned import make_model_runner
from sglang.srt.mem_cache import kv_cache_configurator, qsa_kv_pool
from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool
from sglang.srt.model_executor.pool_configurator import DefaultPoolConfigurator
from sglang.srt.runtime_context import get_context, get_parallel
from sglang_qsa_hisparse.patches.hisparse import pools

# Target model (Qwen4ExpForConditionalGeneration): 48 layers, every fourth is
# full attention; 2 KV heads of dim 256, TP2 -> 1 head per rank; FP8 E4M3 KV.
# QSA indexer: 1 KV head of dim 128, BF16 state, compress ratio 4, budget 2048.
CONTEXT = 262144
PAGE = 64
FULL_LAYER_IDS = list(range(3, 48, 4))
LAYERS = len(FULL_LAYER_IDS)  # 12
KV_HEADS, HEAD_DIM, FP8_BYTES = 1, 256, 1
INDEX_KV_HEADS, INDEX_HEAD_DIM, RATIO, BF16_BYTES = 1, 128, 4, 2

RAW_BYTES_PER_TOKEN = LAYERS * KV_HEADS * (HEAD_DIM + HEAD_DIM) * FP8_BYTES
LOGICAL_BYTES_PER_TOKEN = LAYERS * INDEX_KV_HEADS * INDEX_HEAD_DIM * BF16_BYTES // RATIO
# TP2 for get_parallel().override, which validates the whole topology at v0.5.21.
TP2 = dict(tp_size=2, attn_tp_size=2, moe_tp_size=2)


def staging_tokens(requests):
    # One context of raw K/V, plus a padding row and four C4 members per lease.
    return CONTEXT + requests * (1 + RATIO)


def fixed_bytes(requests):
    raw_staging = RAW_BYTES_PER_TOKEN * (staging_tokens(requests) + PAGE)  # + MHA page pad
    compressed_pad = LOGICAL_BYTES_PER_TOKEN * PAGE
    # Pending C4 ring: ``ratio`` BF16 index keys per lease and layer, plus
    # three int64 values per ring slot.
    ring = requests * RATIO * (INDEX_KV_HEADS * INDEX_HEAD_DIM * BF16_BYTES * LAYERS + 3 * 8)
    return raw_staging + compressed_pad + ring


EXPECTED = {
    # requests: (staging tokens, fixed bytes); logical bytes/token is 768.
    2: (262154, 1_611_141_312),
    4: (262164, 1_611_227_520),
    8: (262184, 1_611_399_936),
}


def test_independent_arithmetic_matches_the_table():
    assert (RAW_BYTES_PER_TOKEN, LOGICAL_BYTES_PER_TOKEN) == (6144, 768)
    for requests, (tokens, fixed) in EXPECTED.items():
        assert (staging_tokens(requests), fixed_bytes(requests)) == (tokens, fixed)


def _qsa_config(config):
    config.indexer_n_heads = 4
    config.indexer_kv_heads = INDEX_KV_HEADS
    config.indexer_head_dim = INDEX_HEAD_DIM
    config.indexer_budget = 2048
    config.indexer_compress_ratio = RATIO
    return config


@pytest.fixture
def case(request):
    return SimpleNamespace(addCleanup=request.addfinalizer)


def _p2_offload(requests, **extra):
    return (
        mock.patch.dict(os.environ, {"SGLANG_QSA_HISPARSE_V3": "p2-offload"}),
        get_context().override_server_args(
            max_running_requests=requests,
            max_total_tokens=requests * CONTEXT,
            page_size=PAGE,
            **extra,
        ),
        get_parallel().override(**TP2),
    )


# C01, C02 ---------------------------------------------------------------------


def _pool_configurator_input(case, requests):
    kvc = make_model_runner(
        case, head_dim=HEAD_DIM, v_head_dim=HEAD_DIM, num_layers=48, page_size=PAGE,
        max_running_requests=requests,
    )
    model_config = kvc.model_config
    model_config.get_num_kv_heads = lambda tp_size, dcp_size=1: 2 // tp_size
    # Hybrid (Mamba-ish) model: the pool covers only the full-attention layers.
    model_config.linear_attn_registry_result = (
        None, SimpleNamespace(full_attention_layer_ids=FULL_LAYER_IDS)
    )
    model_config.hf_text_config = _qsa_config(model_config.hf_config)
    kvc.kv_cache_dtype = torch.float8_e4m3fn
    kvc.kv_cache_dtype_str = "fp8_e4m3"
    return kvc


@pytest.mark.parametrize("requests", sorted(EXPECTED))
def test_configurator_prices_fixed_staging(case, requests):
    kvc = _pool_configurator_input(case, requests)
    logical = requests * CONTEXT
    env, server_args, parallel = _p2_offload(requests)
    with env, server_args, parallel, activated("C01", "C02"):
        cfg = DefaultPoolConfigurator(kvc)
        assert cfg._bias == fixed_bytes(requests) == EXPECTED[requests][1]
        assert cfg._cell_size == LOGICAL_BYTES_PER_TOKEN
        required = fixed_bytes(requests) + LOGICAL_BYTES_PER_TOKEN * logical
        assert cfg.calculate_pool_sizes(required, PAGE).max_total_num_tokens == logical
        assert cfg.calculate_pool_sizes(required - 1, PAGE).max_total_num_tokens == logical - PAGE
        one_page = fixed_bytes(requests) + LOGICAL_BYTES_PER_TOKEN * PAGE
        assert cfg.calculate_pool_sizes(one_page, PAGE).max_total_num_tokens == PAGE
        with pytest.raises(RuntimeError, match="Not enough memory"):
            cfg.calculate_pool_sizes(one_page - 1, PAGE)

    with get_parallel().override(**TP2), activated("C01", "C02"):
        plain = DefaultPoolConfigurator(kvc)  # SGLANG_QSA_HISPARSE_V3 unset
    assert plain._bias == 0
    assert plain._cell_size == RAW_BYTES_PER_TOKEN + LOGICAL_BYTES_PER_TOKEN


@pytest.mark.parametrize(
    "server_args",
    [
        dict(max_running_requests=3, max_total_tokens=3 * CONTEXT),
        dict(max_running_requests=4, max_total_tokens=4 * CONTEXT - PAGE),
        dict(max_running_requests=4, max_total_tokens=4 * CONTEXT, page_size=32),
    ],
)
def test_configurator_requires_bounded_capacity(case, server_args):
    kvc = _pool_configurator_input(case, 4)
    with (
        mock.patch.dict(os.environ, {"SGLANG_QSA_HISPARSE_V3": "p2-offload"}),
        get_context().override_server_args(**{"page_size": PAGE, **server_args}),
        get_parallel().override(**TP2),
        activated("C01", "C02"),
        pytest.raises(ValueError, match="bounded logical capacity"),
    ):
        DefaultPoolConfigurator(kvc)


# K01, K02 ---------------------------------------------------------------------


class _Recorded:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _StagingPool(_Recorded):
    pass


class _QSAPool(_Recorded):
    pass


def _build_kv_pool(requests, max_total_num_tokens):
    """Run the pinned ``_build_token_to_kv_pool`` with recording pool classes."""
    kvc = KVCacheConfigurator.__new__(KVCacheConfigurator)
    kvc.is_draft_worker = False
    kvc.is_hybrid_swa = False
    kvc.use_mla_backend = False
    kvc.post_capture_kv_active = False
    kvc.mambaish_config = SimpleNamespace(full_attention_layer_ids=FULL_LAYER_IDS)
    kvc.layer_info = SimpleNamespace(start_layer=0, end_layer=48, num_effective_layers=48)
    kvc.kv_cache_dtype = torch.float8_e4m3fn
    kvc.kv_cache_dtype_str = "fp8_e4m3"
    kvc.device = "cpu"
    kvc.model_config = SimpleNamespace(
        hf_config=_qsa_config(SimpleNamespace(architectures=["Qwen4ExpForConditionalGeneration"])),
        head_dim=HEAD_DIM,
        get_num_kv_heads=lambda tp_size, dcp_size=1: 2 // tp_size,
    )
    req_to_token_pool = SimpleNamespace(
        req_to_token=torch.zeros(requests + 1, 1), mamba_pool="mamba-pool"
    )
    sizes = SimpleNamespace(
        max_total_num_tokens=max_total_num_tokens, max_running_requests=requests
    )
    with (
        # CUDA has no KV quant method for plain FP8 (the CPU platform adds one).
        mock.patch.object(
            KVCacheConfigurator, "_build_mha_quant_method", lambda self, *, num_layers: None
        ),
        mock.patch.object(kv_cache_configurator, "MHATokenToKVPool", _StagingPool),
        mock.patch.object(pools, "MHATokenToKVPool", _StagingPool),
        mock.patch.object(qsa_kv_pool, "QSATokenToKVPool", _QSAPool),
    ):
        return kvc._build_token_to_kv_pool(
            sizes=sizes,
            is_dsa_model=False,
            is_dsv4_model=False,
            req_to_token_pool=req_to_token_pool,
        )


@pytest.mark.parametrize("requests", sorted(EXPECTED))
def test_raw_staging_pool_matches_priced_bytes(requests):
    env, server_args, parallel = _p2_offload(requests)
    with env, server_args, parallel, activated("K01", "K02"):
        pool = _build_kv_pool(requests, requests * CONTEXT)
    assert isinstance(pool, _QSAPool)
    staging = pool.kwargs["full_kv_pool"]
    assert isinstance(staging, _StagingPool)
    assert staging.kwargs == dict(
        size=EXPECTED[requests][0],
        page_size=PAGE,
        dtype=torch.float8_e4m3fn,
        head_num=KV_HEADS,
        head_dim=HEAD_DIM,
        layer_num=LAYERS,
        device="cpu",
        enable_memory_saver=False,
    )
    # The priced raw term of C01's fixed bias is exactly this pool.
    raw = staging.kwargs
    assert (raw["size"] + raw["page_size"]) * raw["layer_num"] * raw["head_num"] * (
        2 * raw["head_dim"]
    ) * FP8_BYTES == RAW_BYTES_PER_TOKEN * (staging_tokens(requests) + PAGE)
    assert pool.kwargs["size"] == requests * CONTEXT
    assert pool.kwargs["page_size"] == PAGE
    assert pool.kwargs["full_attention_layer_ids"] == FULL_LAYER_IDS
    assert pool.kwargs["full_kv_pool_class"] is _StagingPool
    assert pool.kwargs["mamba_pool"] == "mamba-pool"
    assert (
        pool.kwargs["qsa_index_kv_heads"],
        pool.kwargs["qsa_index_head_dim"],
        pool.kwargs["qsa_compress_ratio"],
        pool.kwargs["qsa_token_topk"],
        pool.kwargs["num_request_slots"],
    ) == (INDEX_KV_HEADS, INDEX_HEAD_DIM, RATIO, 2048, requests + 1)


def test_raw_staging_requires_bounded_capacity():
    env, server_args, parallel = _p2_offload(4)
    with env, server_args, parallel, activated("K01", "K02"):
        with pytest.raises(ValueError, match="bounded logical capacity"):
            _build_kv_pool(4, 4 * CONTEXT - PAGE)
        with pytest.raises(ValueError, match="bounded logical capacity"):
            _build_kv_pool(3, 3 * CONTEXT)


def test_no_raw_staging_without_p2_offload():
    with (
        mock.patch.dict(os.environ, {"SGLANG_QSA_HISPARSE_V3": "p2-resident"}),
        get_context().override_server_args(page_size=PAGE),
        get_parallel().override(**TP2),
        activated("K01", "K02"),
    ):
        pool = _build_kv_pool(4, 4 * CONTEXT)
    assert isinstance(pool, _QSAPool)
    assert "full_kv_pool" not in pool.kwargs
    assert pool.kwargs["size"] == 4 * CONTEXT


def test_adapter_requires_the_enclosing_pool_build():
    kvc = KVCacheConfigurator.__new__(KVCacheConfigurator)
    with activated("K01", "K02"), pytest.raises(LookupError):
        kvc._build_hybrid_linear_kv_pool(
            max_total_num_tokens=CONTEXT, req_to_token_pool=None, mha_pool_class=object
        )


# K03 --------------------------------------------------------------------------


def _small_qsa_pool(**extra):
    return QSATokenToKVPool(
        size=256, dtype=torch.bfloat16, page_size=PAGE, head_num=1, head_dim=64,
        full_attention_layer_ids=[3, 7], device="cpu", mamba_pool=None,
        qsa_index_kv_heads=1, qsa_index_head_dim=64, qsa_compress_ratio=RATIO,
        qsa_token_topk=2048, num_request_slots=3, **extra,
    )


def test_qsa_pool_uses_the_given_full_kv_pool():
    with get_context().override_server_args(page_size=PAGE):
        staging = MHATokenToKVPool(
            size=128, page_size=PAGE, dtype=torch.bfloat16, head_num=1, head_dim=64,
            layer_num=2, device="cpu", enable_memory_saver=False,
        )
        with pytest.raises(TypeError, match="full_kv_pool"):
            _small_qsa_pool(full_kv_pool=staging)  # Pinned signature.
        with activated("K03"):
            given = _small_qsa_pool(full_kv_pool=staging)
            default = _small_qsa_pool()
        with pytest.raises(TypeError, match="full_kv_pool"):
            _small_qsa_pool(full_kv_pool=staging)  # Hooks removed again.
    assert given.full_kv_pool is staging
    assert given.size == 256
    assert type(default.full_kv_pool) is MHATokenToKVPool
    assert default.full_kv_pool is not staging
    assert default.full_kv_pool.size == 256
