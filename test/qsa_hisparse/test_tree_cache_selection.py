"""Tree-cache selection for the QSA private host-prefix contract.

Upstream #42354 routes disabled-radix hybrid-SSM models to UnifiedRadixCache and
rejects any other tree that does not report ``supports_mamba()``. The QSA
offload path keeps its private ChunkCache adapter (logical pages, host
snapshots, lease-aware writes) and releases Mamba slots through
``release_kv_cache``, so the selection and the Mamba exemption must stay narrow:
they apply only to the configured QSA offload pool, and every other hybrid model
still fails the generic guard.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.mem_cache import qsa_kv_pool as qsa_kv_pool_module
from sglang.srt.mem_cache import registry
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.registry import (
    create_tree_cache,
    default_radix_cache_factory,
    qsa_private_host_prefix_active,
)

PREFIX_ENV = "SGLANG_QSA_HISPARSE_PREFIX_CACHE_MB"
MODE_ENV = "SGLANG_QSA_HISPARSE_V3"


class _FakeQsaPool:
    dtype = torch.float8_e4m3fn


class _FakeOtherPool:
    dtype = torch.bfloat16


class _Allocator:
    page_size = 64
    device = torch.device("cpu")

    def __init__(self, pool):
        self._pool = pool

    def get_kvcache(self):
        return self._pool


def _params(pool):
    req_pool = SimpleNamespace(req_to_token=torch.zeros((1, 64), dtype=torch.int32))
    return CacheInitParams(False, req_pool, _Allocator(pool), 64)


def _context(pool, *, is_hybrid_ssm=True, disable_radix_cache=True):
    return SimpleNamespace(
        params=_params(pool),
        is_hybrid_swa=False,
        full_tokens_per_layer=0,
        is_hybrid_ssm=is_hybrid_ssm,
        is_dsa=False,
        enable_hierarchical_cache=False,
        disable_radix_cache=disable_radix_cache,
        effective_chunked_prefill_size=2048,
    )


@pytest.fixture(autouse=True)
def _qsa_pool_type(monkeypatch):
    # ``qsa_private_host_prefix_active`` imports the pool class at call time.
    monkeypatch.setattr(qsa_kv_pool_module, "QSATokenToKVPool", _FakeQsaPool)
    monkeypatch.delenv(MODE_ENV, raising=False)
    monkeypatch.delenv(PREFIX_ENV, raising=False)


@pytest.fixture(autouse=True)
def _config_bags(monkeypatch):
    # The registry reads the published runtime bags; this unit test pins the
    # defaults that keep every optional cache path off.
    monkeypatch.setattr(
        registry,
        "get_memory",
        lambda: SimpleNamespace(
            radix_cache_backend=None,
            enable_hierarchical_cache=False,
            hicache_host_memory_mode=None,
            enable_session_radix_cache=False,
            radix_eviction_policy=None,
        ),
    )
    monkeypatch.setattr(
        registry,
        "get_disagg",
        lambda: SimpleNamespace(disaggregation_decode_retraction_backup=None),
    )
    monkeypatch.setattr(
        registry, "get_serving", lambda: SimpleNamespace(enable_streaming_session=False)
    )


def test_predicate_needs_mode_budget_and_pool(monkeypatch):
    monkeypatch.setenv(MODE_ENV, "p2-offload")
    monkeypatch.setenv(PREFIX_ENV, "64")
    assert qsa_private_host_prefix_active(_params(_FakeQsaPool())) is True

    monkeypatch.delenv(MODE_ENV)
    assert qsa_private_host_prefix_active(_params(_FakeQsaPool())) is False

    monkeypatch.setenv(MODE_ENV, "p2-offload")
    monkeypatch.setenv(PREFIX_ENV, "0")
    assert qsa_private_host_prefix_active(_params(_FakeQsaPool())) is False

    monkeypatch.setenv(PREFIX_ENV, "64")
    assert qsa_private_host_prefix_active(_params(_FakeOtherPool())) is False

    # A single-request mode never installs the host-prefix adapter.
    monkeypatch.setenv(MODE_ENV, "single-offload")
    assert qsa_private_host_prefix_active(_params(_FakeQsaPool())) is False


def test_factory_selects_chunk_cache_only_for_the_private_path(monkeypatch):
    monkeypatch.setenv(MODE_ENV, "p2-offload")
    monkeypatch.setenv(PREFIX_ENV, "64")
    with patch.object(
        registry,
        "create_unified_radix_cache",
        side_effect=AssertionError("QSA host prefixes must not take the unified tree"),
    ):
        cache = default_radix_cache_factory(_context(_FakeQsaPool()))
    assert type(cache).__name__ == "ChunkCache"

    # The same env without the QSA pool keeps upstream's selection.
    unified = SimpleNamespace(supports_mamba=lambda: True, cache_controller=None)
    with patch.object(registry, "create_unified_radix_cache", return_value=unified):
        assert default_radix_cache_factory(_context(_FakeOtherPool())) is unified


def test_create_tree_cache_allows_the_private_mamba_owner(monkeypatch):
    monkeypatch.setenv(MODE_ENV, "p2-offload")
    monkeypatch.setenv(PREFIX_ENV, "64")
    cache = create_tree_cache(_context(_FakeQsaPool()))
    assert type(cache).__name__ == "ChunkCache"
    assert cache.supports_mamba() is False  # Mamba stays with release_kv_cache


def test_create_tree_cache_still_rejects_other_hybrid_ssm_caches(monkeypatch):
    monkeypatch.setenv(MODE_ENV, "p2-offload")
    monkeypatch.setenv(PREFIX_ENV, "64")
    unified = SimpleNamespace(supports_mamba=lambda: False, cache_controller=None)
    with patch.object(registry, "create_unified_radix_cache", return_value=unified):
        with pytest.raises(NotImplementedError, match="mamba state"):
            create_tree_cache(_context(_FakeOtherPool()))


def test_offload_without_host_prefixes_keeps_the_unified_tree(monkeypatch):
    monkeypatch.setenv(MODE_ENV, "p2-offload")
    monkeypatch.setenv(PREFIX_ENV, "0")
    unified = SimpleNamespace(supports_mamba=lambda: True, cache_controller=None)
    with patch.object(registry, "create_unified_radix_cache", return_value=unified):
        assert create_tree_cache(_context(_FakeQsaPool())) is unified
