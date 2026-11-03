"""Tree-cache selection for the QSA private host-prefix contract.

v0.5.22 serves ``--disable-radix-cache`` with a disabled UnifiedRadixCache
(#43198-#43201; ``ChunkCache`` is gone) and rejects any hybrid-SSM tree that
does not report ``supports_mamba()``. The QSA offload path builds its host
prefix cache as a subclass of that cache through upstream's own
``create_unified_radix_cache(ctx, cache_class=...)`` (M05), so the selection
must stay narrow: it applies only to the configured QSA offload pool with a
host-prefix budget, and every other hybrid model still gets upstream's tree
and the generic guard. S01's rejections (HiCache or an enabled radix cache,
the Mamba extra buffer) moved from the wrap into the factory.

Port of production's test/qsa_hisparse/test_tree_cache_selection.py
(897286b12a), run with W2's hooks active (M05 around the default factory).
The predicate is the plugin's copy, and the registry reads the runtime bags
through its module globals, so ``_config_bags`` patches them there.
"""

import ast
import inspect
import textwrap
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from lifecycle.pool_fixtures import allocate_rows, tree_slots

from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache import qsa_kv_pool as qsa_kv_pool_module
from sglang.srt.mem_cache import registry
from sglang.srt.mem_cache.allocator.mamba import MambaSlotAllocator
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.kv.state_storage import StateStorage
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool, ReqToTokenPool
from sglang.srt.mem_cache.registry import (
    create_tree_cache,
    default_radix_cache_factory,
)
from sglang_qsa_hisparse.hisparse.prefix_cache import QSAHostPrefixCache
from sglang_qsa_hisparse.patches.hisparse import lifecycle
from sglang_qsa_hisparse.patches.hisparse.lifecycle import (
    qsa_private_host_prefix_active,
)

pytestmark = pytest.mark.usefixtures("w2_patches")

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


def _params(pool, *, extra_buffer=False):
    req_pool = SimpleNamespace(
        req_to_token=torch.zeros((1, 64), dtype=torch.int32),
        enable_mamba_extra_buffer=extra_buffer,
    )
    return CacheInitParams(False, req_pool, _Allocator(pool), 64)


def _context(
    pool,
    *,
    is_hybrid_ssm=True,
    disable_radix_cache=True,
    hierarchical=False,
    extra_buffer=False,
):
    return SimpleNamespace(
        params=_params(pool, extra_buffer=extra_buffer),
        is_hybrid_swa=False,
        full_tokens_per_layer=0,
        is_hybrid_ssm=is_hybrid_ssm,
        is_dsa=False,
        enable_hierarchical_cache=hierarchical,
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
    memory = SimpleNamespace(
        radix_cache_backend=None,
        enable_hierarchical_cache=False,
        hicache_host_memory_mode=None,
        enable_session_radix_cache=False,
        radix_eviction_policy=None,
        enable_unified_cache_external_linker=False,
    )
    monkeypatch.setattr(registry, "get_memory", lambda: memory)
    monkeypatch.setattr(
        registry,
        "get_disagg",
        lambda: SimpleNamespace(disaggregation_decode_retraction_backup=None),
    )
    serving = SimpleNamespace(enable_streaming_session=False)
    monkeypatch.setattr(registry, "get_serving", lambda: serving)


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


def test_factory_builds_the_host_prefix_cache_only_for_the_private_path(monkeypatch):
    monkeypatch.setenv(MODE_ENV, "p2-offload")
    monkeypatch.setenv(PREFIX_ENV, "64")
    ctx = _context(_FakeQsaPool())
    with (
        patch.object(lifecycle, "create_unified_radix_cache", return_value="host") as build,
        patch.object(
            registry,
            "create_unified_radix_cache",
            side_effect=AssertionError("QSA host prefixes need their cache class"),
        ),
    ):
        assert default_radix_cache_factory(ctx) == "host"
    build.assert_called_once_with(ctx, cache_class=QSAHostPrefixCache)

    # The same env without the QSA pool keeps upstream's selection.
    unified = SimpleNamespace(supports_mamba=lambda: True, cache_controller=None)
    with (
        patch.object(registry, "create_unified_radix_cache", return_value=unified),
        patch.object(lifecycle, "create_unified_radix_cache") as build,
    ):
        assert default_radix_cache_factory(_context(_FakeOtherPool())) is unified
    build.assert_not_called()


@pytest.mark.parametrize(
    "setting, message",
    [
        ({"disable_radix_cache": False}, "disabled radix cache without HiCache"),
        ({"hierarchical": True}, "disabled radix cache without HiCache"),
        ({"extra_buffer": True}, "one mutable Mamba slot"),
    ],
)
def test_host_prefixes_reject_unsupported_tree_settings(monkeypatch, setting, message):
    # S01's rejections, now raised where the cache is built.
    monkeypatch.setenv(MODE_ENV, "p2-offload")
    monkeypatch.setenv(PREFIX_ENV, "64")
    with pytest.raises(ValueError, match=message):
        default_radix_cache_factory(_context(_FakeQsaPool(), **setting))


def test_create_tree_cache_builds_the_mamba_owning_host_prefix_cache(
    monkeypatch, published_context
):
    # Factory time: Q03 has attached the runtime (with its host cache) to the
    # kvcache, and the cache binds it; the disabled tree owns the mamba slot.
    monkeypatch.setenv(MODE_ENV, "p2-offload")
    monkeypatch.setenv(PREFIX_ENV, "64")
    kvcache = _FakeQsaPool()
    kvcache.qsa_hisparse = SimpleNamespace(prefix_cache=SimpleNamespace(epoch=0))
    allocator = PagedTokenToKVPoolAllocator(256, 64, torch.uint8, "cpu", kvcache, False)
    pool = HybridReqToTokenPool.__new__(HybridReqToTokenPool)
    ReqToTokenPool.__init__(pool, 1, 256, "cpu", False)
    pool.state = StateStorage.empty("mamba")
    pool.enable_mamba_extra_buffer = False
    pool.mamba_allocator = MambaSlotAllocator(1, "cpu")
    allocator.state = pool.state
    ctx = _context(kvcache)
    ctx.server_args = None
    ctx.params = CacheInitParams(
        True, pool, allocator, 64, tp_cache_group=None
    )

    cache = create_tree_cache(ctx)

    assert type(cache) is QSAHostPrefixCache
    assert cache.runtime is kvcache.qsa_hisparse
    assert cache.host is kvcache.qsa_hisparse.prefix_cache
    assert cache.tp_group is ctx.params.tp_cache_group
    assert cache.disable and cache.supports_mamba()  # Upstream's guard passes.
    assert not cache.load_back_is_cache_owned()

    kvcache.qsa_hisparse.prefix_cache = None
    with pytest.raises(ValueError, match="host prefix cache"):
        create_tree_cache(ctx)


def _calls(function):
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    return [ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)]


def test_the_runtime_is_attached_before_the_tree_cache_is_built():
    # Q03 attaches kvcache.qsa_hisparse when the attention backends are built,
    # inside init_model_worker; build_kv_cache (create_tree_cache) runs after.
    init = _calls(Scheduler.__init__)
    assert init.index("self.init_model_worker") < init.index(
        "kv_cache_builder.build_kv_cache"
    )
    worker = _calls(Scheduler.init_model_worker)
    assert worker.index("self.init_memory_pools") < worker.index(
        "self.init_all_attention_backends"
    )
    assert "self.tp_worker.init_attention_backends" in _calls(
        Scheduler.init_all_attention_backends
    )


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
