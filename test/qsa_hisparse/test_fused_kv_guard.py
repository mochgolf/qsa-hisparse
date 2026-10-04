"""Boundary regressions for the fused #40972 KV fast path on the QSA pool.

The fused prepare kernel writes the new KV row and packs the sparse selection
itself, so it bypasses the QSA offload contract: ``write_locations`` slot
indirection, the ``after_store`` block/index expansion, the lease-aware
``selected`` packing and the FP8 write divide. The offload runtime must keep the
contract path, and a non-unit FP8 scale must keep it even without the runtime.
"""

from types import SimpleNamespace

import torch

from sglang.srt.layers.attention import qwen_sparse_attn_backend as qsa_backend_module
from sglang.srt.layers.attention.qwen_sparse_attn_backend import QwenSparseAttnBackend
from sglang.srt.model_executor.forward_batch_info import ForwardMode


class _CudaLikeTensor:
    def __init__(self, tensor):
        self._tensor = tensor

    @property
    def is_cuda(self):
        return True

    def reshape(self, *shape):
        return _CudaLikeTensor(self._tensor.reshape(*shape))

    def __getitem__(self, item):
        return _CudaLikeTensor(self._tensor[item])

    def __getattr__(self, name):
        return getattr(self._tensor, name)


class _FakeMhaPool:
    is_quantized_kv_cache = False
    dtype = torch.float8_e4m3fn

    def __init__(self, size=4, kv_heads=1, head_dim=256):
        self.k = torch.zeros((size, kv_heads, head_dim)).to(self.dtype)
        self.v = torch.zeros_like(self.k).to(self.dtype)
        self.writes = []

    def get_key_buffer(self, layer_id):
        return self.k

    def get_value_buffer(self, layer_id):
        return self.v

    def set_kv_buffer(self, layer, loc, k, v, k_scale=None, v_scale=None):
        self.writes.append((loc, k, v, k_scale, v_scale))


def _backend(monkeypatch, pool):
    monkeypatch.setattr(qsa_backend_module, "MHATokenToKVPool", _FakeMhaPool)
    backend = QwenSparseAttnBackend()
    backend.token_to_kv_pool = pool
    backend._fused_kv_pool_eligible = backend._fused_kv_eligible()
    return backend


def _decode_batch():
    return SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        out_cache_loc=torch.arange(1, dtype=torch.int32),
        req_pool_indices=torch.tensor([0]),
    )


def test_offload_runtime_keeps_the_contract_path(monkeypatch):
    pool = _FakeMhaPool()
    backend = _backend(monkeypatch, pool)
    # The QSA C4 contract: token_topk 2048 mirrors block_topk 512.
    backend.qsa_profile = SimpleNamespace(block_topk=512)
    backend.compress_ratio = 4
    assert backend._supports_fused_kv_pool(pool) is True
    assert backend._fused_kv_eligible() is True
    # A decode mode would otherwise defer the block expansion to the fused pack.
    assert backend._can_defer_block_expansion(ForwardMode.DECODE) is True

    backend.qsa_hisparse = SimpleNamespace(
        uses_qsa_hisparse_leases=True,
        offloaded=True,
        graph_enabled=False,
        write_locations=lambda loc: loc,
    )
    assert backend._fused_kv_eligible() is False
    backend._fused_kv_pool_eligible = backend._fused_kv_eligible()
    assert backend._can_defer_block_expansion(ForwardMode.DECODE) is False

    # And the helper itself refuses the offload runtime.
    assert (
        backend._try_fused_kv_attention(
            torch.zeros((1, 2, 256), dtype=torch.bfloat16),
            torch.zeros((1, 1, 256), dtype=torch.bfloat16),
            torch.zeros((1, 1, 256), dtype=torch.bfloat16),
            SimpleNamespace(layer_id=0),
            _decode_batch(),
            torch.zeros((1, 1), dtype=torch.int32),
        )
        is None
    )


def test_fused_path_rejects_non_unit_fp8_scales(monkeypatch):
    pool = _FakeMhaPool()
    backend = _backend(monkeypatch, pool)
    monkeypatch.setattr(
        qsa_backend_module, "_resolve_trtllm_sparse_decode", lambda: object()
    )
    layer = SimpleNamespace(layer_id=0, tp_q_head_num=2, head_dim=256, scaling=0.0625)
    q = _CudaLikeTensor(torch.zeros((1, 2, 256), dtype=torch.bfloat16))
    k = torch.zeros((1, 1, 256), dtype=torch.bfloat16)
    v = torch.zeros((1, 1, 256), dtype=torch.bfloat16)
    indices = torch.zeros((1, 1), dtype=torch.int32)

    class _ReachedPastScaleGuard(Exception):
        pass

    layer.k_scale_float = layer.v_scale_float = 1.0
    backend._resolve_metadata = lambda forward_batch: (_ for _ in ()).throw(
        _ReachedPastScaleGuard()
    )
    try:
        backend._try_fused_kv_attention(q, k, v, layer, _decode_batch(), indices)
    except _ReachedPastScaleGuard:
        pass  # unit scales may take the fused path
    else:
        raise AssertionError("unit-scale FP8 pool did not reach the fused path")

    layer.k_scale_float, layer.v_scale_float = 2.0, 0.5
    assert (
        backend._try_fused_kv_attention(q, k, v, layer, _decode_batch(), indices)
        is None
    )


def test_offload_decode_stores_through_the_runtime_hooks(monkeypatch):
    pool = _FakeMhaPool()
    backend = _backend(monkeypatch, pool)
    backend.qsa_hisparse = SimpleNamespace(
        uses_qsa_hisparse_leases=True,
        offloaded=True,
        graph_enabled=False,
        after_store=lambda layer, **kwargs: backend.stores.append(kwargs),
    )
    backend.stores = []
    backend._store_kv = lambda layer, loc, k, v: backend.stores.append(("store", loc))
    backend._forward_paged_attention = lambda *a, **kw: torch.ones((1, 2))
    backend._resolve_metadata = lambda forward_batch: SimpleNamespace(
        is_cuda_graph=False
    )
    layer = SimpleNamespace(layer_id=0, tp_q_head_num=2, head_dim=256)
    batch = _decode_batch()

    output = backend.forward_decode(
        torch.zeros((1, 2, 256), dtype=torch.bfloat16),
        torch.zeros((1, 1, 256), dtype=torch.bfloat16),
        torch.zeros((1, 1, 256), dtype=torch.bfloat16),
        layer,
        batch,
        topk_indices=torch.zeros((1, 1), dtype=torch.int32),
    )

    assert backend.stores[0][0] == "store"
    assert backend.stores[-1] == {}  # after_store(layer) without the graph flag
    assert output.shape == (1, 2)
