# Reference (production 897286b12a) test/qsa_hisparse/test_fp8_descale.py, run
# with the W4 rows active (tests/qsa/conftest.py). Port edit: the backend
# copies (Q08, Q12) run with the plugin module's globals (inventory section 6,
# G3), so the module the mocks patch is patches/model_compat/qsa_attention.
"""Non-unit FP8 KV descale regressions for the QSA prefill/reference paths.

Frozen contract: a slot stores ``cast_fp8(x / scale)`` and every reader restores
``stored * scale`` with the positive layer scale. The expected values in this
module are independent FP32 attention oracles over the dequantized pool plus
recorded routing boundaries; nothing is derived from the backend under test.

The CUDA-only branches are reached by forcing the ``is_cuda`` predicate on a CPU
tensor; the tensors handed to the kernels stay real CPU tensors, so this is
routing coverage, not GPU validation.
"""

from types import SimpleNamespace

import pytest
import torch

from sglang_qsa_hisparse.patches.model_compat import (  # port edit
    qsa_attention as qsa_backend_module,
)
from sglang.srt.layers.attention.qwen_sparse_attn_backend import QwenSparseAttnBackend
from sglang.srt.model_executor.forward_batch_info import ForwardMode


class _CudaLikeTensor:
    """Force CUDA-only routing branches while the payload stays on CPU."""

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


class _Fp8Pool:
    """Minimal FP8 pool implementing the documented divide-then-cast write."""

    dtype = torch.float8_e4m3fn

    def __init__(self, k_float, v_float, k_scale, v_scale):
        self.k_scale = k_scale
        self.v_scale = v_scale
        self.k = (k_float / k_scale).to(self.dtype)
        self.v = (v_float / v_scale).to(self.dtype)
        self.writes = []

    def get_key_buffer(self, layer_id):
        return self.k

    def get_value_buffer(self, layer_id):
        return self.v

    def set_kv_buffer(self, layer, loc, k, v, k_scale=None, v_scale=None):
        self.writes.append((loc, k, v, k_scale, v_scale))
        self.k[loc] = (k.float() / (1.0 if k_scale is None else k_scale)).to(self.dtype)
        self.v[loc] = (v.float() / (1.0 if v_scale is None else v_scale)).to(self.dtype)

    def dequantized(self):
        return self.k.float() * self.k_scale, self.v.float() * self.v_scale


def _oracle(q, k_dequant, v_dequant, slots, softmax_scale):
    """FP32 sparse GQA over dequantized slots; independent of the backend code."""

    assert k_dequant.shape[1] == 1, "oracle broadcasts a single KV head"
    rows = []
    for row in range(q.shape[0]):
        selected = slots[row][slots[row] >= 0].long()
        keys = k_dequant[selected].squeeze(1)
        values = v_dequant[selected].squeeze(1)
        scores = q[row].float() @ keys.T * softmax_scale
        probabilities = torch.softmax(scores, dim=-1)
        rows.append((probabilities @ values).to(q.dtype))
    return torch.stack(rows)


def _backend(pool):
    # Full construction (no runner) so every attribute the backend reads is
    # initialized; the pool is then swapped for the FP8 test double.
    backend = QwenSparseAttnBackend()
    backend.qsa_hisparse = None
    backend.token_to_kv_pool = pool
    backend.req_to_token_pool = SimpleNamespace(
        req_to_token=torch.arange(pool.k.shape[0], dtype=torch.int32).reshape(1, -1)
    )
    return backend


def _metadata(rows, sequence_length, pool_size):
    return SimpleNamespace(
        token_to_batch_idx=torch.zeros(rows, dtype=torch.int32),
        sequence_lengths=torch.tensor([sequence_length], dtype=torch.int32),
        token_slot_table=torch.arange(pool_size, dtype=torch.int32).reshape(1, -1),
    )


def _layer(heads, head_dim, k_scale, v_scale):
    return SimpleNamespace(
        layer_id=0,
        tp_q_head_num=heads,
        head_dim=head_dim,
        scaling=head_dim**-0.5,
        k_scale_float=k_scale,
        v_scale_float=v_scale,
    )


def _cached_prefix_case(
    total=8, prefix=4, heads=4, head_dim=16, k_scale=2.0, v_scale=0.5
):
    torch.manual_seed(20261004)
    k = torch.randn(total, 1, head_dim)
    v = torch.randn(total, 1, head_dim)
    pool = _Fp8Pool(k, v, k_scale, v_scale)
    q = torch.randn(total - prefix, heads, head_dim).to(torch.bfloat16)
    indices = torch.arange(total, dtype=torch.int32).expand(total - prefix, -1).clone()
    indices[0, -1] = -1
    layer = _layer(heads, head_dim, k_scale, v_scale)
    backend = _backend(pool)
    backend.forward_metadata = _metadata(total - prefix, total, total)
    batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        out_cache_loc=torch.arange(prefix, total),
        req_pool_indices=torch.tensor([0]),
        extend_seq_lens=torch.tensor([total - prefix], dtype=torch.int32),
        extend_seq_lens_cpu=[total - prefix],
        seq_lens_cpu=[total],
    )
    return backend, pool, layer, batch, q, k, v, indices


def test_reference_prefill_restores_non_unit_fp8_scales():
    backend, pool, layer, batch, q, k, v, indices = _cached_prefix_case()
    k_chunk = k[len(k) // 2 :].to(torch.bfloat16)
    v_chunk = v[len(v) // 2 :].to(torch.bfloat16)
    untouched_k, untouched_v = k_chunk.clone(), v_chunk.clone()

    actual = backend.forward_extend(
        q, k_chunk, v_chunk, layer, batch, topk_indices=indices
    )

    # The write itself must keep the live K/V intact for the running attention.
    assert torch.equal(k_chunk, untouched_k)
    assert torch.equal(v_chunk, untouched_v)
    expected = _oracle(q, *pool.dequantized(), indices, layer.scaling)
    torch.testing.assert_close(
        actual.reshape_as(expected), expected, rtol=1e-4, atol=1e-4
    )


def test_reference_prefill_unit_scale_stays_exact():
    backend, pool, layer, batch, q, k, v, indices = _cached_prefix_case(
        k_scale=1.0, v_scale=1.0
    )

    actual = backend.forward_extend(
        q, k[len(k) // 2 :], v[len(v) // 2 :], layer, batch, topk_indices=indices
    )

    expected = _oracle(q, *pool.dequantized(), indices, layer.scaling)
    torch.testing.assert_close(
        actual.reshape_as(expected), expected, rtol=1e-4, atol=1e-4
    )


def test_reference_decode_restores_non_unit_fp8_scales():
    total, heads, head_dim = 6, 4, 16
    torch.manual_seed(7)
    k_scale, v_scale = 2.0, 0.5
    k = torch.randn(total, 1, head_dim)
    v = torch.randn(total, 1, head_dim)
    pool = _Fp8Pool(k, v, k_scale, v_scale)
    layer = _layer(heads, head_dim, k_scale, v_scale)
    backend = _backend(pool)
    backend.forward_metadata = _metadata(1, total, total)
    q = torch.randn(1, heads, head_dim).to(torch.bfloat16)
    indices = torch.tensor([[total - 1, 0, -1]], dtype=torch.int32)
    batch = SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        out_cache_loc=torch.tensor([1]),
        req_pool_indices=torch.tensor([0]),
    )

    actual = backend.forward_decode(
        q, k[:1], v[:1], layer, batch, topk_indices=indices, save_kv_cache=False
    )

    expected = _oracle(q, *pool.dequantized(), indices, layer.scaling)
    torch.testing.assert_close(
        actual.reshape_as(expected), expected, rtol=1e-4, atol=1e-4
    )


def test_cached_prefix_branch_passes_fp8_descales_to_chunk_kernel(monkeypatch):
    recorded = {}

    def fake_chunk_prefill(
        q, k, v, indices, cu_q, cu_k, kv_lens, scale, k_scale=None, v_scale=None
    ):
        recorded.update(
            k=k, v=v, kv_lens=kv_lens, k_scale=k_scale, v_scale=v_scale, scale=scale
        )
        return torch.zeros(q.shape[0], q.shape[1] * q.shape[2], dtype=q.dtype)

    monkeypatch.setattr(
        qsa_backend_module, "sparse_gqa_fwd_interface_triton_ck", fake_chunk_prefill
    )
    backend, pool, layer, batch, q, k, v, indices = _cached_prefix_case()

    actual = backend.forward_extend(
        _CudaLikeTensor(q),
        k[len(k) // 2 :],
        v[len(v) // 2 :],
        layer,
        batch,
        topk_indices=indices,
    )

    # Stored slots are FP8, so the kernel can only recover K/V with the layer
    # scales; 1.0 would silently read every slot at the wrong magnitude.
    assert recorded["k_scale"] == 2.0
    assert recorded["v_scale"] == 0.5
    assert recorded["k"].dtype == torch.float8_e4m3fn
    assert recorded["v"].dtype == torch.float8_e4m3fn
    assert torch.equal(recorded["k"], pool.k)
    assert torch.equal(recorded["v"], pool.v)
    assert recorded["kv_lens"].tolist() == [len(k)]
    assert actual.shape == (q.shape[0], layer.tp_q_head_num * layer.head_dim)


def test_decode_cuda_gather_keeps_fp8_descales(monkeypatch):
    """Protect the already-correct CUDA decode compact-gather scaling."""

    recorded = {}

    def fake_gather(
        k,
        v,
        table,
        requests,
        indices,
        lengths,
        cu_k,
        packed_k,
        packed_v,
        batch,
        topk,
        k_scale=1.0,
        v_scale=1.0,
        **kwargs,
    ):
        recorded.update(k_scale=k_scale, v_scale=v_scale, packed_dtype=packed_k.dtype)

    def fake_flash_attn(**kwargs):
        recorded["softmax_scale"] = kwargs["softmax_scale"]
        return kwargs["q"].contiguous()

    monkeypatch.setattr(
        qsa_backend_module, "qwen_sparse_kv_extraction_compact_triton", fake_gather
    )
    monkeypatch.setattr(
        qsa_backend_module, "qwen_sparse_fa2_cu_seqlens_triton", lambda *a, **k: None
    )
    monkeypatch.setattr(
        qsa_backend_module, "_resolve_trtllm_sparse_decode", lambda: None
    )
    monkeypatch.setattr(qsa_backend_module, "is_hip", lambda: False)
    monkeypatch.setattr(
        qsa_backend_module, "_resolve_flash_attn_varlen_func", lambda: fake_flash_attn
    )

    total, heads, head_dim = 6, 4, 16
    k_scale, v_scale = 2.0, 0.5
    pool = _Fp8Pool(
        torch.randn(total, 1, head_dim),
        torch.randn(total, 1, head_dim),
        k_scale,
        v_scale,
    )
    backend = _backend(pool)
    backend._fa2_scratch = {}
    backend._can_run_fa2_graph = lambda *a, **k: False
    backend.forward_metadata = SimpleNamespace(
        is_cuda_graph=False,
        sequence_lengths=torch.tensor([total], dtype=torch.int32),
        row_req_pool_indices=torch.tensor([0], dtype=torch.int32),
    )
    layer = _layer(heads, head_dim, k_scale, v_scale)
    batch = SimpleNamespace(
        forward_mode=ForwardMode.DECODE, req_pool_indices=torch.tensor([0])
    )

    actual = backend._forward_paged_attention(
        _CudaLikeTensor(torch.randn(1, heads, head_dim).to(torch.bfloat16)),
        layer,
        batch,
        torch.tensor([[0, 2, -1]], dtype=torch.int32),
    )

    assert recorded["k_scale"] == 2.0
    assert recorded["v_scale"] == 0.5
    # The compact gather dequantizes into the query dtype, so no FP8 reaches FA2
    # and the softmax scale must not carry the descales a second time.
    assert recorded["packed_dtype"] == torch.bfloat16
    assert recorded["softmax_scale"] == layer.scaling
    assert actual.shape == (1, heads * head_dim)


def test_store_kv_keeps_live_kv_and_forwards_positive_scales():
    total, heads, head_dim = 4, 2, 8
    k_scale, v_scale = 2.0, 0.5
    pool = _Fp8Pool(
        torch.zeros(total, 1, head_dim),
        torch.zeros(total, 1, head_dim),
        k_scale,
        v_scale,
    )
    backend = _backend(pool)
    layer = _layer(heads, head_dim, k_scale, v_scale)
    k = torch.arange(total * 1 * head_dim, dtype=torch.float32).reshape(
        total, 1, head_dim
    )
    v = -k
    k_live, v_live = k.clone(), v.clone()

    backend._store_kv(layer, torch.arange(total), k, v)

    _, stored_k, stored_v, stored_k_scale, stored_v_scale = pool.writes[0]
    assert stored_k_scale == 2.0
    assert stored_v_scale == 0.5
    # The real pool divides the passed tensors in place before casting.
    assert stored_k is not k and stored_v is not v
    assert torch.equal(k, k_live) and torch.equal(v, v_live)
    torch.testing.assert_close(
        pool.k.float(),
        (k_live / k_scale).to(torch.float8_e4m3fn).float(),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        pool.v.float(),
        (v_live / v_scale).to(torch.float8_e4m3fn).float(),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize(
    ("k_scale", "v_scale", "expected"),
    [(2.0, 0.5, (2.0, 0.5)), (None, None, (1.0, 1.0)), (0.0, -1.0, (1.0, 1.0))],
)
def test_kv_descales_accepts_only_positive_scales(k_scale, v_scale, expected):
    layer = SimpleNamespace(k_scale_float=k_scale, v_scale_float=v_scale)
    assert QwenSparseAttnBackend._kv_descales(layer, torch.float8_e4m3fn) == expected
    assert QwenSparseAttnBackend._kv_descales(layer, torch.bfloat16) == (1.0, 1.0)
