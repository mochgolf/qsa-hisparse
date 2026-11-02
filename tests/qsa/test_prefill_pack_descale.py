"""Phase 6 rule R2: #41933's fused prefill K/V packing versus the FP8 descales.

v0.5.22 packs cached-prefix K/V with ``pack_qsa_prefill_kv(...,
output_dtype=q.dtype)``, a plain cast. Production (bdb935d70f) gathers the raw
FP8 rows and the FP8-aware chunk kernel (A04/A05) applies the layer descales.
The Q08 copy (RESOLVED) packs without the cast, so the descales are applied
exactly once, in the kernel, on bytes identical to production's gather.

Everything runs for real on CPU: the pinned pack kernel and the plugin chunk
kernel under the Triton interpreter (tools/run_cpu_tests.sh sets
TRITON_INTERPRET=1). Only ``q.is_cuda`` is forced, to reach the CUDA branch
of forward_extend. The oracle is FP32 attention over the dequantized pool
(``stored * scale``), built from the test's own slot table.
"""

from types import SimpleNamespace

import pytest

from pool_routes import TypedPoolDouble
import torch

from sglang.srt.layers.attention.qsa.sparse_attn import pack_qsa_prefill_kv
from sglang.srt.layers.attention.qwen_sparse_attn_backend import QwenSparseAttnBackend
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang_qsa_hisparse.kernels.qsa_sparse_attn import (
    sparse_gqa_fwd_interface_triton_ck,
)

FP8 = torch.float8_e4m3fn
# FP16 queries: the interpreter has no BF16 scalar constants for the chunk
# kernel's q scaling. The launchers accept BF16/FP16 queries on the same
# FP8 path (A01's dtype contract), and the pack copies bytes whatever q is.
QUERY_DTYPE = torch.float16
HEADS, KV_HEADS, DIM, POOL, TOPK = 4, 2, 16, 40, 8
# (pool row of the request, cached prefix length, sequence length)
REQUESTS = ((3, 4, 7), (1, 6, 11))


@pytest.fixture(autouse=True)
def production_gpu_name(monkeypatch):
    """The launchers' tile table is chosen by device name (the pinned
    _get_best_config); answer with production's GPU instead of probing CUDA."""
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device=None: "RTX 4090")


class _CudaLikeTensor:
    """Force the CUDA-only branch while the payload stays a CPU tensor."""

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


class _Pool(TypedPoolDouble):
    """KV pool double with the pinned write contract: divide, then cast."""

    def __init__(self, dtype, k, v, k_scale, v_scale):
        self.dtype = dtype
        self.k = (k / k_scale).to(dtype)
        self.v = (v / v_scale).to(dtype)

    def get_key_buffer(self, layer_id):
        return self.k

    def get_value_buffer(self, layer_id):
        return self.v

    def set_kv_buffer(self, layer, loc, k, v, k_scale=None, v_scale=None):
        self.k[loc] = (k.float() / (1.0 if k_scale is None else k_scale)).to(self.dtype)
        self.v[loc] = (v.float() / (1.0 if v_scale is None else v_scale)).to(self.dtype)


class _Runtime:
    """Stand-in HiSparse runtime: identity write locations, recorded slots."""

    offloaded = False

    def __init__(self, table):
        self.table = table
        self.calls = []

    def write_locations(self, loc):
        return loc

    def prefill_slots(self, req_idx, seq_len):
        self.calls.append((req_idx, seq_len))
        return self.table[req_idx, :seq_len].long()


def _case(dtype, k_scale, v_scale):
    gen = torch.Generator().manual_seed(20261019)
    table = torch.zeros(4, 16, dtype=torch.int32)
    slots = torch.randperm(POOL, generator=gen).to(torch.int32)
    used = 0
    for row, _, seq_len in REQUESTS:
        table[row, :seq_len] = slots[used : used + seq_len]
        used += seq_len
    k_float = torch.randn(POOL, KV_HEADS, DIM, generator=gen)
    v_float = torch.randn(POOL, KV_HEADS, DIM, generator=gen)
    pool = _Pool(dtype, k_float, v_float, k_scale, v_scale)
    # The current chunk arrives as live K/V and is stored by forward_extend.
    loc = torch.cat([table[row, prefix:seq_len] for row, prefix, seq_len in REQUESTS])
    k_chunk = torch.randn(len(loc), KV_HEADS, DIM, generator=gen).to(QUERY_DTYPE)
    v_chunk = torch.randn(len(loc), KV_HEADS, DIM, generator=gen).to(QUERY_DTYPE)
    q = torch.randn(len(loc), HEADS, DIM, generator=gen).to(QUERY_DTYPE)
    # Causal sparse selection: each query row picks visible positions of its
    # own request (relative to the request's packed K/V), padded with -1.
    rows = []
    for _, prefix, seq_len in REQUESTS:
        for position in range(prefix, seq_len):
            visible = torch.randperm(position + 1, generator=gen)[:TOPK].sort().values
            row = torch.full((TOPK,), -1, dtype=torch.int32)
            row[: len(visible)] = visible.to(torch.int32)
            rows.append(row)
    indices = torch.stack(rows)
    layer = SimpleNamespace(
        layer_id=0,
        tp_q_head_num=HEADS,
        head_dim=DIM,
        scaling=DIM**-0.5,
        k_scale_float=k_scale,
        v_scale_float=v_scale,
    )
    extend = [seq_len - prefix for _, prefix, seq_len in REQUESTS]
    batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        out_cache_loc=loc.long(),
        req_pool_indices=torch.tensor([row for row, _, _ in REQUESTS]),
        extend_seq_lens=torch.tensor(extend, dtype=torch.int32),
        extend_seq_lens_cpu=extend,
        seq_lens_cpu=[seq_len for _, _, seq_len in REQUESTS],
    )
    backend = QwenSparseAttnBackend()
    backend.qsa_hisparse = None
    backend.token_to_kv_pool = pool
    backend.req_to_token_pool = SimpleNamespace(req_to_token=table)
    return backend, pool, table, layer, batch, q, k_chunk, v_chunk, indices


def _oracle(q, pool, table, indices, scaling, k_scale, v_scale):
    """FP32 sparse GQA over the dequantized pool, one query row at a time."""
    keys_all = pool.k.float() * k_scale
    values_all = pool.v.float() * v_scale
    out, query = [], 0
    for row, prefix, seq_len in REQUESTS:
        for _ in range(prefix, seq_len):
            positions = indices[query][indices[query] >= 0].long()
            slots = table[row, positions].long()
            keys = keys_all[slots].repeat_interleave(HEADS // KV_HEADS, dim=1)
            values = values_all[slots].repeat_interleave(HEADS // KV_HEADS, dim=1)
            scores = torch.einsum("hd,khd->hk", q[query].float(), keys) * scaling
            out.append(torch.einsum("hk,khd->hd", scores.softmax(-1), values))
            query += 1
    return torch.stack(out).reshape(len(out), -1)


def _production_gather(pool, table, batch, layer, q, indices, descales):
    """Production 897286b12a: index_select gather of the raw rows, then A05."""
    lens = batch.seq_lens_cpu
    slots = [table[row, :n].long() for row, n in zip(batch.req_pool_indices.tolist(), lens)]
    lens_tensor = torch.tensor(lens, dtype=torch.int32)
    return sparse_gqa_fwd_interface_triton_ck(
        q,
        torch.cat([pool.k.index_select(0, s) for s in slots]),
        torch.cat([pool.v.index_select(0, s) for s in slots]),
        indices,
        torch.nn.functional.pad(batch.extend_seq_lens.cumsum(0), (1, 0)).int(),
        torch.nn.functional.pad(lens_tensor.cumsum(0), (1, 0)),
        lens_tensor,
        layer.scaling,
        **descales,
    ).reshape(q.shape[0], -1)


@pytest.mark.parametrize("runtime", [False, True], ids=["no-runtime", "runtime"])
@pytest.mark.parametrize(
    "dtype, k_scale, v_scale",
    [(FP8, 2.0, 0.5), (FP8, 1.0, 1.0), (QUERY_DTYPE, 1.0, 1.0)],
    ids=["fp8-calibrated", "fp8-unit", "unquantized"],
)
def test_cached_prefix_prefill_applies_the_descales_exactly_once(
    dtype, k_scale, v_scale, runtime
):
    backend, pool, table, layer, batch, q, k_chunk, v_chunk, indices = _case(
        dtype, k_scale, v_scale
    )
    if runtime:
        backend.qsa_hisparse = _Runtime(table)

    actual = backend.forward_extend(
        _CudaLikeTensor(q), k_chunk, v_chunk, layer, batch, topk_indices=indices
    )

    if runtime:
        assert backend.qsa_hisparse.calls == [(3, 7), (1, 11)]
    descales = (
        {"k_scale": k_scale, "v_scale": v_scale} if dtype == FP8 else {}
    )
    # The pool now holds the stored chunk; production's path reads the same
    # pool, so this is bitwise production's chunk-prefill result.
    assert torch.equal(
        actual, _production_gather(pool, table, batch, layer, q, indices, descales)
    )
    expected = _oracle(q, pool, table, indices, layer.scaling, k_scale, v_scale)
    torch.testing.assert_close(actual.float(), expected, rtol=2e-2, atol=2e-2)
    if (k_scale, v_scale) != (1.0, 1.0):
        # A lost or doubled descale would be far outside the tolerance.
        for lost in (
            _oracle(q, pool, table, indices, layer.scaling, 1.0, 1.0),
            _oracle(
                q, pool, table, indices, layer.scaling, k_scale**2, v_scale**2
            ),
        ):
            assert not torch.allclose(actual.float(), lost, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("dtype", [FP8, torch.bfloat16])
def test_uncast_pack_reproduces_the_index_select_gather_bytes(dtype):
    """pack_qsa_prefill_kv without output_dtype copies the stored bytes."""
    _, pool, table, _, batch, _, _, _, _ = _case(dtype, 2.0, 0.5)
    lens = batch.seq_lens_cpu
    cu_k = torch.nn.functional.pad(torch.tensor(lens).cumsum(0), (1, 0)).int()
    packed_k, packed_v = pack_qsa_prefill_kv(
        k=pool.k,
        v=pool.v,
        req_to_token=table,
        req_indices=batch.req_pool_indices,
        cu_k=cu_k,
        total_k=sum(lens),
        max_k=max(lens),
    )
    slots = [table[row, :n].long() for row, n in zip(batch.req_pool_indices.tolist(), lens)]
    gathered_k = torch.cat([pool.k.index_select(0, s) for s in slots])
    gathered_v = torch.cat([pool.v.index_select(0, s) for s in slots])
    assert packed_k.dtype == packed_v.dtype == dtype
    assert torch.equal(packed_k.view(torch.uint8), gathered_k.view(torch.uint8))
    assert torch.equal(packed_v.view(torch.uint8), gathered_v.view(torch.uint8))
