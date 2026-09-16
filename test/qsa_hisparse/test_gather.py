"""Independent gather oracle for the upstream/QSA HiSparse merge.

Run with TRITON_INTERPRET=1. NaN scratch exposes missed zero-fill stores;
non-unit scales expose accidental loss of the offload FP8 scale contract.
"""

import os
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.attention.qsa.sparse_attn import (
    qwen_sparse_kv_extraction_compact_triton,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("TRITON_INTERPRET") != "1",
    reason="requires the explicit CPU Triton interpreter",
)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("strided", [False, True])
def test_gather_preserves_scales_and_padding(dtype, strided):
    batch, topk, heads, dim, stride = 3, 7, 2, 8, 16
    # Small exact values keep this a byte/layout test, without a tuned tolerance.
    values = (torch.arange(64 * heads * dim) % 13 - 6).reshape(64, heads, dim)
    k = values.to(dtype)
    v = (-values).to(dtype)
    table = torch.arange(63, -1, -1, dtype=torch.int32).reshape(4, 16)
    requests = torch.tensor([2, 0, 3], dtype=torch.int32)
    lengths = torch.tensor([3, 7, 1], dtype=torch.int32)
    indices = torch.full((batch, topk), -1, dtype=torch.int32)
    for row, count in enumerate(lengths.tolist()):
        indices[row, :count] = torch.arange(count)
    offsets = (
        torch.arange(batch + 1, dtype=torch.int32) * stride
        if strided
        else torch.tensor([0, 3, 10, 11], dtype=torch.int32)
    )
    packed_k = torch.full(
        (batch * stride, heads, dim), float("nan"), dtype=torch.bfloat16
    )
    packed_v = packed_k.clone()
    qwen_sparse_kv_extraction_compact_triton(
        k,
        v,
        table,
        requests,
        indices,
        lengths,
        offsets,
        packed_k,
        packed_v,
        batch,
        topk,
        k_scale=0.5,
        v_scale=2.0,
        zero_fill_cols=stride if strided else 0,
    )

    for row, count in enumerate(lengths.tolist()):
        start = int(offsets[row])
        slots = table[requests[row], :count].long()
        for pool, actual, scale in ((k, packed_k, 0.5), (v, packed_v, 2.0)):
            expected = pool.float()[slots]
            if dtype == torch.float8_e4m3fn:
                expected *= scale
            torch.testing.assert_close(
                actual[start : start + count],
                expected.to(torch.bfloat16),
                rtol=0,
                atol=0,
            )
            if strided:
                assert torch.count_nonzero(actual[start + count : start + stride]) == 0
    if not strided:
        assert torch.isnan(packed_k[int(offsets[-1]) :]).all()
        assert torch.isnan(packed_v[int(offsets[-1]) :]).all()


def test_paged_backend_passes_fp8_scales_to_gather():
    from sglang.srt.layers.attention.qwen_sparse_attn_backend import (
        QwenSparseAttnBackend,
    )

    backend = QwenSparseAttnBackend.__new__(QwenSparseAttnBackend)
    backend._fa2_scratch = {}
    backend._trtllm_sparse_tables = {}
    backend._trtllm_workspace = torch.zeros(1, dtype=torch.uint8)
    backend.req_to_token_pool = SimpleNamespace(
        req_to_token=torch.tensor([[2, 1, 0]], dtype=torch.int32)
    )
    q = torch.ones((1, 2, 8), dtype=torch.bfloat16)
    k = torch.full((3, 1, 8), 4, dtype=torch.float8_e4m3fn)
    v = torch.full((3, 1, 8), 3, dtype=torch.float8_e4m3fn)
    layer = SimpleNamespace(k_scale_float=0.5, v_scale_float=2.0, scaling=0.25)
    metadata = SimpleNamespace(
        is_cuda_graph=False,
        sequence_lengths=torch.tensor([2], dtype=torch.int32),
        row_req_pool_indices=torch.tensor([0], dtype=torch.int32),
    )

    def decode(**kwargs):
        packed_k, packed_v = kwargs["kv_cache"]
        assert torch.all(packed_k[0, 0, :2] == 2)
        assert torch.all(packed_v[0, 0, :2] == 6)
        assert torch.count_nonzero(packed_k[0, 0, 2:]) == 0
        assert torch.count_nonzero(packed_v[0, 0, 2:]) == 0
        assert kwargs["seq_lens"].tolist() == [2]
        return q

    actual = backend._forward_trtllm_sparse(
        q,
        k,
        v,
        layer,
        SimpleNamespace(),
        metadata,
        torch.tensor([[0, 1, -1]], dtype=torch.int32),
        decode,
    )
    torch.testing.assert_close(actual, q.reshape(1, -1), rtol=0, atol=0)
