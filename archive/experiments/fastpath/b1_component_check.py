#!/usr/bin/env python3
"""One FP8 numerical check for the SM89 ragged-FA2 B1 primitive."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source.resolve() / "python"))

    import torch
    from flashinfer.prefill import BatchPrefillWithRaggedKVCacheWrapper
    from sglang.kernels.ops.attention.flash_attention import flash_attn_varlen_func
    from sglang.srt.layers.attention.qsa.sparse_attn import (
        qwen_sparse_fa2_cu_seqlens_triton,
        qwen_sparse_kv_extraction_compact_triton,
    )

    if torch.cuda.get_device_capability() != (8, 9):
        raise RuntimeError("B1 ragged FA2 check requires SM89")
    q_heads, kv_heads, head_dim, topk = 12, 1, 256, 2051
    k_scale, v_scale = 0.5, 1.75
    generator = torch.Generator(device="cuda").manual_seed(2026091101)
    q = torch.randn(
        (1, q_heads, head_dim), device="cuda", dtype=torch.bfloat16, generator=generator
    )
    k32 = torch.randn((topk, kv_heads, head_dim), device="cuda", generator=generator)
    v32 = torch.randn((topk, kv_heads, head_dim), device="cuda", generator=generator)
    k8, v8 = (
        (k32 / k_scale).to(torch.float8_e4m3fn),
        (v32 / v_scale).to(torch.float8_e4m3fn),
    )
    indices = torch.randperm(topk, device="cuda", generator=generator)[None].to(
        torch.int32
    )
    seq_lens = torch.tensor([topk], dtype=torch.int32, device="cuda")
    req_table = torch.arange(topk, dtype=torch.int32, device="cuda")[None]
    rows = torch.zeros(1, dtype=torch.int32, device="cuda")
    counts = torch.empty(1, dtype=torch.int32, device="cuda")
    cu_q = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    cu_k = torch.tensor([0, topk], dtype=torch.int32, device="cuda")
    packed_k = torch.empty(
        (topk, kv_heads, head_dim), dtype=torch.bfloat16, device="cuda"
    )
    packed_v = torch.empty_like(packed_k)
    qwen_sparse_fa2_cu_seqlens_triton(seq_lens, indices, counts, cu_k, 1, topk)
    qwen_sparse_kv_extraction_compact_triton(
        k8,
        v8,
        req_table,
        rows,
        indices,
        seq_lens,
        cu_k,
        packed_k,
        packed_v,
        1,
        topk,
        k_scale=k_scale,
        v_scale=v_scale,
    )

    permutation = indices[0].long()
    expected_k = (k8.float() * k_scale)[permutation]
    expected_v = (v8.float() * v_scale)[permutation]
    gather_error = max(
        (packed_k.float() - expected_k).abs().max().item(),
        (packed_v.float() - expected_v).abs().max().item(),
    )
    workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    wrapper = BatchPrefillWithRaggedKVCacheWrapper(
        workspace,
        "NHD",
        use_cuda_graph=True,
        qo_indptr_buf=cu_q,
        kv_indptr_buf=cu_k,
        backend="fa2",
    )
    scale = 1.0 / math.sqrt(head_dim)
    wrapper.plan(
        cu_q.clone(),
        cu_k.clone(),
        q_heads,
        kv_heads,
        head_dim,
        head_dim_vo=head_dim,
        causal=False,
        q_data_type=q.dtype,
        kv_data_type=q.dtype,
        o_data_type=q.dtype,
        sm_scale=scale,
    )
    ragged = wrapper.run(q, packed_k, packed_v)[0].float()
    baseline = flash_attn_varlen_func(
        q=q,
        k=packed_k,
        v=packed_v,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=1,
        max_seqlen_k=topk,
        softmax_scale=scale,
        causal=True,
    )[0].float()
    reference = (
        torch.softmax(q[0].float() @ expected_k[:, 0].T * scale, dim=-1)
        @ expected_v[:, 0]
    )
    torch.testing.assert_close(ragged, reference, atol=0.005, rtol=0.02)
    torch.testing.assert_close(baseline, reference, atol=0.005, rtol=0.02)
    result = {
        "status": "PASS",
        "gpu": torch.cuda.get_device_name(),
        "capability": [8, 9],
        "shape": [1, q_heads, kv_heads, head_dim],
        "topk": topk,
        "k_scale": k_scale,
        "v_scale": v_scale,
        "gather_max_abs": gather_error,
        "ragged_max_abs_vs_fp32": (ragged - reference).abs().max().item(),
        "baseline_max_abs_vs_fp32": (baseline - reference).abs().max().item(),
    }
    if gather_error != 0:
        raise AssertionError(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
