#!/usr/bin/env python3
"""Dynamic B2/B8 CUDA-graph check for the HiSparse ragged-FA2 path."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys


TOPK = 2051
Q_HEADS = 12
KV_HEADS = 1
HEAD_DIM = 256
K_SCALE = 0.5
V_SCALE = 1.75
CASES = {
    2: ([2048, 2051], [2050, 2049]),
    8: (
        [2051, 2048, 2050, 2049, 2051, 2048, 2050, 2049],
        [2048, 2051, 2049, 2050, 2048, 2051, 2049, 2050],
    ),
}


def reference(torch, q, k8, v8, lengths):
    out = torch.empty_like(q)
    scale = 1.0 / math.sqrt(HEAD_DIM)
    for row, length in enumerate(lengths):
        start = row * TOPK
        k = k8[start : start + length, 0].float() * K_SCALE
        v = v8[start : start + length, 0].float() * V_SCALE
        out[row] = (
            torch.softmax(q[row].float() @ k.T * scale, dim=-1) @ v
        ).to(q.dtype)
    return out


def run_batch(torch, wrapper_cls, flash_attn, cu_kernel, gather_kernel, batch):
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(2026091102 + batch)
    q = torch.randn(
        (batch, Q_HEADS, HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    k32 = torch.randn(
        (batch * TOPK, KV_HEADS, HEAD_DIM), device=device, generator=generator
    )
    v32 = torch.randn(
        (batch * TOPK, KV_HEADS, HEAD_DIM), device=device, generator=generator
    )
    k8 = (k32 / K_SCALE).to(torch.float8_e4m3fn)
    v8 = (v32 / V_SCALE).to(torch.float8_e4m3fn)
    del k32, v32

    seq_lens = torch.empty(batch, dtype=torch.int32, device=device)
    indices = torch.empty((batch, TOPK), dtype=torch.int32, device=device)
    counts = torch.empty(batch, dtype=torch.int32, device=device)
    cu_q = torch.arange(batch + 1, dtype=torch.int32, device=device)
    cu_k = torch.empty(batch + 1, dtype=torch.int32, device=device)
    rows = torch.arange(batch, dtype=torch.int32, device=device)
    req_table = torch.arange(
        batch * TOPK, dtype=torch.int32, device=device
    ).view(batch, TOPK)
    packed_k = torch.empty(
        (batch * TOPK, KV_HEADS, HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
    )
    packed_v = torch.empty_like(packed_k)
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = wrapper_cls(
        workspace,
        "NHD",
        use_cuda_graph=True,
        qo_indptr_buf=cu_q,
        kv_indptr_buf=cu_k,
        backend="fa2",
    )
    wrapper.plan(
        cu_q.clone(),
        cu_q.clone() * TOPK,
        Q_HEADS,
        KV_HEADS,
        HEAD_DIM,
        head_dim_vo=HEAD_DIM,
        causal=False,
        q_data_type=q.dtype,
        kv_data_type=q.dtype,
        o_data_type=q.dtype,
        sm_scale=1.0 / math.sqrt(HEAD_DIM),
    )
    if not bool(wrapper._plan_info[-1]):
        raise AssertionError("expected default split-KV plan")

    def set_lengths(lengths):
        values = torch.tensor(lengths, dtype=torch.int32, device=device)
        seq_lens.copy_(values)
        columns = torch.arange(TOPK, dtype=torch.int32, device=device)[None]
        indices.copy_(torch.where(columns < values[:, None], columns, -1))

    def compute():
        packed_k.fill_(float("nan"))
        packed_v.fill_(float("nan"))
        cu_kernel(seq_lens, indices, counts, cu_k, batch, TOPK)
        gather_kernel(
            k8,
            v8,
            req_table,
            rows,
            indices,
            seq_lens,
            cu_k,
            packed_k,
            packed_v,
            batch,
            TOPK,
            k_scale=K_SCALE,
            v_scale=V_SCALE,
        )
        return wrapper.run(q, packed_k, packed_v)

    set_lengths(CASES[batch][0])
    compute()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out = compute()

    records = []
    for lengths in CASES[batch]:
        set_lengths(lengths)
        graph.replay()
        torch.cuda.synchronize()
        first = graph_out.clone()
        graph.replay()
        torch.cuda.synchronize()
        if not torch.equal(first, graph_out):
            raise AssertionError(f"B{batch} replay is not bitwise stable: {lengths}")
        expected_cu = [0]
        for length in lengths:
            expected_cu.append(expected_cu[-1] + length)
        if cu_k.cpu().tolist() != expected_cu:
            raise AssertionError(f"B{batch} cu_k mismatch: {cu_k.cpu().tolist()}")
        expected_k = torch.cat(
            [
                (k8[row * TOPK : row * TOPK + length].float() * K_SCALE).to(
                    packed_k.dtype
                )
                for row, length in enumerate(lengths)
            ]
        )
        expected_v = torch.cat(
            [
                (v8[row * TOPK : row * TOPK + length].float() * V_SCALE).to(
                    packed_v.dtype
                )
                for row, length in enumerate(lengths)
            ]
        )
        if not torch.equal(packed_k[: expected_cu[-1]], expected_k) or not torch.equal(
            packed_v[: expected_cu[-1]], expected_v
        ):
            raise AssertionError(f"B{batch} compact K/V differs: {lengths}")
        expected = reference(torch, q, k8, v8, lengths)
        fallback = flash_attn(
            q=q,
            k=packed_k,
            v=packed_v,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=1,
            max_seqlen_k=TOPK,
            softmax_scale=1.0 / math.sqrt(HEAD_DIM),
            causal=True,
        )
        torch.cuda.synchronize()
        if not torch.isfinite(graph_out).all() or not torch.isfinite(fallback).all():
            raise AssertionError(f"B{batch} non-finite output: {lengths}")
        torch.testing.assert_close(graph_out.float(), expected.float(), atol=0.005, rtol=0.02)
        torch.testing.assert_close(fallback.float(), expected.float(), atol=0.005, rtol=0.02)
        records.append(
            {
                "lengths": list(lengths),
                "cu_k": expected_cu,
                "gather_max_abs": 0.0,
                "ragged_max_abs_vs_fp32": (graph_out.float() - expected.float())
                .abs()
                .max()
                .item(),
                "fallback_max_abs_vs_fp32": (fallback.float() - expected.float())
                .abs()
                .max()
                .item(),
                "replay_bitwise_stable": True,
            }
        )
    return {"batch": batch, "split_kv": True, "replays": records}


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
        raise RuntimeError("B>1 ragged FA2 check requires SM89")
    batches = [
        run_batch(
            torch,
            BatchPrefillWithRaggedKVCacheWrapper,
            flash_attn_varlen_func,
            qwen_sparse_fa2_cu_seqlens_triton,
            qwen_sparse_kv_extraction_compact_triton,
            batch,
        )
        for batch in CASES
    ]
    result = {
        "status": "PASS",
        "source_commit": "687a39bbe0af00240fecb61872e8643ccb076cb7",
        "gpu": torch.cuda.get_device_name(),
        "valid_count_range": [2048, 2051],
        "batches": batches,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
