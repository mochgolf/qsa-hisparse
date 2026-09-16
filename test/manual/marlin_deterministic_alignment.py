#!/usr/bin/env python3
"""One-GPU regression for Marlin split-K/alignment, without a model load.

Contract: every expert has the SAME symmetric GPTQ W4/group128 matrix W.
For token t and route k, GEMM output equals x[t] @ dequantized(W), with BF16
dequantization/tensor-core rounding. Alignment may change execution placement,
never the token/expert mapping. Fixed alignment must repeat bitwise; no claim
is made that different batch shapes or different placements round identically.

Provenance: on 2026-09-16 the root diagnostic on SM89, M76/topk10,
K2560/N640/E512, active64, BF16, atomic=false, FP32 reduction yielded one
output in nine frozen-native/stable trials, versus six outputs/nine alignments
with fresh native alignment (max delta 0.00390625). Independent float64 CPU
dequantized reference max error was 0.00780035 in all three arms. This test
asserts repeated bits, and reports reference error without fitting tolerances.
"""

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
# Load the production pure helper directly so --cpu-only never imports GPU packages.
helper = ROOT / "python/sglang/srt/layers/moe/fused_moe_triton/stable_align.py"
spec = importlib.util.spec_from_file_location(
    "marlin_stable_alignment_regression", helper
)
helper_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper_module)
stable_align = helper_module.moe_align_block_size_stable


def reference_fixture(m, k, n, e, topk, active_experts):
    gen = torch.Generator().manual_seed(791223)
    x = (torch.randn(m, k, generator=gen) / 3).to(torch.bfloat16)
    w = (torch.randn(k, n, generator=gen) / 20).to(torch.bfloat16)
    # Keep the declared E512 alignment profile, but skew traffic to64 experts.
    # This creates multi-block experts, so native atomic order can move a row
    # between Marlin blocks with different split-K partitions. Uniform E512
    # traffic is a useful control, usually containing no multi-block experts.
    scores = torch.rand(m, e, generator=gen)
    scores[:, active_experts:] += 2
    ids = torch.argsort(scores, dim=-1)[:, :topk].int()
    aligned = stable_align(ids, 8, e)
    ref = []
    flat = ids.flatten().tolist()
    for expert in range(e):
        rows = [index for index, item in enumerate(flat) if item == expert]
        ref.extend(rows + [m * topk] * (-len(rows) % 8))
    assert aligned[0][: len(ref)].tolist() == ref
    assert int(aligned[2][0]) == len(ref)
    return x, w, ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path.cwd() / "marlin-deterministic-alignment.json",
    )
    parser.add_argument("--experts", type=int, default=512)
    parser.add_argument("--active-experts", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()
    m, topk, k, n = 76, 10, 2560, 640
    if not topk <= args.active_experts <= args.experts:
        parser.error("topk10 <= active-experts <= experts is required")
    x_cpu, w_cpu, ids_cpu = reference_fixture(
        m, k, n, args.experts, topk, args.active_experts
    )
    counts = torch.bincount(ids_cpu.flatten().long(), minlength=args.experts)
    report = dict(
        m=m,
        topk=topk,
        k=k,
        n=n,
        experts=args.experts,
        block_m=8,
        dtype="bfloat16",
        atomic=False,
        fp32_reduce=True,
        active_experts=args.active_experts,
        max_expert_population=int(counts.max()),
        multiblock_experts=int((counts > 8).sum()),
    )
    if args.cpu_only:
        output = x_cpu.double() @ w_cpu.double()
        report.update(
            cpu_only=True,
            reference_finite=bool(torch.isfinite(output).all()),
            stable_alignment_reference_pass=True,
        )
    else:
        from sgl_kernel.scalar_type import scalar_types
        from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
        from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
        from sglang.test.test_marlin_utils import marlin_quantize

        torch.cuda.set_device(args.device)
        # Quantize/repack ONE CPU matrix, then replicate its packed form only.
        # At E512 this uses about420 MB of packed GPU weights, not a model load.
        w_ref, packed, scales, _, _, _ = marlin_quantize(
            w_cpu, scalar_types.uint4b8, 128, False
        )
        qweights = packed.unsqueeze(0).repeat(args.experts, 1, 1).to(args.device)
        all_scales = scales.unsqueeze(0).repeat(args.experts, 1, 1).to(args.device)
        hidden = x_cpu.to(args.device)
        ids = ids_cpu.to(args.device)
        routing_weights = torch.full((m, topk), 1 / topk, device=args.device)
        workspace = torch.zeros(
            torch.cuda.get_device_properties(args.device).multi_processor_count * 4,
            dtype=torch.int32,
            device=args.device,
        )
        output = torch.empty((m * topk, n), dtype=torch.bfloat16, device=args.device)
        frozen = moe_align_block_size(ids, 8, args.experts)
        expected = (x_cpu.double() @ w_ref.double()).repeat_interleave(topk, dim=0)
        results = []
        for mode, fn in (
            ("frozen_native", None),
            ("fresh_native", moe_align_block_size),
            ("stable", stable_align),
        ):
            outputs, alignments = [], []
            reference = None
            max_delta = 0.0
            for trial in range(args.repeats):
                alignment = frozen if fn is None else fn(ids, 8, args.experts)
                result = moe_wna16_marlin_gemm(
                    hidden,
                    output,
                    qweights,
                    None,
                    all_scales,
                    None,
                    None,
                    None,
                    None,
                    workspace,
                    *alignment,
                    routing_weights,
                    moe_block_size=8,
                    top_k=topk,
                    mul_topk_weights=False,
                    is_ep=False,
                    b_q_type=scalar_types.uint4b8,
                    size_m=m,
                    size_n=n,
                    size_k=k,
                    is_k_full=True,
                    use_atomic_add=False,
                    use_fp32_reduce=True,
                    is_zp_float=False,
                )
                cpu = result.cpu()
                outputs.append(
                    hashlib.sha256(cpu.view(torch.uint8).numpy().tobytes()).hexdigest()
                )
                aligned = alignment[0].cpu()
                alignments.append(
                    hashlib.sha256(
                        aligned.view(torch.uint8).numpy().tobytes()
                    ).hexdigest()
                )
                if reference is None:
                    reference = cpu.float()
                    torch.save(
                        cpu,
                        args.output.with_name(args.output.stem + "-" + mode + ".pt"),
                    )
                else:
                    max_delta = max(
                        max_delta, float((cpu.float() - reference).abs().max())
                    )
                assert bool(torch.isfinite(cpu).all()), mode
                assert not bool(workspace.any()), "Marlin locks did not return to zero"
            results.append(
                dict(
                    mode=mode,
                    distinct_outputs=len(set(outputs)),
                    distinct_alignments=len(set(alignments)),
                    max_abs_delta=max_delta,
                    cpu_dequant_reference_max_abs=float(
                        (reference.double() - expected).abs().max()
                    ),
                    output_digests=outputs,
                    alignment_digests=alignments,
                )
            )
        report.update(
            device=args.device,
            capability=list(torch.cuda.get_device_capability(args.device)),
            repeats=args.repeats,
            results=results,
            fixed_alignment_repeatable=results[0]["distinct_outputs"]
            == results[2]["distinct_outputs"]
            == 1,
        )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not args.cpu_only:
        assert report["fixed_alignment_repeatable"], (
            "Fixed Marlin alignment changed repeated output bits"
        )


if __name__ == "__main__":
    main()
