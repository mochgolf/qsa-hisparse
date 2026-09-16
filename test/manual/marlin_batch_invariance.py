#!/usr/bin/env python3
"""Qualify opt-in whole-K Marlin against identical-row batch placement.

No model load. The fixed contract is y[t,k] = x[t] @ dequant(W[expert[t,k]]),
then SiLU(gate)*up, then down @ dequant(V[expert[t,k]]) * route_weight[t,k],
and a fixed-order top10 sum. Every expert uses the same independently generated
W/V, so changing other rows or routing cannot change the target's real-valued
answer. Native BF16 operations may round differently; this probe reports every
bit mismatch without fitting a tolerance or treating it as cache corruption.

Actual earlier Qwen4 layer0 metadata: W4 uint4b8/group64, BF16, E512/top10,
gate-up K2560/N640, down K320/N2560, blockM8, atomic=false, FP32 reduction.
Independent frozen precursor repro_marlin_batch_invariance.py on actualgroup64
observed162/237 differingbatchcases, allfixedcaserepeatsstable (SM89BF16).
The older group128 M76 fixture characterized unstable alignment and is a
separate diagnostic profile. This version keeps precursor fixtures/equations
and enables only whole-K CTA reduction; --native is its explicit control.
Bitwise target agreement is required across the selected shapes/backgrounds;
this establishes only this GEMM/MoE fixture domain, not model-wide invariance.
--cpu-only checks fixtures/oracle without GPU imports.
"""

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import torch


def digest(tensor):
    raw = tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def delta(value, reference):
    difference = (value.float() - reference.float()).abs()
    return dict(
        bitwise_equal=digest(value) == digest(reference),
        changed_elements=int((value != reference).sum()),
        max_abs_delta=float(difference.max()),
    )


def fixture(
    m, row, pattern, e, topk, target_x, target_down, target_ids, target_weights
):
    gen = torch.Generator().manual_seed(66107 + m)
    x = (torch.randn(m, target_x.numel(), generator=gen) / 3).bfloat16()
    down = (torch.randn(m, topk, target_down.shape[-1], generator=gen) / 3).bfloat16()
    ids = torch.argsort(torch.rand(m, e, generator=gen), dim=1)[:, :topk].int()
    weights = torch.rand(m, topk, generator=gen)
    weights /= weights.sum(dim=1, keepdim=True)
    if pattern in ("identical", "same_routes"):
        ids[:] = target_ids
        weights[:] = target_weights
    if pattern == "identical":
        x[:] = target_x
        down[:] = target_down
    if pattern == "shifted_routes":
        # Move the target's expert blocks in the flattened expert ordering.
        ids = (
            torch.arange(topk).unsqueeze(0) + torch.arange(m).unsqueeze(1) * topk
        ) % e
        ids = ids.int()
    x[row], down[row], ids[row], weights[row] = (
        target_x,
        target_down,
        target_ids,
        target_weights,
    )
    assert torch.equal(x[row], target_x) and torch.equal(down[row], target_down)
    assert torch.equal(ids[row], target_ids) and torch.equal(
        weights[row], target_weights
    )
    assert all(len(set(r.tolist())) == topk for r in ids)
    return x, down, ids, weights


def cases(args):
    for m in args.batch_sizes:
        rows = (
            range(m) if m <= 8 else sorted({0, 7, 8, 11, m // 2, m - 1} & set(range(m)))
        )
        for pattern in args.patterns:
            for row in rows:
                yield m, row, pattern


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    p.add_argument(
        "--output", type=Path, default=Path.cwd() / "marlin-batch-invariance.json"
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--experts", type=int, default=512)
    p.add_argument("--group-size", type=int, default=64)
    p.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=list(range(1, 9)) + [12, 24, 48, 96],
    )
    p.add_argument(
        "--patterns",
        nargs="+",
        choices=["identical", "same_routes", "spread_routes", "shifted_routes"],
        default=["identical", "same_routes", "spread_routes", "shifted_routes"],
    )
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--cpu-only", action="store_true")
    p.add_argument(
        "--cuda-graphs",
        action="store_true",
        help="Capture alignment and both GEMMs; replay also changes background routing",
    )
    p.add_argument(
        "--native",
        action="store_true",
        help="Control: original split-K reduction; characterize, do not require batch agreement",
    )
    args = p.parse_args()
    if args.experts < 10 or min(args.batch_sizes) < 1 or args.repeats < 1:
        p.error("experts>=10, batches>=1, repeats>=1 required")
    if 2560 % args.group_size or 320 % args.group_size:
        p.error("group-size must divide both K2560 and K320; actual model uses64")
    sys.path.insert(0, str(args.source / "python"))
    helper = (
        args.source / "python/sglang/srt/layers/moe/fused_moe_triton/stable_align.py"
    )
    spec = importlib.util.spec_from_file_location("batch_probe_stable_align", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    align = module.moe_align_block_size_stable
    gen = torch.Generator().manual_seed(791223)
    target_x = (torch.randn(2560, generator=gen) / 3).bfloat16()
    target_down = (torch.randn(10, 320, generator=gen) / 3).bfloat16()
    target_ids = torch.randperm(args.experts, generator=gen)[:10].int()
    target_weights = torch.rand(10, generator=gen)
    target_weights /= target_weights.sum()
    w1_cpu = (torch.randn(2560, 640, generator=gen) / 20).bfloat16()
    w2_cpu = (torch.randn(320, 2560, generator=gen) / 20).bfloat16()
    report = dict(
        contract=__doc__,
        source=str(args.source.resolve()),
        helper_sha256=hashlib.sha256(helper.read_bytes()).hexdigest(),
        experts=args.experts,
        group_size=args.group_size,
        use_deterministic_reduce=not args.native,
        cuda_graphs=args.cuda_graphs,
        topk=10,
        block_m=8,
        shapes=[[2560, 640], [320, 2560]],
        dtype="bfloat16",
        atomic=False,
        fp32_reduce=True,
        target_ids=target_ids.tolist(),
        target_weights=target_weights.tolist(),
        results=[],
    )
    expected_up = target_x.double() @ w1_cpu.double()
    expected_down = (
        target_down.double() @ w2_cpu.double() * target_weights.double().unsqueeze(1)
    )
    if args.cpu_only:
        for m, row, pattern in cases(args):
            x, down, ids, weights = fixture(
                m,
                row,
                pattern,
                args.experts,
                10,
                target_x,
                target_down,
                target_ids,
                target_weights,
            )
            aligned = align(ids, 8, args.experts)
            padded = int(aligned[2][0])
            actual = aligned[0][:padded].tolist()
            expected_ids = []
            for expert in range(args.experts):
                positions = [
                    i for i, item in enumerate(ids.flatten().tolist()) if item == expert
                ]
                expected_ids.extend(positions + [m * 10] * (-len(positions) % 8))
            assert actual == expected_ids
            # Target oracle equations are independent of alignment and batch.
            assert torch.equal(x[row].double() @ w1_cpu.double(), expected_up)
            assert torch.equal(
                down[row].double()
                @ w2_cpu.double()
                * weights[row].double().unsqueeze(1),
                expected_down,
            )
            report["results"].append(
                dict(m=m, row=row, pattern=pattern, expert_blocks=padded // 8)
            )
        report.update(
            cpu_only=True,
            cpu_fixture_and_integer_oracle_pass=True,
            cases=len(report["results"]),
        )
    else:
        from sgl_kernel import moe_sum_reduce, silu_and_mul
        from sgl_kernel.scalar_type import scalar_types
        from sglang.kernels.ops.moe.moe_align_single_token import moe_align_single_token
        from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
        from sglang.test.test_marlin_utils import marlin_quantize

        torch.cuda.set_device(args.device)
        quantized = []
        refs = []
        for weights in (w1_cpu, w2_cpu):
            ref, packed, scales, _, _, _ = marlin_quantize(
                weights, scalar_types.uint4b8, args.group_size, False
            )
            refs.append(ref)
            quantized.append(
                (
                    packed.unsqueeze(0).repeat(args.experts, 1, 1).to(args.device),
                    scales.unsqueeze(0).repeat(args.experts, 1, 1).to(args.device),
                )
            )
        expected_up = (target_x.double() @ refs[0].double()).repeat(10, 1)
        expected_down = (
            target_down.double()
            @ refs[1].double()
            * target_weights.double().unsqueeze(1)
        )
        workspace = torch.zeros(
            torch.cuda.get_device_properties(args.device).multi_processor_count * 4,
            dtype=torch.int32,
            device=args.device,
        )

        def gemm(hidden, alignment, routing, which, down=False):
            size_m, size_k = hidden.shape
            size_n = 2560 if down else 640
            top_k = 1 if down else 10
            output = torch.empty(
                (size_m * top_k, size_n), dtype=torch.bfloat16, device=args.device
            )
            packed, scales = quantized[which]
            return moe_wna16_marlin_gemm(
                hidden,
                output,
                packed,
                None,
                scales,
                None,
                None,
                None,
                None,
                workspace,
                *alignment,
                routing,
                moe_block_size=8,
                top_k=top_k,
                mul_topk_weights=down,
                is_ep=False,
                b_q_type=scalar_types.uint4b8,
                size_m=size_m,
                size_n=size_n,
                size_k=size_k,
                is_k_full=True,
                use_atomic_add=False,
                use_fp32_reduce=True,
                is_zp_float=False,
                use_deterministic_reduce=not args.native,
            )

        references = None
        saved = {}
        # First run fixes the M1/single-warp baseline used by accepted forward.
        ordered = [(1, 0, "identical")] + list(cases(args))
        for ordinal, (m, row, pattern) in enumerate(ordered):
            x, down, ids, routing = fixture(
                m,
                row,
                pattern,
                args.experts,
                10,
                target_x,
                target_down,
                target_ids,
                target_weights,
            )
            x, down, ids, routing = [v.to(args.device) for v in (x, down, ids, routing)]
            alignment = (
                moe_align_single_token(ids, 8)
                if ordinal == 0
                else align(ids, 8, args.experts)
            )
            padded = int(alignment[2].cpu()[0])
            positions = {
                int(pair): index
                for index, pair in enumerate(alignment[0][:padded].cpu().tolist())
                if pair < m * 10
            }
            placement = [
                dict(
                    pair=row * 10 + route,
                    block=positions[row * 10 + route] // 8,
                    lane=positions[row * 10 + route] % 8,
                )
                for route in range(10)
            ]
            digests = {}
            first = None

            def execute():
                current = (
                    (
                        moe_align_single_token(ids, 8)
                        if ordinal == 0
                        else align(ids, 8, args.experts)
                    )
                    if args.cuda_graphs
                    else alignment
                )
                up = gemm(x, current, routing, 0)
                isolated = gemm(
                    down.reshape(m * 10, 320).contiguous(), current, routing, 1, True
                )
                activated = torch.empty(
                    (m * 10, 320), dtype=torch.bfloat16, device=args.device
                )
                silu_and_mul(up, activated)
                pipeline = gemm(activated, current, routing, 1, True)
                reduced = torch.empty(
                    (m, 2560), dtype=torch.bfloat16, device=args.device
                )
                moe_sum_reduce(pipeline.view(m, 10, 2560), reduced, 1.0)
                return {
                    "gate_up": up.view(m, 10, 640)[row],
                    "down_isolated": isolated.view(m, 10, 2560)[row],
                    "silu_mul": activated.view(m, 10, 320)[row],
                    "pipeline_down": pipeline.view(m, 10, 2560)[row],
                    "pipeline_sum": reduced[row],
                }

            graph = None
            if args.cuda_graphs:
                # Initialize JIT/modules before capture. Capture alone does not
                # execute kernels, so the first inspected values follow replay.
                execute()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = execute()
            for repeat in range(args.repeats):
                if graph is not None:
                    graph.replay()
                    device_values = captured
                else:
                    device_values = execute()
                values = {name: value.cpu() for name, value in device_values.items()}
                for name, value in values.items():
                    assert bool(torch.isfinite(value).all()), (m, row, pattern, name)
                    digests.setdefault(name, []).append(digest(value))
                if first is None:
                    first = values
                assert not bool(workspace.any()), "Marlin workspace locks not zero"
            if references is None:
                references = first
            stats = {
                name: {
                    **delta(value, references[name]),
                    "distinct_repeated_outputs": len(set(digests[name])),
                    "sha256": digests[name][0],
                }
                for name, value in first.items()
            }
            stats["gate_up"]["cpu_float64_dequant_max_abs"] = float(
                (first["gate_up"].double() - expected_up).abs().max()
            )
            stats["down_isolated"]["cpu_float64_dequant_max_abs"] = float(
                (first["down_isolated"].double() - expected_down).abs().max()
            )
            graph_changed_background_equal = None
            if graph is not None:
                changed_ids = ((ids.cpu().long() + 79) % args.experts).int()
                changed_ids[row] = target_ids
                ids.copy_(changed_ids.to(args.device))
                graph.replay()
                graph_changed_background_equal = all(
                    digest(value.cpu()) == digest(first[name])
                    for name, value in captured.items()
                )
                if not args.native:
                    assert graph_changed_background_equal, (
                        m,
                        row,
                        pattern,
                        "graph changed routing",
                    )
                assert not bool(workspace.any()), "Marlin graph locks not zero"
            item = dict(
                ordinal=ordinal,
                m=m,
                row=row,
                pattern=pattern,
                alignment="single_warp" if ordinal == 0 else "stable",
                expert_blocks=padded // 8,
                target_placement=placement,
                stages=stats,
                graph_changed_background_equal=graph_changed_background_equal,
            )
            report["results"].append(item)
            saved[str(ordinal)] = first
            # Write progress, so a failed/terminated run retains bounded evidence.
            args.output.write_text(json.dumps(report, indent=2) + "\n")
        report.update(
            device=args.device,
            capability=list(torch.cuda.get_device_capability(args.device)),
            repeats=args.repeats,
            cases=len(report["results"]),
            differing_cases=sum(
                any(not stat["bitwise_equal"] for stat in item["stages"].values())
                for item in report["results"]
            ),
            fixed_case_repeatable=all(
                stat["distinct_repeated_outputs"] == 1
                for item in report["results"]
                for stat in item["stages"].values()
            ),
        )
        torch.save(
            dict(
                target_inputs=dict(
                    x=target_x, down=target_down, ids=target_ids, weights=target_weights
                ),
                references=references,
                outputs=saved,
            ),
            args.output.with_suffix(".pt"),
        )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key not in ("results", "contract")
            },
            indent=2,
        )
    )
    if not args.cpu_only:
        assert report["fixed_case_repeatable"], (
            "Fixed-input repeatability confound; inspect digests"
        )
        if not args.native:
            assert report["differing_cases"] == 0, (
                "Whole-K target differs by batch placement; inspect stage vectors"
            )


if __name__ == "__main__":
    main()
