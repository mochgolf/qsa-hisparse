"""Independent top-k repeatability probe; GPU execution is explicitly external.

Contract fixed before candidate results: exact largest-score set, smaller index
at score ties, ascending logical-index output and trailing -1. Native selection
has unspecified order/ties and is observed without imposing this new contract.
No model scores, packing helpers, or candidate top-k populate the Python oracle.
"""

import argparse
import json
import random
import time
from pathlib import Path

import torch

from sglang.srt.layers.attention.qsa.kernel import qsa_fast_topk


def expected(scores, starts, ends, topk):
    result = []
    for row, start, end in zip(scores.tolist(), starts.tolist(), ends.tolist()):
        values = row[start:end]
        indices = sorted(range(len(values)), key=lambda index: (-values[index], index))
        chosen = sorted(indices[:topk])
        result.append(chosen + [-1] * (topk - len(chosen)))
    return torch.tensor(result, dtype=torch.int32).reshape(len(result), topk)


def fixture(topk, width, mode):
    starts = torch.tensor([0, 7, 11, 3, 5, 13, 31, 1], dtype=torch.int32)
    capacities = width - starts - 7
    lengths = torch.minimum(
        torch.tensor([0, 1, topk - 1, topk, topk + 1, topk + 13, width, 3]),
        capacities,
    ).to(torch.int32)
    scores = torch.full((8, 2 * width), float("nan"), dtype=torch.float32)
    for row, (start, capacity) in enumerate(zip(starts.tolist(), capacities.tolist())):
        if mode == "untied":
            values = [float(index - capacity // 2) / 16 for index in range(capacity)]
            random.Random(131 + row).shuffle(values)
        elif mode == "constant":
            values = [0.0] * capacity
        else:
            values = [float((index * 11 + row) % 7 - 3) for index in range(capacity)]
        scores[row, start : start + capacity] = torch.tensor(values)
    return scores, starts, starts + lengths


def run_case(topk, width, mode, strided, device, repeats):
    host_base, host_starts, host_ends = fixture(topk, width, mode)
    reference = expected(host_base, host_starts, host_ends, topk)
    base = host_base.to(device)
    scores = base[:, :width]
    if not strided:
        scores = scores.contiguous()
    starts, ends = host_starts.to(device), host_ends.to(device)
    deterministic_outputs, native_outputs = [], []
    for _ in range(repeats):
        deterministic_outputs.append(
            qsa_fast_topk(scores, starts, ends, topk, deterministic=True).cpu()
        )
        native_outputs.append(qsa_fast_topk(scores, starts, ends, topk).cpu())
    exact = all(torch.equal(output, reference) for output in deterministic_outputs)
    if not exact:
        raise AssertionError(
            f"deterministic oracle mismatch: {topk=} {width=} {mode=} {strided=}"
        )
    first = native_outputs[0]
    graph_runs, graph_dynamic_length = 0, None
    if device.type == "cuda":
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output = qsa_fast_topk(scores, starts, ends, topk, deterministic=True)
        for _ in range(repeats):
            graph.replay()
            if not torch.equal(graph_output.cpu(), reference):
                raise AssertionError("CUDA graph deterministic oracle mismatch")
            graph_runs += 1
        # The captured input shape is fixed while a row crosses the top-k threshold.
        alternate_ends = host_ends.clone()
        alternate_ends[3] += 1
        ends.copy_(alternate_ends.to(device))
        graph.replay()
        alternate_reference = expected(host_base, host_starts, alternate_ends, topk)
        graph_dynamic_length = torch.equal(graph_output.cpu(), alternate_reference)
        if not graph_dynamic_length:
            raise AssertionError("CUDA graph dynamic length crossing oracle mismatch")
    return {
        "topk": topk,
        "width": width,
        "mode": mode,
        "strided": strided,
        "score_stride": list(scores.stride()),
        "lengths": (host_ends - host_starts).tolist(),
        "deterministic_exact": exact,
        "deterministic_repeats": repeats,
        "native_order_changes": sum(
            not torch.equal(output, first) for output in native_outputs[1:]
        ),
        "native_set_changes": sum(
            not torch.equal(output.sort().values, first.sort().values)
            for output in native_outputs[1:]
        ),
        "cuda_graph_exact_replays": graph_runs,
        "cuda_graph_dynamic_threshold_exact": graph_dynamic_length,
    }


def large_prefill_timing(device):
    """One shot, identical input rows checked by the unchanged Python oracle."""
    rows, width, topk = 2048, 65536, 512
    host_row = torch.arange(width, dtype=torch.float32).reshape(1, width)
    # All rows have byte-identical scores and intervals. Apply the unchanged
    # exact oracle to that input and compare every output row to its result.
    reference_row = expected(host_row, torch.tensor([0]), torch.tensor([width]), topk)
    reference = reference_row.expand(rows, -1)
    scores = host_row.to(device).expand(rows, -1).contiguous()
    starts = torch.zeros(rows, dtype=torch.int32, device=device)
    ends = torch.full((rows,), width, dtype=torch.int32, device=device)
    baseline_bytes = None
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        baseline_bytes = torch.cuda.memory_allocated(device)
    began = time.perf_counter()
    output = qsa_fast_topk(scores, starts, ends, topk, deterministic=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - began
    peak_extra_bytes = (
        None
        if baseline_bytes is None
        else torch.cuda.max_memory_allocated(device) - baseline_bytes
    )
    exact = torch.equal(output.cpu(), reference)
    if not exact:
        raise AssertionError("large prefill exact Python oracle mismatch")
    return {
        "shape": [rows, width],
        "topk": topk,
        "timed_calls": 1,
        "seconds": seconds,
        "deterministic_exact": exact,
        "logits_bytes": scores.nbytes,
        "output_bytes": output.nbytes,
        "peak_extra_allocated_bytes": peak_extra_bytes,
        "input_rows": "byte-identical",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--large-prefill",
        action="store_true",
        help="time one exact 2048x65536 deterministic selection separately",
    )
    args = parser.parse_args()
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    device = torch.device(args.device)
    rows = []
    for topk in (512, 2048):
        for width in (topk + 64, 8192 + 64, 65536):
            for mode in ("untied", "constant", "few_levels"):
                for strided in (False, True):
                    rows.append(
                        run_case(
                            topk,
                            width,
                            mode,
                            strided,
                            device,
                            min(args.repeats, 2) if width == 65536 else args.repeats,
                        )
                    )
    empty_shapes = []
    for shape in ((0, 513), (3, 0)):
        vector = torch.zeros(shape[0], dtype=torch.int32, device=device)
        output = qsa_fast_topk(
            torch.empty(shape, device=device), vector, vector, 512, True
        )
        if output.shape != (shape[0], 512) or not bool((output == -1).all()):
            raise AssertionError("empty shape contract mismatch")
        empty_shapes.append(list(shape))
    report = {
        "device": str(device),
        "torch": torch.__version__,
        "cases": rows,
        "empty_shapes_exact": empty_shapes,
    }
    if args.large_prefill:
        report["large_prefill"] = large_prefill_timing(device)
    if device.type == "cuda":
        import flashinfer

        report["flashinfer"] = flashinfer.__version__
        report["gpu"] = torch.cuda.get_device_name(device)
    body = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(body)
    print(body, end="")


if __name__ == "__main__":
    main()
