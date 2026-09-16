"""Integer oracle for the Marlin run-to-run determinism regression.

GPU graph checks are explicitly root-run: SGLANG_TEST_MARLIN_GPU=1. The
standalone test/manual/marlin_deterministic_alignment.py tests the real GEMM.
"""

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
HELPER = ROOT / "python/sglang/srt/layers/moe/fused_moe_triton/stable_align.py"
# Load the actual pure helper without importing the model/GPU package tree.
spec = importlib.util.spec_from_file_location("marlin_stable_align_test", HELPER)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
align = module.moe_align_block_size_stable


def oracle(ids, block, experts):
    values = ids.reshape(-1).tolist()
    buckets = [[] for _ in range(experts + 1)]
    for pair, expert in enumerate(values):
        buckets[expert + 1].append(pair)
    sorted_ids, expert_ids = [], []
    for bucket, matching in enumerate(buckets):
        sorted_ids.extend(matching)
        sorted_ids.extend([len(values)] * (-len(matching) % block))
        expert_ids.extend([bucket - 1] * ((len(matching) + block - 1) // block))
    return sorted_ids, expert_ids


def assert_contract(ids, result, block, experts):
    sorted_ids, expert_ids, total = (tensor.cpu() for tensor in result)
    expected, expected_experts = oracle(ids.cpu(), block, experts)
    assert int(total[0]) == len(expected)
    assert sorted_ids[: len(expected)].tolist() == expected
    assert expert_ids[: len(expected_experts)].tolist() == expected_experts
    assert bool((sorted_ids[len(expected) :] == ids.numel()).all())
    assert bool((expert_ids[len(expected_experts) :] == -1).all())
    assert sorted(sorted_ids[sorted_ids < ids.numel()].tolist()) == list(
        range(ids.numel())
    )
    assert sorted_ids.dtype == expert_ids.dtype == total.dtype == torch.int32


@pytest.mark.parametrize(
    "tokens,topk,experts,block,negative",
    [
        (76, 10, 512, 8, False),
        (1, 10, 512, 8, False),
        (64, 1, 1000, 16, False),
        (76, 10, 512, 8, True),
        (0, 10, 512, 8, False),
        (17, 1, 3, 1, True),
        (256, 10, 512, 8, False),
        (2048, 10, 512, 8, False),
    ],
)
def test_integer_order_padding_and_pair_identity(
    tokens, topk, experts, block, negative
):
    gen = torch.Generator().manual_seed(tokens + experts + block)
    ids = torch.randint(-1 if negative else 0, experts, (tokens, topk), generator=gen)
    result = align(ids, block, experts)
    assert_contract(ids, result, block, experts)
    assert all(
        torch.equal(first, second)
        for first, second in zip(result, align(ids, block, experts))
    )


def test_all_filtered_and_strided_pairs():
    assert_contract(
        torch.full((76, 10), -1), align(torch.full((76, 10), -1), 8, 512), 8, 512
    )
    ids = torch.arange(760 * 2).reshape(76, 20)[:, ::2] % 64
    ids = ids.t().contiguous().t()
    assert not ids.is_contiguous()
    assert_contract(ids, align(ids, 8, 512), 8, 512)


@pytest.mark.parametrize(
    "published,deterministic,dtype,capability,atomic,broken",
    [
        (False, True, torch.bfloat16, 9, True, False),
        (True, False, torch.bfloat16, 8, False, False),
        (True, False, torch.bfloat16, 9, True, False),
        (True, False, torch.float16, 8, True, False),
        (True, True, torch.bfloat16, 9, False, False),
        (True, True, torch.float16, 8, False, False),
        (True, True, torch.bfloat16, 9, False, True),
    ],
)
@pytest.mark.parametrize("tokens", [76, 2048])
def test_server_flag_and_unpublished_standalone_policy(
    monkeypatch, published, deterministic, dtype, capability, atomic, broken, tokens
):
    # Execute the actual fused implementation with CPU GEMM/activation stubs.
    # An unpublished kernel call must never consult the execution bag; a
    # published server must propagate its flag to alignment and both GEMMs.
    __import__("triton")  # Initialize the CPU decorator before substituting packages.

    calls = {"native_align": 0, "exec": 0, "gemms": []}

    def stub(name, **attributes):
        fake = ModuleType(name)
        fake.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, fake)
        return fake

    def get_exec():
        calls["exec"] += 1
        assert published, "Uninitialized standalone call accessed runtime config"
        if broken:
            raise ValueError("Published execution config is invalid")
        return SimpleNamespace(
            deterministic=SimpleNamespace(enable_deterministic_inference=deterministic)
        )

    def native_align(*args):
        calls["native_align"] += 1
        return align(*args)

    def gemm(*args, **kwargs):
        calls["gemms"].append(
            (
                kwargs["use_atomic_add"],
                kwargs["use_deterministic_reduce"],
                kwargs["moe_block_size"],
            )
        )
        args[1].fill_(1)
        return args[1]

    def reduce(input, output, scale):
        output.copy_(input.sum(1) * scale)

    stub(
        "sglang.srt.layers",
        zero_copy_context=SimpleNamespace(get_moe_output=lambda _: None),
    )
    stub("sglang.srt.layers.moe.fused_moe_triton", moe_align_block_size=native_align)
    monkeypatch.setitem(
        sys.modules, "sglang.srt.layers.moe.fused_moe_triton.stable_align", module
    )
    stub(
        "sglang.srt.runtime_context",
        get_exec=get_exec,
        get_context=lambda: SimpleNamespace(
            is_config_namespace_published=lambda _: published
        ),
    )
    stub("sglang.srt.utils", is_cuda=lambda: True)
    stub(
        "sglang.srt.utils.custom_op", register_custom_op=lambda *a, **kw: lambda fn: fn
    )
    stub("sgl_kernel", moe_sum_reduce=reduce)
    stub("sgl_kernel.scalar_type", scalar_types=SimpleNamespace(uint4b8=object()))
    stub(
        "sglang.kernels.ops.activation.activation",
        silu_and_mul=lambda inp, out: out.fill_(1),
    )
    stub("sglang.kernels.ops.moe.moe_wna16_marlin", moe_wna16_marlin_gemm=gemm)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (capability, 0))
    # No actual GPU operation can be issued by this test.
    monkeypatch.setattr(
        torch.cuda, "_lazy_init", lambda: pytest.fail("GPU initialization forbidden")
    )
    source = HELPER.with_name("fused_marlin_moe.py")
    fused_spec = importlib.util.spec_from_file_location(
        "marlin_cpu_policy_test", source
    )
    fused = importlib.util.module_from_spec(fused_spec)
    fused_spec.loader.exec_module(fused)
    hidden = torch.ones(tokens, 16, dtype=dtype)
    inputs = dict(
        hidden_states=hidden,
        w1=torch.zeros(512, 1, 32),
        w2=torch.zeros(512, 8, 32),
        w1_scale=torch.ones(512, 1, 1, dtype=dtype),
        w2_scale=torch.ones(512, 1, 1, dtype=dtype),
        gating_output=torch.zeros(tokens, 512),
        topk_weights=torch.ones(tokens, 10),
        topk_ids=(torch.arange(tokens * 10).reshape(tokens, 10) % 64).int(),
        workspace=torch.zeros(4, dtype=torch.int32),
        num_bits=4,
    )
    if broken:
        with pytest.raises(ValueError, match="Published execution config is invalid"):
            fused.fused_marlin_moe(**inputs)
        assert calls["exec"] == 1
        assert calls["native_align"] == 0
        assert calls["gemms"] == []
        return
    result = fused.fused_marlin_moe(**inputs)
    assert calls["exec"] == int(published)
    assert calls["native_align"] == int(not (published and deterministic))
    block = 8 if tokens == 76 or published and deterministic else 48
    assert calls["gemms"] == [(atomic, published and deterministic, block)] * 2
    assert torch.equal(result, torch.full_like(hidden, 10))


@pytest.fixture(scope="module")
def whole_k_scheduler(tmp_path_factory):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("CPU C++ compiler unavailable")
    folder = tmp_path_factory.mktemp("marlin_integer_schedule")
    header = ROOT / "python/sglang/kernels/jit/csrc/gemm/marlin_moe/stripe_schedule.h"
    source = folder / "oracle.cpp"
    source.write_text(
        '#include <iostream>\n#include "' + str(header) + '"\n'
        "int main(int argc, char**) { using C=sglang::device::marlin_moe::whole_k_launch_config; "
        'if(argc>1) { std::cout << C::thread_k << " " << C::thread_n << " " << C::num_threads << " " << C::blocks_per_sm; return 0; } '
        "int k,n,p,b; while(std::cin >> k >> n >> p >> b) "
        'std::cout << sglang::device::marlin_moe::whole_k_stripe_iters(k,n,p,b) << "\\n"; }\n'
    )
    executable = folder / "oracle"
    subprocess.run(
        [compiler, "-std=c++17", str(source), "-o", str(executable)],
        check=True,
        capture_output=True,
    )
    return executable


def output_slice_owners(k_tiles, output_slices, blocks, stripe_iters):
    # Independent ownership oracle: enumerate the actual flattened K work
    # assigned to CTAs and group it by output. Split-K gives multiple owners.
    owners = [set() for _ in range(output_slices)]
    for block in range(blocks):
        for tile in range(
            block * stripe_iters,
            min((block + 1) * stripe_iters, k_tiles * output_slices),
        ):
            owners[tile // k_tiles].add(block)
    return owners


def test_actual_integer_scheduler_has_one_cta_per_complete_k_slice(whole_k_scheduler):
    cases = [
        (k, n, parallel, blocks)
        for k, n in [(5, 20), (40, 5), (40, 20)]
        for parallel in [0, 1, 10, 60, 120, 513]
        for blocks in [1, 4, 114, 142, 284, 568]
    ]
    output = subprocess.run(
        [str(whole_k_scheduler)],
        input="".join(f"{k} {n} {p} {b}\n" for k, n, p, b in cases),
        text=True,
        check=True,
        capture_output=True,
    ).stdout.splitlines()
    assert len(output) == len(cases)
    for (k, n, parallel, blocks), value in zip(cases, output):
        stripe_iters = int(value)
        owners = output_slice_owners(k, n * parallel, blocks, stripe_iters)
        assert all(len(matching) == 1 for matching in owners)
        assert stripe_iters % k == 0
    # Reject the old scheduler: a plausible reversion cuts real K20/N5 work
    # inside an output slice, despite fixed native lock order/repeated bits.
    native_iters = (20 * 5 * 60 + 568 - 1) // 568
    assert any(
        len(matching) > 1
        for matching in output_slice_owners(20, 5 * 60, 568, native_iters)
    )


def test_fixed_launch_dimensions_cover_both_actual_group64_gemms(whole_k_scheduler):
    dimensions = [
        int(value)
        for value in subprocess.run(
            [str(whole_k_scheduler), "config"],
            text=True,
            check=True,
            capture_output=True,
        ).stdout.split()
    ]
    thread_k, thread_n, threads, blocks_per_sm = dimensions
    assert threads % 32 == 0 and blocks_per_sm == 1
    for k, n in [(2560, 640), (320, 2560)]:
        assert k % 64 == 0  # Actual quantization group size, separate from128 fixture.
        assert k % thread_k == n % thread_n == 0
        for sm_count in [114, 142]:
            grid = sm_count * blocks_per_sm
            for parallel in [10, 80, 120, 2560]:
                output = subprocess.run(
                    [str(whole_k_scheduler)],
                    input=f"{k // thread_k} {n // thread_n} {parallel} {grid}\n",
                    text=True,
                    check=True,
                    capture_output=True,
                ).stdout
                owners = output_slice_owners(
                    k // thread_k, n // thread_n * parallel, grid, int(output)
                )
                assert all(len(matching) == 1 for matching in owners)


@pytest.mark.parametrize(
    "explicit,block,atomic,valid",
    [
        (None, 16, True, True),
        (True, 8, False, True),
        (True, 16, False, False),
        (True, 8, True, False),
    ],
)
def test_standalone_gemm_default_and_explicit_reduction_contract(
    monkeypatch, explicit, block, atomic, valid
):
    selections = []
    launches = []
    utils = ModuleType("sglang.kernels.jit.utils")
    utils.cache_once = lambda fn: fn
    utils.make_cpp_args = lambda *args: args
    utils.load_jit = lambda *args, **kwargs: None
    logging = ModuleType("sglang.kernels.kernel_api_logging")
    logging.debug_kernel_api = lambda fn: fn
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    monkeypatch.setitem(sys.modules, logging.__name__, logging)
    source = ROOT / "python/sglang/kernels/ops/moe/moe_wna16_marlin.py"
    spec = importlib.util.spec_from_file_location("standalone_marlin_cpu_api", source)
    raw = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(raw)

    def select(dtype, ep, bias, deterministic=False):
        selections.append(deterministic)
        return SimpleNamespace(
            moe_wna16_marlin_gemm=lambda *args: launches.append(args)
        )

    raw._jit_moe_wna16_marlin_module = select
    monkeypatch.setattr(
        torch.cuda, "_lazy_init", lambda: pytest.fail("GPU initialization forbidden")
    )
    extra = {} if explicit is None else dict(use_deterministic_reduce=explicit)
    call = lambda: raw.moe_wna16_marlin_gemm(
        torch.ones(1, 64),
        None,
        torch.ones(2, 4, 128),
        None,
        torch.ones(2, 1, 128),
        None,
        None,
        None,
        None,
        torch.zeros(4, dtype=torch.int32),
        torch.zeros(10 * block, dtype=torch.int32),
        torch.zeros(10, dtype=torch.int32),
        torch.tensor([10 * block], dtype=torch.int32),
        torch.ones(1, 10),
        moe_block_size=block,
        top_k=10,
        mul_topk_weights=False,
        is_ep=False,
        b_q_type=SimpleNamespace(id=1),
        size_m=1,
        size_n=128,
        size_k=64,
        use_atomic_add=atomic,
        **extra,
    )
    if valid:
        result = call()
        assert result.shape == (10, 128)
        assert selections == [bool(explicit)] and len(launches) == 1
    else:
        with pytest.raises(ValueError, match="blockM8 and no atomics"):
            call()
        assert selections == launches == []


@pytest.mark.skipif(
    os.environ.get("SGLANG_TEST_MARLIN_GPU") != "1", reason="root-only GPU execution"
)
@pytest.mark.parametrize("tokens", [76, 256, 2048])
def test_cuda_graph_replay_updates_integer_mapping(tokens):
    gen = torch.Generator().manual_seed(792213 + tokens)
    cpu = torch.randint(-1, 512, (tokens, 10), generator=gen)
    gpu = cpu.cuda()
    align(gpu, 8, 512)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = align(gpu, 8, 512)
    graph.replay()
    assert_contract(cpu, captured, 8, 512)
    for _ in range(3):
        cpu = torch.randint(-1, 512, (tokens, 10), generator=gen)
        gpu.copy_(cpu)
        graph.replay()
        assert_contract(cpu, captured, 8, 512)


@pytest.mark.skipif(
    os.environ.get("SGLANG_TEST_MARLIN_GPU") != "1", reason="root-only GPU execution"
)
def test_whole_k_actual_group64_cuda_graph_target_batch_reference(tmp_path):
    script = ROOT / "test/manual/marlin_batch_invariance.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--cuda-graphs",
            "--batch-sizes",
            "1",
            "8",
            "96",
            "2048",
            "--patterns",
            "identical",
            "spread_routes",
            "--output",
            str(tmp_path / "marlin-graphs.json"),
        ],
        text=True,
        capture_output=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stdout + result.stderr
