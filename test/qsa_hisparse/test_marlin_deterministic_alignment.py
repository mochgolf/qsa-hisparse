"""Integer oracle for the Marlin run-to-run determinism regression.

GPU graph checks are explicitly root-run: SGLANG_TEST_MARLIN_GPU=1. The
standalone test/manual/marlin_deterministic_alignment.py tests the real GEMM.
"""

import importlib.util
import os
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
def test_server_flag_and_unpublished_standalone_policy(
    monkeypatch, published, deterministic, dtype, capability, atomic, broken
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
        calls["gemms"].append(kwargs["use_atomic_add"])
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
    hidden = torch.ones(76, 16, dtype=dtype)
    inputs = dict(
        hidden_states=hidden,
        w1=torch.zeros(512, 1, 32),
        w2=torch.zeros(512, 8, 32),
        w1_scale=torch.ones(512, 1, 1, dtype=dtype),
        w2_scale=torch.ones(512, 1, 1, dtype=dtype),
        gating_output=torch.zeros(76, 512),
        topk_weights=torch.ones(76, 10),
        topk_ids=(torch.arange(760).reshape(76, 10) % 64).int(),
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
    assert calls["gemms"] == [atomic, atomic]
    assert torch.equal(result, torch.full_like(hidden, 10))


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
