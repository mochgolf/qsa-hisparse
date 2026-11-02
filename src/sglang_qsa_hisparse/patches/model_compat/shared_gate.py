"""Reference shared-expert gate reduction geometry for deterministic Qwen4.

D02 retains the production 897286b12a launcher verbatim, including its launch
constants. Main #42087 caps small-batch global-invariant calls at eight warps;
the reference uses sixteen at hidden width 2560. The kernel remains native.
"""

import torch
import triton

from sglang.kernels.ops.elementwise.elementwise import (
    _fused_gate_sigmoid_mul_add_kernel,
    _is_hip,
)
from sglang.kernels.jit.utils import is_arch_support_pdl
from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled
from sglang.srt.utils.common import is_cuda
from sglang_qsa_hisparse import scope
from sglang_qsa_hisparse.features import MODEL_COMPAT
from sglang_qsa_hisparse.patching import patch

_MODULE = "sglang.kernels.ops.elementwise.elementwise"


def _launch_fused_gate_sigmoid_mul(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    shared_output: torch.Tensor,
    output: torch.Tensor,
    *,
    do_add: bool,
) -> None:
    assert hidden_states.is_contiguous(), "hidden_states must be contiguous"
    assert gate_weight.is_contiguous(), "gate_weight must be contiguous"
    assert shared_output.is_contiguous(), "shared_output must be contiguous"
    assert output.is_contiguous(), "output must be contiguous"

    num_tokens, hidden_dim = hidden_states.shape
    assert gate_weight.shape == (hidden_dim,)
    assert shared_output.shape == (num_tokens, hidden_dim)
    assert output.shape == (num_tokens, hidden_dim)

    max_warps = 16 if _is_hip else 32
    config = {
        "BLOCK_SIZE": triton.next_power_of_2(hidden_dim),
        "num_warps": max(
            min(triton.next_power_of_2(triton.cdiv(hidden_dim, 256)), max_warps), 4
        ),
    }

    if num_tokens >= 1024:
        config["num_warps"] = min(config["num_warps"], 8)

    use_pdl = is_arch_support_pdl()
    pdl_kwargs = {"launch_pdl": True} if use_pdl else {}

    _fused_gate_sigmoid_mul_add_kernel[(num_tokens,)](
        hidden_states,
        gate_weight,
        shared_output,
        output,
        hidden_dim=hidden_dim,
        DO_ADD=do_add,
        USE_PDL=use_pdl,
        **config,
        **pdl_kwargs,
    )


@patch(
    f"{_MODULE}._launch_fused_gate_sigmoid_mul",
    "around",
    feature=MODEL_COMPAT,
    row="D02",
    depends=(
        f"{_MODULE}._fused_gate_sigmoid_mul_add_kernel",
        "sglang.kernels.jit.utils.arch.is_arch_support_pdl",
        "sglang.srt.batch_invariant_ops.batch_invariant_ops.is_batch_invariant_mode_enabled",
        "sglang.srt.utils.common.is_cuda",
    ),
    reason=(
        "Main-only reference compatibility: #42087 caps shared-gate reduction "
        "at eight warps whenever global batch-invariant mode is on. Reference "
        "geometry at width2560 uses sixteen warps for rows<1024 and eight "
        "otherwise. Only target NVIDIA CUDA calls with global mode enabled "
        "run the frozen reference launcher on the native kernel; other calls "
        "delegate. The local helper digest guards every launch constant and "
        "the target/full-module and kernel fingerprints guard native arithmetic."
    ),
)
def _shared_gate_launch(original, hidden_states, gate_weight, shared_output, output, *, do_add):
    if not is_cuda() or not is_batch_invariant_mode_enabled() or not scope.target_model_active():
        return original(hidden_states, gate_weight, shared_output, output, do_add=do_add)
    return _launch_fused_gate_sigmoid_mul(
        hidden_states, gate_weight, shared_output, output, do_add=do_add
    )
