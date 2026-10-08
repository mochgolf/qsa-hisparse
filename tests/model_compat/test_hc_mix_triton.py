"""The fork's addition to test/registered/kernel/hyperconnection/test_hc_mix_triton.py.

Ported from ee8fe158d6 with its assertion unchanged; the helpers are the
fork file's own. Upstream moved ``hc_mix_triton`` unchanged to
``sglang.kernels.ops.gemm.hc_mix`` (#41243), so the import names it there. ``fused_hc_mix(..., stable=True)`` reaches the plugin's
stable kernel (H03/H06) through the patched pinned function, so W5's rows are
activated for the target model. Needs CUDA (the persistent kernel's grid
barrier cannot run under the Triton interpreter).
"""

import pytest
import torch

from sglang.kernels.ops.gemm.hc_mix import fused_hc_mix

HC_COUNT = 4
HIDDEN_SIZE = 2560
LOWRANK = 320


def _make_inputs(num_tokens: int, dtype: torch.dtype):
    torch.manual_seed(0)
    x = torch.randn(num_tokens, HC_COUNT * HIDDEN_SIZE, dtype=dtype, device="cuda")
    w_down = (
        torch.randn(LOWRANK, HC_COUNT * HIDDEN_SIZE, dtype=dtype, device="cuda") * 0.02
    )
    w_up = (
        torch.randn(HC_COUNT * HIDDEN_SIZE, LOWRANK, dtype=dtype, device="cuda") * 0.02
    )
    return x, w_down, w_up


@pytest.mark.gpu
def test_stable_fused_hc_mix_is_exact_across_batch_sizes(compat, target):
    x, w_down, w_up = _make_inputs(8, torch.bfloat16)
    one = fused_hc_mix(x[:1], w_down, w_up, HC_COUNT, HIDDEN_SIZE, stable=True)
    eight = fused_hc_mix(x, w_down, w_up, HC_COUNT, HIDDEN_SIZE, stable=True)
    assert torch.equal(one, eight[:1])
