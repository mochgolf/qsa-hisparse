"""Reference launch geometry survives main global batch-invariant mode."""

import pytest
import torch

from sglang.srt.batch_invariant_ops import batch_invariant_ops as invariant
from sglang.srt.layers.attention.qsa import sparse_attn as native
from sglang_qsa_hisparse.kernels import qsa_sparse_attn as plugin


@pytest.mark.parametrize('global_invariant', [False, True])
@pytest.mark.parametrize('rows, expected', [
    (1, (32, 8, 2)), (33, (64, 8, 2)), (65, (64, 4, 2)),
    (129, (32, 4, 2)), (513, (16, 1, 2)), (2048, (16, 1, 2)),
])
def test_plugin_keeps_reference_sm89_launch_with_global_invariant_on(
    monkeypatch, global_invariant, rows, expected
):
    # Changes only the global flag read by the native selector, never CUDA
    # dispatch registrations or GPU state. Expected tuples are the frozen
    # reference 897286b12a SM89 table, independent of the imported candidate.
    monkeypatch.setattr(invariant, '_batch_invariant_MODE', global_invariant)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda device=None: 'RTX 4090')
    assert invariant.is_batch_invariant_mode_enabled() is global_invariant
    if global_invariant:
        assert native._get_best_config(rows) == (32, 4, 2)
    assert plugin._get_best_config(rows) == expected
