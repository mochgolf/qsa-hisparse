"""Plugin-owned copy of the JIT fast top-k op wrapper (inventory T05).

Production (fork PR #6, ``773f3c2d84``) keeps every candidate when a radix
threshold bin overflows the 4K shared-memory stage (full-row refinement)
instead of dropping them; the fix is in ``fast_topk.cuh`` only, shipped here
byte for byte as ``csrc/fast_topk/fast_topk.cuh``. This wrapper is
``python/sglang/kernels/ops/attention/fast_topk.py`` (identical at the pin and
in production) with mechanical edits (rule 3), all in ``_jit_fast_topk_module``
and the module call, so the plugin module never shares a JIT module name,
cache directory or export symbol with the in-tree op (as for the Marlin copy,
inventory section 3):
- ``load_jit`` marker ``"fast_topk"`` -> ``"qsa_hisparse_fast_topk"``;
- ``cuda_files`` names the plugin copy of ``fast_topk.cuh`` by absolute path
  (its includes are ``<sgl_kernel/...>`` and ``<tvm/...>`` only);
- export ``fast_topk`` -> ``qsa_hisparse_fast_topk`` (also at the
  ``module.<export>(...)`` call).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_CSRC = Path(__file__).resolve().parent / "csrc"

_FAST_TOPK_SUPPORTED_K = (512, 2048)


@cache_once
def _jit_fast_topk_module(topk: int) -> Module:
    """Compile and cache the JIT fast top-k module for a given k."""
    # Checks on the compile key live here, not in `fast_topk`: `cache_once`
    # keys on `topk`, so this runs once per specialisation.
    if topk not in _FAST_TOPK_SUPPORTED_K:
        raise RuntimeError(
            f"Unsupported topk {topk}. Supported: {_FAST_TOPK_SUPPORTED_K}"
        )
    args = make_cpp_args(topk, is_arch_support_pdl())
    return load_jit(
        "qsa_hisparse_fast_topk",
        *args,
        cuda_files=[str(_CSRC / "fast_topk" / "fast_topk.cuh")],
        cuda_wrappers=[("qsa_hisparse_fast_topk", f"FastTopKKernel<{args}>::run")],
    )


def fast_topk(
    score: torch.Tensor,
    lengths: torch.Tensor,
    topk: int,
    row_starts: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Per-row top-k selection over a fp32 score matrix.

    Row b selects the `topk` largest values in
    ``score[b, row_starts[b] : row_starts[b] + lengths[b]]`` and returns their
    indices relative to ``row_starts[b]``. Slots beyond ``lengths[b]`` are -1.
    Output order within a row is unspecified (atomic collection order).

    Parameters
    ----------
    score      : CUDA fp32 tensor [B, L]
    lengths    : CUDA int32 tensor [B]
    topk       : number of indices per row; 512 or 2048
    row_starts : optional CUDA int32 tensor [B]; defaults to zeros

    Returns
    -------
    CUDA int32 tensor [B, topk]
    """
    batch = score.shape[0]
    if row_starts is None:
        row_starts = torch.zeros(batch, dtype=torch.int32, device=score.device)
    indices = score.new_empty((batch, topk), dtype=torch.int32)

    module = _jit_fast_topk_module(topk)
    module.qsa_hisparse_fast_topk(score, row_starts, indices, lengths)
    return indices
