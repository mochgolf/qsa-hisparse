"""PLE gather from a host table with int8 and per-row scale support (inventory E04).

``_gather_ple_embedding_from_pinned_kernel`` copied verbatim from fork
``python/sglang/srt/models/qwen4_exp.py`` at ee8fe158d6; E07's ``gather`` copy
launches it instead of the pinned kernel.
"""

import triton
import triton.language as tl


@triton.jit
def _gather_ple_embedding_from_pinned_kernel(
    weight_ptr,
    ids_ptr,
    output_ptr,
    row_scale_ptr,
    embedding_dim,
    tp_vocab_start,
    tp_vocab_end,
    is_fp8: tl.constexpr,
    is_int8: tl.constexpr,
    has_row_scale: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row_id = tl.program_id(0)
    global_idx = tl.load(ids_ptr + row_id)
    in_range = (global_idx >= tp_vocab_start) & (global_idx < tp_vocab_end)
    local_idx = tl.where(in_range, global_idx - tp_vocab_start, 0)
    offsets = tl.arange(0, BLOCK_D)
    mask = offsets < embedding_dim
    if is_fp8:
        weight_ptr = weight_ptr.to(tl.int64).to(tl.pointer_type(tl.float8e4nv))
    elif is_int8:
        weight_ptr = weight_ptr.to(tl.int64).to(tl.pointer_type(tl.int8))
    else:
        weight_ptr = weight_ptr.to(tl.int64).to(tl.pointer_type(tl.bfloat16))
    values = tl.load(
        weight_ptr + local_idx * embedding_dim + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.bfloat16)
    if has_row_scale:
        row_scale_ptr = row_scale_ptr.to(tl.int64).to(tl.pointer_type(tl.bfloat16))
        scale = tl.load(row_scale_ptr + local_idx).to(tl.float32)
        values = (values.to(tl.float32) * scale).to(tl.bfloat16)
    tl.store(
        output_ptr + row_id * embedding_dim + offsets,
        tl.where(in_range, values, 0.0),
        mask=mask,
    )
