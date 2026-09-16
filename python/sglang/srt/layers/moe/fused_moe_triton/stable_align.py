"""Canonical integer token placement for deterministic Marlin MoE execution."""

from typing import Tuple

import torch


def moe_align_block_size_stable(
    topk_ids: torch.Tensor, block_size: int, num_experts: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad expert buckets and order their pairs by flattened token/route index.

    Matches the native default contract: expert -1 occupies the filtered EP
    bucket, padding holds ``topk_ids.numel()``, and expert IDs mark each block.
    Valid input IDs are [-1, num_experts). All allocation shapes depend on input
    shapes, with no device-to-host synchronization or floating point reductions.

    Stable placement fixes run-to-run split-K grouping for identical inputs.
    Different batch shapes can still change Marlin's K-stripe partition.
    """
    flat = topk_ids.reshape(-1)
    num_pairs = flat.numel()
    capacity = (
        num_pairs * block_size
        if num_pairs < num_experts + 1
        else num_pairs + (num_experts + 1) * (block_size - 1)
    )
    buckets, order = torch.sort(flat.to(torch.int64) + 1, stable=True)
    bare_prefix = torch.searchsorted(
        buckets,
        torch.arange(num_experts + 2, device=flat.device, dtype=torch.int64),
    )
    counts = bare_prefix[1:] - bare_prefix[:-1]
    padded_counts = ((counts + block_size - 1) // block_size) * block_size
    padded_ends = torch.cumsum(padded_counts, dim=0)
    padding_offsets = padded_ends - padded_counts - bare_prefix[:-1]
    positions = torch.arange(num_pairs, device=flat.device) + padding_offsets[buckets]
    sorted_ids = torch.full(
        (capacity,), num_pairs, dtype=torch.int32, device=flat.device
    )
    # Stable sorting produces one unique destination for every live pair.
    sorted_ids.scatter_(0, positions, order.to(torch.int32))
    block_starts = (
        torch.arange(
            (capacity + block_size - 1) // block_size,
            device=flat.device,
            dtype=torch.int64,
        )
        * block_size
    )
    expert_ids = torch.searchsorted(padded_ends, block_starts, right=True) - 1
    expert_ids = torch.where(block_starts < padded_ends[-1], expert_ids, -1).to(
        torch.int32
    )
    return sorted_ids, expert_ids, padded_ends[-1:].to(torch.int32)
