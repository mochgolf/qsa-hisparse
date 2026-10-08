"""Exact stable QSA top-k (inventory T01), moved from fork ``qsa/kernel.py``.

Copied verbatim from the reference (production ``897286b12a``, unchanged
since fork ``ee8fe158d6``) lines 12-59. Selection is routed here
by the ``qsa_fast_topk`` hook (T02) when deterministic inference is enabled.
"""

import torch

# Account conservatively for gathered fp32 scores, masks, int64 indices, sort
# outputs/workspace and the preceding tile. This bounds wide explicit tensors;
# actual allocator peaks remain part of GPU qualification, not this estimate.
_QSA_DETERMINISTIC_TOPK_TILE_BYTES = 16 * 1024 * 1024
_QSA_DETERMINISTIC_TOPK_BYTES_PER_SCORE = 64


def _qsa_deterministic_topk_tile_rows(rows: int, width: int) -> int:
    if width <= 0:
        return max(rows, 1)
    return max(
        1,
        min(
            rows,
            _QSA_DETERMINISTIC_TOPK_TILE_BYTES
            // (width * _QSA_DETERMINISTIC_TOPK_BYTES_PER_SCORE),
        ),
    )


def _qsa_stable_topk(logits, starts, lengths, topk):
    """Exact stable selection with static row tiles and no device scalar reads."""
    rows, width = logits.shape
    output = torch.full((rows, topk), -1, dtype=torch.int32, device=logits.device)
    if rows == 0 or width == 0:
        return output
    columns = torch.arange(width, device=logits.device, dtype=torch.int64)
    tile_rows = _qsa_deterministic_topk_tile_rows(rows, width)
    selected_width = min(topk, width)
    sentinel = torch.iinfo(torch.int64).max
    for begin in range(0, rows, tile_rows):
        end = min(begin + tile_rows, rows)
        tile_lengths = lengths[begin:end, None]
        # Place each valid interval first in relative-index order. Stable ties
        # therefore prefer its lower logical indices, including valid -inf
        # scores ahead of the -inf padding that follows the interval.
        absolute = starts[begin:end, None].long() + columns
        absolute.clamp_(max=width - 1)
        scores = logits[begin:end].gather(1, absolute)
        scores.masked_fill_(columns >= tile_lengths, -float("inf"))
        ranked = torch.argsort(scores, dim=-1, descending=True, stable=True)
        selected = ranked[:, :selected_width]
        selected = torch.where(selected < tile_lengths, selected, sentinel)
        ordered = selected.sort(dim=-1).values
        output[begin:end, :selected_width] = torch.where(
            ordered == sentinel, -1, ordered
        ).to(torch.int32)
    return output
