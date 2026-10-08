"""Image requests as the scheduler holds them (``Req.multimodal_inputs``).

Items are real ``MultimodalDataItem`` objects shaped like the Qwen-VL fast
path's output after transport (one inclusive offset span, an
``image_grid_thw`` of shape ``[1, 3]`` and the I1 artifact key).
"""

from types import SimpleNamespace

import torch

from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle
from sglang_qsa_hisparse.hisparse.image_request import ARTIFACT_KEY

PAD = 7  # One pad value for every image: tokens collide unless stated.


def item(key, start, stop, grid=(1, 4, 8), *, modality=Modality.IMAGE, **fields):
    """A transported item covering tokens ``[start, stop)``."""
    data = {"image_grid_thw": torch.tensor([grid])}
    if key is not None:
        data[ARTIFACT_KEY] = key
    return MultimodalDataItem(
        modality=modality,
        offsets=[(start, stop - 1)],
        model_specific_data=data,
        **fields,
    )


def tokens(length, items, pad=PAD):
    ids = list(range(1000, 1000 + length))
    for it in items:
        ((start, end),) = it.offsets
        ids[start : end + 1] = [pad] * (end + 1 - start)
    return ids


def positions(length, shift_from=None):
    """Text-like M-RoPE positions; ``shift_from`` changes those at and after it."""
    value = torch.arange(length).repeat(3, 1)
    if shift_from is not None:
        value[1, shift_from:] += 1
    return value


def mm_inputs(items, length):
    return MultimodalInputs(mm_items=list(items), mrope_positions=positions(length))


def image_req(name, items, length, attempt=0):
    return SimpleNamespace(
        rid=name,
        cache_request_handle=CacheRequestHandle(name, attempt),
        multimodal_inputs=mm_inputs(items, length),
    )
