"""Track I (I-B): Qwen-VL-like image prompts and a stand-in for I-A's identity.

A prompt is laid out as the Qwen-VL processor lays it out: ``<vision_start>``,
the image placeholder tokens (replaced by the item's pad value, as the
scheduler pads them), ``<vision_end>``. The M-RoPE table comes from the
pinned ``get_rope_index`` over the unpadded ids, like ``compute_mrope_positions``.
"""

import hashlib
from array import array
from types import SimpleNamespace

import torch

from sglang.srt.layers.rotary_embedding.mrope_rope_index import get_rope_index
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang_qsa_hisparse.hisparse.image_identity import (
    ImagePrefixIdentity,
    ImageRecord,
    page_digests,
)

VOCAB = 512
IMAGE_TOKEN, VISION_START, VISION_END, VIDEO_TOKEN = 500, 501, 502, 503
MERGE = 2  # Qwen-VL spatial merge: an (t, h, w) grid yields t*(h/2)*(w/2) tokens.
BYPASS = object()  # Any non-identity value I-A may return for unsupported inputs.

# Page64 boundaries of STANDARD: 64 between A and B, 128 inside B, 192 at C's
# first token, 256 after every image.
STANDARD = (("A", 21, (1, 8, 8)), ("B", 90, (1, 20, 20)), ("C", 192, (1, 12, 12)))


def image_prompt(length, images=STANDARD):
    """``images``: (artifact key, first image token, grid) in token order."""
    unpadded = [i * 7 % 400 + 1 for i in range(length)]
    padded = list(unpadded)
    items, spans = [], []
    for key, start, grid in images:
        stop = start + grid[0] * (grid[1] // MERGE) * (grid[2] // MERGE)
        unpadded[start - 1], unpadded[stop] = VISION_START, VISION_END
        unpadded[start:stop] = [IMAGE_TOKEN] * (stop - start)
        item = MultimodalDataItem(modality=Modality.IMAGE, offsets=[(start, stop - 1)])
        item.set_hash(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little"))
        item.feature = torch.full((stop - start, 4), float(item.hash % 251))
        item.image_grid_thw = torch.tensor([grid])
        item.artifact_key = key
        padded[start - 1 : stop + 1] = unpadded[start - 1 : stop + 1]
        padded[start:stop] = [item.pad_value] * (stop - start)
        items.append(item)
        spans.append((key, start, stop, grid))
    positions, delta = get_rope_index(
        spatial_merge_size=MERGE,
        image_token_id=IMAGE_TOKEN,
        video_token_id=VIDEO_TOKEN,
        vision_start_token_id=VISION_START,
        model_type="qwen4_exp",
        input_ids=torch.tensor([unpadded]),
        image_grid_thw=torch.tensor([grid for _, _, grid in images]),
    )
    return SimpleNamespace(
        length=length,
        ids=array("q", padded),
        spans=spans,
        mm=MultimodalInputs(
            mm_items=items,
            mrope_positions=positions.squeeze(1),
            mrope_position_delta=delta,
        ),
    )


def identity_of(prompt):
    """What I-A's identity_for returns for a supported image request."""
    return ImagePrefixIdentity(
        records=tuple(
            ImageRecord(key, order, start, stop, grid)
            for order, (key, start, stop, grid) in enumerate(prompt.spans)
        ),
        page_digests=page_digests(prompt.mm.mrope_positions, prompt.length),
    )
