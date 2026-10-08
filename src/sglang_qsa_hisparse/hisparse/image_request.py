"""Image identity of a scheduler request (Track I: I1 transport, I2 key).

Row I1 (``patches/hisparse/image_identity.py``) records each Qwen-VL image
item's full artifact key in ``model_specific_data[ARTIFACT_KEY]``; it reaches
the scheduler with the item. ``identity_for(req)`` turns a request's items
and M-RoPE positions into the ``ImagePrefixIdentity`` of ``image_identity.py``.
"""

from sglang.srt.environ import envs
from sglang_qsa_hisparse.hisparse.image_identity import (
    ImagePrefixIdentity,
    ImageRecord,
    page_digests,
)

ARTIFACT_KEY = "artifact_key"
# Multimodal input that host prefixes do not support: the request bypasses
# host prefixes entirely (no match, no capture).
BYPASS = object()


def identity_for(req):
    """``ImagePrefixIdentity``, ``None`` for a text request, or ``BYPASS``.

    Computed once per request attempt and M-RoPE length (a retracted request
    extends its positions over its output tokens), cached on the request.
    """
    mm = getattr(req, "multimodal_inputs", None)
    if mm is None:
        return None
    positions = getattr(mm, "mrope_positions", None)
    attempt = (
        req.cache_request_handle,
        None if positions is None else positions.shape[-1],
    )
    cached = getattr(req, "qsa_image_identity", None)
    if cached is None or cached[0] != attempt:
        cached = req.qsa_image_identity = (attempt, _identity(mm, positions))
    return cached[1]


def _identity(mm, positions):
    items = getattr(mm, "mm_items", None)
    if positions is None or not items or envs.SGLANG_MM_SKIP_COMPUTE_HASH.get():
        return BYPASS
    records = []
    for order, item in enumerate(items):
        key = item.model_specific_data.get(ARTIFACT_KEY)
        if (
            not item.is_image()
            or item.is_precomputed_embedding()
            or item.precomputed_embeddings is not None
            or item.offsets is None
            or len(item.offsets) != 1
            or key is None
        ):
            return BYPASS
        ((start, end),) = item.offsets  # SGLang offsets end inclusively.
        grid = item.model_specific_data["image_grid_thw"].reshape(-1).tolist()
        records.append(ImageRecord(key, order, start, end + 1, tuple(grid)))
    records.sort(key=lambda record: record.start)
    return ImagePrefixIdentity(
        tuple(records), page_digests(positions, positions.shape[-1])
    )
