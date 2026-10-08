"""Row I1 (Track I): carry each Qwen-VL image item's full artifact key.

At the pin the artifact key (content digest, modality, processor fingerprint
and every preprocessing kwarg) identifies the processor output, but only a
64-bit hash of it reaches ``MultimodalDataItem.hash`` and a 30-bit value the
prompt (``pad_value``). This hook records the full key in the item's
``model_specific_data``, which the tokenizer-to-scheduler transport (msgpack
or pickle) and the TP broadcast carry unchanged. Items without the key (the
uncached processor path, tokenizer-worker subprocesses, which do not load
plugins at the pin) bypass host prefixes (``hisparse.image_request``).
"""

from sglang_qsa_hisparse.hisparse.image_request import ARTIFACT_KEY
from sglang_qsa_hisparse.patching import patch

QWEN_VL = "sglang.srt.multimodal.processors.qwen_vl.QwenVLImageProcessor"
MEDIA = "sglang.srt.multimodal.media_artifacts.base.MediaArtifactCacheMixin"


@patch(
    f"{QWEN_VL}.compose_image_artifacts",
    "after",
    feature="hisparse",
    row="I1",
    depends=(
        "sglang.srt.multimodal.cache.identity.build_artifact_key",
        f"{MEDIA}._artifact_key",
        "sglang.srt.managers.schedule_batch.MultimodalDataItem",
        "sglang.srt.managers.schedule_batch.MultimodalInputs.from_processor_output",
    ),
    reason=(
        "hisparse: only host prefixes read the key. Upstream reads named "
        "model_specific_data keys (none is 'artifact_key') and passes other "
        "values through (IPC proxy scans, item splits), so the string is inert. "
        "Not scoped (rule 9 covers model_compat; the tokenizer process publishes "
        "no served-model config). After hook: the method builds one item per "
        "artifact, in artifact order, from a deepcopy of the artifact's "
        "model_specific_data (so the cached artifact is not modified), and "
        "returns None when it falls back to the uncached path."
    ),
)
def record_artifact_keys(composed, processor, input_text, artifacts):
    if composed is not None:
        for item, artifact in zip(composed.mm_items, artifacts, strict=True):
            item.model_specific_data[ARTIFACT_KEY] = artifact.artifact_key
