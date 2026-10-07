"""Multimodal inputs on rebuilt HiSparse decode batches (inventory S03)."""

from sglang_qsa_hisparse import scope
from sglang_qsa_hisparse.features import MODEL_COMPAT
from sglang_qsa_hisparse.patching import patch


@patch(
    "sglang.srt.managers.scheduler.Scheduler._build_hisparse_decode_batch",
    "after",
    feature=MODEL_COMPAT,
    row="S03",
    depends=(
        "sglang.srt.managers.schedule_batch.ScheduleBatch.init_new",
        "sglang.srt.sampling.sampling_batch_info.SamplingBatchInfo.from_schedule_batch",
    ),
    reason=(
        "model_compat: changes upstream --enable-hisparse decode batches for "
        "multimodal requests. After hook: nothing later in the body reads "
        "batch.multimodal_inputs (SamplingBatchInfo.from_schedule_batch does "
        "not), so setting it on the returned batch equals the fork's insertion "
        "after ScheduleBatch.init_new. Scope (rule 9): the upstream HiSparse "
        "decode path is generic, so non-target models keep the pinned batch. "
        "Fork scheduler.py 3563-3564 verbatim, with `batch` as the result."
    ),
)
def _carry_multimodal_inputs(batch, self, reqs):
    if not scope.target_model_active():
        return None
    # Rebuilt batches bypass prepare_for_extend, which normally sets these rows.
    batch.multimodal_inputs = [req.multimodal_inputs for req in reqs]
    return batch
