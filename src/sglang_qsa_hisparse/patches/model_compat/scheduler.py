"""Rebuilt HiSparse decode batches (inventory S03; deviation D5)."""

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
        "Production scheduler.py 3620-3621 verbatim, with `batch` as the result. "
        "Deviation D5 (owner-accepted 2026-10-08): with return_logprob the "
        "pinned body (and the fork) set token_ids_logprobs to every prompt "
        "token id; image pad ids exceed the vocabulary and crash decode, and "
        "text requests waste a whole-prompt gather whose result is dropped. "
        "The hook uses each request's own token_ids_logprob, as ScheduleBatch "
        "builds decode batches; nothing later in the body reads the field "
        "(SamplingBatchInfo.from_schedule_batch does not)."
    ),
)
def _carry_multimodal_inputs(batch, self, reqs):
    if not scope.target_model_active():
        return None
    # Rebuilt batches bypass prepare_for_extend, which normally sets these rows.
    batch.multimodal_inputs = [req.multimodal_inputs for req in reqs]
    if batch.return_logprob:  # D5: each request's own token-id logprob list.
        batch.token_ids_logprobs = [req.logprob.token_ids_logprob for req in reqs]
    return batch
