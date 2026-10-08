"""Monotonic request-row generations across flushes (inventory M04)."""

from sglang_qsa_hisparse import scope
from sglang_qsa_hisparse.features import MODEL_COMPAT
from sglang_qsa_hisparse.patching import patch


@patch(
    "sglang.srt.mem_cache.memory_pool.ReqToTokenPool.clear",
    "around",
    feature=MODEL_COMPAT,
    row="M04",
    depends=("sglang.srt.mem_cache.memory_pool.ReqToTokenPool.alloc_rows",),
    reason=(
        "model_compat: request-row generations read by overlap_utils and the "
        "DSpark planner stay monotonic across flushes, with or without the "
        "HiSparse runtime. Scope (rule 9): request-row generations are a generic "
        "path, so the pinned clear runs unchanged unless target_model_active(). "
        "Around hook (P3 narrowing of the fork's replace, which deletes the "
        "req_generation.zero_() line, memory_pool.py 348-350): the pinned clear "
        "zeroes req_generation in place and nothing else in it (free_slots, the "
        "MiniCPM aux cache) reads the generations, so restoring the snapshot "
        "afterwards equals not zeroing; HybridReqToTokenPool.clear reaches it "
        "through super().clear() as it reached the fork's body."
    ),
)
def _keep_generations(original, self):
    if not scope.target_model_active():
        return original(self)
    # Row identities remain monotonic across flushes. Physical lease owners
    # retain generations to reject callbacks from a released request; a
    # flush must not make a newly allocated row impersonate that request.
    generations = self.req_generation.clone()
    original(self)
    self.req_generation.copy_(generations)
