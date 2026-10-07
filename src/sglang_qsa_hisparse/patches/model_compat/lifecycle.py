"""Monotonic request-row generations across flushes (inventory M04)."""

from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang_qsa_hisparse import scope
from sglang_qsa_hisparse.features import MODEL_COMPAT
from sglang_qsa_hisparse.patching import patch

# The pinned method for non-target models. Held in a dict because HookRegistry
# rebinds module-level references to a patched function to its wrapper.
_PINNED = {"clear": ReqToTokenPool.clear}


# Fork mem_cache/memory_pool.py:346-352, verbatim after the added scope check.
@patch(
    "sglang.srt.mem_cache.memory_pool.ReqToTokenPool.clear",
    "replace",
    feature=MODEL_COMPAT,
    row="M04",
    depends=("sglang.srt.mem_cache.memory_pool.ReqToTokenPool.alloc_rows",),
    reason=(
        "model_compat: request-row generations read by overlap_utils and the "
        "DSpark planner stay monotonic across flushes, with or without the "
        "HiSparse runtime. Scope (rule 9): request-row generations are a generic "
        "path, so the pinned clear runs unless target_model_active(); that check "
        "is the only edit to the fork body. Replace: an around that snapshots "
        "and restores req_generation would be equivalent but heavier."
    ),
)
def clear(self):
    if not scope.target_model_active():
        return _PINNED["clear"](self)
    self.free_slots = list(range(1, self._alloc_size))
    # Row identities remain monotonic across flushes. Physical lease owners
    # retain generations to reject callbacks from a released request; a
    # flush must not make a newly allocated row impersonate that request.
    if self._aux_cache is not None:
        self._aux_cache.clear()
