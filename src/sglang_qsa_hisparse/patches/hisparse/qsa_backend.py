"""QSA backend runtime construction and hooks (inventory Q03, Q06, Q07).

The attention bodies that consume ``self.qsa_hisparse`` are model_compat
REPLACE copies (``patches/model_compat/qsa_attention.py``, rule 2); this
feature only builds the runtime, keeps the pin's fused #40972 KV path off
while it is attached (the reference's ``_fused_kv_eligible``), and adds
before/after hooks.
"""

import os

from sglang_qsa_hisparse import scope
from sglang_qsa_hisparse.errors import PluginActivationError
from sglang_qsa_hisparse.features import HISPARSE
from sglang_qsa_hisparse.hisparse.depends import RUNTIME_DEPENDS
from sglang_qsa_hisparse.patching import attach_value, patch

_BACKEND = "sglang.srt.layers.attention.qwen_sparse_attn_backend.QwenSparseAttnBackend"


def _fused_kv_eligible(self) -> bool:
    """Whether the fused #40972 KV fast path may own this pool.

    It writes the row and packs the sparse selection in one kernel, which
    would bypass the QSA offload contract (write_locations, after_store,
    selected, lease-aware packing) that our ChunkCache/host-prefix runtime
    depends on.
    """
    return self.qsa_hisparse is None and self._supports_fused_kv_pool(
        self.token_to_kv_pool
    )


_Q03_DEPENDS = tuple(RUNTIME_DEPENDS) + (
    f"{_BACKEND}._supports_fused_kv_pool",
    f"{_BACKEND}._can_defer_block_expansion",
)
attach_value(
    _BACKEND,
    "_fused_kv_eligible",
    _fused_kv_eligible,
    feature=HISPARSE,
    row="Q03",
    depends=_Q03_DEPENDS,
    reason=(
        "hisparse: member added by the reference (production merge "
        "80dc48ddfc); with no runtime it equals the pin's "
        "_supports_fused_kv_pool(token_to_kv_pool). Called by the Q03 hook; "
        "the reference's other call (the late pool binding in "
        "_capture_cuda_graph_metadata) is reached only without a runtime "
        "(attaching one needs the pool at construction), where the pin's "
        "expression is equal, so it needs no hook."
    ),
)


@patch(
    f"{_BACKEND}.__init__",
    "after",
    feature=HISPARSE,
    row="Q03",
    depends=_Q03_DEPENDS,
    reason=(
        "hisparse: env-gated (SGLANG_QSA_HISPARSE_V3). after: the reference's "
        "block is followed only by plain assignments that nothing in __init__ "
        "reads. Reference lines 282-299 with the runtime imports rewritten to "
        "sglang_qsa_hisparse.hisparse; the preceding qsa_hisparse = None is the "
        "model_compat Q02 hook, which sets it only when absent, so hook order is "
        "irrelevant (section 2, C3). The reference computes "
        "_fused_kv_pool_eligible after attaching the runtime (moved down from "
        "the pin's first lines); with a runtime the hook recomputes it, "
        "otherwise the pin's value is the reference's. Plugin addition: "
        "rejects a non-target model (unsupported combination)."
    ),
)
def _attach_hisparse_runtime(result, self, runner=None):
    if runner is not None and os.environ.get("SGLANG_QSA_HISPARSE_V3"):
        # The runtime relies on the model_compat bodies, which run only for
        # the target model (rule 9); other models are unsupported.
        if not scope.target_model_active():
            raise PluginActivationError(
                "SGLANG_QSA_HISPARSE_V3 requires the target model "
                f"{sorted(scope.TARGET_ARCHITECTURES)}"
            )
        from sglang_qsa_hisparse.hisparse.single_request import QSAHiSparseSingleRequest

        mode = os.environ["SGLANG_QSA_HISPARSE_V3"]
        if mode in ("p2-offload", "p2-resident"):
            from sglang_qsa_hisparse.hisparse.runtime import QSAHiSparseRuntime

            self.qsa_hisparse = QSAHiSparseRuntime(runner, mode)
        else:
            self.qsa_hisparse = QSAHiSparseSingleRequest(runner, mode)
        self.token_to_kv_pool.qsa_hisparse = self.qsa_hisparse
        # The fused #40972 write/pack path owns the KV row and the deferred
        # block expansion; the QSA offload runtime owns both instead
        # (write_locations/after_store/selected and the FP8 descales), so it
        # stays off whenever the runtime is live.
        self._fused_kv_pool_eligible = self._fused_kv_eligible()


@patch(
    f"{_BACKEND}.init_forward_metadata",
    "before",
    feature=HISPARSE,
    row="Q06",
    reason=(
        "hisparse: begin_batch on the runtime. before: the fork inserts it "
        "after the idle early return and before any other statement, so the "
        "hook skips idle batches."
    ),
)
def _begin_hisparse_batch(self, forward_batch):
    if forward_batch.forward_mode.is_idle():
        return None
    if self.qsa_hisparse is not None:
        self.qsa_hisparse.begin_batch(forward_batch)
    return None


@patch(
    f"{_BACKEND}._capture_cuda_graph_metadata",
    "after",
    feature=HISPARSE,
    row="Q07",
    depends=(f"{_BACKEND}._is_speculative_paged_mode",),
    reason=(
        "hisparse: plans the SM89 FlashInfer ragged FA2 graph wrapper for P2 "
        "graph decode. after: the fork's block is the last statement; all inputs "
        "are keyword-only arguments and metadata_rows is recomputed with the "
        "fork's expression (fork line 932, pure). Calls the model_compat Q09 "
        "members (section 2, C2)."
    ),
)
def _plan_fa2_graph_wrapper(
    result, self, *, bs, num_tokens, req_pool_indices, seq_lens, forward_mode, spec_info
):
    metadata_rows = num_tokens if self._is_speculative_paged_mode(forward_mode) else bs
    if (
        metadata_rows >= 1
        and forward_mode.is_decode()
        and spec_info is None
        and self.qsa_profile is not None
        and self.qsa_hisparse is not None
        and getattr(self.qsa_hisparse, "uses_qsa_hisparse_leases", False)
        and getattr(self.qsa_hisparse, "graph_enabled", False)
    ):
        shape = self._qsa_local_head_shape()
        if shape is not None:
            self._ensure_fa2_graph_wrapper(metadata_rows, *shape[1:])
