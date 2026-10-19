"""KV allocation, release, invalidation and tree-cache selection (inventory W2).

Rows B06, M01, M02, M03 and M05. Release order for a request with a QSA
lease: the tree's ``claim_kv_row`` (a pending host restore must drain through
rollback first), prefix capture in the tree's ``checkpoint`` (whose disabled
UnifiedRadixCache then frees a finished request's mamba slot), runtime release
(drains GPU and copy events), the tree cache's ``free_kv_row`` (logical free,
deferred inside a free group), ``on_release`` (frees the mamba slot when the
request was not checkpointed), ``req_to_token_pool.free``,
``mark_kv_released``, ``after_release(lease)``, then the allocator's
``free_group_end`` and ``after_logical_flush``, after which the physical slot
can be reused. M05 builds ``QSAHostPrefixCache`` (a UnifiedRadixCache) under
``--disable-radix-cache``; M01 records and rolls back its allocations. The
REPLACE hook is v0.5.22's definition plus production's change (PLAN.md rules
3 and P1) and runs with this module's globals, which are imported from the
pinned module that defines the target (inventory 6, G3).
"""

from __future__ import annotations

import os

from sglang.srt.mem_cache.common import _release_overallocated_kv_indices
from sglang.srt.mem_cache.registry import create_unified_radix_cache
from sglang_qsa_hisparse.features import HISPARSE
from sglang_qsa_hisparse.hisparse.prefix_cache import QSAHostPrefixCache
from sglang_qsa_hisparse.patching import patch

ALLOCATION = "sglang.srt.mem_cache.allocation"
REGISTRY = "sglang.srt.mem_cache.registry"
URC = "sglang.srt.mem_cache.unified_radix_cache.UnifiedRadixCache"
WEIGHT_UPDATER = (
    "sglang.srt.managers.scheduler_components.weight_updater."
    "SchedulerWeightUpdaterManager"
)


@patch(
    f"{WEIGHT_UPDATER}._observe_weight_load",
    "before",
    feature=HISPARSE,
    row="B06",
    depends=(
        f"{WEIGHT_UPDATER}.update_weights_from_disk",
        f"{WEIGHT_UPDATER}.update_weights_from_distributed",
        f"{WEIGHT_UPDATER}.update_weights_from_tensor",
        f"{WEIGHT_UPDATER}.update_weights_from_ipc",
    ),
    reason=(
        "hisparse: only QSAHostPrefixCache defines invalidate_model. Before hook "
        "on the @contextmanager: all four callers use `with "
        "self._observe_weight_load(...)`, so this runs directly before the "
        "generator prefix the fork extended; an exception leaves the with "
        "statement before its body and metrics in both. Production "
        "weight_updater.py 111-116 verbatim."
    ),
)
def _invalidate_host_prefixes(self, source):
    # Host checkpoints cannot survive any attempted model mutation, even
    # when the caller requests no ordinary KV flush or a partial load fails.
    cache = getattr(self.scheduler, "tree_cache", None)
    invalidate = getattr(cache, "invalidate_model", None)
    if invalidate is not None:
        invalidate()


@patch(
    "sglang.srt.mem_cache.allocator.paged.PagedTokenToKVPoolAllocator.free_group_end",
    "after",
    feature=HISPARSE,
    row="M02",
    depends=(
        "sglang.srt.mem_cache.allocator.paged.PagedTokenToKVPoolAllocator."
        "_release_page_ids",
        "sglang.srt.mem_cache.allocator.base.BaseTokenToKVPoolAllocator.free_group_end",
    ),
    reason=(
        "hisparse: requires kvcache.qsa_hisparse, which no upstream kvcache has. "
        "After hook: the fork inserts at the end of the body, which has no early "
        "return. Production paged.py 336-340 verbatim."
    ),
)
def _notify_logical_flush(result, self):
    adapter = getattr(self.get_kvcache(), "qsa_hisparse", None)
    if getattr(adapter, "uses_qsa_hisparse_leases", False):
        adapter.after_logical_flush()
    elif adapter is not None and adapter.pending_release is not None:
        adapter.after_release(adapter.pending_release)


@patch(
    f"{ALLOCATION}.alloc_for_extend",
    "around",
    feature=HISPARSE,
    row="M01",
    depends=(
        f"{ALLOCATION}._prefix_kv_indices",
        f"{ALLOCATION}.alloc_req_slots",
        f"{ALLOCATION}.write_cache_indices",
        f"{ALLOCATION}.alloc_paged_token_slots_extend",
        f"{URC}.prefix_device_indices",
        f"{URC}.maybe_hand_to_session",
    ),
    reason=(
        "hisparse: acts only for QSAHostPrefixCache. Around hook (v0.5.22 "
        "seams; was a replace with a rollback wrapper): production's "
        "prefix steps are cache methods upstream now calls itself "
        "(prefix_device_indices before alloc_req_slots for a request without a "
        "row and prefix_len > 0, maybe_hand_to_session per request after the "
        "row write and the kv_allocated_len update), so the hook only records "
        "each request first (prepare_prefix_for_extend, production's per-request "
        "TP convergence) and, on any exception, rolls back (production's "
        "wrapper). The rollback reads a request's new suffix from its row once "
        "kv_allocated_len moved; before that upstream either allocated no "
        "suffix or freed out_cache_loc itself (alloc_aux_to_lengths, which has "
        "no aux cache on QSA pools)."
    ),
)
def _roll_back_failed_extend(original, batch):
    cache = batch.tree_cache
    if not isinstance(cache, QSAHostPrefixCache):
        return original(batch)
    try:
        cache.prepare_prefix_for_extend(batch.reqs)
        return original(batch)
    except BaseException as error:
        try:
            cache.rollback_prefix_for_extend(batch.reqs)
        except BaseException as cleanup_error:
            error.add_note(f"QSA prefix allocation rollback retained ownership: {cleanup_error!r}")
        raise


# v0.5.22 mem_cache/common.py:378-422 with production's change.
@patch(
    "sglang.srt.mem_cache.common.release_kv_cache",
    "replace",
    feature=HISPARSE,
    row="M03",
    depends=(
        "sglang.srt.mem_cache.common._release_overallocated_kv_indices",
        "sglang.srt.mem_cache.memory_pool.ReqToTokenPool.free",
        "sglang.srt.mem_cache.base_prefix_cache.BasePrefixCache.claim_kv_row",
        "sglang.srt.mem_cache.base_prefix_cache.BasePrefixCache.checkpoint",
        "sglang.srt.mem_cache.base_prefix_cache.BasePrefixCache.free_kv_row",
        f"{URC}.checkpoint_into_tree",
        f"{URC}.on_release",
        "sglang.srt.mem_cache.unified_cache.components.mamba.MambaComponent."
        "cleanup_after_caching_req",
    ),
    reason=(
        "hisparse: getattr on the kvcache; without the QSA runtime nothing is "
        "added. Replace: the lease release sits between the tree's checkpoint "
        "(where QSAHostPrefixCache captures the prefix, observing the live "
        "lease) and the logical free in free_kv_row, and after_release follows "
        "req_to_token_pool.free / mark_kv_released. A streaming session that "
        "claims the row returns before any QSA call (claim_kv_row is the first "
        "release hook). Kept at v0.5.22 (design 3.1.6 fallback): the drain also "
        "serves QSA leases without host prefixes, whose tree is upstream's "
        "disabled UnifiedRadixCache, so a QSAHostPrefixCache free_kv_row "
        "override cannot carry it, and an upstream seam would need the lease "
        "threaded between calls. Hand merge (RESOLVED): production's mamba "
        "free for a tree without mamba support is dropped; the disabled "
        "UnifiedRadixCache (and QSAHostPrefixCache) frees the slot in "
        "checkpoint or on_release, and create_tree_cache rejects every other "
        "hybrid-SSM tree."
    ),
)
def release_kv_cache(req: Req, tree_cache: BasePrefixCache, *, checkpoint: bool):
    """Give the request's kv row back; with ``checkpoint`` the tree first keeps
    what it can key."""
    assert (not req.kv.holds_kv) == req.kv.is_kv_released
    # A mamba-capable cache may alloc mamba state before alloc KV cache
    if not req.kv.holds_kv:
        assert tree_cache.supports_mamba(), (
            "Only a mamba-capable tree cache allows freeing before alloc"
        )
        # TODO (csy, hanming): clean up this early allocation logic
        if req.kv.holds_mamba:
            tree_cache.req_to_token_pool.mamba_allocator.free(
                req.kv.mamba_pool_idx.unsqueeze(-1)
            )
            req.kv.mamba_pool_idx = None
        return
    if tree_cache.claim_kv_row(req):
        # A streaming session detached the kv record to keep the row.
        assert not req.kv.holds_kv
        return

    # The QSA host-prefix adapter captures its CPU snapshot inside the tree's
    # ``checkpoint`` hook, and that capture must observe the live request lease.
    qsa_hisparse = getattr(
        tree_cache.token_to_kv_pool_allocator.get_kvcache(), "qsa_hisparse", None
    )
    owned_kv_len = req.owned_kv_len()
    checkpoint = checkpoint and not req.skip_radix_cache_insert
    if checkpoint:
        # A tree that takes over component state (mamba) must see the request
        # finished, or the insert forks the state and the slot leaks.
        assert req.finished() or not tree_cache.supports_mamba(), (
            f"releasing unfinished request {req.rid} into a mamba tree"
        )
        # The fill-id array lags output_ids until the next prepare_for_decode.
        req.refresh_fill_ids()
        tree_cache.checkpoint(req, up_to=owned_kv_len)
    if qsa_hisparse is not None:
        qsa_lease = qsa_hisparse.release(req.kv.req_pool_idx, req.rid)
    # The protected prefix is not this req's to free.
    tree_cache.free_kv_row(req.kv, [(req.kv.cache_protected_len, owned_kv_len)])
    tree_cache.unlock(req.lock)
    req.lock = None
    _release_overallocated_kv_indices(
        req, owned_kv_len, req.kv.kv_allocated_len, tree_cache
    )
    tree_cache.on_release(req, checkpointed=checkpoint)

    # The DSV4-NPU ReqToTokenPool subclass's free() additionally releases the
    # c4/c128 state pages; other ReqToTokenPool subclasses are a no-op here.
    tree_cache.req_to_token_pool.free(req)
    req.kv.mark_kv_released()
    if qsa_hisparse is not None:
        qsa_hisparse.after_release(qsa_lease)


# Production mem_cache/registry.py:81-102, verbatim except the config import,
# rewritten to the plugin package. Called by M05's hook.
def qsa_private_host_prefix_active(params: CacheInitParams) -> bool:
    """Whether this build serves the private QSA host-prefix adapter.

    All three conditions must hold, so the override cannot leak into other
    hybrid models that happen to have an env var set:

    * ``SGLANG_QSA_HISPARSE_V3`` selects the multi-request offload runtime that
      owns the lease/DMA contract (``qsa_hisparse/config.py`` rejects a prefix
      budget on any other mode);
    * a non-zero host-prefix budget, which is what installs the adapter;
    * the pool is the QSA pool, whose layout the adapter's logical pages and
      the scheduler type guard assume.
    """
    from sglang_qsa_hisparse.hisparse.config import prefix_cache_options
    from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool

    if os.environ.get("SGLANG_QSA_HISPARSE_V3") not in ("p2-offload", "p2-resident"):
        return False
    budget, _ = prefix_cache_options()
    if budget <= 0:
        return False
    return isinstance(params.token_to_kv_pool_allocator.get_kvcache(), QSATokenToKVPool)


@patch(
    f"{REGISTRY}.default_radix_cache_factory",
    "around",
    feature=HISPARSE,
    row="M05",
    depends=(
        f"{REGISTRY}.create_tree_cache",
        f"{REGISTRY}.create_unified_radix_cache",
        "sglang.srt.mem_cache.qsa_kv_pool.QSATokenToKVPool",
        # The runtime is attached (Q03) before the cache is built.
        "sglang.srt.managers.scheduler.Scheduler.__init__",
        "sglang.srt.managers.scheduler.Scheduler.init_model_worker",
        "sglang.srt.managers.scheduler.Scheduler.init_all_attention_backends",
        "sglang.srt.managers.schedule_batch.Req.init_next_round_input",
        "sglang.srt.mem_cache.common.match_kv_cache",
        "sglang.srt.mem_cache.common.checkpoint_kv_cache",
        "sglang.srt.mem_cache.base_prefix_cache.BasePrefixCache.finish",
        URC,
        f"{URC}.__init__",
        f"{URC}.reset",
        f"{URC}.match_prefix",
        f"{URC}.init_load_back",
        f"{URC}.prefix_device_indices",
        f"{URC}.maybe_hand_to_session",
        f"{URC}.claim_kv_row",
        f"{URC}.checkpoint",
        f"{URC}.on_release",
        f"{URC}.release_aborted_request",
        f"{URC}.release_host_resources",
        "sglang.srt.session.streaming_session.StreamingSession.take",
        # Read by prefix_cache.py (and image_request.py) itself: match
        # parameters and result, the key, force-miss and Track I env switches,
        # the rows, slots and pages the rollback frees.
        "sglang.srt.mem_cache.base_prefix_cache.MatchPrefixParams",
        "sglang.srt.mem_cache.base_prefix_cache.MatchResult",
        "sglang.srt.mem_cache.radix_cache.RadixKey",
        "sglang.srt.environ.Envs",
        "sglang.srt.mem_cache.memory_pool.ReqToTokenPool.free",
        "sglang.srt.mem_cache.memory_pool.HybridReqToTokenPool.free_mamba_cache",
        "sglang.srt.mem_cache.allocator.paged.PagedTokenToKVPoolAllocator.alloc",
        "sglang.srt.mem_cache.allocator.paged.PagedTokenToKVPoolAllocator.free",
    ),
    reason=(
        "hisparse: qsa_private_host_prefix_active is False unless "
        "SGLANG_QSA_HISPARSE_V3 is a P2 mode, the host-prefix budget is "
        "positive and the KV pool is QSATokenToKVPool, so the original runs "
        "otherwise. Around hook: under --disable-radix-cache the original "
        "returns create_unified_radix_cache(ctx) for every model that is not "
        "pure SWA (the QSA pool's model is hybrid SSM), after only reading "
        "ctx; the hook returns the same construction with "
        "cache_class=QSAHostPrefixCache (no CLI change). S01's guards (no "
        "HiCache or enabled radix cache, one mutable Mamba slot per request) "
        "move here from the wrap, which v0.5.22 no longer needs (the cache "
        "reads its runtime from the kvcache, attached by Q03 during "
        "init_model_worker, before build_kv_cache). The cache's own "
        "overrides rely on the depends listed (design (a1)+H)."
    ),
)
def _build_host_prefix_cache(original, ctx):
    if not qsa_private_host_prefix_active(ctx.params):
        return original(ctx)
    if not ctx.disable_radix_cache or ctx.enable_hierarchical_cache:
        raise ValueError(
            "QSA host prefixes require the disabled radix cache without HiCache"
        )
    if ctx.params.req_to_token_pool.enable_mamba_extra_buffer:
        raise ValueError("QSA host prefixes require one mutable Mamba slot per request")
    return create_unified_radix_cache(ctx, cache_class=QSAHostPrefixCache)
