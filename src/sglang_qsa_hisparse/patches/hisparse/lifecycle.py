"""KV allocation, release, invalidation and tree-cache selection (inventory W2).

Rows B06, M01, M02, M03, M05 and M06. Release order for a request with a QSA
lease: the tree's ``claim_kv_row`` (a pending host restore must drain through
rollback first), prefix capture in the tree's ``checkpoint``, runtime release
(drains GPU and copy events), the tree cache's ``free_kv_row`` (logical free,
deferred inside a free group), ``on_release``, the mamba slot when the tree
does not manage it, ``req_to_token_pool.free``, ``mark_kv_released``,
``after_release(lease)``, then the allocator's ``free_group_end`` and
``after_logical_flush``, after which the physical slot can be reused. M05 and
M06 keep the ChunkCache that ``QSAHostPrefixCache`` wraps (S01) on hybrid-SSM
models under ``--disable-radix-cache``. REPLACE hooks are production's
definitions (PLAN.md rules 3, P1 and Q1) and run with this module's globals,
which are imported from the pinned module that defines each target
(inventory 6, G3).
"""

from __future__ import annotations

import os

import torch

from sglang.srt.mem_cache.allocation import (
    _alloc_extend_loc_with_kv_reuse,
    _alloc_page_size,
    _is_npu,
    alloc_paged_token_slots_extend,
    alloc_req_slots,
    alloc_token_slots,
    is_pin_memory_available,
    maybe_write_dsv4_extend,
    write_cache_indices,
)
from sglang.srt.mem_cache.common import _release_overallocated_kv_indices
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
from sglang.srt.mem_cache.registry import (
    default_radix_cache_factory,
    get_memory,
    get_radix_cache_factory,
    get_serving,
    logger,
    registered_radix_cache_backends,
)
from sglang_qsa_hisparse.features import HISPARSE
from sglang_qsa_hisparse.patching import patch

ALLOCATION = "sglang.srt.mem_cache.allocation"
REGISTRY = "sglang.srt.mem_cache.registry"
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


# Production mem_cache/allocation.py:344-357, verbatim.
@patch(
    "sglang.srt.mem_cache.allocation.alloc_for_extend",
    "replace",
    feature=HISPARSE,
    row="M01",
    depends=(
        f"{ALLOCATION}.alloc_req_slots",
        f"{ALLOCATION}.write_cache_indices",
        f"{ALLOCATION}.alloc_paged_token_slots_extend",
        f"{ALLOCATION}.alloc_token_slots",
        f"{ALLOCATION}._alloc_extend_loc_with_kv_reuse",
        f"{ALLOCATION}._alloc_page_size",
        "sglang.srt.hardware_backend.npu.dsv4.dsv4_common_hooks.maybe_write_dsv4_extend",
        "sglang.srt.utils.common.is_pin_memory_available",
    ),
    reason=(
        "hisparse: every new call is a getattr on the tree cache, and only "
        "QSAHostPrefixCache defines them; reading prefix_tensors after "
        "prepare_prefix_for_extend is pure because alloc_req_slots never "
        "reassigns req.prefix_indices. Replace: insertions at four mid-function "
        "points. The fork's rollback wrapper and its renamed body "
        "_alloc_for_extend are both copied as module functions."
    ),
)
def alloc_for_extend(
    batch: ScheduleBatch,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rollback = getattr(batch.tree_cache, "rollback_prefix_for_extend", None)
    if rollback is None:
        return _alloc_for_extend(batch)
    try:
        return _alloc_for_extend(batch)
    except BaseException as error:
        try:
            rollback(batch.reqs)
        except BaseException as cleanup_error:
            error.add_note(f"QSA prefix allocation rollback retained ownership: {cleanup_error!r}")
        raise


# Production mem_cache/allocation.py:360-478, verbatim.
def _alloc_for_extend(
    batch: ScheduleBatch,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Allocate KV cache for extend batch and write to req_to_token_pool.

    Returns ``(out_cache_loc, req_pool_indices_device, req_pool_indices_cpu)``
    (the last is the host/CPU mirror). ``alloc_req_slots`` raises ``RuntimeError``
    if the pool can't satisfy the batch (fail-loud — see its docstring).
    """
    # free out-of-window swa tokens
    batch.maybe_evict_swa()

    reuse_kv = None
    if batch.is_dllm():
        reuse_kv = [r.kv.holds_kv and bool(r.dllm_incomplete_ids) for r in batch.reqs]

    # Create tensors for allocation
    pin_memory = is_pin_memory_available(batch.device)
    prefix_lens_cpu = torch.tensor(
        batch.prefix_lens, dtype=torch.int64, pin_memory=pin_memory
    )
    extend_lens_cpu = torch.tensor(
        batch.extend_lens, dtype=torch.int64, pin_memory=pin_memory
    )
    prefix_lens_device = prefix_lens_cpu.to(batch.device, non_blocking=True)
    extend_lens_device = extend_lens_cpu.to(batch.device, non_blocking=True)

    # Allocate req slots (raises RuntimeError if the pool is exhausted)
    req_pool_indices = alloc_req_slots(
        batch.req_to_token_pool, batch.reqs, batch.tree_cache
    )
    prepare_prefix = getattr(batch.tree_cache, "prepare_prefix_for_extend", None)
    if prepare_prefix is not None:
        prepare_prefix(batch.reqs)
    prefix_tensors = [r.prefix_indices for r in batch.reqs]
    req_pool_indices_cpu = torch.tensor(
        req_pool_indices, dtype=torch.int64, pin_memory=pin_memory
    )
    req_pool_indices_device = req_pool_indices_cpu.to(batch.device, non_blocking=True)

    # Allocate KV cache (throws exception on failure)
    alloc_page_size = _alloc_page_size(batch)
    if reuse_kv is not None and any(reuse_kv):
        out_cache_loc = _alloc_extend_loc_with_kv_reuse(
            batch,
            reuse_kv,
            req_pool_indices_cpu,
            prefix_lens_cpu,
            extend_lens_cpu,
            req_pool_indices_device,
            alloc_page_size,
        )
    elif alloc_page_size == 1:
        out_cache_loc = alloc_token_slots(batch.tree_cache, batch.extend_num_tokens)
    else:
        # Paged allocation - build last_loc
        last_loc = [
            (t[-1:] if len(t) > 0 else torch.full((1,), -1, device=batch.device))
            for t in prefix_tensors
        ]
        out_cache_loc = alloc_paged_token_slots_extend(
            tree_cache=batch.tree_cache,
            prefix_lens=prefix_lens_device,
            prefix_lens_cpu=prefix_lens_cpu,
            seq_lens=batch.seq_lens,
            seq_lens_cpu=batch.seq_lens_cpu,
            last_loc=torch.cat(last_loc),
            extend_num_tokens=batch.extend_num_tokens,
            req_pool_indices=req_pool_indices_device,
            batch=batch,
        )

    note_allocation = getattr(batch.tree_cache, "note_extend_allocation", None)
    if note_allocation is not None:
        note_allocation(batch.reqs, out_cache_loc)

    # Write to req_to_token_pool
    write_cache_indices(
        out_cache_loc,
        req_pool_indices_device,
        req_pool_indices_cpu,
        prefix_lens_device,
        prefix_lens_cpu,
        batch.seq_lens,
        batch.seq_lens_cpu,
        extend_lens_device,
        extend_lens_cpu,
        prefix_tensors,
        batch.req_to_token_pool,
    )
    try:
        batch.req_to_token_pool.alloc_aux_to_lengths(
            req_pool_indices_cpu=req_pool_indices_cpu,
            target_seq_lens_cpu=batch.seq_lens_cpu,
        )
    except Exception:
        if note_allocation is None:
            batch.tree_cache.token_to_kv_pool_allocator.free(out_cache_loc)
        raise

    # DSV4-NPU hook: no-op on non-DSV4 paths.
    if _is_npu:
        maybe_write_dsv4_extend(
            batch,
            req_pool_indices_cpu,
            prefix_lens_cpu,
            batch.seq_lens_cpu,
        )

    for req, seq_len in zip(batch.reqs, batch.seq_lens_cpu.tolist()):
        req.kv.kv_allocated_len = seq_len
        req.kv.kv_committed_len = seq_len

    restore_prefix = getattr(batch.tree_cache, "restore_prefix_for_extend", None)
    if restore_prefix is not None:
        restore_prefix(batch.reqs)

    return out_cache_loc, req_pool_indices_device, req_pool_indices_cpu


# Production mem_cache/common.py:297-360, verbatim.
@patch(
    "sglang.srt.mem_cache.common.release_kv_cache",
    "replace",
    feature=HISPARSE,
    row="M03",
    depends=(
        "sglang.srt.mem_cache.common._release_overallocated_kv_indices",
        "sglang.srt.mem_cache.memory_pool.ReqToTokenPool.free",
        "sglang.srt.mem_cache.memory_pool.HybridReqToTokenPool.free_mamba_cache",
        "sglang.srt.mem_cache.base_prefix_cache.BasePrefixCache.claim_kv_row",
        "sglang.srt.mem_cache.base_prefix_cache.BasePrefixCache.checkpoint",
        "sglang.srt.mem_cache.base_prefix_cache.BasePrefixCache.free_kv_row",
    ),
    reason=(
        "hisparse: getattr on the kvcache; without the QSA runtime nothing is "
        "added, and the mamba free is reached only by a tree that does not "
        "manage mamba state on a HybridReqToTokenPool, which upstream's "
        "create_tree_cache rejects (only M06's QSA host-prefix ChunkCache gets "
        "there). Replace: the lease release sits between the tree's checkpoint "
        "(where QSAHostPrefixCache captures the prefix, observing the live "
        "lease) and the logical free in free_kv_row; the mamba free and "
        "after_release follow on_release and req_to_token_pool.free / "
        "mark_kv_released. A streaming session that claims the row returns "
        "before any QSA call (claim_kv_row is the first release hook)."
    ),
)
def release_kv_cache(req: Req, tree_cache: BasePrefixCache, is_insert: bool = True):
    """Give the request's kv row back; with ``is_insert`` the tree first keeps
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
    is_insert = is_insert and not req.skip_radix_cache_insert
    if is_insert:
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
    tree_cache.unpin(req)
    _release_overallocated_kv_indices(
        req, owned_kv_len, req.kv.kv_allocated_len, tree_cache
    )
    tree_cache.on_release(req, inserted=is_insert)

    # A tree that does not own component state must give the mamba slot back
    # here. Upstream moved that ownership into UnifiedRadixCache (#42354); the
    # private QSA chunk-cache adapter reports supports_mamba() == False.
    if isinstance(tree_cache.req_to_token_pool, HybridReqToTokenPool) and (
        not tree_cache.supports_mamba()
    ):
        assert req.kv.holds_mamba, (
            "mamba state is freed while the tree cache does not manage mamba states"
        )
        tree_cache.req_to_token_pool.free_mamba_cache(req)

    # The DSV4-NPU ReqToTokenPool subclass's free() additionally releases the
    # c4/c128 state pages; other ReqToTokenPool subclasses are a no-op here.
    tree_cache.req_to_token_pool.free(req)
    req.kv.mark_kv_released()
    if qsa_hisparse is not None:
        qsa_hisparse.after_release(qsa_lease)


# Production mem_cache/registry.py:81-102, verbatim except the config import,
# rewritten to the plugin package. Called by M05's hook and M06's copy.
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
        "sglang.srt.mem_cache.chunk_cache.ChunkCache",
        "sglang.srt.mem_cache.qsa_kv_pool.QSATokenToKVPool",
    ),
    reason=(
        "hisparse: qsa_private_host_prefix_active is False unless "
        "SGLANG_QSA_HISPARSE_V3 is a P2 mode, the host-prefix budget is "
        "positive and the KV pool is QSATokenToKVPool, so the original runs "
        "otherwise. Around hook: production inserts its branch after "
        "`params = ctx.params` and the is_pure_swa comparison, which have no "
        "side effects, so returning ChunkCache(ctx.params) before calling the "
        "original equals it; the only caller is create_tree_cache (M06), "
        "whose copy reaches this hook through HookRegistry's propagation. "
        "Production registry.py 109-112."
    ),
)
def _keep_qsa_chunk_cache(original, ctx):
    if ctx.disable_radix_cache and qsa_private_host_prefix_active(ctx.params):
        from sglang.srt.mem_cache.chunk_cache import ChunkCache

        return ChunkCache(ctx.params)
    return original(ctx)


# Production mem_cache/registry.py:266-356, verbatim.
@patch(
    f"{REGISTRY}.create_tree_cache",
    "replace",
    feature=HISPARSE,
    row="M06",
    depends=(
        f"{REGISTRY}.default_radix_cache_factory",
        f"{REGISTRY}.get_radix_cache_factory",
        f"{REGISTRY}.registered_radix_cache_backends",
        "sglang.srt.mem_cache.base_prefix_cache.BasePrefixCache.supports_mamba",
    ),
    reason=(
        "hisparse: the hybrid-SSM guard is skipped only when "
        "qsa_private_host_prefix_active (M05), which is False without the "
        "QSA runtime's P2 mode, host-prefix budget and pool. Replace: the "
        "change turns a mid-function raise into a log line; the guard reads "
        "the factory's local result, so no hook seam reproduces it "
        "(supports_mamba is also read by release_kv_cache and the "
        "schedulers)."
    ),
)
def create_tree_cache(ctx: TreeCacheBuildContext) -> BasePrefixCache:
    """Route to the matching factory to construct Radix Cache."""
    name = get_memory().radix_cache_backend
    if name:
        factory = get_radix_cache_factory(name)
        if factory is None:
            raise ValueError(
                f"--radix-cache-backend={name!r} is not registered. "
                f"Registered backends: {registered_radix_cache_backends()}. "
                "External backends must call register_radix_cache_backend(...) at import time."
            )
        cache = factory(ctx)
        source = f"registered({name!r})"
    else:
        cache = default_radix_cache_factory(ctx)
        source = "default"

    if (
        get_memory().enable_hierarchical_cache
        and get_memory().hicache_host_memory_mode == "buffer_only"
    ):
        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

        if not isinstance(cache, UnifiedRadixCache):
            raise ValueError(
                "--hicache-host-memory-mode buffer_only is only implemented for "
                f"the unified radix tree; this model selected {type(cache).__name__}."
            )

    if get_memory().enable_session_radix_cache and not getattr(
        cache, "enable_session_radix_cache", False
    ):
        raise ValueError(
            "--enable-session-radix-cache requires UnifiedRadixCache, but "
            f"tree_cache is {type(cache).__name__}. Drop the flag or the "
            "option that selected another tree cache for this model."
        )

    if get_memory().radix_eviction_policy == "tlru":
        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

        # T-LRU's per-node tail bookkeeping only exists on the unified tree;
        # any other cache would silently fall back to LRU ordering.
        if not isinstance(cache, UnifiedRadixCache):
            raise ValueError(
                "--radix-eviction-policy tlru requires UnifiedRadixCache, but "
                f"tree_cache is {type(cache).__name__}. Drop the flag or the "
                "option that selected another tree cache for this model."
            )

    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    if get_serving().enable_streaming_session and not isinstance(
        cache, UnifiedRadixCache
    ):
        raise NotImplementedError(
            f"--enable-streaming-session is not verified with {type(cache).__name__}; "
            "streaming sessions run on UnifiedRadixCache. Please open an issue or "
            "a PR at https://github.com/sgl-project/sglang if you need this."
        )

    if ctx.is_hybrid_ssm and not cache.supports_mamba():
        if not qsa_private_host_prefix_active(ctx.params):
            raise NotImplementedError(
                f"Models with mamba state are not verified with {type(cache).__name__}; "
                "mamba state lives in UnifiedRadixCache. Please open an issue or a PR "
                "at https://github.com/sgl-project/sglang if you need this."
            )
        # Verified exception: the QSA private host-prefix adapter does not take
        # Mamba ownership away from the request lifecycle. ``release_kv_cache``
        # frees the mamba slot whenever the tree reports supports_mamba() ==
        # False, the adapter never shares GPU radix pages, and
        # ``qsa_hisparse/config.py`` pins this path to the plain FP8 C4 QSA
        # pool (TP2, page 64, radix disabled, no overlap/speculation).
        logger.info(
            "Tree cache %s runs the QSA private host-prefix path; mamba slots "
            "are released through release_kv_cache",
            type(cache).__name__,
        )

    hicache_attached = cache.cache_controller is not None
    logger.info(
        "Tree cache initialized: source=%s impl=%s hybrid_swa=%s hybrid_ssm=%s "
        "hicache_attached=%s",
        source,
        type(cache).__name__,
        ctx.is_hybrid_swa,
        ctx.is_hybrid_ssm,
        hicache_attached,
    )
    return cache
