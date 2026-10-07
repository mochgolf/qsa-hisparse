"""KV allocation, release and invalidation lifecycle (inventory W2).

Rows B06, M01, M02 and M03. Release order for a request with a QSA lease:
prefix capture (``before_release``), runtime release (drains GPU and copy
events), ``cache_finished_req`` (logical free, deferred inside a free group),
``req_to_token_pool.free``, ``mark_kv_released``, ``after_release(lease)``,
then the allocator's ``free_group_end`` and ``after_logical_flush``, after
which the physical slot can be reused. REPLACE hooks are the fork's bodies
copied verbatim (PLAN.md rule 3) and run with this module's globals, which are
imported from the pinned module that defines each target (inventory 6, G3).
"""

from __future__ import annotations

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
from sglang.srt.mem_cache.common import (
    HybridReqToTokenPool,
    _release_overallocated_kv_indices,
)
from sglang_qsa_hisparse.features import HISPARSE
from sglang_qsa_hisparse.patching import patch

ALLOCATION = "sglang.srt.mem_cache.allocation"
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
        "statement before its body and metrics in both. Fork weight_updater.py "
        "95-100 verbatim."
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
        "return. Fork paged.py 336-340 verbatim."
    ),
)
def _notify_logical_flush(result, self):
    adapter = getattr(self.get_kvcache(), "qsa_hisparse", None)
    if getattr(adapter, "uses_qsa_hisparse_leases", False):
        adapter.after_logical_flush()
    elif adapter is not None and adapter.pending_release is not None:
        adapter.after_release(adapter.pending_release)


# Fork mem_cache/allocation.py:344-357, verbatim.
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


# Fork mem_cache/allocation.py:360-478, verbatim.
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


# Fork mem_cache/common.py:254-307, verbatim.
@patch(
    "sglang.srt.mem_cache.common.release_kv_cache",
    "replace",
    feature=HISPARSE,
    row="M03",
    depends=(
        "sglang.srt.mem_cache.common._release_overallocated_kv_indices",
        "sglang.srt.mem_cache.memory_pool.ReqToTokenPool.free",
    ),
    reason=(
        "hisparse: getattrs on the tree cache and the kvcache; without "
        "QSAHostPrefixCache and the QSA runtime nothing is added. Replace: the "
        "first insertion follows an early-return branch and precedes "
        "cache_finished_req; after_release must follow req_to_token_pool.free "
        "and mark_kv_released."
    ),
)
def release_kv_cache(req: Req, tree_cache: BasePrefixCache, is_insert: bool = True):
    assert (not req.kv.holds_kv) == req.kv.is_kv_released
    # MambaRadixCache may alloc mamba state before alloc KV cache
    if not req.kv.holds_kv:
        assert tree_cache.supports_mamba(), (
            "Only MambaRadixCache allow freeing before alloc"
        )
        # TODO (csy, hanming): clean up this early allocation logic
        if req.kv.holds_mamba:
            tree_cache.req_to_token_pool.mamba_allocator.free(
                req.kv.mamba_pool_idx.unsqueeze(-1)
            )
            req.kv.mamba_pool_idx = None
        return

    qsa_hisparse = getattr(
        tree_cache.token_to_kv_pool_allocator.get_kvcache(), "qsa_hisparse", None
    )
    before_release = getattr(tree_cache, "before_release", None)
    if before_release is not None:
        before_release(req, is_insert and not getattr(req, "skip_radix_cache_insert", False))
    if qsa_hisparse is not None:
        qsa_lease = qsa_hisparse.release(req.kv.req_pool_idx, req.rid)

    effective_kv_committed_len = req.effective_kv_committed_len()
    tree_cache.cache_finished_req(
        req,
        is_insert=is_insert and not getattr(req, "skip_radix_cache_insert", False),
        kv_len_to_handle=effective_kv_committed_len,
    )

    # StreamingSession.cache_finished_req handles speculative tail trim
    # internally, then sets req_pool_idx = None.
    assert (not req.kv.holds_kv) == req.kv.is_kv_released
    if not req.kv.holds_kv:
        return

    start_p, end_p = effective_kv_committed_len, req.kv.kv_allocated_len
    _release_overallocated_kv_indices(req, start_p, end_p, tree_cache)

    # If the prefix cache doesn't manage mamba states, we must free them here.
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
