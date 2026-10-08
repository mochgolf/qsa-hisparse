"""Scheduler, batch-result and prefill-admission integration (inventory W2).

Rows S01, S02, S04-S09, B02-B05, P01 and P02. B01 (an unused import) is a
drop. REPLACE hooks are production's definitions (PLAN.md rules 3, P1 and
Q1); they run with this module's globals, so every name they use
is imported below from the pinned module that defines the target (inventory
6, G3). ``release_kv_cache`` and the other patched callees are rebound to
their patched versions by HookRegistry's propagation.
"""

from __future__ import annotations

import time

import torch

from sglang.srt.managers.schedule_policy import (
    CLIP_MAX_NEW_TOKENS,
    CacheTransferPhase,
    ComponentType,
    InitLoadBackParams,
    UnifiedRadixCache,
    _PrefillAdmission,
    nullcontext,
)
from sglang.srt.managers.scheduler import (
    LOAD_STALL_REFRESH_S,
    SCHEDULER_STAGE_GET_NEXT_BATCH,
    SCHEDULER_STAGE_IDLE,
    SCHEDULER_STAGE_SANITY_CHECK_CACHE,
    TEST_RETRACT,
    TEST_RETRACT_NO_PREFILL_BS,
    AddReqResult,
    DisaggregationMode,
    NextBatchPlan,
    PrefillAdder,
    PrefillStats,
    ScheduleBatch,
    envs,
    get_schedule,
    get_spec,
    scheduler_stage_method,
    set_schedule_time_batch,
    set_time_batch,
)
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    FINISH_MATCHED_TOKEN,
    BaseSpecWorker,
    checkpoint_kv_cache,
    get_disagg,
    get_exec,
    release_kv_cache,
)
from sglang_qsa_hisparse.features import HISPARSE
from sglang_qsa_hisparse.patching import attach, patch

SCHEDULER = "sglang.srt.managers.scheduler.Scheduler"
PROCESSOR = (
    "sglang.srt.managers.scheduler_components.batch_result_processor."
    "SchedulerBatchResultProcessor"
)
ADDER = "sglang.srt.managers.schedule_policy.PrefillAdder"
RELEASE_KV_CACHE = "sglang.srt.mem_cache.common.release_kv_cache"
GET_DISAGG = "sglang.srt.runtime_context.get_disagg"
STAGE_METHOD = "sglang.srt.observability.scheduler_stage_metrics.scheduler_stage_method"
SAME_GATE = (
    "hisparse: upstream creates the HiSparse coordinator exactly when "
    "enable_hisparse is set (ModelRunner.maybe_init_hisparse_coordinator; "
    "TpModelWorker.register_hisparse_coordinator has no callers), so gating on "
    "`hisparse_coordinator is not None` is equivalent without the QSA runtime. "
)


# Declared before S01, so the coordinator is adopted before the cache is wrapped
# (the fork's order; neither hook reads the other's result).
@patch(
    f"{SCHEDULER}.init_hisparse_coordinator",
    "after",
    feature=HISPARSE,
    row="S02",
    depends=(
        "sglang.srt.model_executor.model_runner.ModelRunner."
        "maybe_init_hisparse_coordinator",
    ),
    reason=(
        "hisparse: without a QSA kvcache adapter the fork's gate reduces to "
        "`not enable_hisparse`. After hook (P3 narrowing of the fork's 9-line "
        "replace): the original returns with hisparse_coordinator None exactly "
        "when enable_hisparse is unset, and only then does the fork differ, by "
        "adopting the runner's coordinator under QSA leases; the hook does that "
        "with production scheduler.py 1277-1278, 1281-1283. The None guard on "
        "set_decode_producer_stream is unreachable with enable_hisparse set "
        "(ModelRunner.maybe_init_hisparse_coordinator builds the coordinator "
        "whenever enable_hisparse is set, R01 only swaps in another one, and "
        "TpModelWorker.register_hisparse_coordinator has no callers)."
    ),
)
def _adopt_qsa_coordinator(result, self):
    qsa = getattr(self.token_to_kv_pool_allocator.get_kvcache(), "qsa_hisparse", None)
    if not self.enable_hisparse and getattr(qsa, "uses_qsa_hisparse_leases", False):
        self.hisparse_coordinator = self.tp_worker.model_runner.hisparse_coordinator
        if self.hisparse_coordinator is not None:
            self.hisparse_coordinator.set_decode_producer_stream(self.forward_stream)


@patch(
    f"{SCHEDULER}.init_hisparse_coordinator",
    "after",
    feature=HISPARSE,
    row="S01",
    depends=(
        f"{SCHEDULER}.__init__",
        f"{SCHEDULER}.init_batch_result_processor",
        "sglang.srt.mem_cache.chunk_cache.ChunkCache",
    ),
    reason=(
        "hisparse: requires kvcache.qsa_hisparse.prefix_cache, which no upstream "
        "kvcache has. After hook: Scheduler.__init__ (pin 616) is the only caller "
        "and the fork inserts directly after that call, so every later capture "
        "of tree_cache (e.g. init_batch_result_processor, pin 707) sees the "
        "wrapped cache in both. At this pin the ChunkCache it wraps on hybrid-SSM "
        "models comes from M05/M06. Production scheduler.py "
        "617-632 verbatim; the QSAHostPrefixCache import is rewritten to the "
        "plugin package."
    ),
)
def _wrap_host_prefix_cache(result, self):
    qsa = getattr(
        self.token_to_kv_pool_allocator.get_kvcache(), "qsa_hisparse", None
    )
    if getattr(qsa, "prefix_cache", None) is not None:
        from sglang.srt.mem_cache.chunk_cache import ChunkCache
        from sglang_qsa_hisparse.hisparse.prefix_cache import QSAHostPrefixCache

        if type(self.tree_cache) is not ChunkCache or self.enable_hierarchical_cache:
            raise ValueError(
                "QSA host prefixes require the private chunk-cache scheduler"
            )
        if self.req_to_token_pool.enable_mamba_extra_buffer:
            raise ValueError(
                "QSA host prefixes require one mutable Mamba slot per request"
            )
        self.tree_cache = QSAHostPrefixCache(self.tree_cache, qsa, self.tp_cpu_group)


@patch(
    f"{SCHEDULER}.get_num_allocatable_reqs",
    "after",
    feature=HISPARSE,
    row="S05",
    depends=(
        "sglang.srt.beam_search.coordinator.BeamCoordinator.pending_member_rows",
        "sglang.srt.sampling.sampling_params.SamplingParams.verify",
    ),
    reason=(
        "hisparse: gated on coordinator.uses_qsa_hisparse_leases. After hook: the "
        "original has no side effects, and SamplingParams.verify rejects "
        "beam_width < 1, so with beam width None or 1 its beam cap is a no-op; "
        "capping the returned value afterwards equals the fork, which caps before "
        "the beam cap. Production scheduler.py 3825-3831 verbatim."
    ),
)
def _cap_allocatable_reqs(res, self, running_bs, beam_width=None, running_batch=None):
    if getattr(self.hisparse_coordinator, "uses_qsa_hisparse_leases", False):
        if beam_width is not None and beam_width != 1:
            raise ValueError("QSA P2 does not support beam/prefix ownership sharing")
        slots = self.hisparse_coordinator.adapter.slots
        res = min(res, 1, len(slots.free_slots))
        if slots.prefill_owner is not None:
            res = 0
    return res


@patch(
    f"{SCHEDULER}.is_fully_idle",
    "after",
    feature=HISPARSE,
    row="S08",
    reason=(
        SAME_GATE
        + "After hook: the original is a conjunction of side-effect-free terms "
        "and QSAHiSparseCoordinator.has_ongoing_staging is pure, so the fork's "
        "extra term can be applied after return; with enable_hisparse set the "
        "original already applies it. v0.5.21 added ignore_waiting, which only "
        "drops the waiting-queue term. Production scheduler.py 5088."
    ),
)
def _wait_for_staging(idle, self, for_health_check=False, ignore_waiting=False):
    if (
        not for_health_check
        and not self.enable_hisparse
        and self.hisparse_coordinator is not None
    ):
        idle &= not self.hisparse_coordinator.has_ongoing_staging()
    return idle


@patch(
    f"{SCHEDULER}.collect_inflight_reqs",
    "after",
    feature=HISPARSE,
    row="S09",
    depends=(f"{SCHEDULER}.abort_request", f"{SCHEDULER}.record_weight_version_change"),
    reason=(
        "hisparse: gated on coordinator.uses_qsa_hisparse_leases. After hook: the "
        "callers are abort_request (the fork adds the staging requests to this "
        "same set) and record_weight_version_change (already unions "
        "ack_staging_queue whenever a coordinator exists, so its set is "
        "unchanged); the running-timeout sweep reads _collect_inflight_batches, "
        "not this method. Production scheduler.py 5457-5460 verbatim."
    ),
)
def _add_staging_reqs(inflight, self):
    if getattr(self.hisparse_coordinator, "uses_qsa_hisparse_leases", False):
        inflight.update(item.req for item in self.hisparse_coordinator.ack_staging_queue)
    return inflight


# Production managers/scheduler.py:3667-3809, verbatim.
@patch(
    f"{SCHEDULER}.get_next_batch_to_run",
    "replace",
    feature=HISPARSE,
    row="S04",
    depends=(
        f"{SCHEDULER}.update_running_batch",
        f"{SCHEDULER}.get_new_batch_prefill",
        f"{SCHEDULER}.stash_chunked_request",
        f"{SCHEDULER}.process_pending_chunked_abort",
        f"{SCHEDULER}._arm_prefill_decode_interval",
        f"{SCHEDULER}._build_hisparse_decode_batch",
        f"{SCHEDULER}._process_hicache_events",
        f"{SCHEDULER}._should_defer_prefill",
        "sglang.srt.managers.schedule_batch.NextBatchPlan",
        "sglang.srt.managers.schedule_batch.ScheduleBatch",
        "sglang.srt.observability.req_time_stats.set_schedule_time_batch",
        STAGE_METHOD,
        "sglang.srt.runtime_context.get_spec",
    ),
    reason=(
        SAME_GATE
        + "Replace: two mid-function predicates; flipping enable_hisparse in an "
        "around hook would leak into callees (update_running_batch, the PP "
        "mixin, disaggregation). The @scheduler_stage_method decorator is kept."
    ),
)
@scheduler_stage_method(SCHEDULER_STAGE_GET_NEXT_BATCH)
def get_next_batch_to_run(
    self, running_batch: ScheduleBatch, last_batch: Optional[ScheduleBatch]
) -> NextBatchPlan:
    self.process_pending_chunked_abort()
    self._process_hicache_events()

    if self.enable_fpm:
        self._fpm_batch_t0 = time.monotonic()
    if self.dllm_config is not None:
        self.dllm_manager.filter_finished_reqs()

    # Merge the prefill batch into the running batch
    chunked_req_to_exclude = set()

    if self.dllm_config is not None and self.dllm_manager.any_staging_reqs():
        chunked_req_to_exclude.update(self.dllm_manager.staging_queue)
        for req in self.dllm_manager.staging_queue:
            self.finish_dllm_forward(req)

    if self.chunked_req is not None:
        # Move the chunked request out of the batch so that we can merge
        # only finished requests to running_batch.
        chunked_req_to_exclude.add(self.chunked_req)

        # Stash (cache) the previous chunk only when it produced new KV
        # beyond what is already cached. A parked chunk (add_chunked_req
        # hybrid-SWA early-return) leaves extend_range.end ==
        # len(prefix_indices), so there is nothing new to cache and
        # stashing would be a no-op.
        if self.chunked_req.extend_range.end > len(self.chunked_req.prefix_indices):
            self.stash_chunked_request(self.chunked_req)

    # HiSparse has its own prefill-to-decode transition; skip last_batch merge.
    if self.hisparse_coordinator is not None:
        ready_reqs = self.hisparse_coordinator.collect_ready_reqs()
        if len(ready_reqs) > 0:
            new_batch = self._build_hisparse_decode_batch(ready_reqs)
            if running_batch.is_empty():
                running_batch = new_batch
            else:
                running_batch.merge_batch(new_batch)
            running_batch.hisparse_coordinator = self.hisparse_coordinator
        # Reset batch_is_full so the scheduler can schedule more prefills.
        running_batch.batch_is_full = False

    if (
        self.hisparse_coordinator is None
        and last_batch
        and last_batch.forward_mode.is_extend()
    ):
        if last_batch.chunked_req is not None:
            # In the context pipeline parallelism, after the last chunk, the current microbatch still track outdated chunked_req.
            # We need to discard it.
            chunked_req_to_exclude.add(last_batch.chunked_req)

        if self.dllm_config is not None and last_batch.reqs:
            chunked_req_to_exclude.update(last_batch.reqs)

        # Filter batch
        last_bs = last_batch.batch_size()
        last_batch.filter_batch(chunked_req_to_exclude=list(chunked_req_to_exclude))
        if last_batch.batch_size() < last_bs:
            running_batch.batch_is_full = False

        # Merge the new batch into the running batch.
        if not last_batch.is_empty():
            if running_batch.is_empty():
                running_batch = last_batch
            else:
                # Merge running_batch with prefill batch
                running_batch.merge_batch(last_batch)

    # For prefill-only batch, filter out finished requests since they
    # won't go through the decode step. This keeps running_batch accurate
    # for load reporting (num_running_reqs via /v1/loads).
    # Runs outside the last_batch block so stale requests are cleaned
    # even when no new batches arrive (e.g. traffic stops).
    if running_batch.is_prefill_only:
        running_batch.filter_batch()
        if running_batch.is_empty():
            running_batch.batch_is_full = False

    if self.dllm_config is not None:
        new_batch = self.get_new_batch_dllm(running_batch)
    elif self._should_defer_prefill():
        new_batch = None
    else:
        prefill_plan = self.get_new_batch_prefill(running_batch)
        new_batch = prefill_plan.batch_to_run
        running_batch = prefill_plan.running_batch

    need_mlp_sync = self.require_mlp_sync
    if (
        need_mlp_sync
        and not self.spec_algorithm.is_none()
        and not get_spec().speculative_skip_dp_mlp_sync
        and not envs.SGLANG_ENABLE_DP_SPEC_PREFILL_COORDINATION.get()
    ):
        # NOTE: This branch makes sure prefill and decode batches will not be mixed when spec and dp-attn is enabled.
        # Before merging the new batch into running batch:
        # 1. All new batches are none -> need_mlp_sync remains true (sync is needed for decode batch).
        # 2. All new batches are some (prefill / idle) -> we do not need prepare mlp sync one more time.
        new_batch = self.dp_attn_adapter.maybe_prepare_mlp_sync_batch(new_batch)
        need_mlp_sync = new_batch is None

    if new_batch is not None:
        # Run prefill first if possible
        ret = new_batch
    else:
        # Run decode (skip for prefill-only batches)
        if not running_batch.is_empty() and not running_batch.is_prefill_only:
            running_batch = self.update_running_batch(running_batch)
            ret = running_batch if not running_batch.is_empty() else None
        else:
            ret = None

    # Handle DP attention and log stats
    ret = self.dp_attn_adapter.maybe_prepare_mlp_sync_batch(
        ret, need_sync=need_mlp_sync
    )
    # Decode->extend conversion keeps a heterogeneous dp step replayable.
    converted = self.dp_attn_adapter.maybe_convert_decode_to_extend(ret)
    if converted is running_batch and converted.forward_mode.is_extend():
        # The converted batch re-enters via the last_batch extend-merge
        # next iteration; empty running_batch or it merges with itself.
        running_batch = ScheduleBatch(
            reqs=[], batch_is_full=running_batch.batch_is_full
        )
    ret = converted
    self._arm_prefill_decode_interval(ret)

    # Handle ngram embedding
    ret = self.ngram_embedding_manager.prepare_for_forward(
        ret, chunked_req=self.chunked_req
    )

    if ret:
        set_schedule_time_batch(ret)
        if self.enable_fpm:
            ret.fpm_start_time = self._fpm_batch_t0

    return NextBatchPlan(batch_to_run=ret, running_batch=running_batch)


# Production managers/scheduler.py:3865-4196, verbatim.
@patch(
    f"{SCHEDULER}._get_new_batch_prefill_raw",
    "replace",
    feature=HISPARSE,
    row="S06",
    depends=(
        f"{ADDER}.__init__",
        f"{SCHEDULER}.get_num_allocatable_reqs",
        f"{SCHEDULER}._add_request_to_queue",
        f"{SCHEDULER}._prefetch_after_device_hit_loss",
        f"{SCHEDULER}.can_schedule_lora_req",
        "sglang.srt.managers.schedule_batch.ScheduleBatch",
        "sglang.srt.observability.req_time_stats.set_time_batch",
        "sglang.srt.runtime_context.get_schedule",
    ),
    reason=(
        "hisparse: gated on coordinator.uses_qsa_hisparse_leases. Replace: the "
        "prefill_max_requests argument is computed inline in the PrefillAdder "
        "call. The inventory's alternative (a before hook on PrefillAdder.__init__) "
        "is not taken: dllm/mixin/scheduler.py also builds a PrefillAdder, so it "
        "would not be equivalent on every pinned path."
    ),
)
def _get_new_batch_prefill_raw(
    self,
    prefill_delayer_single_pass: Optional[PrefillDelayerSinglePassExecutor],
    running_batch: ScheduleBatch,
) -> Tuple[Optional[ScheduleBatch], ScheduleBatch]:
    # Check if the grammar is ready in the grammar queue
    if self.grammar_manager.has_waiting_grammars():
        ready_grammar_requests = self.grammar_manager.get_ready_grammar_requests()
        for req in ready_grammar_requests:
            self._add_request_to_queue(req)

    if self.enable_priority_preemption or self.is_hybrid_swa:
        # Reset batch_is_full to try preemption with a prefill adder.
        running_batch.batch_is_full = False

    if (
        running_batch.batch_is_full or len(self.waiting_queue) == 0
    ) and self.chunked_req is None:
        return None, running_batch

    running_bs = len(running_batch.reqs)
    # Skipped during a chunked prefill: that pass must proceed regardless.
    if (
        self.min_free_slots_delayer is not None
        and self.chunked_req is None
        and self.min_free_slots_delayer.should_delay(
            running_bs=running_bs,
            num_allocatable_reqs=self.get_num_allocatable_reqs(
                running_bs, running_batch=running_batch
            ),
        )
    ):
        return None, running_batch

    # Ignore the check if self.chunked_req is not None.
    # In the non-PP case, when self.chunked_req is not None, num_allocatable_reqs should always be greater than 0,
    # as the space for the chunked requests has just been released.
    # In PP case, chunked requests (or dllm requests) can start in one microbatch and end in another microbatch, so the max_running_requests per microbatch should not be strict.
    # Instead, we should always allow chunked requests to be added, otherwise, there will be a memory leak.
    if (
        self.get_num_allocatable_reqs(running_bs, running_batch=running_batch) <= 0
        and self.chunked_req is None
        and not self.enable_priority_preemption
    ):
        running_batch.batch_is_full = True
        return None, running_batch

    # Get priority queue
    self.policy.calc_priority(
        self.waiting_queue,
        running_batch,
        processed_tokens=self.processed_tokens_counter,
    )

    if TEST_RETRACT and running_bs > TEST_RETRACT_NO_PREFILL_BS:
        # If we are testing retraction and the running batch size exceeds
        # TEST_RETRACT_NO_PREFILL_BS, we skip the prefill to keep the requests
        # in the waiting queue.
        return None, running_batch

    # Determine chunked_prefill_size for this batch
    chunked_prefill_size = self.chunked_prefill_size
    if self.chunked_req is not None and self.dynamic_chunk_sizer is not None:
        history_len = len(self.chunked_req.prefix_indices)
        dynamic_size = self.dynamic_chunk_sizer.predict(history_len)
        if dynamic_size is not None:
            chunked_prefill_size = dynamic_size

    # Prefill policy
    # Get BLOCK_M from the backend for tile-budget admission logic
    attn_backend = self.tp_worker.model_runner.attn_backend
    if hasattr(attn_backend, "extend_attention_block_m"):
        prefill_tile_block_m = attn_backend.extend_attention_block_m
    else:
        prefill_tile_block_m = 64  # Fallback for non-Triton backends

    adder = PrefillAdder(
        self.page_size,
        self.tree_cache,
        self.token_to_kv_pool_allocator,
        running_batch,
        self.new_token_ratio_tracker.current,
        self.max_prefill_tokens,
        chunked_prefill_size,
        running_bs if self.is_mixed_chunk else 0,
        self.priority_scheduling_preemption_threshold,
        max_prefill_bs=int(self.max_prefill_bs),
        max_running_requests=self.max_running_requests,
        prefill_max_requests=(1 if getattr(self.hisparse_coordinator, "uses_qsa_hisparse_leases", False)
                              else get_schedule().prefill_max_requests),
        prefill_delayer_single_pass=prefill_delayer_single_pass,
        dllm_config=self.dllm_config,
        waiting_queue_len=len(self.waiting_queue),
        prefill_tile_block_m=prefill_tile_block_m,
    )

    if self.chunked_req is not None:
        self.chunked_req.init_next_round_input()
        adder.chunked_req_limit = self.policy.shortest_prefill_chunk_limit(
            self.chunked_req,
            self.waiting_queue,
            adder.rem_chunk_tokens or 0,
            self.page_size,
        )
        self.chunked_req = adder.add_chunked_req(self.chunked_req)

    if self.enable_lora:
        running_loras = {
            req.lora_id for req in running_batch.reqs if not req.finished()
        }
        # Account for LoRAs that are already loaded in the adder, such as chunked requests
        running_loras.update(req.lora_id for req in adder.can_run_list)

        if self.lora_drainer:
            self.lora_drainer.update_draining_state(
                self.waiting_queue,
                running_batch.reqs,
            )

    mamba_allocator = getattr(self.req_to_token_pool, "mamba_allocator", None)
    if mamba_allocator is not None:
        mamba_allocator.alloc_group_begin(len(self.waiting_queue))
    buffer_pipeline = self.tree_cache.buffer_pipeline
    # Get requests from the waiting queue to a new prefill batch
    for req in self.waiting_queue:
        if self.enable_lora and not self.can_schedule_lora_req(req, running_loras):
            continue

        # A forward batch runs one pooling mode, so setwise readout requests
        # (token_indices_to_pool) cannot share a batch with last-token ones.
        # Admit only the first request's mode; the other stays queued.
        if adder.can_run_list and (
            (req.token_indices_to_pool is not None)
            != (adder.can_run_list[0].token_indices_to_pool is not None)
        ):
            continue

        running_bs = len(running_batch.reqs)
        candidate_beam_width = (
            req.beam_group.beam_width if req.beam_group is not None else None
        )
        if len(adder.can_run_list) >= self.get_num_allocatable_reqs(
            running_bs,
            candidate_beam_width,
            running_batch=running_batch,
        ):
            running_batch.batch_is_full = True
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            # In prefill mode, prealloc queue and transfer queue can also take memory,
            # so we need to check if the available size for the actual available size.
            if len(adder.can_run_list) >= self.req_to_token_pool.available_size():
                running_batch.batch_is_full = True

        if running_batch.batch_is_full:
            if not self.enable_priority_preemption or not adder.preempt_to_schedule(
                req
            ):
                break

        if self.enable_hicache_storage or self.enable_lmcache:
            prefetch_done = self.tree_cache.check_prefetch_progress(
                req.cache_request_handle
            )
            if not prefetch_done:
                # skip staging requests that are ongoing prefetch
                continue
            # Pop the L3-loaded span. Unified cache exposes its absolute
            # start so cache-mode L2/L3 attribution survives L3-tail eviction.
            loaded_tokens, loaded_start = self.tree_cache.pop_prefetch_loaded_span(
                req.cache_request_handle
            )
            if loaded_tokens > 0:
                req.storage_hit_length = loaded_tokens
                req.storage_hit_start = loaded_start
                # Cache-mode host memory is a resident L2 tier. Buffer mode
                # marks the staged span below once it is surfaced.
                req.host_hit_is_storage = False

        req.init_next_round_input(self.tree_cache)
        if self.enable_hicache_storage and (
            self._prefetch_after_device_hit_loss(req)
        ):
            continue
        if (
            self.enable_hicache_storage
            and buffer_pipeline is not None
            and not buffer_pipeline.prepare_staged_prefetch(req)
        ):
            continue
        res = adder.add_one_req(
            req,
            has_chunked_req=(self.chunked_req is not None),
            truncation_align_size=self.truncation_align_size,
        )

        if self.enable_lora:
            running_loras.add(req.lora_id)

        if res != AddReqResult.CONTINUE:
            if res == AddReqResult.NO_TOKEN:
                if (
                    self.enable_hierarchical_cache
                    or self.enable_lmcache
                    or self.enable_unified_cache_external_linker
                ):
                    # Set batch_is_full after making sure there are requests that can be served
                    running_batch.batch_is_full = len(adder.can_run_list) > 0 or (
                        not running_batch.is_empty()
                    )
                else:
                    running_batch.batch_is_full = True
            # revert matched mamba idx to avoid memory leak, if req is not added.
            # Only free if the slot was freshly allocated in this batch (not
            # pre-existing from a session). Session-held slots have their own
            # lifecycle and freeing them here causes double-free.
            added = len(adder.can_run_list) > 0 and req is adder.can_run_list[-1]
            if not added:
                # init_next_round_input() may stage deferred Mamba COW/clear
                # metadata before add_one_req() rejects the request.
                req.kv.mamba_cow_src_index = None
                req.kv.mamba_needs_clear = False
                if req.kv.holds_mamba and not getattr(req, "session", None):
                    self.tree_cache.req_to_token_pool.mamba_allocator.free(
                        req.kv.mamba_pool_idx.unsqueeze(-1)
                    )
                    req.kv.mamba_pool_idx = None
            break

    if mamba_allocator is not None:
        mamba_allocator.alloc_group_end()

    # Update waiting queue
    can_run_list: List[Req] = adder.can_run_list
    if len(can_run_list) == 0:
        return None, running_batch

    can_run_set = set(can_run_list)
    retries = self.tree_cache.storage_prefetch_retries
    if self.enable_hicache_storage and retries is not None:
        for req in can_run_list:
            retries.cancel(req.rid)
    self.waiting_queue = [x for x in self.waiting_queue if x not in can_run_set]
    if adder.preempt_list:
        for req in adder.preempt_list:
            self._add_request_to_queue(req)

    if adder.new_chunked_req is not None:
        # Update chunked prefill
        assert self.chunked_req is None
        self.chunked_req = adder.new_chunked_req

    if self.chunked_req is not None:
        self.chunked_req.inflight_middle_chunks += 1

    set_time_batch(can_run_list, "set_forward_entry_time")

    # Create a new batch
    new_batch = ScheduleBatch.init_new(
        can_run_list,
        self.req_to_token_pool,
        self.token_to_kv_pool_allocator,
        self.tree_cache,
        self.model_config,
        self.enable_overlap,
        self.spec_algorithm,
        chunked_req=self.chunked_req,
    )

    new_batch.contains_last_prefill_chunk = (
        self.chunked_req is None or len(can_run_list) != 1
    )

    if self.enable_hierarchical_cache or self.enable_unified_cache_external_linker:
        # todo (zhiqiang): disable cuda graph execution if hicache loading triggered
        new_batch.hicache_consumer_index = (
            self.tree_cache.ready_to_load_host_cache()
        )

    new_batch.prepare_for_extend()

    if self.tp_worker.model_runner.prefill_aware_swa:
        for req in can_run_list:
            req.kv.swa_evict_floor = req.extend_range.end

    # Record prefill stats for logging after forward.
    new_batch.prefill_stats = PrefillStats.from_adder(
        adder,
        running_batch.reqs,
        self.enable_priority_scheduling,
        num_pending_tokens=self.load_inquirer._get_num_pending_tokens(
            chunk_deduct=(
                self.chunked_req.extend_range.length
                if self.chunked_req is not None
                else 0
            ),
        ),
    )

    # Mixed-style chunked prefill
    if (
        self.is_mixed_chunk
        and not running_batch.is_empty()
        and not (new_batch.return_logprob or running_batch.return_logprob)
        # mix_with_running cats input_ids but not input_embeds — shapes would mismatch
        and new_batch.input_embeds is None
        # Beam member rows are not supported inside a mixed extend batch.
        and all(r.beam_group is None for r in running_batch.reqs)
    ):
        # TODO (lianmin): support return_logprob + mixed chunked prefill
        running_batch.filter_batch()
        if not running_batch.is_empty():
            running_batch.prepare_for_decode()
            new_batch.mix_with_running(running_batch)
            new_batch.decoding_reqs = running_batch.reqs
            if not self.enable_overlap and not self.spec_algorithm.is_none():
                # Non-overlap spec never writes the relay; stash the
                # tails' pending tokens for the mixed input resolve.
                last_tokens = torch.tensor(
                    [r.output_ids[-1] for r in running_batch.reqs],
                    dtype=torch.int64,
                    device=self.device,
                )
                self.future_map.stash_bonus_tokens(
                    running_batch.req_pool_indices, last_tokens
                )
            running_batch = ScheduleBatch(
                reqs=[], batch_is_full=running_batch.batch_is_full
            )
    else:
        new_batch.decoding_reqs = None

    return new_batch, running_batch


# Production managers/scheduler.py:4948-5036, verbatim.
@patch(
    f"{SCHEDULER}.on_idle",
    "replace",
    feature=HISPARSE,
    row="S07",
    depends=(
        f"{SCHEDULER}.is_fully_idle",
        f"{SCHEDULER}.maybe_send_health_check_signal",
        f"{SCHEDULER}.maybe_sleep_on_idle",
        f"{SCHEDULER}.publish_load_snapshot",
        "sglang.srt.managers.scheduler_components.invariant_checker."
        "SchedulerInvariantChecker",
        STAGE_METHOD,
    ),
    reason=(
        SAME_GATE
        + "Replace: the predicate gates a mid-function block. The "
        "@scheduler_stage_method decorator is kept."
    ),
)
@scheduler_stage_method(SCHEDULER_STAGE_IDLE)
def on_idle(self):
    """Idle housekeeping: guard, check, metrics, reset, sleep."""
    # Flush any health-check signal deferred while the engine was busy.
    self.maybe_send_health_check_signal()

    # Publish before the fully-idle gate: a no-batch-but-not-idle stall
    # (queues parked under KV pressure / disagg transfer) has no
    # process_batch_result to publish the growing gauge, and gating here
    # froze /get_loads, DP balancing, and the LoadStat for the stall. This
    # path is polled repeatedly, so a wall-clock floor bounds the
    # O(queue) get_loads for both sinks; the fully-idle publish runs
    # post-flush below.
    fully_idle = self.is_fully_idle()
    if not fully_idle:
        self.metrics_reporter.record_scheduler_active(time.monotonic_ns())
        now = time.monotonic()
        if now - self._last_stall_publish_ts >= LOAD_STALL_REFRESH_S:
            self._last_stall_publish_ts = now
            snapshot = self.publish_load_snapshot(force=True)
            self.load_publisher.publish_load_stat(
                self.load_inquirer.get_loads, force=True, snapshot=snapshot
            )
        if (
            self.enable_hicache_storage
            or self.disaggregation_mode != DisaggregationMode.NULL
            or self.enable_lmcache
        ):
            # Storage and transfer workers need the GIL between I/O calls.
            # Singleton PD polls no longer yield through a collective.
            time.sleep(0)
        return
    self.metrics_reporter.record_scheduler_idle()

    if self.enable_unified_memory:
        try:
            self.token_to_kv_pool_allocator.flush_opportunistic()
        except Exception:
            pass

    # memory leak check (skipped for hisparse — pool counters intentionally
    # diverge during host-backup, see _get_swa_token_info clamp).
    # Also skipped while deferred KV releases are pending: they hold pages out
    # of the allocator by design, so the pool is transiently below `total` and
    # would trip the idle leak invariant. Resumes once the holds resolve.
    deferred_pending = (
        self.disaggregation_mode == DisaggregationMode.DECODE
        and self.disagg_decode_transfer_queue.has_pending_deferred_releases()
    )
    with self.scheduler_stage_metrics.record(SCHEDULER_STAGE_SANITY_CHECK_CACHE):
        if self.hisparse_coordinator is None and not deferred_pending:
            has_leak, messages = self.invariant_checker._check_all_pools(
                self.pool_stats_observer.get_pool_stats(),
            )
            if has_leak:
                self.invariant_checker._report_leak("pool", "\n".join(messages))
            self.invariant_checker._check_req_pool()
            # Byte-conservation diagnostic (allocator-owned; static pools
            # return [] — the token identity above can't see byte leaks).
            byte_violations = (
                self.token_to_kv_pool_allocator.verify_byte_accounting()
            )
            if byte_violations:
                self.invariant_checker._report_leak(
                    "pool-bytes", "\n".join(byte_violations)
                )

        # tree cache sanity check
        self.invariant_checker._check_tree_cache()

    # metrics every 30s
    self.metrics_reporter._maybe_log_idle_metrics()

    # kv event publishing
    self.kv_events_publisher.publish_kv_events()

    # reset token ratio
    self.new_token_ratio_tracker.reset()

    # Fully-idle publish, post-flush so the gauge reflects compacted KV.
    # Forced (immediate) so the busy->idle transition is never delayed.
    snapshot = self.publish_load_snapshot(force=True)
    self.load_publisher.publish_load_stat(
        self.load_inquirer.get_loads, force=True, snapshot=snapshot
    )

    # sleep until next event
    self.maybe_sleep_on_idle()
    self.metrics_reporter.record_scheduler_idle()


# Production managers/scheduler_components/batch_result_processor.py:116-133, verbatim.
@patch(
    f"{PROCESSOR}.process_batch_result_prebuilt",
    "replace",
    feature=HISPARSE,
    row="B02",
    depends=(RELEASE_KV_CACHE, GET_DISAGG),
    reason=(
        SAME_GATE
        + "Replace: the predicate is mid-function, and get_memory cannot be "
        "rebound for one module because HookRegistry propagation rebinds every "
        "module's reference."
    ),
)
def process_batch_result_prebuilt(self, batch: ScheduleBatch):
    assert self.disaggregation_mode == DisaggregationMode.DECODE
    use_free_group = get_disagg().disaggregation_decode_enable_radix_cache
    if use_free_group:
        self.token_to_kv_pool_allocator.free_group_begin()
    for req in batch.reqs:
        req.time_stats.set_decode_prebuilt_finish_time()
        req.update_finish_state()
        if req.finished():
            req.time_stats.set_quick_finish_time()
            if self.hisparse_coordinator is not None:
                self.hisparse_coordinator.request_finished(req)
            release_kv_cache(req, self.tree_cache)

    # Note: Logprobs should be handled on the prefill engine.
    self.output_streamer.stream_output(batch.reqs, batch.return_logprob)
    if use_free_group:
        self.token_to_kv_pool_allocator.free_group_end()


# Production managers/scheduler_components/batch_result_processor.py:257-494, verbatim.
@patch(
    f"{PROCESSOR}.process_batch_result_prefill",
    "replace",
    feature=HISPARSE,
    row="B03",
    depends=(
        f"{PROCESSOR}._append_prefill_hidden_states",
        f"{PROCESSOR}._apply_chunked_prefill_logprobs",
        f"{PROCESSOR}._apply_prefill_grammar",
        f"{PROCESSOR}._apply_prefill_logprobs",
        f"{PROCESSOR}._convert_embeddings",
        f"{PROCESSOR}._get_prefill_hidden_capture_mode",
        f"{PROCESSOR}._maybe_collect_customized_info",
        f"{PROCESSOR}._maybe_collect_indexer_topk",
        f"{PROCESSOR}._maybe_collect_routed_experts",
        f"{PROCESSOR}._maybe_update_reasoning_tokens",
        f"{PROCESSOR}._validate_pp_skip_output_comm",
        f"{PROCESSOR}.add_sampling_mask_return_values",
        f"{PROCESSOR}.consume_auxiliary_output",
        f"{PROCESSOR}.get_sampling_mask_finish_reason",
        f"{PROCESSOR}.materialize_sampling_mask_output",
        f"{PROCESSOR}.move_logprobs_to_cpu",
        f"{PROCESSOR}.snapshot_auxiliary_output_starts",
        "sglang.srt.mem_cache.common.checkpoint_kv_cache",
        RELEASE_KV_CACHE,
    ),
    reason=SAME_GATE + "Replace: the predicate is mid-function (B02 argument).",
)
def process_batch_result_prefill(
    self,
    batch: ScheduleBatch,
    result: Union[GenerationBatchResult, EmbeddingBatchResult],
):
    skip_stream_req = None
    self.token_to_kv_pool_allocator.free_group_begin()

    if self.is_generation:
        if result.copy_done is not None:
            result.copy_done.synchronize()
        auxiliary_output_starts = self.snapshot_auxiliary_output_starts(
            batch, result
        )
        auxiliary_output = result.auxiliary_host_output
        if result.routed_experts_output is not None:
            result.routed_experts_output.finalize()
            result.routed_experts_output = None
        if result.indexer_topk_output is not None:
            result.indexer_topk_output.finalize()
            result.indexer_topk_output = None

        (
            logits_output,
            next_token_ids,
            extend_input_len_per_req,
            extend_logprob_start_len_per_req,
        ) = (
            result.logits_output,
            result.next_token_ids,
            result.extend_input_len_per_req,
            result.extend_logprob_start_len_per_req,
        )

        # Move next_token_ids and logprobs to cpu
        next_token_ids = next_token_ids.tolist()
        self.move_logprobs_to_cpu(batch=batch, logits_output=logits_output)
        self.materialize_sampling_mask_output(batch.reqs, logits_output)

        self._validate_pp_skip_output_comm(batch, result)

        hidden_state_offset = 0
        prefill_hidden_capture_mode = self._get_prefill_hidden_capture_mode(
            batch,
        )

        # Check finish conditions
        logprob_pt = 0

        for i, (req, next_token_id) in enumerate(zip(batch.reqs, next_token_ids)):
            should_commit_output = (
                not req.finished()
                and not req.is_retracted
                and req.inflight_middle_chunks <= 0
            )
            sampling_mask_finish_reason = None
            if should_commit_output and req.return_sampling_mask:
                assert logits_output is not None
                statuses = logits_output.next_token_sampling_mask_status
                status = None if statuses is None else statuses[i]
                sampling_mask_finish_reason = self.get_sampling_mask_finish_reason(
                    status=status
                )
            if (
                batch.return_hidden_states
                and logits_output.hidden_states is not None
            ):
                assert extend_input_len_per_req is not None
                hidden_state_offset = self._append_prefill_hidden_states(
                    req=req,
                    logits_output=logits_output,
                    hidden_state_offset=hidden_state_offset,
                    capture_hidden_mode=prefill_hidden_capture_mode,
                    extend_input_len=extend_input_len_per_req[i],
                    store=(
                        should_commit_output and sampling_mask_finish_reason is None
                    ),
                )

            if (
                req.finished() and req.inflight_middle_chunks <= 0
            ) or req.is_retracted:
                # Decode req in a mixed batch, or a retracted req. Keep an
                # aborted middle chunk in the chunked branch long enough to
                # drain its accounting without streaming it.
                continue

            if req.inflight_middle_chunks <= 0:
                req.time_stats.set_prefill_finished_time()

                if sampling_mask_finish_reason is not None:
                    req.to_finish = sampling_mask_finish_reason
                    req.update_finish_state(0)
                elif req.beam_group is not None:
                    # The relay point already replaced the sampled-token
                    # append; the group owns all finish semantics.
                    self.beam_coordinator.commit_prefill(
                        req, up_to_tick=batch.forward_iter
                    )
                else:
                    # req output_ids are set here
                    req.output_ids.append(next_token_id)

                    self._maybe_update_reasoning_tokens(req, next_token_id)

                    req.update_finish_state()
                # A mixed spec tail committed its pending bonus token; advance
                # so the next spec prepare_for_decode reserves from the right base.
                if (
                    not req.finished()
                    and batch.decoding_reqs
                    and req in batch.decoding_reqs
                    and not batch.spec_algorithm.is_none()
                ):
                    req.kv.kv_committed_len += 1
                if req.finished():
                    if sampling_mask_finish_reason is None:
                        self._maybe_collect_routed_experts(req)
                        self._maybe_collect_indexer_topk(req)
                    release_kv_cache(
                        req,
                        self.tree_cache,
                        is_insert=sampling_mask_finish_reason is None,
                    )
                    req.time_stats.set_completion_time()
                elif not batch.decoding_reqs or req not in batch.decoding_reqs:
                    checkpoint_kv_cache(req, self.tree_cache)
                    if self.hisparse_coordinator is not None:
                        self.hisparse_coordinator.admit_request_into_staging(req)

                if sampling_mask_finish_reason is None:
                    self._maybe_collect_customized_info(i, req, logits_output)

                if batch.return_logprob:
                    logprob_pt = self._apply_prefill_logprobs(
                        req=req,
                        i=i,
                        logits_output=logits_output,
                        extend_input_len_per_req=extend_input_len_per_req,
                        extend_logprob_start_len_per_req=extend_logprob_start_len_per_req,
                        next_token_ids=next_token_ids,
                        logprob_pt=logprob_pt,
                        store=sampling_mask_finish_reason is None,
                    )

                if sampling_mask_finish_reason is not None:
                    continue

                if req.return_sampling_mask:
                    self.add_sampling_mask_return_values(i, req, logits_output)

                if req.grammar is not None:
                    self._apply_prefill_grammar(
                        req=req,
                        next_token_id=next_token_id,
                        already_advanced=result.grammar_advanced,
                    )

            else:
                # being chunked reqs' prefill is not finished
                req.inflight_middle_chunks -= 1
                # There is only at most one request being currently chunked.
                # Because this request does not finish prefill,
                # we don't want to stream the request currently being chunked.
                skip_stream_req = req

                # Incrementally update input logprobs.
                if batch.return_logprob:
                    logprob_pt = self._apply_chunked_prefill_logprobs(
                        req=req,
                        i=i,
                        logits_output=logits_output,
                        extend_input_len_per_req=extend_input_len_per_req,
                        extend_logprob_start_len_per_req=extend_logprob_start_len_per_req,
                        logprob_pt=logprob_pt,
                    )

                req.time_stats.set_last_chunked_prefill_finish_time()

        if auxiliary_output is not None:
            self.consume_auxiliary_output(
                batch,
                auxiliary_output,
                auxiliary_output_starts,
            )

    else:  # embedding or reward model
        if result.copy_done is not None:
            result.copy_done.synchronize()

        embeddings = self._convert_embeddings(result=result)
        phs = result.pooled_hidden_states

        if phs is not None:
            if isinstance(phs, list):
                phs = [t.cpu().detach() for t in phs]
            else:
                phs = phs.cpu().detach()

        # Check finish conditions
        for i, req in enumerate(batch.reqs):
            if req.is_retracted:
                continue

            req.embedding = embeddings[i]
            if req.return_pooled_hidden_states and phs is not None:
                req.pooled_hidden_state = phs[i]
            if req.inflight_middle_chunks <= 0:
                req.time_stats.set_prefill_finished_time()
                # Dummy output token for embedding models
                req.output_ids.append(0)
                req.update_finish_state()

                if req.finished():
                    release_kv_cache(req, self.tree_cache)
                    req.time_stats.set_completion_time()
                else:
                    checkpoint_kv_cache(req, self.tree_cache)
            else:
                # being chunked reqs' prefill is not finished
                req.inflight_middle_chunks -= 1
                req.time_stats.set_last_chunked_prefill_finish_time()

    self.token_to_kv_pool_allocator.free_group_end()
    self.output_streamer.stream_output(
        batch.reqs, batch.return_logprob, skip_stream_req
    )

    can_run_cuda_graph = result.can_run_cuda_graph
    # None on decode->extend converted batches; they are decode work and
    # have no prefill stats to report.
    if batch.prefill_stats is not None:
        self.metrics_reporter.report_prefill_stats(
            batch=batch,
            prefill_stats=batch.prefill_stats,
            can_run_cuda_graph=can_run_cuda_graph,
            dp_cooperation_info=batch.dp_cooperation_info,
        )


# Production managers/scheduler_components/batch_result_processor.py:1325-1340, verbatim.
@patch(
    f"{PROCESSOR}._handle_sampling_mask_abort",
    "replace",
    feature=HISPARSE,
    row="B04",
    depends=(RELEASE_KV_CACHE, GET_DISAGG),
    reason=SAME_GATE + "Replace: the predicate is mid-function (B02 argument).",
)
def _handle_sampling_mask_abort(self, req: Req) -> None:
    """Release a request whose sampled token must not be committed."""
    if req.multimodal_inputs is not None and req.session is None:
        req.multimodal_inputs.release_features()
    if get_disagg().disaggregation_decode_enable_offload_kvcache:
        self.decode_offload_manager.finalize_release_on_finish(req)
    else:
        if self.hisparse_coordinator is not None:
            self.hisparse_coordinator.request_finished(req)
        prepare_release = getattr(
            self.model_worker, "prepare_for_kv_cache_release", None
        )
        if callable(prepare_release):
            prepare_release(req)
        release_kv_cache(req, self.tree_cache, is_insert=False)
    req.time_stats.set_completion_time()


# Production managers/scheduler_components/batch_result_processor.py:1342-1435, verbatim.
@patch(
    f"{PROCESSOR}._handle_finish_state_updated_req",
    "replace",
    feature=HISPARSE,
    row="B05",
    depends=(
        f"{PROCESSOR}._mamba_prefix_cache_update",
        f"{PROCESSOR}._maybe_collect_customized_info",
        f"{PROCESSOR}._maybe_collect_indexer_topk",
        f"{PROCESSOR}._maybe_collect_routed_experts",
        RELEASE_KV_CACHE,
        GET_DISAGG,
        "sglang.srt.runtime_context.get_exec",
    ),
    reason=SAME_GATE + "Replace: the predicate is mid-function (B02 argument).",
)
def _handle_finish_state_updated_req(
    self,
    req: Req,
    batch: ScheduleBatch,
    result: GenerationBatchResult,
    i: int,
    logits_output: LogitsProcessorOutput,
):
    lazy = get_exec().mamba.enable_mamba_extra_buffer_lazy
    known_mamba_boundary = None
    completed_mamba_boundary = None
    lookahead = 0
    if batch.mamba_track_mask_cpu is not None:
        completed_mamba_boundary = bool(batch.mamba_track_mask_cpu[i])
        lookahead = req.decode_batch_idx - batch.mamba_decode_batch_idx_cpu[i]
        assert lookahead in (0, 1), (
            f"mamba result lookahead={lookahead} for req {req.rid}; "
            "overlap advanced more than one decode batch"
        )
        if lookahead == 0:
            known_mamba_boundary = bool(batch.mamba_track_mask_cpu[i])
        else:
            known_mamba_boundary = bool(batch.mamba_track_mask_next_cpu[i])

        if completed_mamba_boundary and not lazy:
            req.kv.mamba_last_track_idx = batch.mamba_track_buffer_indices[i]
            # The slot that stops being the latest still holds its
            # checkpoint; name it so a short key can fall back to it.
            req.kv.mamba_prev_track_seqlen = req.kv.mamba_last_track_seqlen
            req.kv.mamba_last_track_seqlen = req.kv.kv_committed_len - lookahead
        elif (
            req.finished()
            and lazy
            and lookahead == 1
            and known_mamba_boundary
            and req.kv.mamba_next_track_idx == req.kv.mamba_last_track_idx
        ):
            req.mamba_lazy_is_insert = False

    # Called here (after update_finish_state) so req.finished() is valid
    # for mamba_lazy_post_decode_at_boundary inside.
    should_update = completed_mamba_boundary if lazy else known_mamba_boundary
    if should_update is None or should_update:
        self._mamba_prefix_cache_update(
            req,
            batch,
            result,
            i,
            known_boundary=not lazy and known_mamba_boundary is True,
        )

    if (
        get_disagg().disaggregation_decode_enable_offload_kvcache
        and not req.finished()
    ):
        self.decode_offload_manager.offload_kv_cache(req)

    if req.finished():
        # isinstance narrowing: create_worker may also return plain
        # TpModelWorker-based drafts, which carry no spec-worker hooks.
        if isinstance(self.draft_worker, BaseSpecWorker):
            self.draft_worker.note_request_finished(
                rid=req.rid,
                natural_stop=isinstance(req.finished_reason, FINISH_MATCHED_TOKEN),
            )

        # delete feature to save memory
        if req.multimodal_inputs is not None and req.session is None:
            req.multimodal_inputs.release_features()
        self._maybe_collect_routed_experts(req)
        self._maybe_collect_indexer_topk(req)

        if get_disagg().disaggregation_decode_enable_offload_kvcache:
            # Asynchronously offload KV cache; release_kv_cache will be called after Device->Host transfer completes
            if not self.decode_offload_manager.offload_kv_cache(req):
                self.decode_offload_manager.finalize_release_on_finish(req)
        else:
            if self.hisparse_coordinator is not None:
                self.hisparse_coordinator.request_finished(req)
            prepare_release = getattr(
                self.model_worker, "prepare_for_kv_cache_release", None
            )
            if callable(prepare_release):
                prepare_release(req)
            is_insert = (
                req.mamba_lazy_is_insert
                if get_exec().mamba.enable_mamba_extra_buffer_lazy
                else True
            )
            release_kv_cache(req, self.tree_cache, is_insert=is_insert)

        req.time_stats.set_completion_time()

    self._maybe_collect_customized_info(i, req, logits_output)


@patch(
    f"{ADDER}.add_chunked_req",
    "before",
    feature=HISPARSE,
    row="P01",
    depends=(f"{ADDER}.__init__", f"{SCHEDULER}._get_new_batch_prefill_raw"),
    reason=(
        "hisparse: a getattr on the tree cache; no upstream cache defines "
        "prefill_checkpoint_limit, so the chunk is unchanged without "
        "QSAHostPrefixCache. Before hook (P3 narrowing of the fork's replace, "
        "via v0.5.21's chunked_req_limit seam): the fork caps the local chunk "
        "budget at the checkpoint limit after the budget early return and "
        "before the prefill delayer; v0.5.21 applies min(budget, "
        "chunked_req_limit) after the delayer, which does not read the budget, "
        "so lowering chunked_req_limit to the limit first is equivalent. "
        "chunked_req_limit is initialized to None in PrefillAdder.__init__, "
        "set by the scheduler just before its single add_chunked_req call per "
        "adder (S06), and read nowhere else; prefill_checkpoint_limit returns a "
        "positive int or None (upstream asserts > 0) and, apart from memoizing "
        "the request's image identity, only reads the request, so calling it "
        "before the budget computation changes nothing. Production "
        "schedule_policy.py 1149-1153 verbatim, then the limit is stored "
        "instead of applied."
    ),
)
def _cap_chunk_at_checkpoint(self, req):
    checkpoint_limit = getattr(self.tree_cache, "prefill_checkpoint_limit", None)
    if (
        checkpoint_limit is not None
        and (limit := checkpoint_limit(req)) is not None
    ):
        if self.chunked_req_limit is not None:
            limit = min(limit, self.chunked_req_limit)
        self.chunked_req_limit = limit


# Production managers/schedule_policy.py:1373-1402, verbatim.
@patch(
    f"{ADDER}.add_one_req",
    "replace",
    feature=HISPARSE,
    row="P02",
    depends=(
        f"{ADDER}.add_one_req_ignore_eos",
        f"{ADDER}._select_prefill_admission",
        f"{ADDER}._commit_prefill_admission",
        f"{ADDER}.budget_state",
        f"{ADDER}._lock_node",
        f"{ADDER}._mamba_gap_budget_for_req",
        f"{ADDER}._kv_shard_reserve_scratch",
        "sglang.srt.managers.schedule_policy._PrefillAdmission",
        "sglang.srt.mem_cache.base_prefix_cache.InitLoadBackParams",
    ),
    reason=(
        "hisparse: getattrs on the tree cache; with upstream caches the charge is "
        "0, the limit None and the ignore_eos predicate unchanged. Replace: the "
        "ignore_eos predicate is mid-function. Production renamed the original "
        "body to PrefillAdder._add_one_req, which is attached (absent at the "
        "pin), so the copy keeps `self._add_one_req(...)` verbatim. "
        "_add_one_req is production's renamed body: the pinned add_one_req "
        "with the fork's ignore_eos change (tests/lifecycle/test_renamed_copies.py "
        "checks it)."
    ),
)
def add_one_req(
    self, req: Req, has_chunked_req: bool, truncation_align_size: Optional[int]
):
    pending_prefix = getattr(self.tree_cache, "pending_prefix_tokens", None)
    charge = 0 if pending_prefix is None else pending_prefix(req)
    checkpoint_limit = getattr(self.tree_cache, "prefill_checkpoint_limit", None)
    limit = None if checkpoint_limit is None else checkpoint_limit(req)
    original_chunk_tokens = self.rem_chunk_tokens
    capped = limit is not None and (
        original_chunk_tokens is None or limit < original_chunk_tokens
    )
    if capped:
        self.rem_chunk_tokens = limit
    # Host snapshots still require fresh logical/index pages. Charge those
    # pages before the existing suffix/decode admission checks.
    self.memory_budget.total_offset += charge
    self.memory_budget.current_offset += charge
    try:
        return self._add_one_req(req, has_chunked_req, truncation_align_size)
    finally:
        if capped:
            spent = limit - self.rem_chunk_tokens
            self.rem_chunk_tokens = (
                None
                if original_chunk_tokens is None
                else original_chunk_tokens - spent
            )
        if req not in self.can_run_list:
            self.memory_budget.total_offset -= charge
            self.memory_budget.current_offset -= charge


# Production PrefillAdder._add_one_req (managers/schedule_policy.py:1404-1566), verbatim.
@attach(ADDER, feature=HISPARSE, row="P02")
def _add_one_req(
    self, req: Req, has_chunked_req: bool, truncation_align_size: Optional[int]
):
    if (x := self.prefill_max_requests) is not None and len(self.can_run_list) >= x:
        return AddReqResult.OTHER

    pending_prefix = getattr(self.tree_cache, "pending_prefix_tokens", None)
    if (
        req.sampling_params.ignore_eos
        and getattr(self.tree_cache, "disable", True)
        and not (pending_prefix is not None and pending_prefix(req))
    ):
        return self.add_one_req_ignore_eos(req)

    # Reserve page_size for page-alignment overhead: the paged allocator may
    # consume one extra page per request (see alloc_extend), which
    # _update_prefill_budget also deducts.
    max_new = min(
        max(req.sampling_params.max_new_tokens - len(req.output_ids), 0),
        CLIP_MAX_NEW_TOKENS,
    )
    cand_extend_input_len = len(req.full_untruncated_fill_ids) - len(
        req.prefix_indices
    )
    total_tokens = cand_extend_input_len + max_new + self.per_req_token_overhead
    # Shared Mamba pool: fold the new mamba state's shared-gap cost into
    # `total_tokens` so both `rem_total_tokens` gates reflect the joint budget.
    # Read before `init_load_back` binds `req.mamba_pool_idx` — after that
    # this returns 0, so the debit sites below reuse the value.
    mamba_gap_reserve = self._mamba_gap_budget_for_req(req)
    total_tokens += mamba_gap_reserve

    # The temporary pin excludes this prefix from the evictable budget.
    # Selection itself neither allocates slots nor materializes host hits.
    with self._lock_node(req.last_node):
        admission = self._select_prefill_admission(
            req,
            total_tokens=total_tokens,
            host_hit_length=req.host_hit_length,
            swa_host_hit_length=req.swa_host_hit_length,
            truncation_align_size=truncation_align_size,
            has_chunked_req=has_chunked_req,
        )
        if isinstance(admission, AddReqResult):
            return admission

        # A rejected candidate must not report prefillable or queue H2D.
        if (self.prefill_delayer_single_pass is not None) and (
            not self.prefill_delayer_single_pass.negotiate_should_allow_prefill(
                local_prefillable=True,
                running_batch=self.running_batch.batch_size(),
                max_prefill_bs=self.max_prefill_bs,
                max_running_requests=self.max_running_requests,
                waiting_queue_len=self.waiting_queue_len,
            )
        ):
            return AddReqResult.OTHER

        if req.needs_host_load_back():
            load_max_new = min(max_new, admission.max_new_tokens)
            # Reclaim can write back device victims and evict host leaves.
            # Pin the selected host/aux match until load-back owns its locks.
            with (
                self._lock_node(req.best_match_node, lock_host=True)
                if isinstance(self.tree_cache, UnifiedRadixCache)
                else nullcontext()
            ):
                full_load_tokens = req.host_hit_length
                if (
                    isinstance(self.tree_cache, UnifiedRadixCache)
                    and self.tree_cache.buffer_pipeline is None
                    and not (
                        self.tree_cache.linker is not None
                        and self.tree_cache.linker.has_hit(req.rid)
                    )
                ):
                    # Host hits can include resident FULL behind host-only aux.
                    # Reuse the FULL transfer spec to count only new slots.
                    full_transfer = (
                        self.tree_cache.tree_core.build_hicache_transfers(
                            ComponentType.FULL,
                            req.best_match_node,
                            CacheTransferPhase.LOAD_BACK,
                        )[0]
                    )
                    full_load_tokens = len(full_transfer.host_indices)
                if not self.memory_budget.prepare_load_back(
                    full_tokens=(
                        full_load_tokens
                        + admission.extend_len
                        + load_max_new
                        + self.page_size
                        + mamba_gap_reserve
                    ),
                    extend_input_len=admission.extend_len,
                    max_new_tokens=load_max_new,
                    swa_host_hit_length=req.swa_host_hit_length,
                    chunk_limit=self.rem_chunk_tokens,
                ):
                    return AddReqResult.NO_TOKEN
                promised_host_hit = req.host_hit_length
                loaded = self.tree_cache.init_load_back(
                    InitLoadBackParams(
                        best_match_node=req.best_match_node,
                        host_hit_length=req.host_hit_length,
                        req=req,
                    )
                )
            if loaded is None:
                return AddReqResult.OTHER
            new_indices, req.last_node = loaded
            req.host_loaded_length = len(new_indices)
            if 0 < req.host_loaded_length < promised_host_hit:
                raise RuntimeError(
                    "HiCache load-back must commit all promised FULL tokens or none: "
                    f"req={req.rid} promised={promised_host_hit} "
                    f"loaded={req.host_loaded_length}"
                )
            if req.host_loaded_length > promised_host_hit:
                # A load can expose resident FULL behind host-only aux state; its
                # H2D is queued, so keep the approved budget and shrink the work.
                prefix_len = len(req.prefix_indices) + req.host_loaded_length
                extend_len = admission.extend_len
                if self.dllm_config is None:
                    extend_len = min(
                        extend_len, len(req.full_untruncated_fill_ids) - prefix_len
                    )
                is_chunked = admission.is_chunked and (
                    prefix_len + extend_len < len(req.full_untruncated_fill_ids)
                )
                max_new_tokens = admission.max_new_tokens
                if admission.is_chunked and not is_chunked:
                    max_new_tokens = min(
                        req.sampling_params.max_new_tokens, CLIP_MAX_NEW_TOKENS
                    )
                admission = _PrefillAdmission(
                    prefix_len, extend_len, max_new_tokens, is_chunked
                )
            elif req.host_loaded_length < promised_host_hit:
                # No FULL was loaded; recomputation may no longer fit.
                admission = self._select_prefill_admission(
                    req,
                    total_tokens=total_tokens,
                    host_hit_length=0,
                    swa_host_hit_length=0,
                    truncation_align_size=truncation_align_size,
                    has_chunked_req=has_chunked_req,
                )
                if isinstance(admission, AddReqResult):
                    return admission
            req.prefix_indices = torch.cat([req.prefix_indices, new_indices])
            req.kv.cache_protected_len = len(req.prefix_indices)

        # Sharded pools cannot load host KV; reserve scratch after all other gates.
        if not self._kv_shard_reserve_scratch(
            prefix_len=admission.prefix_len, extend_len=admission.extend_len
        ):
            return AddReqResult.OTHER

        self._commit_prefill_admission(req, admission, mamba_gap_reserve)

    # This verdict controls the next candidate, not the committed request.
    return self.budget_state()
