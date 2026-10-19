"""Scheduler, batch-result and prefill-admission integration (inventory W2).

Rows S02, S04-S09 and B02-B05; P01 and P02 are methods of S06's
``QSAPrefillAdder``. B01 (an unused import) is a drop; S01 is removed at
v0.5.22 (M05 builds the host prefix cache). REPLACE hooks are v0.5.22's
definitions plus production's change (PLAN.md rules 3 and P1); they run
with this module's globals, so every name they use is imported below from
the pinned module that defines the target (inventory 6, G3).
``release_kv_cache`` and the other patched callees are rebound to their
patched versions by HookRegistry's propagation.
"""

from __future__ import annotations

import time

from sglang.srt.managers.schedule_policy import PrefillAdder
from sglang.srt.managers.scheduler import (
    LOAD_STALL_REFRESH_S,
    SCHEDULER_STAGE_GET_NEXT_BATCH,
    SCHEDULER_STAGE_IDLE,
    SCHEDULER_STAGE_SANITY_CHECK_CACHE,
    DisaggregationMode,
    NextBatchPlan,
    envs,
    get_spec,
    scheduler_stage_method,
    set_schedule_time_batch,
)
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    FINISH_MATCHED_TOKEN,
    BaseSpecWorker,
    checkpoint_kv_cache,
    get_disagg,
    get_exec,
    get_memory,
    release_kv_cache,
)
from sglang_qsa_hisparse.features import HISPARSE
from sglang_qsa_hisparse.hisparse.prefix_cache import QSAHostPrefixCache
from sglang_qsa_hisparse.patching import patch

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
        "TpModelWorker.register_hisparse_coordinator has no callers). "
        "Unchanged at v0.5.22 (the runner's builder gained num_draft_tokens)."
    ),
)
def _adopt_qsa_coordinator(result, self):
    qsa = getattr(self.token_to_kv_pool_allocator.get_kvcache(), "qsa_hisparse", None)
    if not self.enable_hisparse and getattr(qsa, "uses_qsa_hisparse_leases", False):
        self.hisparse_coordinator = self.tp_worker.model_runner.hisparse_coordinator
        if self.hisparse_coordinator is not None:
            self.hisparse_coordinator.set_decode_producer_stream(self.forward_stream)


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
        "original already applies it. ignore_waiting only replaces the "
        "waiting-queue term (v0.5.22: by a side-effect-free check that no "
        "waiting request holds a row). Production scheduler.py 5088."
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


# v0.5.22 managers/scheduler.py:3765-3905 with production's change.
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
    reqs_to_exclude = set()

    if self.dllm_config is not None and self.dllm_manager.any_staging_reqs():
        reqs_to_exclude.update(self.dllm_manager.staging_queue)
        for req in self.dllm_manager.staging_queue:
            self.finish_dllm_forward(req)

    if self.chunked_req is not None:
        # Move the chunked request out of the batch so that we can merge
        # only finished requests to running_batch.
        reqs_to_exclude.add(self.chunked_req)

        # A parked chunk (add_chunked_req hybrid-SWA early-return) leaves
        # extend_end at prefix_len: it computed no new KV, so nothing to stash.
        if self.chunked_req.extend_end > self.chunked_req.prefix_len:
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
            reqs_to_exclude.add(last_batch.chunked_req)

        if self.dllm_config is not None and last_batch.reqs:
            reqs_to_exclude.update(last_batch.reqs)

        # Filter batch
        last_bs = last_batch.batch_size()
        last_batch.filter_batch(reqs_to_exclude=list(reqs_to_exclude))
        if last_batch.batch_size() < last_bs:
            running_batch.batch_is_full = False

        # Merge the new batch into the running batch. A running batch that
        # DP attention converted to extend is last_batch itself: filtered
        # in place above, nothing to merge.
        if not last_batch.is_empty() and last_batch is not running_batch:
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
            decode_batch = self.update_running_batch(running_batch)
            ret = (
                decode_batch
                if decode_batch is not None and not decode_batch.is_empty()
                else None
            )
        else:
            ret = None

    # Handle DP attention and log stats
    ret = self.dp_attn_adapter.maybe_prepare_mlp_sync_batch(
        ret, need_sync=need_mlp_sync
    )
    # Decode->extend conversion keeps a heterogeneous dp step replayable.
    # A converted running batch stays the running batch; next step it comes
    # back as last_batch and is filtered in place, not merged with itself.
    ret = self.dp_attn_adapter.maybe_convert_decode_to_extend(ret)
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


class QSAPrefillAdder(PrefillAdder):
    """Prefill admission under QSA leases (S06), with host prefixes (P01, P02).

    One request per prefill pass. With ``QSAHostPrefixCache`` a prefill stops
    at the request's last reusable page (its host-prefix checkpoint), and an
    admitted host hit is charged and admitted as production did.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prefill_max_requests = 1

    def _checkpoint_limit(self, req):
        if not isinstance(self.tree_cache, QSAHostPrefixCache):
            return None
        return self.tree_cache.prefill_checkpoint_limit(req)

    def add_chunked_req(self, req):
        # P01: chunked_req_limit caps the chunk after the prefill delayer, which
        # does not read the budget, so it equals production's earlier cap.
        limit = self._checkpoint_limit(req)
        if limit is not None:
            if self.chunked_req_limit is not None:
                limit = min(limit, self.chunked_req_limit)
            self.chunked_req_limit = limit
        return super().add_chunked_req(req)

    def add_one_req(
        self, req, has_chunked_req: bool, truncation_align_size: Optional[int]
    ):
        if not isinstance(self.tree_cache, QSAHostPrefixCache):
            return super().add_one_req(req, has_chunked_req, truncation_align_size)
        pending = self.tree_cache.pending_prefix_tokens(req)
        limit = self.tree_cache.prefill_checkpoint_limit(req)
        original_chunk_tokens = self.rem_chunk_tokens
        capped = limit is not None and (
            original_chunk_tokens is None or limit < original_chunk_tokens
        )
        if capped:
            self.rem_chunk_tokens = limit
        sampling = req.sampling_params
        ignore_eos = sampling.ignore_eos
        if pending:
            # Production admits an ignore_eos request with a pending host prefix
            # through the normal path, which reuses the prefix. The fast path is
            # chosen on this flag, which nothing else in add_one_req reads.
            sampling.ignore_eos = False
        try:
            result = super().add_one_req(req, has_chunked_req, truncation_align_size)
            if pending and req in self.can_run_list:
                # Admission demand already counts the restored pages (prefix_len
                # is 0 until init_load_back); production also kept them in both
                # offsets for the verdict and the rest of the pass.
                self.memory_budget.total_offset += pending
                self.memory_budget.current_offset += pending
                result = self.budget_state()
            return result
        finally:
            sampling.ignore_eos = ignore_eos
            if capped:
                spent = limit - self.rem_chunk_tokens
                self.rem_chunk_tokens = (
                    None
                    if original_chunk_tokens is None
                    else original_chunk_tokens - spent
                )


@patch(
    f"{SCHEDULER}.get_prefill_adder_class",
    "after",
    feature=HISPARSE,
    row="S06",
    depends=(
        f"{SCHEDULER}._get_new_batch_prefill_raw",
        f"{ADDER}.__init__",
        f"{ADDER}.add_chunked_req",
        f"{ADDER}.add_one_req",
        f"{ADDER}.add_one_req_ignore_eos",
        f"{ADDER}.budget_state",
        f"{ADDER}._select_prefill_admission",
        f"{ADDER}._commit_prefill_admission",
        f"{ADDER}._update_prefill_budget",
        "sglang.srt.mem_cache.prefill_budget.PrefillBudget",
        "sglang.srt.mem_cache.allocator.base.BaseTokenToKVPoolAllocator."
        "create_prefill_budget",
    ),
    reason=(
        "hisparse: gated on coordinator.uses_qsa_hisparse_leases; QSAPrefillAdder "
        "changes admission only for that coordinator and QSAHostPrefixCache. "
        "After hook (v0.5.22's get_prefill_adder_class seam, #43870; was a "
        "replace of _get_new_batch_prefill_raw): its only caller builds the "
        "adder with prefill_max_requests=get_schedule().prefill_max_requests, "
        "which PrefillAdder.__init__ only stores and add_one_req alone reads, "
        "so setting it to 1 after super().__init__ equals production's "
        "argument (S06). P01: chunked_req_limit is set by the scheduler just "
        "before add_chunked_req and read only there, after the prefill delayer, "
        "which does not read the budget. P02: a host hit is a host_hit_length "
        "(prefix_len 0), so admission's total_tokens already counts the "
        "restored pages: (F+X) < avail-off equals production's "
        "(F-L+X) < avail-(off+L); after admission L is added to both offsets "
        "and budget_state() recomputed, as production left them. An "
        "ignore_eos request with a pending hit takes the normal path "
        "(production's predicate), by clearing sampling_params.ignore_eos, "
        "which add_one_req reads only to choose the fast path. The pinned "
        "base PrefillBudget (create_prefill_budget of the paged allocator) "
        "checks only remaining_total in check_prefill."
    ),
)
def _qsa_prefill_adder(result, self):
    if getattr(self.hisparse_coordinator, "uses_qsa_hisparse_leases", False):
        return QSAPrefillAdder
    return result


# v0.5.22 managers/scheduler.py:5071-5160 with production's change.
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
    self.dp_attn_adapter.drop_sync_wait_carry()
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


# v0.5.22 managers/scheduler_components/batch_result_processor.py:117-134 with production's change.
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
            release_kv_cache(req, self.tree_cache, checkpoint=True)

    # Note: Logprobs should be handled on the prefill engine.
    self.output_streamer.stream_output(batch.reqs, batch.return_logprob)
    if use_free_group:
        self.token_to_kv_pool_allocator.free_group_end()


# v0.5.22 managers/scheduler_components/batch_result_processor.py:258-515 with production's change.
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
        f"{PROCESSOR}._prepare_kv_cache_release",
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
    reason=(
        SAME_GATE
        + "Replace: the predicate is mid-function (B02 argument). Only the "
        "staging admission gate is production's; v0.5.22's request_finished "
        "gate for requests finishing in a prefill batch (#37771) keeps "
        "enable_hisparse, which QSA leaves unset, so QSA leases are not "
        "notified there, as in production."
    ),
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
            # A reason here finishes the request without committing this
            # forward's output: a failed KV load or an unusable sampling mask.
            discard_finish_reason = (
                req.discard_output_reason if should_commit_output else None
            )
            if (
                should_commit_output
                and discard_finish_reason is None
                and req.return_sampling_mask
            ):
                assert logits_output is not None
                statuses = logits_output.next_token_sampling_mask_status
                status = None if statuses is None else statuses[i]
                discard_finish_reason = self.get_sampling_mask_finish_reason(
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
                    store=(should_commit_output and discard_finish_reason is None),
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

                if discard_finish_reason is not None:
                    req.to_finish = discard_finish_reason
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
                    if discard_finish_reason is None:
                        self._maybe_collect_routed_experts(req)
                        self._maybe_collect_indexer_topk(req)
                    if get_memory().enable_hisparse:
                        self.hisparse_coordinator.request_finished(req)
                    if batch.decoding_reqs and req in batch.decoding_reqs:
                        self._prepare_kv_cache_release(req)
                    release_kv_cache(
                        req,
                        self.tree_cache,
                        checkpoint=discard_finish_reason is None,
                    )
                    req.time_stats.set_completion_time()
                elif not batch.decoding_reqs or req not in batch.decoding_reqs:
                    checkpoint_kv_cache(req, self.tree_cache)
                    if self.hisparse_coordinator is not None:
                        self.hisparse_coordinator.admit_request_into_staging(req)

                if discard_finish_reason is None:
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
                        store=discard_finish_reason is None,
                    )

                if discard_finish_reason is not None:
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

            if (
                req.discard_output_reason is not None
                and req.inflight_middle_chunks <= 0
            ):
                req.to_finish = req.discard_output_reason
                req.update_finish_state(0)
                release_kv_cache(req, self.tree_cache, checkpoint=False)
                req.time_stats.set_completion_time()
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
                    release_kv_cache(req, self.tree_cache, checkpoint=True)
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


# v0.5.22 managers/scheduler_components/batch_result_processor.py:1354-1365 with production's change.
@patch(
    f"{PROCESSOR}._handle_sampling_mask_abort",
    "replace",
    feature=HISPARSE,
    row="B04",
    depends=(RELEASE_KV_CACHE, GET_DISAGG, f"{PROCESSOR}._prepare_kv_cache_release"),
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
        self._prepare_kv_cache_release(req)
        release_kv_cache(req, self.tree_cache, checkpoint=False)
    req.time_stats.set_completion_time()


# v0.5.22 managers/scheduler_components/batch_result_processor.py:1367-1456 with production's change.
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
        f"{PROCESSOR}._prepare_kv_cache_release",
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
            req.mamba_lazy_checkpoint = False

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
            self._prepare_kv_cache_release(req)
            checkpoint = (
                req.mamba_lazy_checkpoint
                if get_exec().mamba.enable_mamba_extra_buffer_lazy
                else True
            )
            release_kv_cache(req, self.tree_cache, checkpoint=checkpoint)

        req.time_stats.set_completion_time()

    self._maybe_collect_customized_info(i, req, logits_output)
