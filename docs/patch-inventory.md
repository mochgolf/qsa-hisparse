# Patch inventory (P0-A)

Every change in `fork:` `ee8fe158d6` relative to the pin `76e06febab` under
`python/sglang`, mapped to a plugin mechanism. Source of truth for hunks:
`git -C ../qsa-hisparse diff -U0 76e06febab ee8fe158d6 -- python/sglang`
(196 hunks in 31 modified files: 28 under `srt/`, the Marlin op wrapper, and
two Marlin headers; plus 15 new files). Appendix A maps every
hunk to exactly one row and was produced by a script that fails on an
unmapped or doubly mapped hunk.

Conventions:

- **Fork lines** are new-side lines at `ee8fe158d6`. File paths are relative
  to `python/sglang/srt/` unless they start with `kernels/`.
- **Targets** are dotted paths verified at the pin with
  `sglang_qsa_hisparse.fingerprint.resolve_file` and `_locate` (all resolve;
  `pin` gives the definition's first line).
- **Types**: `before` / `after` / `around` / `replace` follow HookRegistry.
  `replace (class)` substitutes a subclass. `attach` means adding a new member
  to a pinned class; HookRegistry cannot do that (see §6, G1).
  `moved/owned` means a plugin-owned copy with no hook. `none` means another
  row covers the change. `drop` means the change is not reproduced (with a
  reason).
- **Copy size** (in parentheses after `replace`) is the length of the fork
  definition that is copied, including decorators.
- A narrower hook placed at a different point than the fork's insertion is
  proposed only when the row states why it is equivalent on every pinned path.
  Each such row names a REPLACE fallback in case review rejects the argument.
- **depends** lists the key pinned definitions whose behavior the copy or hook
  assumes. Appendix B has the complete auto-derived callee list for each
  REPLACE copy.
- **WS**: workstreams W1–W7 from PLAN.md. **U**: the upstream PR U1–U9 that
  would remove the REPLACE or hook. "gap" means no planned PR covers it.

## 1. Inventory

> **Reconciled with [DEVIATIONS.md](DEVIATIONS.md) after G0 (2026-10-07).**
> F01 is now a hisparse `after` hook on `ForwardBatch.init_new` (D3); U01 and
> Z05 are drops (D2, D1). Counts in §5 predate this change: one fewer
> model_compat class replace and one fewer model_compat function replace.
> Generic shared-path rows (M04, J03, Z01, Z02, Z03, E-rows) are subject to
> the target-model scope in PLAN.md rule 9.

### W2: scheduler and lifecycle

| ID | Fork file:lines | Behavior | Feature: justification | Hook target (pin) | Type: why | depends (key) | WS | U |
|---|---|---|---|---|---|---|---|---|
| S01 | managers/scheduler.py:630-645 | Once the coordinator is set, wrap the ChunkCache in `QSAHostPrefixCache` when the runtime has a host prefix cache. Reject the hierarchical cache and the Mamba extra buffer | hisparse: needs `kvcache.qsa_hisparse.prefix_cache`; no upstream kvcache has `qsa_hisparse` | `sglang.srt.managers.scheduler.Scheduler.init_hisparse_coordinator` (pin 1267) | after: the only call site is `Scheduler.__init__` (pin 618), immediately before the fork's insertion, so every later capture of `tree_cache` (e.g. `init_batch_result_processor`, pin 713) sees the wrapped cache | `Scheduler.__init__`, `Scheduler.init_batch_result_processor`, `mem_cache.chunk_cache.ChunkCache` | W2 (W6) | U23 |
| S02 | managers/scheduler.py:1281-1282, 1283 (del), 1286-1287 | Adopt the runner's coordinator when the kvcache adapter uses QSA leases. Guard `set_decode_producer_stream` on None | hisparse: with no adapter the gate reduces to `not enable_hisparse`. Upstream the coordinator is non-None exactly when `enable_hisparse` (`ModelRunner.maybe_init_hisparse_coordinator`; `TpModelWorker.register_hisparse_coordinator` has no callers) | `sglang.srt.managers.scheduler.Scheduler.init_hisparse_coordinator` (pin 1267-1274) | after (P4: narrowed from replace (9)): the original leaves `hisparse_coordinator` None exactly when `enable_hisparse` is unset, and only then does the fork differ (adopt the runner's coordinator under QSA leases); with `enable_hisparse` set the fork's None guard is unreachable. Declared before S01, so adoption precedes the wrap as in the fork | `ModelRunner.maybe_init_hisparse_coordinator`, R01 | W2 | U1 |
| S03 | managers/scheduler.py:3563-3564 | A rebuilt HiSparse decode batch carries each request's `multimodal_inputs` | model_compat: changes upstream `--enable-hisparse` decode batches for multimodal requests | `sglang.srt.managers.scheduler.Scheduler._build_hisparse_decode_batch` (pin 3584) | after: nothing later in the body reads `batch.multimodal_inputs` (`SamplingBatchInfo.from_schedule_batch` reads only `batch.reqs` and `batch.device`), so `result.multimodal_inputs = [r.multimodal_inputs for r in reqs]` | `ScheduleBatch.init_new`, `SamplingBatchInfo.from_schedule_batch` | W2 | U9 |
| S04 | managers/scheduler.py:3647, 3660 | The staging-to-decode transition and the last-batch merge gate on `hisparse_coordinator is not None` instead of `enable_hisparse` | hisparse: equivalent upstream (coordinator present ⇔ enable_hisparse) | `sglang.srt.managers.scheduler.Scheduler.get_next_batch_to_run` (pin 3640-3782) | replace (143): two mid-function predicates. Flipping `self.enable_hisparse` in an around would leak into callees (the PP mixin, disagg). P4: keep; the transition must follow the pending-chunked-abort, HiCache-event and chunked-stash steps and rebinds the local `running_batch`, so no hook seam reproduces it | S03, `Scheduler.update_running_batch`, `.get_new_batch_prefill`, `.stash_chunked_request`, `.process_pending_chunked_abort`; Appx B | W2 | U1 |
| S05 | managers/scheduler.py:3770-3776 | Under QSA leases, allocate at most 1 request and no more than the free QSA slots, and 0 while a prefill owns a slot. Raise for beam width > 1 | hisparse: gated on `coordinator.uses_qsa_hisparse_leases` | `sglang.srt.managers.scheduler.Scheduler.get_num_allocatable_reqs` (pin 3784) | after: the original has no side effects, and with beam width None or 1 its beam cap is a no-op. So `min(result, 1, free_slots)`, or 0, applied after return equals the fork. Fallback: replace (26) | `BeamCoordinator.pending_member_rows` | W2 | U1 |
| S06 | managers/scheduler.py:3898-3899 | `PrefillAdder(prefill_max_requests=1)` under QSA leases | hisparse | `sglang.srt.managers.scheduler.Scheduler._get_new_batch_prefill_raw` (pin 3831-4161) | replace (332): the kwarg is computed inline in the `PrefillAdder(...)` call. Alternative: a before hook on `PrefillAdder.__init__` keyed on the allocator's kvcache adapter. It is equivalent for this call, but `dllm/mixin/scheduler.py:256` also builds a PrefillAdder. P4: keep (fork-faithful; the owner declined D4) | `schedule_policy.PrefillAdder.__init__`, `Scheduler.get_num_allocatable_reqs`; Appx B | W2 | U1 |
| S07 | managers/scheduler.py:4816 | Skip the idle pool-leak and invariant checks whenever a coordinator exists | hisparse | `sglang.srt.managers.scheduler.Scheduler.on_idle` (pin 4915-5003) | replace (89): the predicate gates a mid-function block. P4: keep; hooks on the gated callees would also change their other callers (`SchedulerInvariantChecker._check_all_pools` is called again at `invariant_checker.py:540`) | `Scheduler.is_fully_idle`, invariant checker methods; Appx B | W2 | U1 |
| S08 | managers/scheduler.py:4905 | Fully-idle also waits for coordinator staging when a coordinator exists | hisparse | `sglang.srt.managers.scheduler.Scheduler.is_fully_idle` (pin 5011) | after: `if not for_health_check and not self.enable_hisparse and self.hisparse_coordinator is not None: result = result and not has_ongoing_staging()`. The original is a conjunction of side-effect-free terms. v0.5.21's `ignore_waiting` parameter only drops the waiting-queue term; the hook accepts it. Fallback: replace (65) | plugin `QSAHiSparseCoordinator.has_ongoing_staging` (pure) | W2 | U1 |
| S09 | managers/scheduler.py:5410-5413 | `abort_request` also considers QSA staging requests (`ack_staging_queue`) | hisparse | `sglang.srt.managers.scheduler.Scheduler.collect_inflight_reqs` (pin 5438) | after: under leases, add staging requests to the returned set. The callers are `abort_request` (same set object and same update as the fork) and `record_weight_version_change` (already unions `ack_staging_queue` when a coordinator exists, so it gets the same set). Fallback: replace `abort_request` (138) | `Scheduler.abort_request`, `Scheduler.record_weight_version_change` | W2 | U1 |
| B01 | managers/scheduler_components/batch_result_processor.py:42 (del) | Drop the unused `get_memory` import | drop: no behavior. The pinned unit test patches this name | none | none | none | W2 | none |
| B02 | batch_result_processor.py:125 | `request_finished` gate: `get_memory().enable_hisparse` becomes `self.hisparse_coordinator is not None` | hisparse: equivalent upstream (S04 argument) | `sglang.srt.managers.scheduler_components.batch_result_processor.SchedulerBatchResultProcessor.process_batch_result_prebuilt` (pin 116) | replace (18): the predicate is mid-function. `get_memory` cannot be rebound for one module because `_propagate_patch` rebinds every module's reference. P4: keep; the next callee, `release_kv_cache`, is also reached from abort and retract paths that do not call `request_finished` | `mem_cache.common.release_kv_cache` | W2 | U1 |
| B03 | batch_result_processor.py:383 | `admit_request_into_staging` gate on coordinator presence | hisparse | `sglang.srt.managers.scheduler_components.batch_result_processor.SchedulerBatchResultProcessor.process_batch_result_prefill` (pin 257-494) | replace (238): same reason as B02 (P4: keep) | `mem_cache.common.maybe_cache_unfinished_req`, `release_kv_cache`; Appx B | W2 | U1 |
| B04 | batch_result_processor.py:1245 | `request_finished` gate on coordinator presence | hisparse | `sglang.srt.managers.scheduler_components.batch_result_processor.SchedulerBatchResultProcessor._handle_sampling_mask_abort` (pin 1249) | replace (16): same reason as B02 (P4: keep) | `release_kv_cache` | W2 | U1 |
| B05 | batch_result_processor.py:1329 | `request_finished` gate on coordinator presence | hisparse | `sglang.srt.managers.scheduler_components.batch_result_processor.SchedulerBatchResultProcessor._handle_finish_state_updated_req` (pin 1266) | replace (91): same reason as B02 (P4: keep) | `release_kv_cache`, `_mamba_prefix_cache_update`; Appx B | W2 | U1 |
| B06 | managers/scheduler_components/weight_updater.py:95-100 | Before any weight update, call `tree_cache.invalidate_model()` if it exists | hisparse: only `QSAHostPrefixCache` defines it | `sglang.srt.managers.scheduler_components.weight_updater.SchedulerWeightUpdaterManager._observe_weight_load` (pin 102) | before: a `@contextmanager`. All four call sites are `with self._observe_weight_load(...)`, so the generator prefix runs at `__enter__`, directly after the call. v0.5.21's `begin_weight_update`/`end_weight_update` session (P2P/RDMA writes) bypasses `_observe_weight_load`, as it would in the fork (open question to the orchestrator) | the four `update_weights_from_*` call sites (pin 129/185/225/250) | W2 (W6) | U23 |
| P01 | managers/schedule_policy.py:960-966 | `add_chunked_req` caps `_rem_tokens` at `tree_cache.prefill_checkpoint_limit(req)` | hisparse: `getattr` on the tree cache; upstream caches lack it | `sglang.srt.managers.schedule_policy.PrefillAdder.add_chunked_req` (pin 1114) | before (P4: narrowed from replace (58)): v0.5.21 caps the chunk at `self.chunked_req_limit` after the prefill delayer, which does not read the budget, so lowering `chunked_req_limit` to the checkpoint limit before the call equals the fork's cap before the delayer. The scheduler sets `chunked_req_limit` just before its only `add_chunked_req` call per adder (S06); nothing else reads it | `PrefillAdder.__init__`, `Scheduler._get_new_batch_prefill_raw` | W2 (W6) | U23 |
| P02 | managers/schedule_policy.py:1165-1195, 1200-1205 | New `add_one_req` wrapper: charge pending host-prefix pages to the memory budget offsets, cap `rem_chunk_tokens` at the checkpoint limit, then restore both. The original body is renamed `_add_one_req`, whose `ignore_eos` fast path is skipped while a host prefix is pending | hisparse: with upstream caches the charge is 0, the limit None, and the predicate unchanged | `sglang.srt.managers.schedule_policy.PrefillAdder.add_one_req` (pin 1347-1463) | replace (30 wrapper + 122 plugin-local `_add_one_req` = 152): the wrapper alone could be an around, but the `ignore_eos` predicate is mid-function (P4: keep) | `PrefillAdder.add_one_req_ignore_eos`, `._select_prefill_admission`, `._commit_prefill_admission`, `.budget_state`, `._kv_shard_reserve_scratch`; Appx B | W2 (W6) | U23 |
| M01 | mem_cache/allocation.py:346-361, 372 (del), 392-395, 433-436, 457-458, 474-477 | `alloc_for_extend` gains a rollback wrapper. `prepare_prefix_for_extend` runs after request-slot allocation, and `prefix_tensors` is read after it. `note_extend_allocation` runs after KV allocation. `out_cache_loc` is not freed on a write failure when the cache owns it. `restore_prefix_for_extend` runs at the end | hisparse: every hook is a `getattr` on the tree cache. Moving `prefix_tensors` is pure: `alloc_req_slots` never reassigns `req.prefix_indices` | `sglang.srt.mem_cache.allocation.alloc_for_extend` (pin 344-451) | replace (14 wrapper + 119 `_alloc_for_extend` = 133): insertions at four mid-function points (P4: keep) | `allocation.alloc_req_slots`, `.write_cache_indices`, `.alloc_paged_token_slots_extend`, `.alloc_token_slots`, `._alloc_extend_loc_with_kv_reuse`; Appx B | W2 (W6) | U23 |
| M02 | mem_cache/allocator/paged.py:336-340 | After `free_group_end`, notify the QSA adapter (`after_logical_flush`, or `after_release(pending_release)`) | hisparse: requires `kvcache.qsa_hisparse` | `sglang.srt.mem_cache.allocator.paged.PagedTokenToKVPoolAllocator.free_group_end` (pin 328) | after: the insertion is at the end of the body, which has no early return | `PagedTokenToKVPoolAllocator._release_page_ids` | W2 | U23 |
| M03 | mem_cache/common.py:269-277, 306-307 | `release_kv_cache` calls the `before_release(req, is_insert)` hook and takes a QSA lease release before the tree cache's release calls (`cache_finished_req` at the fork base; `claim_kv_row`, `insert_req`, `free_kv_row`, `unpin`, `on_release` at v0.5.21), then `after_release(lease)` after the request-pool free | hisparse: `getattr` on the tree cache and the kvcache | `sglang.srt.mem_cache.common.release_kv_cache` (pin 292-336) | replace (56): the first insertion follows an early-return branch and precedes `claim_kv_row`; `after_release` follows `mark_kv_released` and is skipped when a streaming session claims the row (P4: keep) | `common._release_overallocated_kv_indices`, `ReqToTokenPool.free`, `BasePrefixCache.claim_kv_row`, `.free_kv_row` | W2 (W6) | U23 |
| M04 | mem_cache/memory_pool.py:348-350 | `ReqToTokenPool.clear` no longer zeroes `req_generation`, so generations stay monotonic across flushes | model_compat: changes generation values seen by `managers/overlap_utils.py` and the DSpark planner after any flush | `sglang.srt.mem_cache.memory_pool.ReqToTokenPool.clear` (pin 365) | around (P4: narrowed from replace (7)): snapshot `req_generation`, run the pinned clear, restore it. The pinned clear zeroes it in place and nothing else in `clear` (or in `HybridReqToTokenPool.clear` after `super().clear()`) reads it | `ReqToTokenPool.alloc_rows` (increments generation) | W2 | U9 |

### W3: pools and graph

| ID | Fork file:lines | Behavior | Feature: justification | Hook target (pin) | Type: why | depends (key) | WS | U |
|---|---|---|---|---|---|---|---|---|
| K01 | mem_cache/kv_cache_configurator.py:1318 | Pass `max_running_requests=sizes.max_running_requests` to `_build_hybrid_linear_kv_pool` | hisparse: only K02's env-gated branch reads it | `sglang.srt.mem_cache.kv_cache_configurator.KVCacheConfigurator._build_token_to_kv_pool` (pin 1214) | around: `sizes` is a keyword argument. Bind `sizes.max_running_requests` in a contextvar for K02's adapter. `_build_hybrid_linear_kv_pool` has exactly one caller (pin 1315). Fallback: replace (123) | `KVCacheConfigurator._build_hybrid_linear_kv_pool` | W3 | U4 |
| K02 | kv_cache_configurator.py:6, 1860, 1936-1953 | With `SGLANG_QSA_HISPARSE_V3=p2-offload`, validate the bounded logical capacity and build a raw staging `MHATokenToKVPool` passed as `full_kv_pool` | hisparse: env-gated | `sglang.srt.mem_cache.kv_cache_configurator.KVCacheConfigurator._build_hybrid_linear_kv_pool` (pin 1854-1953) | replace (119), with an adapter that supplies `max_running_requests` from K01: `full_kv_pool` must enter `extra_args` mid-function, before `pool_class(...)` | `memory_pool.MHATokenToKVPool`, `qsa_kv_pool.QSATokenToKVPool.__init__`, `KVCacheConfigurator._build_mha_quant_method`; Appx B | W3 | U4 |
| K03 | mem_cache/qsa_kv_pool.py:71, 103 | `QSATokenToKVPool.__init__` accepts `full_kv_pool` and forwards it to `HybridLinearKVPool` | hisparse: the default None equals upstream | `sglang.srt.mem_cache.qsa_kv_pool.QSATokenToKVPool.__init__` (pin 49) + `sglang.srt.mem_cache.memory_pool.HybridLinearKVPool.__init__` (pin 3776) | around (QSA: pop `full_kv_pool` into a contextvar) + before (HybridLinear: inject it while the contextvar is set). The pinned `HybridLinearKVPool.__init__` already accepts `full_kv_pool` (memory_pool.py:3805), and QSA's `super().__init__` is the only nested pool construction. Fallback: replace (141) | `HybridLinearKVPool.__init__` | W3 | U4 |
| F01 | model_executor/forward_batch_info.py:479-482, 815-820, 840-841 | `ForwardBatch` gains `req_pool_indices_cpu` (read by `hisparse/runtime.py:1283`) and `kv_allocated_lens_cpu` (never read), filled in `init_new` | hisparse (deviation D3): instance attribute only while the HiSparse runtime is active; `kv_allocated_lens_cpu` dropped | `sglang.srt.model_executor.forward_batch_info.ForwardBatch.init_new` | after: set `ret.req_pool_indices_cpu = batch.req_pool_indices_cpu` when the KV pool carries a `qsa_hisparse` runtime; no class replace, so two-batch overlap is unaffected. Second after hook on `sglang.srt.model_executor.runner.eager_runner.EagerRunner.load_batch` copies the attribute onto its `dataclasses.replace` copy (found by W3) | `sglang.srt.managers.schedule_batch.ScheduleBatch` (class; the field is not a definition) | W3 | U5 |
| R01 | model_executor/model_runner.py:1037-1043 | `init_attention_backends` creates `QSAHiSparseCoordinator` when the kvcache adapter uses leases | hisparse | `sglang.srt.model_executor.model_runner.ModelRunner.init_attention_backends` (pin 1013) | after: the remaining tail only prepares the DCP-replicated `q_proj` and does not touch the coordinator | `ModelRunner._prepare_replicated_q_proj` | W3 | U1 |
| R02 | model_executor/model_runner.py:1811-1817, 1825-1828 | Raise when QSA full-graph decode cannot run the graph. Skip `num_real_reqs.fill_` when the QSA graph is enabled | hisparse: gated on `hisparse_coordinator.adapter`, which upstream `HiSparseCoordinator` lacks | `sglang.srt.model_executor.model_runner.ModelRunner._forward_raw` (pin 1782-1880) | replace (109): the first change needs the local `can_run_graph`; the second guards a mid-function statement | `DecodeCudaGraphRunner.can_run_graph`, `.execute`, `ModelRunner._prepare_eager_forward_batch`; Appx B | W3 | U5 |
| C01 | model_executor/pool_configurator.py:17, 208, 298-346 | `DefaultPoolConfigurator`: `_bias = 0`. With p2-offload, bias = the fixed raw + ring bytes, and the cell size becomes compressed-only | hisparse: env-gated; `_bias` is 0 otherwise | `sglang.srt.model_executor.pool_configurator.DefaultPoolConfigurator.__init__` (pin 205) | after: the block is appended at the end of the body, which has no early return. The class has no subclasses | `DefaultPoolConfigurator._compute_qsa_cell_size`, `layers.attention.qsa.config.parse_qsa_profile`, `sglang.srt.mem_cache.qsa_kv_pool.QSATokenToKVPool` (class; `index_state_dtype` is a class attribute, not a definition) | W3 | U4 |
| C02 | pool_configurator.py:588 | `calculate_pool_sizes` subtracts `self._bias` | hisparse | `sglang.srt.model_executor.pool_configurator.DefaultPoolConfigurator.calculate_pool_sizes` (pin 534) | before: pass `available_bytes - self._bias`. The first statement re-clamps it with `max(..., 0)` and is the argument's only use | none | W3 | U4 |
| C03 | pool_configurator.py:203 | Class docstring | drop: docstring only | none | none | none | W3 | none |
| G01 | model_executor/runner/decode_cuda_graph_runner.py:90, 1130-1139, 1205 (del), 1212-1218 | `capture_one_shape`: require `FullCudaGraphBackend` under the QSA graph. Compute `shape_key` before `forward_context` (moved out of `canary_ctx`). Enter `qsa.graph_capture`. Chain `qsa.after_graph_warmup` after the attention warmup hook | hisparse: everything is gated on the adapter, and the `shape_key` move is pure (§4 item 3) | `sglang.srt.model_executor.runner.decode_cuda_graph_runner.DecodeCudaGraphRunner.capture_one_shape` (pin 1105-1219) | replace (125): the context enters mid-function, after `capture_prepare`, and the warmup hook is a closure | `DecodeCudaGraphRunner._make_graph_key`, `._capture_graph_size`, `.capture_prepare`, `runner_backend.full_cuda_graph_backend.FullCudaGraphBackend`, `maybe_flashinfer_autotune_speculative_draft` | W3 | U5 |
| G02 | decode_cuda_graph_runner.py:1253-1257, 1324-1326, 1405 | `load_batch`: reject external preplanning under the QSA graph. Call `qsa.prepare_graph_replay` before the buffer fill. Skip `num_real_reqs.fill_` | hisparse | `sglang.srt.model_executor.runner.decode_cuda_graph_runner.DecodeCudaGraphRunner.load_batch` (pin 1228-1394) | replace (175): three mid-function insertions | `build_replay_fb_view`, the `buffer_registry.fill_from` contract; Appx B | W3 | U5 |
| G03 | decode_cuda_graph_runner.py:1431-1434, 1454-1456 | `execute`: enter `qsa.graph_replay_scope` inside `replay_session`. Call `qsa.finish_graph_replay` directly after `backend.replay` | hisparse | `sglang.srt.model_executor.runner.decode_cuda_graph_runner.DecodeCudaGraphRunner.execute` (pin 1401-1467) | replace (73): the scope must nest inside `timer_ctx` and `replay_session`, and `finish` must precede `_publish_read_done` | `DecodeCudaGraphRunner.load_batch`, `._publish_read_done`, `FullCudaGraphBackend.replay` | W3 | U5 |

### W4: QSA attention

| ID | Fork file:lines | Behavior | Feature: justification | Hook target (pin) | Type: why | depends (key) | WS | U |
|---|---|---|---|---|---|---|---|---|
| Q01 | layers/attention/qwen_sparse_attn_backend.py:95-103 | On SM86/SM89 without `flash_attn`, use SGLang's vendored varlen flash-attention instead of FA4 cute | model_compat: changes the kernel on consumer GPUs | `sglang.srt.layers.attention.qwen_sparse_attn_backend._resolve_flash_attn_varlen_func` (pin 65) | replace (39): inserted between two try-blocks. The copy must keep `@lru_cache(maxsize=1)`, and the wrapper must expose `cache_clear` (§6, G5) | `sglang.kernels.ops.attention.flash_attention.flash_attn_varlen_func`, `utils.is_sm121` | W4 | U6 |
| Q02 | qwen_sparse_attn_backend.py:239-243 | `__init__`: FA2 graph-wrapper state (`_fa2_graph_wrappers`, `_workspace`, `_shape`, `_unavailable`, `_active_logged`) | model_compat (rule 2): Q12's compat-owned verbatim body reads `_fa2_graph_wrappers` and `_fa2_graph_active_logged`; inert without the runtime | `sglang.srt.layers.attention.qwen_sparse_attn_backend.QwenSparseAttnBackend.__init__` (pin 177) | after: also sets `qsa_hisparse = None` only if absent (§2, C3). The fork's statements after the block are plain None assignments | `QwenSparseAttnBackend.__init__` | W4 | U6 |
| Q03 | qwen_sparse_attn_backend.py:11, 246-257 | `__init__`: with `SGLANG_QSA_HISPARSE_V3` set, build `QSAHiSparseRuntime`/`SingleRequest` and attach it as `token_to_kv_pool.qsa_hisparse` | hisparse: env-gated | `sglang.srt.layers.attention.qwen_sparse_attn_backend.QwenSparseAttnBackend.__init__` (pin 177) | after: same argument as Q02 | `QwenSparseAttnBackend.__init__` | W4 | U23/U6 |
| Q04 | qwen_sparse_attn_backend.py:34, 262-291 | New `_kv_descales` and `_store_kv`: for FP8 KV with non-unit scales, `set_kv_buffer` gets scales and cloned K/V; with a runtime, `write_locations` remaps | model_compat: FP8 KV path. The `qsa_hisparse` branch is inert (rule 2) | new members on `sglang.srt.layers.attention.qwen_sparse_attn_backend.QwenSparseAttnBackend` | attach: new methods (§6, G1) | `memory_pool.HybridLinearKVPool.set_kv_buffer`, `MHATokenToKVPool.set_kv_buffer` (divides in place), A01 | W4 | U6 |
| Q05 | qwen_sparse_attn_backend.py:704, 741-749, 790 | Eager plain decode (no spec) sets `decode_score_width = ceil(max_blocks/page)*page`, matching the CUDA graph | model_compat: changes eager decode score width (§4 item 6) | `sglang.srt.layers.attention.qwen_sparse_attn_backend.QwenSparseAttnBackend._metadata_from_forward_batch` (pin 540) | after: when the batch is not idle or empty, `should_reuse_mtp_sparse_indices` is false (pure), the mode is decode, and `spec_info` is None, `msgspec.structs.replace` `indexer_metadata` with the width. Requires T04. Fallback: replace (193) | `QwenSparseAttnBackend.should_reuse_mtp_sparse_indices`, `._empty_metadata`, T04 | W4 | U8 (gap: not named) |
| Q06 | qwen_sparse_attn_backend.py:808-809 | `init_forward_metadata` calls `qsa_hisparse.begin_batch(fb)` for non-idle batches | hisparse | `sglang.srt.layers.attention.qwen_sparse_attn_backend.QwenSparseAttnBackend.init_forward_metadata` (pin 723) | before: skip when the batch is idle (the fork's early return) | none | W4 | U6 |
| Q07 | qwen_sparse_attn_backend.py:995-1006 | `_capture_cuda_graph_metadata` plans an SM89 FlashInfer ragged FA2 graph wrapper for P2 graph decode | hisparse | `sglang.srt.layers.attention.qwen_sparse_attn_backend.QwenSparseAttnBackend._capture_cuda_graph_metadata` (pin 828) | after: all inputs are keyword arguments, and `metadata_rows = bs` for decode with no spec. The insertion is at the end | `._is_speculative_paged_mode`, Q09 | W4 | U5/U6 |
| Q08 | qwen_sparse_attn_backend.py:1370, 1429-1434, 1436, 1440 | `forward_extend`: store through `_store_kv`. Chunk prefill gathers raw slots via `qsa_hisparse.prefill_slots` | model_compat (both features; rule 2 copies the inert `qsa_hisparse` branch) | `sglang.srt.layers.attention.qwen_sparse_attn_backend.QwenSparseAttnBackend.forward_extend` (pin 1261-1362) | replace (102): mid-function call changes | Q04, A03, A05, `kernel.qsa_sparse_attention`, `._resolve_metadata`, `._forward_paged_attention` | W4 | U6 |
| Q09 | qwen_sparse_attn_backend.py:50-60, 1492-1593 | `_resolve_flashinfer_qsa_ragged` (module) + `_qsa_local_head_shape`, `_ensure_fa2_graph_wrapper`, `_can_run_fa2_graph` | model_compat (rule 2): `_can_run_fa2_graph` is called by Q12's verbatim body. All four are inert without the runtime; only Q07 (hisparse) calls the other two | new members on `sglang.srt.layers.attention.qwen_sparse_attn_backend.QwenSparseAttnBackend` | attach (3 methods); the module function lives in the plugin | flashinfer `BatchPrefillWithRaggedKVCacheWrapper` (external) | W4 | U6 |
| Q10 | qwen_sparse_attn_backend.py:1650, 1667-1668 | `_forward_trtllm_sparse` passes FP8 descales to compact extraction | model_compat: FP8 KV | `sglang.srt.layers.attention.qwen_sparse_attn_backend.QwenSparseAttnBackend._forward_trtllm_sparse` (pin 1412-1497) | replace (89): extra arguments in a mid-function call | A09, `._get_trtllm_sparse_tables`, `._get_fa2_scratch`, `sparse_attn.qwen_sparse_valid_counts_triton` | W4 | U6 |
| Q11 | qwen_sparse_attn_backend.py:1713-1719 | `forward_decode`: store through `_store_kv`, then `qsa_hisparse.after_store(layer[, graph=True])` | model_compat (both; rule 2) | `sglang.srt.layers.attention.qwen_sparse_attn_backend.QwenSparseAttnBackend.forward_decode` (pin 1499) | replace (23): `after_store` must follow the store and precede paged attention | Q04, Q12, `._resolve_metadata` | W4 | U6 |
| Q12 | qwen_sparse_attn_backend.py:42, 1741-1753, 1766-1845 | `_forward_paged_attention`: `qsa_hisparse.selected()` buffers. trtllm is disabled under HiSparse. NVTX ranges. FP8 scratch dtype and descales. FA2 graph wrapper. `capture_decode` | model_compat (both; rule 2) | `sglang.srt.layers.attention.qwen_sparse_attn_backend.QwenSparseAttnBackend._forward_paged_attention` (pin 1519-1611) | replace (124) | Q01, Q09, Q10, A09, `_resolve_trtllm_sparse_decode`, `qwen_sparse_fa2_cu_seqlens_triton`, `._get_fa2_scratch`, `utils.nvtx_utils.profile_range` (`operations_nvtx_range` is a module-level `partial`, not fingerprintable) | W4 | U6 |
| T01 | layers/attention/qsa/kernel.py:12-61 | Exact stable top-k (`_qsa_stable_topk`) with tile sizing | model_compat (deterministic inference) | none | moved/owned: `kernels/qsa_topk.py` | none | W4 | U8 |
| T02 | qsa/kernel.py:78, 80-85, 89-90, 123-124; qsa/metadata.py:15, 123-132; qsa/qsa_indexer.py:25, 444-448, 475 | `qsa_fast_topk(deterministic=)` routes to the stable top-k. `topk_transform` and `select_prefill_tokens` pass the exec-config flag. The kernel.py:123-124 `torch.topk` rewrite is behavior-identical | model_compat: active whenever `enable_deterministic_inference` | `sglang.srt.layers.attention.qsa.kernel.qsa_fast_topk` (pin 23) | around: accept `deterministic`. When it is omitted, compute it with the fork's expression (`get_context().is_config_namespace_published("exec") and get_exec().deterministic.enable_deterministic_inference`). At the pin the only callers are these two and T03, and in the fork each passes exactly that expression. False delegates to the original. Fallback: replace `topk_transform` (24) and `select_prefill_tokens` (61) | `QSAIndexerMetadata.topk_transform`, `QSAIndexer.select_prefill_tokens` (rely on the default) | W4 | U8 |
| T03 | qsa/qsa_indexer.py:505-509, 523-527 | `select_decode_tokens`: under deterministic inference, skip the JIT `fast_topk` and pass the flag | model_compat | `sglang.srt.layers.attention.qsa.qsa_indexer.QSAIndexer.select_decode_tokens` (pin 481) | replace (48): mid-function branch predicate | T02, `qsa.mqa.qsa_mqa_decode`, `qsa.kernel.expand_qsa_block_indices`, `sglang.kernels.ops.elementwise.fast_topk.fast_topk` | W4 | U8 |
| T04 | qsa/metadata.py:86-88, 219-220, 232-233 | `QSAIndexerMetadata.decode_score_width` field; `get_decode_mqa_inputs` uses it as the score stride | model_compat (with Q05) | `sglang.srt.layers.attention.qsa.metadata.QSAIndexerMetadata` (pin 46) | replace (class): a frozen msgspec subclass adding the field (default None) and overriding `get_decode_mqa_inputs` verbatim (53). A Struct field cannot be added otherwise | `metadata.compressed_decode_view` | W4 | U8 |
| A01 | layers/attention/qsa/sparse_attn.py:9-40 | `is_fp8_kv_dtype`, `_validate_sparse_gqa_dtypes`, `_unit_scale` | model_compat | none | moved/owned: `kernels/qsa_sparse_attn.py` | none | W4 | U6 |
| A02 | sparse_attn.py:70-71, 93, 135-137, 143-148, 152-160 | `_sparse_gqa_prefill` Triton kernel: FP8 K/V cast, k/v scales, separate accumulation | model_compat | none (JITFunction, not hookable) | moved/owned | none | W4 | U6 |
| A03 | sparse_attn.py:174-185, 200-201, 223 | `sparse_gqa_fwd_interface_triton`: dtype validation, scales, `KV_IS_FP8`, launch of the plugin kernel | model_compat | `sglang.srt.layers.attention.qsa.sparse_attn.sparse_gqa_fwd_interface_triton` (pin 125) | replace (54): the kernel signature changed, so the wrapper must launch the plugin copy | `sparse_attn._get_best_config` | W4 | U6 |
| A04 | sparse_attn.py:241-242, 264, 307-308, 314-319, 323-331 | `_sparse_gqa_chunk_prefill` kernel: FP8-only cast and scales (the unconditional cast is removed) | model_compat | none | moved/owned | none | W4 | U6 |
| A05 | sparse_attn.py:345-356, 358, 376-377, 399 | `sparse_gqa_fwd_interface_triton_ck`: validation, scales, plugin kernel | model_compat | `sglang.srt.layers.attention.qsa.sparse_attn.sparse_gqa_fwd_interface_triton_ck` (pin 273) | replace (59): same reason as A03 | `sparse_attn._get_best_config` | W4 | U6 |
| A06 | sparse_attn.py:440-442 | Comment | drop: comment only | none | none | none | W4 | none |
| A07 | sparse_attn.py:474-475, 484, 519-525 | `_compact_kv` kernel: FP8 dequant with scales; implicit store cast | model_compat | none | moved/owned | none | W4 | U6 |
| A08 | sparse_attn.py:529-531 | Docstring | drop: docstring only | none | none | none | W4 | none |
| A09 | sparse_attn.py:555-556, 575-577, 592-593, 602 | `qwen_sparse_kv_extraction_compact_triton`: scale kwargs, dtype-pair check, `DEQUANTIZE_FP8`, plugin kernel | model_compat | `sglang.srt.layers.attention.qsa.sparse_attn.qwen_sparse_kv_extraction_compact_triton` (pin 458) | replace (63): same reason as A03 | none beyond A07 | W4 | U6 |
| A10 | sparse_attn.py:609 | `__all__` adds `is_fp8_kv_dtype` | drop: the plugin exports it from its own module | none | none | none | W4 | none |

### W5: model compatibility

| ID | Fork file:lines | Behavior | Feature: justification | Hook target (pin) | Type: why | depends (key) | WS | U |
|---|---|---|---|---|---|---|---|---|
| J01 | kernels/ops/moe/moe_wna16_marlin.py:20-23, 25, 73, 75-84, 153-155 | Op wrapper: `use_deterministic_reduce` kwarg and a 4th template arg. Requires blockM8 and no atomics | model_compat | none | moved/owned: `kernels/marlin_moe.py`, with its own JIT module (§3) | `sglang.kernels.jit.utils.load_jit` / `make_cpp_args` / `cache_once` | W5 | U8 |
| J02 | kernels/jit/csrc/gemm/marlin_moe/marlin_template.h (4 hunks), moe_wna16_marlin.cuh (10 hunks), stripe_schedule.h (new) | `kDeterministicReduce` template parameter: whole-K stripe iteration and a fixed launch config | model_compat | none | moved/owned: `kernels/csrc/marlin_moe/` (§3) | in-tree `csrc/gemm/marlin/*.cuh` headers | W5 | U8 |
| J03 | layers/moe/fused_moe_triton/fused_marlin_moe.py:6 (del), 8-11, 18 (del), 197-206, 255-258, 317-323, 352, 416 | Under deterministic inference: stable token alignment, `block_size_m = 8`, no atomic add, `use_deterministic_reduce` | model_compat: any Marlin MoE model with deterministic inference | `sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe.fused_marlin_moe` (pin 134) | replace (315): mid-function. The pinned attribute is the eager custom-op packet `torch.ops.sglang.fused_marlin_moe`, so the copy must be registered with `register_custom_op(op_name=<distinct>, out_shape="hidden_states")` (§6, G4) | `fused_moe_triton.moe_align_block_size`, `fused_marlin_moe.get_scalar_type`, `situ_and_mul`, `swiglu_limit_func`, J01, J04 (`register_custom_op` cannot be fingerprinted: 5 overloads) | W5 | U8 |
| J04 | layers/moe/fused_moe_triton/stable_align.py (new) | `moe_align_block_size_stable` | model_compat | none | moved: `kernels/stable_align.py` (done) | none | W5 | U8 |
| Z01 | hardware_backend/gpu/quantization/gptq_kernels.py:3-4, 279-280, 289-290, 294-299, 316-317 | Call `gc.collect()` and `torch.cuda.empty_cache()` after each repack. The w13 scale permute uses `size_k = scales.shape[1] * group` instead of `intermediate_size_per_partition` | model_compat: every GPTQ Marlin MoE load | `sglang.srt.hardware_backend.gpu.quantization.gptq_kernels.GPTQMarlinMoEKernel.process_weights_after_loading` (pin 225) | replace (91): mid-function | `gptq_kernels.gptq_marlin_moe_repack`, `layers.quantization.marlin_utils.marlin_moe_permute_scales`, `layers.quantization.utils.replace_parameter` | W5 | U7 |
| Z02 | layers/quantization/gptq/schemes/gptq_moe.py:200, 239, 247 | w2 group scales are sized without `moe_tp_size`; w13/w2 scales use `params_dtype` instead of fp16 | model_compat | `sglang.srt.layers.quantization.gptq.schemes.gptq_moe.GPTQMarlinMoEScheme.create_weights` (pin 180) | replace (142): mid-function values | `utils.common.set_weight_attrs`, `FusedMoeWeightScaleSupported` | W5 | U7 |
| Z03 | layers/quantization/auto_round.py:524-532 | When Marlin rejects g128 but accepts g64, use Marlin g64 and set `layer._marlin_g64_expand_scales` | model_compat | `sglang.srt.layers.quantization.auto_round.AutoRoundConfig.apply_gptq_quant_layer` (pin 437) | replace (142): mutates mid-function locals | `check_moe_marlin_supports_layer`, `AutoRoundConfig.get_layer_config`, `.get_gptq_config_kwargs` | W5 | U7 |
| Z04 | layers/moe/fused_moe_triton/layer.py:1024-1028 | `weight_loader` applies `repeat_interleave(2, dim=0)` to `*_scales`/`*_qzeros` on flagged layers | model_compat (part of Z03) | `sglang.srt.layers.moe.fused_moe_triton.layer.FusedMoE.weight_loader` (pin 999) | before: the only preceding branch is the static-mxfp4 path, unreachable for AutoRound-flagged layers. Must handle positional and keyword calls. The bound `weight_loader` is captured at layer construction, after activation | `FusedMoE.weight_loader` | W5 | U7 |
| Z05 | layers/moe/fused_moe_triton/layer.py:497 | Deferred-finalize log message changes from debug to info | drop (deviation D1, owner-accepted 2026-10-07): log-only; reproducing it needs a 235-line replace of `FusedMoE.__init__` | none | none | none | W5 | none |
| H01 | layers/hc_mix_triton.py:3-27 | Module docstring | drop | none | none | none | W5 | none |
| H02 | hc_mix_triton.py:67 | Kernel docstring | drop: changes only the Triton source hash | none | none | none | W5 | none |
| H03 | hc_mix_triton.py:161-297 | `_hc_mix_stable_persistent_kernel` (fixed-order split-K) | model_compat | none | moved/owned: `kernels/hc_mix.py` (calls the pinned `_grid_barrier`) | `hc_mix_triton._grid_barrier` | W5 | U8 |
| H04 | hc_mix_triton.py:317 (del) | Blank line | drop | none | none | none | W5 | none |
| H05 | hc_mix_triton.py:319-323, 325-327 | `fused_hc_mix_supported(stable=)`: the stable path bypasses the deterministic early return | model_compat | `sglang.srt.layers.hc_mix_triton.fused_hc_mix_supported` (pin 161) | replace (23): the early return is mid-function; an around would re-implement the predicate | `hc_mix_triton._deterministic_inference` | W5 | U8 |
| H06 | hc_mix_triton.py:349-351, 359-365, 369, 378-411 | `fused_hc_mix(stable=, stable_splits=)` launches the stable kernel | model_compat | `sglang.srt.layers.hc_mix_triton.fused_hc_mix` (pin 182) | around: `stable=False` goes to the original, which is identical to the fork's non-stable branch; `stable=True` runs the plugin copy of the fork body (70) | `hc_mix_triton._get_counters`, H03 | W5 | U8 |
| H07 | layers/hyperconnection.py:93 | `HyperConnectionBase.mix(stable=)` signature | drop: unreachable. `GatedResidual`, the only subclass, overrides `mix`, and no caller passes `stable` to the base | none | none | none | W5 | none |
| H08 | hyperconnection.py:222, 236-258 | `GatedResidual.mix(stable=)`: stable fused kernel, falling back to the torch.compile chain | model_compat | `sglang.srt.layers.hyperconnection.GatedResidual.mix` (pin 222) | around: `stable` defaults to E01's `_stable_hc()` when omitted. False goes to the original (identical to the fork's non-stable branch); True runs the plugin copy (79). Fallback: replace (79) plus E02's three replaces | H05, H06, `GroupedGemmaRMSNorm.forward` | W5 | U8 |
| E01 | models/qwen4_exp.py:71, 103-109 | `_stable_hc()` (reads `get_exec`) | model_compat | none | moved/owned: plugin helper used by H08 | none | W5 | U8 |
| E02 | qwen4_exp.py:1407-1409, 1421-1423, 1772-1774 | The three `mix()` calls pass `stable=_stable_hc()` | model_compat | none (covered by H08's default; these are the only `mix()` call sites at the pin) | none. Fallback: replace `Qwen4ExpLayerExtensionMixin._prepare_qwen4_exp_attn` (34), `._prepare_qwen4_exp_mlp` (13), `Qwen4ExpModel.forward` (56) | H08 | W5 | U8 |
| E03 | qwen4_exp.py:517-531, 535-552 | `Qwen4ExpNGramEmbedding.__init__`: the embedding is built on `meta` whenever `ple_offload_embedding`. int8/int8_row dtype. `ple_row_scale_mode` validation | model_compat | `sglang.srt.models.qwen4_exp.Qwen4ExpNGramEmbedding.__init__` (pin 446) | replace (100): mid-function. Rewrite the zero-argument `super()` (§6, G2) | `VocabParallelEmbedding.__init__`, `qwen4_exp._ple_table_is_fp8`, `._use_attn_tp_ngram`, `Qwen4ExpNGramEmbedding._build_head_vocab_and_offsets` | W5 | U7 |
| E04 | qwen4_exp.py:769, 774-775, 786-787, 795-798 | `_gather_ple_embedding_from_pinned_kernel`: int8 pointer and per-row scale | model_compat | none | moved/owned: `kernels/ple_gather.py` | none | W5 | U7 |
| E05 | qwen4_exp.py:811-813 | Class docstring | drop | none | none | none | W5 | none |
| E06 | qwen4_exp.py:847-851, 853-854, 891-904 | `Qwen4ExpPinnedHostEmbedding.__init__`: accept int8; pinned NaN-filled `row_scale` buffer | model_compat | `sglang.srt.models.qwen4_exp.Qwen4ExpPinnedHostEmbedding.__init__` (pin 794) | replace (72): mid-function | `qwen4_exp_ple_table.allocate_ple_host_table`, `make_ple_file_prefetcher`, `make_ple_file_rss_trimmer` | W5 | U7 |
| E07 | qwen4_exp.py:950, 955-956 | `gather` passes the `row_scale` pointer and the `is_int8`/`has_row_scale` constexprs to E04 | model_compat | `sglang.srt.models.qwen4_exp.Qwen4ExpPinnedHostEmbedding.gather` (pin 861) | replace (40): kernel arguments changed | `Qwen4ExpPinnedHostEmbedding.allocate_output`, E04 | W5 | U7 |
| E08 | qwen4_exp.py:1979-1981, 1990-2015, 2057-2067, 2258-2271 | `load_weights`: `row_scale` shards, int8 storage consistency errors, post-load NaN coverage check | model_compat | `sglang.srt.models.qwen4_exp.Qwen4ExpForConditionalGeneration.load_weights` (pin 1823) | replace (392): changes are inside the nested closure `load_qwen4_exp_ple_shard`, which cannot be hooked | `Qwen4ExpForConditionalGeneration._load_qwen4_exp_ple_buffer`, `layers.utils.common.get_layer_id`, E06 | W5 | U7 |
| U01 | utils/common.py:4001 | `freeze_gc` logs at info instead of debug (fork commit 39a3373d77) | drop (deviation D2, owner-accepted 2026-10-07): log-only | none | none | none | W5 | none |

### W1: new files (moved)

| ID | Fork file | Feature | Plugin location | WS |
|---|---|---|---|---|
| N01 | mem_cache/qsa_hisparse/{__init__,config,coordinator,layout,runtime,single_request,slots}.py | moved | `hisparse/` (done; imports rewritten) | W1 |
| N02 | mem_cache/qsa_hisparse/{prefix,prefix_cache}.py | moved | `hisparse/` (done); behavior is owned by W6 | W1/W6 |
| N03 | layers/attention/qsa/hisparse_graph.py | moved | `hisparse/graph.py` (done) | W1 |
| N04 | mem_cache/qsa_hisparse_{p2,slots,v3}.py | drop: aliases for legacy import paths; nothing in fork `python/`, `test/` or `scripts/` imports them | none | W1 |

W6 owns no fork hunks outside N02. The call sites its prefix cache depends on
are S01, B06, P01, P02, M01 and M03 (all W2); W6 must verify them.

## 2. Cross-feature conflicts (rule 2)

No target receives a REPLACE from both features, and hisparse adds no hook to
any model_compat REPLACE target.

| # | Target(s) | Model compat | Hisparse | Resolution |
|---|---|---|---|---|
| C1 | `QwenSparseAttnBackend.forward_extend` / `forward_decode` / `_forward_paged_attention` (Q08, Q11, Q12) | REPLACE with the verbatim fork body, including the `qsa_hisparse` branches (`prefill_slots`, `after_store`, `selected`, `capture_decode`, FA2 graph) | none | Rule 2. Hisparse behavior arrives only through `self.qsa_hisparse`, set by Q03 |
| C2 | New members `_store_kv`, `_kv_descales`, `_can_run_fa2_graph`, `_qsa_local_head_shape`, `_ensure_fa2_graph_wrapper` (Q04, Q09) | attach, verbatim and inert without the runtime | Q07's after hook calls `_ensure_fa2_graph_wrapper` and `_qsa_local_head_shape` | Model compat owns the definitions because its verbatim bodies call them; hisparse only calls them. Needs the G1 mechanism |
| C3 | `QwenSparseAttnBackend.__init__` (Q02, Q03) | after: FA2 state; `qsa_hisparse = None` | after: build and attach the runtime | Two after hooks, no replace. HookRegistry runs them in registration order, and `patching.collect` imports model_compat first, so compat runs first. To make the order irrelevant, the compat hook must set `qsa_hisparse` only when absent |
| C4 | `ForwardBatch.init_new` (F01) | none (D3) | after: set `req_pool_indices_cpu` when the runtime is present | Resolved by deviation D3: hisparse-only instance attribute, no class replace |
| C5 | `Scheduler._build_hisparse_decode_batch` (S03) | after | S04's replace calls it through `self`, which reaches the hooked method | Composition; nothing to resolve |
| C6 | `DecodeCudaGraphRunner.capture_one_shape` (G01) | none | replace | The `shape_key` relocation is pure (§4 item 3), so model_compat has no claim. If review reclassifies it, model_compat would own this replace and hisparse would have to move its capture context into an around, which cannot reproduce the context placement |
| C7 | `Scheduler.init_hisparse_coordinator` (S01, S02) | none | replace + after | Same feature. HookRegistry applies the replace first, then wraps it with the after hook |
| C8 | `QSAIndexerMetadata` (T04) and `qsa_fast_topk` (T02) | class replace; around | none | none |

## 3. Non-Python changes: a plugin-owned Marlin MoE JIT copy

Files: `kernels/jit/csrc/gemm/marlin_moe/{marlin_template.h, moe_wna16_marlin.cuh}`
(modified) and `stripe_schedule.h` (new), plus `kernels/ops/moe/moe_wna16_marlin.py`.
With `kDeterministicReduce = false` (the template default, gated by
`if constexpr`) the fork's kernel is source-equivalent to upstream. The plugin
therefore needs its copy only for J03, and the in-tree op is left unpatched.

**Confirmed feasible** at the pin:

- `load_jit(*args, cuda_files=..., cuda_wrappers=..., extra_include_paths=..., ...)`
  is at `kernels/jit/utils/compile/loader.py:48`. `resolve_sources`
  (`spec.py:125-134`) passes absolute paths through unchanged; relative paths
  resolve against the in-tree `csrc/`. With `header_only=True` (the default)
  the generated `cuda.cu` does `#include "<absolute path>"` and exports each
  wrapper with `TVM_FFI_DLL_EXPORT_TYPED_FUNC` inside `namespace sglang`
  (`spec.py:111-122`).
- Include flags are plain `-I` in this order: toolchain base, then
  `DEFAULT_INCLUDE = [KERNEL_PATH/include]` (`paths.py:30`), then
  `extra_include_paths` (`loader.py:89`, `ninja.py:83-84`). Quoted includes
  search the including file's directory first, then the `-I` list.
- The plugin copy lives at `kernels/csrc/marlin_moe/` and holds all four files:
  the fork's `moe_wna16_marlin.cuh`, `marlin_template.h` and
  `stripe_schedule.h`, plus `kernel.h`, which is unchanged but copied so that
  no `marlin_moe` sibling can resolve to the in-tree copy. It is shipped by the
  existing `kernels/csrc/**/*` package-data rule.
- The fork headers include `"../marlin/{dequant.h,marlin.cuh,marlin_dtypes.cuh}"`,
  and `kernel.h` includes `"../marlin/..."`. Pass
  `extra_include_paths=[str(KERNEL_PATH / "csrc" / "gemm" / "marlin")]` so that
  `../marlin/x` resolves to the installed in-tree `csrc/gemm/marlin/x`. Those
  headers include only siblings and `<sgl_kernel/...>` (from `DEFAULT_INCLUDE`).
  Do not use `.../gemm/marlin_moe` as the include path: a header the plugin
  forgot to copy would then silently resolve to the unmodified in-tree file.
  With `.../gemm/marlin` a missing `kernel.h` resolves to the dense-Marlin
  `kernel.h` and fails to compile loudly instead.
- Cache identity. `module_name = "sgl_kernel_jit_" + "_".join(args)`
  (`spec.py:20,55-57`). The cache directory is
  `<SGLANG_JIT_CACHE_DIR>/<target>/<module_name>/build-<build_key>/deps-<deps_key>/`
  (`cache.py:306`). `build_key` hashes `module_args`, the direct source
  contents, the generated ninja text and the environment (`cache.py:259-289`).
  `deps_key` hashes the transitive includes, so an edit to the in-tree
  `../marlin` headers rebuilds the plugin module. Pruning is scoped per
  build-key directory (`cache.py:441-466`).
- **Args that avoid collisions**: use a distinct marker as the first
  positional arg and append the fourth template arg:
  `load_jit("qsa_hisparse_moe_wna16_marlin", *make_cpp_args(dtype, is_ep, has_bias, use_deterministic_reduce), cuda_files=[str(PLUGIN_CSRC / "marlin_moe/moe_wna16_marlin.cuh")], cuda_wrappers=[("qsa_moe_wna16_marlin_gemm", f"moe_wna16_marlin_gemm<{args}>")], extra_include_paths=[...])`.
  The module name becomes
  `sgl_kernel_jit_qsa_hisparse_moe_wna16_marlin_<dtype>_<ep>_<bias>_<det>`,
  which differs from the in-tree `sgl_kernel_jit_moe_wna16_marlin_<dtype>_<ep>_<bias>`.
  The `.so` basename and cache directory are therefore distinct. A distinct
  export name (`qsa_moe_wna16_marlin_gemm`) also avoids two loaded libraries
  exporting the same `extern "C"` `__tvm_ffi_*` symbol. `build_key` would
  differ even with the same marker, but the shared `<module_name>` directory
  and `.so` basename are avoidable.
- Residual risk: both libraries define C++ entities in
  `sglang::device::marlin_moe`. Every template the fork changed gained the
  extra `bool` parameter, so its mangled names differ. The remaining
  non-template inline code is identical, so symbol interposition would be
  harmless. W5 should still confirm how `tvm_ffi.load_module` calls `dlopen`
  (`RTLD_LOCAL` avoids interposition entirely). `SGLANG_CRASH_ON_JIT_COMPILE`
  (`loader.py:131`) also applies to the plugin module: GPU windows that set
  it need the cache seeded.

## 4. Hidden shared-path changes (behavior changes with HiSparse unset)

Status: **confirmed** (as described), **corrected**, or **new** (not in the
known list).

1. **Confirmed (M04).** `ReqToTokenPool.clear` no longer zeroes
   `req_generation`. Readers: `managers/overlap_utils.py:212,232`
   (stale-result generation ring) and `speculative/dspark_components/dspark_planner.py:349,398`.
   The disaggregated `DecodeReqToTokenPool.clear` (`disaggregation/decode.py:230`)
   still zeroes its own copy, which the fork leaves inconsistent.
2. **Confirmed, and stronger (F01).** The new `ForwardBatch` fields are not
   just passive. `req_pool_indices_cpu` is non-None for scheduled batches, and
   `kv_allocated_lens_cpu` is non-None whenever `seq_lens_cpu` is None.
   `TboForwardBatchPreparer.filter_batch` raises for any non-None field it does
   not handle (`batch_overlap/two_batch_overlap.py:845-851`), so with the fork
   two-batch overlap fails for every model. `kv_allocated_lens_cpu` has no
   reader anywhere in the fork, and it adds a per-batch Python list build.
3. **Corrected (G01).** Moving `shape_key` out of `canary_ctx` is
   behavior-identical. `_make_graph_key` builds a `ShapeKey` from its arguments,
   and `_capture_graph_size` reads `self.ragged_verify_mode`, set at
   construction. Every subclass that overrides `_make_graph_key` (EAGLE draft,
   draft-extend, multi-layer EAGLE, frozen-KV MTP) also overrides
   `capture_one_shape`. XPU and NPU inherit both. G01 is therefore hisparse.
4. **Confirmed, with detail (Q04, Q10, Q12, A01-A09).** QSA FP8 descale:
   - `_store_kv` passes k/v scales and cloned K/V to `set_kv_buffer` for FP8
     caches with non-unit `k_scale_float`/`v_scale_float`.
   - Decode and trtllm extraction dequantize with those scales.
   - The FP8 path of the prefill Triton kernels changes accumulation order
     (`acc*alpha + dot*v_scale`) even with unit scales: callers pass no scales.
   - New `ValueError`s: K/V dtype mismatch, queries not BF16/FP16, non-FP8 K/V
     dtype different from Q.
   - The decode scratch dtype becomes `k_buffer.dtype` for non-FP8 KV
     (previously `q.dtype`). This is identical when the dtypes match.
5. **Confirmed (Q01).** The SM86/SM89 fallback applies only when FA2
   `flash_attn` is not importable. It selects SGLang's vendored
   `flash_attn_varlen_func` instead of FA4 cute. Separately, the SM89 FlashInfer
   ragged FA2 path (Q07, Q09) is hisparse-only. PLAN's "SM89 fallback" covers
   both SM86 and SM89 for the first.
6. **Confirmed (Q05, T04).** `decode_score_width` changes only eager plain
   decode, without spec decoding or MTP reuse: the MQA score stride becomes
   `ceil(ceil(context_len/ratio)/page)*page`, matching the graph, instead of the
   batch's page-table width. Graph decode is unchanged.
7. **Confirmed, plus new items (Z01, Z02).** In `create_weights`, the w2 group
   scales are sized without `moe_tp_size` and the scales are `params_dtype`
   instead of fp16. **New:** in `process_weights_after_loading`, the w13
   `marlin_moe_permute_scales(size_k=...)` now derives K from the scale tensor
   instead of `intermediate_size_per_partition`, and each repack is followed by
   `gc.collect()` and `torch.cuda.empty_cache()`. This changes load-time peak
   memory for every GPTQ Marlin MoE model.
8. **Confirmed (Z03, Z04).** For AutoRound g128 MoE layers that Marlin rejects
   (TP2, K=320), the fork switches to Marlin g64 and expands each scale and
   qzero row twice at load. Upstream would use the non-Marlin MoE path.
9. **Confirmed, plus new items (E03, E06, E07, E08).** INT8-row PLE. **New:**
   `Qwen4ExpNGramEmbedding` now builds its table on the `meta` device for
   *every* `ple_offload_embedding` config, including bf16 and fp8 (a
   memory-peak change). `load_weights` adds hard errors for int8 storage
   mismatches and NaN row-scale coverage.
10. **Confirmed (U01), plus a new item (Z05).** Log level: `freeze_gc` changes
    from debug to info. **New:** `FusedMoE.__init__`'s deferred-finalize log
    also changes from debug to info.
11. **Confirmed (S03).** `multimodal_inputs` is set on the HiSparse decode
    batch. This affects only upstream `--enable-hisparse` with multimodal
    requests.
12. **New (J01-J04).** Deterministic Marlin MoE: with
    `enable_deterministic_inference`, every Marlin MoE model uses stable token
    alignment, `block_size_m = 8`, no atomic add, and the whole-K reduce.
13. **New (T01-T03).** Deterministic QSA top-k: with deterministic inference,
    selection is the stable top-k and decode skips the JIT `fast_topk`.
14. **New (H03-H08, E01-E02).** Stable HC: with deterministic inference,
    Qwen4Exp's `GatedResidual.mix` uses the stable fused kernel when it is
    supported. Previously the fused path was disabled and the torch.compile
    chain ran.
15. **New, negligible (Q12).** NVTX ranges in `_forward_paged_attention` are
    emitted only when operations NVTX profiling is enabled.

Verified inert with HiSparse unset (classified hisparse): the scheduler and
batch-result gates. Upstream `hisparse_coordinator` is non-None exactly when
`enable_hisparse`, and `register_hisparse_coordinator` is never called. Also
inert: S05, S08, S09, M01 (the `prefix_tensors` move is pure), M02, M03, P01,
P02, B06, K01-K03, C01, C02, R01, R02, G01-G03, Q03, Q06 and Q07.

## 5. Counts and risk

- Rows: **89**. All 196 hunks in 31 modified files are mapped (Appendix A),
  plus 15 new files (rows J02, J04, N01-N04).
- After G0 reconciliation (D1–D3): 37 replace (36 function or method,
  1 class: T04), 13 after (F01 added), 5 before (B06, C02, Q06, Z04, and K03's
  half), 5 around (K01, T02, H06, H08, and K03's half), 2 attach rows
  (5 methods), 14 moved/owned, 14 drop (Z05, U01 added), 1 none. The
  executable form is `src/sglang_qsa_hisparse/manifest.json`, generated by
  `tools/manifest.py` from this file.
- REPLACE per workstream (copied lines):

| WS | Count | Rows | Copied lines |
|---|---|---|---|
| W2 | 13 | S02, S04, S06, S07, B02-B05, P01, P02, M01, M03, M04 | ~1314 |
| W3 | 5 | K02, R02, G01, G02, G03 | ~601 |
| W4 | 10 | Q01, Q08, Q10, Q11, Q12, T03, T04 (class), A03, A05, A09 | ~657 |
| W5 | 9 | J03, Z01, Z02, Z03, H05, E03, E06, E07, E08 | ~1317 |
| W1, W6, W7, W8 | 0 | | |
| **Total** | **37** (20 model_compat, 17 hisparse) | | **~3890** |

  If review rejects the equivalence-based narrow hooks, the fallbacks add up
  to 10 more replaces: S05, S08, S09 (`abort_request`), K01, K03, Q05, T02
  (two), and E02 (three). H08 becomes a replace of the same size.
- **10 riskiest REPLACE targets** (largest copy × upstream churn):

| # | Target | Copied lines | Row | Feature / U | Why risky |
|---|---|---|---|---|---|
| 1 | `Qwen4ExpForConditionalGeneration.load_weights` | 392 | E08 | compat / U7 | Changes sit in a nested closure; active model loader |
| 2 | `Scheduler._get_new_batch_prefill_raw` | 316 | S06 | hisparse / U1 | A one-kwarg change in the highest-churn scheduler function |
| 3 | `fused_marlin_moe` | 315 | J03 | compat / U8 | Eager custom-op re-registration; MoE hot path |
| 4 | `SchedulerBatchResultProcessor.process_batch_result_prefill` | 238 | B03 | hisparse / U1 | A one-line gate; high churn |
| 5 | `DecodeCudaGraphRunner.load_batch` | 175 | G02 | hisparse / U5 | Graph replay contract; high churn |
| 6 | `Scheduler.get_next_batch_to_run` | 148 | S04 | hisparse / U1 | Core loop |
| 7 | `PrefillAdder.add_one_req` + `_add_one_req` | 145 | P02 | hisparse / U23 | Fork split the function; admission budget logic |
| 8 | `AutoRoundConfig.apply_gptq_quant_layer` | 142 | Z03 | compat / U7 | Quant config dispatch |
| 9 | `GPTQMarlinMoEScheme.create_weights` | 142 | Z02 | compat / U7 | Parameter shapes (checkpoint compatibility) |
| 10 | `alloc_for_extend` + `_alloc_for_extend` | 133 | M01 | hisparse / U23 | Fork split; ownership and rollback ordering |

  Next by size: `capture_one_shape` 125, `_forward_paged_attention` 124,
  `_build_hybrid_linear_kv_pool` 119, `_forward_raw` 109, `forward_extend` 102,
  `Qwen4ExpNGramEmbedding.__init__` 100.

## 6. Implementation constraints and deviations from PLAN.md

- **G1. New members on pinned classes cannot be patched.** HookRegistry
  resolves `getattr(owner, name)`, so the attribute must already exist. A
  failure is logged and skipped, and `activate()` then raises "Hooks were not
  applied". This affects the new methods in Q04 and Q09, the `_add_one_req`
  split in P02 (a plugin function is enough there), and the dataclass/Struct
  fields in F01 and T04. PLAN rule 3 ("copied verbatim") cannot hold for
  bodies that call `self._store_kv(...)` and similar unless the framework
  gains an `attach` declaration: set a new attribute only after verifying it
  is absent at the pin, recorded as an "absent" fingerprint. The alternatives
  are a class replace or mechanical edits that call plugin functions.
  Orchestrator decision.
- **G2. Zero-argument `super()`** in a copied method fails outside its class.
  This affects E03 (`Qwen4ExpNGramEmbedding.__init__`), which must use
  `super(Qwen4ExpNGramEmbedding, self)`. That is a mechanical edit to a
  "verbatim" body.
- **G3. Copies run with the plugin module's globals.** Every private helper or
  constant a copy uses must be imported into the plugin module, for example
  `_TRTLLM_SPARSE_PAGE_SIZE`, `_get_best_config` and `logger`. Bindings made
  before `HookRegistry.apply_hooks` are rebound to patched objects by
  `_propagate_patch`. That is how copies reach other patched targets
  (Q12→Q01/A09, Q08→A03/A05, T03→T02).
- **G4. `fused_marlin_moe` is an eager custom op.** `register_custom_op`
  defaults to `eager=True`, so the module attribute is
  `torch.ops.sglang.fused_marlin_moe`. The copy must be registered under a
  distinct `op_name`; reusing the name would silently route to the upstream
  implementation (`CustomOpWrapper.real_impl` checks
  `hasattr(torch.ops.sglang, name)`). `register_custom_op` itself cannot be
  fingerprinted, because `fingerprint._locate` rejects its 5 `@overload`
  definitions.
- **G5. `lru_cache` targets.** The HookRegistry wrapper does not expose
  `cache_clear`. Fork `test/qsa_hisparse/test_flash_attention.py` and pinned
  `test/registered/kernel/qsa/test_qsa.py:162` call
  `_resolve_flash_attn_varlen_func.cache_clear()`. With the plugin on, W4 must
  expose `cache_clear` (and `_resolve_flashinfer_qsa_ragged`'s own cache)
  through the wrapper.
- **G6. Class REPLACE propagation (T04).** `_propagate_patch` rebinds
  every module-level reference to the original class, including the plugin's
  own alias. Define the subclass against an attribute read at class-creation
  time rather than a stored alias. (F01 no longer replaces `ForwardBatch`;
  see deviation D3.)
- **G7. Unit tests with hisparse on.** The pinned
  `test/registered/unit/managers/test_scheduler_chunked_req_gate.py` builds a
  Scheduler without `hisparse_coordinator`, which S04's copy reads. The fork
  edited that test (+1 line). W7's "installed but off" run is unaffected.
- **Contradictions with PLAN.md:**
  (a) Rule 3 vs G1/G2/G4/G5: some copies cannot be both verbatim and
  activatable.
  (b) U-list gaps: F01's `ForwardBatch` fields and Q05/T04's
  `decode_score_width` match no PR; the log-level changes (U01, Z05) have no
  PR; U7 should also list Z01's w13 `size_k` and gc changes and E03's
  meta-device NGram embedding.
  (c) The known hidden-change list includes the `shape_key` move, which is
  pure (§4 item 3).
  (d) PLAN W5 scope implies `moe/.../layer.py` is reproduced in full; Z05 (a
  log-only line) is proposed as drop.
  (e) PLAN says W4 handles "SM89 fallback"; the fork's fallback is SM86 and
  SM89 (§4 item 5).

## 7. Fork changes outside `python/sglang` (noted, not inventoried)

- `test/qsa_hisparse/*` are ported per PLAN by W1 (runtime, slots,
  single_request, gather), W4 (deterministic_topk, flash_attention), W5
  (marlin alignment, model_compatibility), W6 (prefix_cache) and W2/W7
  (service_control; assigned to W7).
- `test/registered/*`: test_hc_mix_triton (+7, W5), test_qsa (+36, W4),
  test_pool_configurator (+69, W3), test_batch_result_processor_hidden_states
  (−get_memory patch; needed only by the fork's import removal, B01) and
  test_scheduler_chunked_req_gate (+1, G7, W2).
- `test/manual/*` (GPU scripts: Marlin batch invariance and alignment, top-k
  probe, prefix acceptance, concurrency, latency, ledger, lifecycle) are
  Phase 2 inputs (P0-B).
- `scripts/qsa_service.py` and its tests: ported by W7 to use the plugin
  launcher. `scripts/test_qsa_hisparse_cpu.sh`, `examples/qsa_hisparse/*`:
  replaced by `tools/run_cpu_tests.sh` and the launcher.
- `python/pyproject.toml`: description and URLs only. `python/LICENSE` and
  `python/README.md` are added.
- Top-level docs and `archive/` (design notes, experiments, results): no code
  impact.

## 8. Track I: image prefix reuse (plugin-only, no fork hunks)

Rows added in Phase 3 (PLAN.md "Track I"). They map no fork hunk, so they are
outside §1 and Appendix A. The prefix matching rule and signatures (I2) are
plugin code in `hisparse/{image_request,prefix,prefix_cache,runtime}.py`, not
hooks.

| ID | Fork file:lines | Behavior | Feature: justification | Hook target (pin) | Type: why | depends (key) | WS | U |
|---|---|---|---|---|---|---|---|---|
| I1 | none | Record each Qwen-VL fast-path image item's full artifact key (content digest, modality, processor fingerprint, preprocessing kwargs) in `model_specific_data["artifact_key"]`, which the tokenizer-to-scheduler transport carries | hisparse: only host prefixes read the key; the extra string is inert upstream | `sglang.srt.multimodal.processors.qwen_vl.QwenVLImageProcessor.compose_image_artifacts` (pin 864) | after: the method builds one item per artifact, in order, from a deepcopy of the artifact's `model_specific_data`, and returns None on fallback | `cache.identity.build_artifact_key`, `MediaArtifactCacheMixin._artifact_key`, `MultimodalDataItem`, `MultimodalInputs.from_processor_output` | I-A | #41792 (`MultimodalDataItem.identity`) |

## Appendix A. Hunk → row map (generated; every hunk exactly once)

Generated from `git diff -U0 76e06febab ee8fe158d6 -- python/sglang`: 31 modified files, 196 hunks, 15 new files, 89 rows. Hunk = new-side `+start,count`.

| File (under python/sglang/) | Hunk | Row |
|---|---|---|
| kernels/jit/csrc/gemm/marlin_moe/marlin_template.h | +27,1 (L27) | J02 |
| kernels/jit/csrc/gemm/marlin_moe/marlin_template.h | +59,2 (L59-60) | J02 |
| kernels/jit/csrc/gemm/marlin_moe/marlin_template.h | +303,2 (L303-304) | J02 |
| kernels/jit/csrc/gemm/marlin_moe/marlin_template.h | +400,6 (L400-405) | J02 |
| kernels/jit/csrc/gemm/marlin_moe/moe_wna16_marlin.cuh | +318,2 (L318-319) | J02 |
| kernels/jit/csrc/gemm/marlin_moe/moe_wna16_marlin.cuh | +443,1 (L443) | J02 |
| kernels/jit/csrc/gemm/marlin_moe/moe_wna16_marlin.cuh | +479,1 (L479) | J02 |
| kernels/jit/csrc/gemm/marlin_moe/moe_wna16_marlin.cuh | +542,1 (L542) | J02 |
| kernels/jit/csrc/gemm/marlin_moe/moe_wna16_marlin.cuh | +580,1 (L580) | J02 |
| kernels/jit/csrc/gemm/marlin_moe/moe_wna16_marlin.cuh | +713,10 (L713-722) | J02 |
| kernels/jit/csrc/gemm/marlin_moe/moe_wna16_marlin.cuh | +729,1 (L729) | J02 |
| kernels/jit/csrc/gemm/marlin_moe/moe_wna16_marlin.cuh | +801,1 (L801) | J02 |
| kernels/jit/csrc/gemm/marlin_moe/moe_wna16_marlin.cuh | +852,1 (L852) | J02 |
| kernels/jit/csrc/gemm/marlin_moe/moe_wna16_marlin.cuh | +1089,1 (L1089) | J02 |
| kernels/jit/csrc/gemm/marlin_moe/stripe_schedule.h | (new file) | J02 |
| kernels/ops/moe/moe_wna16_marlin.py | +20,4 (L20-23) | J01 |
| kernels/ops/moe/moe_wna16_marlin.py | +25,1 (L25) | J01 |
| kernels/ops/moe/moe_wna16_marlin.py | +73,1 (L73) | J01 |
| kernels/ops/moe/moe_wna16_marlin.py | +75,10 (L75-84) | J01 |
| kernels/ops/moe/moe_wna16_marlin.py | +153,3 (L153-155) | J01 |
| srt/hardware_backend/gpu/quantization/gptq_kernels.py | +3,2 (L3-4) | Z01 |
| srt/hardware_backend/gpu/quantization/gptq_kernels.py | +279,2 (L279-280) | Z01 |
| srt/hardware_backend/gpu/quantization/gptq_kernels.py | +289,2 (L289-290) | Z01 |
| srt/hardware_backend/gpu/quantization/gptq_kernels.py | +294,6 (L294-299) | Z01 |
| srt/hardware_backend/gpu/quantization/gptq_kernels.py | +316,2 (L316-317) | Z01 |
| srt/layers/attention/qsa/hisparse_graph.py | (new file) | N03 |
| srt/layers/attention/qsa/kernel.py | +12,50 (L12-61) | T01 |
| srt/layers/attention/qsa/kernel.py | +78,1 (L78) | T02 |
| srt/layers/attention/qsa/kernel.py | +80,6 (L80-85) | T02 |
| srt/layers/attention/qsa/kernel.py | +89,2 (L89-90) | T02 |
| srt/layers/attention/qsa/kernel.py | +123,2 (L123-124) | T02 |
| srt/layers/attention/qsa/metadata.py | +15,1 (L15) | T02 |
| srt/layers/attention/qsa/metadata.py | +86,3 (L86-88) | T04 |
| srt/layers/attention/qsa/metadata.py | +123,10 (L123-132) | T02 |
| srt/layers/attention/qsa/metadata.py | +219,2 (L219-220) | T04 |
| srt/layers/attention/qsa/metadata.py | +232,2 (L232-233) | T04 |
| srt/layers/attention/qsa/qsa_indexer.py | +25,1 (L25) | T02 |
| srt/layers/attention/qsa/qsa_indexer.py | +444,5 (L444-448) | T02 |
| srt/layers/attention/qsa/qsa_indexer.py | +475,1 (L475) | T02 |
| srt/layers/attention/qsa/qsa_indexer.py | +505,5 (L505-509) | T03 |
| srt/layers/attention/qsa/qsa_indexer.py | +523,5 (L523-527) | T03 |
| srt/layers/attention/qsa/sparse_attn.py | +9,32 (L9-40) | A01 |
| srt/layers/attention/qsa/sparse_attn.py | +70,2 (L70-71) | A02 |
| srt/layers/attention/qsa/sparse_attn.py | +93,1 (L93) | A02 |
| srt/layers/attention/qsa/sparse_attn.py | +135,3 (L135-137) | A02 |
| srt/layers/attention/qsa/sparse_attn.py | +143,6 (L143-148) | A02 |
| srt/layers/attention/qsa/sparse_attn.py | +152,9 (L152-160) | A02 |
| srt/layers/attention/qsa/sparse_attn.py | +174,12 (L174-185) | A03 |
| srt/layers/attention/qsa/sparse_attn.py | +200,2 (L200-201) | A03 |
| srt/layers/attention/qsa/sparse_attn.py | +223,1 (L223) | A03 |
| srt/layers/attention/qsa/sparse_attn.py | +241,2 (L241-242) | A04 |
| srt/layers/attention/qsa/sparse_attn.py | +264,1 (L264) | A04 |
| srt/layers/attention/qsa/sparse_attn.py | +307,2 (L307-308) | A04 |
| srt/layers/attention/qsa/sparse_attn.py | +314,6 (L314-319) | A04 |
| srt/layers/attention/qsa/sparse_attn.py | +323,9 (L323-331) | A04 |
| srt/layers/attention/qsa/sparse_attn.py | +345,12 (L345-356) | A05 |
| srt/layers/attention/qsa/sparse_attn.py | +358,1 (L358) | A05 |
| srt/layers/attention/qsa/sparse_attn.py | +376,2 (L376-377) | A05 |
| srt/layers/attention/qsa/sparse_attn.py | +399,1 (L399) | A05 |
| srt/layers/attention/qsa/sparse_attn.py | +440,3 (L440-442) | A06 |
| srt/layers/attention/qsa/sparse_attn.py | +474,2 (L474-475) | A07 |
| srt/layers/attention/qsa/sparse_attn.py | +484,1 (L484) | A07 |
| srt/layers/attention/qsa/sparse_attn.py | +519,7 (L519-525) | A07 |
| srt/layers/attention/qsa/sparse_attn.py | +529,3 (L529-531) | A08 |
| srt/layers/attention/qsa/sparse_attn.py | +555,2 (L555-556) | A09 |
| srt/layers/attention/qsa/sparse_attn.py | +575,3 (L575-577) | A09 |
| srt/layers/attention/qsa/sparse_attn.py | +592,2 (L592-593) | A09 |
| srt/layers/attention/qsa/sparse_attn.py | +602,1 (L602) | A09 |
| srt/layers/attention/qsa/sparse_attn.py | +609,1 (L609) | A10 |
| srt/layers/attention/qwen_sparse_attn_backend.py | +11,1 (L11) | Q03 |
| srt/layers/attention/qwen_sparse_attn_backend.py | +34,1 (L34) | Q04 |
| srt/layers/attention/qwen_sparse_attn_backend.py | +42,1 (L42) | Q12 |
| srt/layers/attention/qwen_sparse_attn_backend.py | +50,11 (L50-60) | Q09 |
| srt/layers/attention/qwen_sparse_attn_backend.py | +95,9 (L95-103) | Q01 |
| srt/layers/attention/qwen_sparse_attn_backend.py | +239,5 (L239-243) | Q02 |
| srt/layers/attention/qwen_sparse_attn_backend.py | +246,12 (L246-257) | Q03 |
| srt/layers/attention/qwen_sparse_attn_backend.py | +262,30 (L262-291) | Q04 |
| srt/layers/attention/qwen_sparse_attn_backend.py | +704,1 (L704) | Q05 |
| srt/layers/attention/qwen_sparse_attn_backend.py | +741,9 (L741-749) | Q05 |
| srt/layers/attention/qwen_sparse_attn_backend.py | +790,1 (L790) | Q05 |
| srt/layers/attention/qwen_sparse_attn_backend.py | +808,2 (L808-809) | Q06 |
| srt/layers/attention/qwen_sparse_attn_backend.py | +995,12 (L995-1006) | Q07 |
| srt/layers/attention/qwen_sparse_attn_backend.py | +1370,1 (L1370) | Q08 |
| srt/layers/attention/qwen_sparse_attn_backend.py | +1429,6 (L1429-1434) | Q08 |
| srt/layers/attention/qwen_sparse_attn_backend.py | +1436,1 (L1436) | Q08 |
| srt/layers/attention/qwen_sparse_attn_backend.py | +1440,1 (L1440) | Q08 |
| srt/layers/attention/qwen_sparse_attn_backend.py | +1492,102 (L1492-1593) | Q09 |
| srt/layers/attention/qwen_sparse_attn_backend.py | +1650,1 (L1650) | Q10 |
| srt/layers/attention/qwen_sparse_attn_backend.py | +1667,2 (L1667-1668) | Q10 |
| srt/layers/attention/qwen_sparse_attn_backend.py | +1713,7 (L1713-1719) | Q11 |
| srt/layers/attention/qwen_sparse_attn_backend.py | +1741,13 (L1741-1753) | Q12 |
| srt/layers/attention/qwen_sparse_attn_backend.py | +1766,80 (L1766-1845) | Q12 |
| srt/layers/hc_mix_triton.py | +3,25 (L3-27) | H01 |
| srt/layers/hc_mix_triton.py | +67,1 (L67) | H02 |
| srt/layers/hc_mix_triton.py | +161,137 (L161-297) | H03 |
| srt/layers/hc_mix_triton.py | +317,0 (deletion) | H04 |
| srt/layers/hc_mix_triton.py | +319,5 (L319-323) | H05 |
| srt/layers/hc_mix_triton.py | +325,3 (L325-327) | H05 |
| srt/layers/hc_mix_triton.py | +349,3 (L349-351) | H06 |
| srt/layers/hc_mix_triton.py | +359,7 (L359-365) | H06 |
| srt/layers/hc_mix_triton.py | +369,1 (L369) | H06 |
| srt/layers/hc_mix_triton.py | +378,34 (L378-411) | H06 |
| srt/layers/hyperconnection.py | +93,1 (L93) | H07 |
| srt/layers/hyperconnection.py | +222,1 (L222) | H08 |
| srt/layers/hyperconnection.py | +236,23 (L236-258) | H08 |
| srt/layers/moe/fused_moe_triton/fused_marlin_moe.py | +6,0 (deletion) | J03 |
| srt/layers/moe/fused_moe_triton/fused_marlin_moe.py | +8,4 (L8-11) | J03 |
| srt/layers/moe/fused_moe_triton/fused_marlin_moe.py | +18,0 (deletion) | J03 |
| srt/layers/moe/fused_moe_triton/fused_marlin_moe.py | +197,10 (L197-206) | J03 |
| srt/layers/moe/fused_moe_triton/fused_marlin_moe.py | +255,4 (L255-258) | J03 |
| srt/layers/moe/fused_moe_triton/fused_marlin_moe.py | +317,7 (L317-323) | J03 |
| srt/layers/moe/fused_moe_triton/fused_marlin_moe.py | +352,1 (L352) | J03 |
| srt/layers/moe/fused_moe_triton/fused_marlin_moe.py | +416,1 (L416) | J03 |
| srt/layers/moe/fused_moe_triton/layer.py | +497,1 (L497) | Z05 |
| srt/layers/moe/fused_moe_triton/layer.py | +1024,5 (L1024-1028) | Z04 |
| srt/layers/moe/fused_moe_triton/stable_align.py | (new file) | J04 |
| srt/layers/quantization/auto_round.py | +524,9 (L524-532) | Z03 |
| srt/layers/quantization/gptq/schemes/gptq_moe.py | +200,1 (L200) | Z02 |
| srt/layers/quantization/gptq/schemes/gptq_moe.py | +239,1 (L239) | Z02 |
| srt/layers/quantization/gptq/schemes/gptq_moe.py | +247,1 (L247) | Z02 |
| srt/managers/schedule_policy.py | +960,7 (L960-966) | P01 |
| srt/managers/schedule_policy.py | +1165,31 (L1165-1195) | P02 |
| srt/managers/schedule_policy.py | +1200,6 (L1200-1205) | P02 |
| srt/managers/scheduler.py | +630,16 (L630-645) | S01 |
| srt/managers/scheduler.py | +1281,2 (L1281-1282) | S02 |
| srt/managers/scheduler.py | +1283,0 (deletion) | S02 |
| srt/managers/scheduler.py | +1286,2 (L1286-1287) | S02 |
| srt/managers/scheduler.py | +3563,2 (L3563-3564) | S03 |
| srt/managers/scheduler.py | +3647,1 (L3647) | S04 |
| srt/managers/scheduler.py | +3660,1 (L3660) | S04 |
| srt/managers/scheduler.py | +3770,7 (L3770-3776) | S05 |
| srt/managers/scheduler.py | +3898,2 (L3898-3899) | S06 |
| srt/managers/scheduler.py | +4816,1 (L4816) | S07 |
| srt/managers/scheduler.py | +4905,1 (L4905) | S08 |
| srt/managers/scheduler.py | +5410,4 (L5410-5413) | S09 |
| srt/managers/scheduler_components/batch_result_processor.py | +42,0 (deletion) | B01 |
| srt/managers/scheduler_components/batch_result_processor.py | +125,1 (L125) | B02 |
| srt/managers/scheduler_components/batch_result_processor.py | +383,1 (L383) | B03 |
| srt/managers/scheduler_components/batch_result_processor.py | +1245,1 (L1245) | B04 |
| srt/managers/scheduler_components/batch_result_processor.py | +1329,1 (L1329) | B05 |
| srt/managers/scheduler_components/weight_updater.py | +95,6 (L95-100) | B06 |
| srt/mem_cache/allocation.py | +346,16 (L346-361) | M01 |
| srt/mem_cache/allocation.py | +372,0 (deletion) | M01 |
| srt/mem_cache/allocation.py | +392,4 (L392-395) | M01 |
| srt/mem_cache/allocation.py | +433,4 (L433-436) | M01 |
| srt/mem_cache/allocation.py | +457,2 (L457-458) | M01 |
| srt/mem_cache/allocation.py | +474,4 (L474-477) | M01 |
| srt/mem_cache/allocator/paged.py | +336,5 (L336-340) | M02 |
| srt/mem_cache/common.py | +269,9 (L269-277) | M03 |
| srt/mem_cache/common.py | +306,2 (L306-307) | M03 |
| srt/mem_cache/kv_cache_configurator.py | +6,1 (L6) | K02 |
| srt/mem_cache/kv_cache_configurator.py | +1318,1 (L1318) | K01 |
| srt/mem_cache/kv_cache_configurator.py | +1860,1 (L1860) | K02 |
| srt/mem_cache/kv_cache_configurator.py | +1936,18 (L1936-1953) | K02 |
| srt/mem_cache/memory_pool.py | +348,3 (L348-350) | M04 |
| srt/mem_cache/qsa_hisparse/__init__.py | (new file) | N01 |
| srt/mem_cache/qsa_hisparse/config.py | (new file) | N01 |
| srt/mem_cache/qsa_hisparse/coordinator.py | (new file) | N01 |
| srt/mem_cache/qsa_hisparse/layout.py | (new file) | N01 |
| srt/mem_cache/qsa_hisparse/prefix.py | (new file) | N02 |
| srt/mem_cache/qsa_hisparse/prefix_cache.py | (new file) | N02 |
| srt/mem_cache/qsa_hisparse/runtime.py | (new file) | N01 |
| srt/mem_cache/qsa_hisparse/single_request.py | (new file) | N01 |
| srt/mem_cache/qsa_hisparse/slots.py | (new file) | N01 |
| srt/mem_cache/qsa_hisparse_p2.py | (new file) | N04 |
| srt/mem_cache/qsa_hisparse_slots.py | (new file) | N04 |
| srt/mem_cache/qsa_hisparse_v3.py | (new file) | N04 |
| srt/mem_cache/qsa_kv_pool.py | +71,1 (L71) | K03 |
| srt/mem_cache/qsa_kv_pool.py | +103,1 (L103) | K03 |
| srt/model_executor/forward_batch_info.py | +479,4 (L479-482) | F01 |
| srt/model_executor/forward_batch_info.py | +815,6 (L815-820) | F01 |
| srt/model_executor/forward_batch_info.py | +840,2 (L840-841) | F01 |
| srt/model_executor/model_runner.py | +1037,7 (L1037-1043) | R01 |
| srt/model_executor/model_runner.py | +1811,7 (L1811-1817) | R02 |
| srt/model_executor/model_runner.py | +1825,4 (L1825-1828) | R02 |
| srt/model_executor/pool_configurator.py | +17,1 (L17) | C01 |
| srt/model_executor/pool_configurator.py | +203,1 (L203) | C03 |
| srt/model_executor/pool_configurator.py | +208,1 (L208) | C01 |
| srt/model_executor/pool_configurator.py | +298,49 (L298-346) | C01 |
| srt/model_executor/pool_configurator.py | +588,1 (L588) | C02 |
| srt/model_executor/runner/decode_cuda_graph_runner.py | +90,1 (L90) | G01 |
| srt/model_executor/runner/decode_cuda_graph_runner.py | +1130,10 (L1130-1139) | G01 |
| srt/model_executor/runner/decode_cuda_graph_runner.py | +1205,0 (deletion) | G01 |
| srt/model_executor/runner/decode_cuda_graph_runner.py | +1212,7 (L1212-1218) | G01 |
| srt/model_executor/runner/decode_cuda_graph_runner.py | +1253,5 (L1253-1257) | G02 |
| srt/model_executor/runner/decode_cuda_graph_runner.py | +1324,3 (L1324-1326) | G02 |
| srt/model_executor/runner/decode_cuda_graph_runner.py | +1405,1 (L1405) | G02 |
| srt/model_executor/runner/decode_cuda_graph_runner.py | +1431,4 (L1431-1434) | G03 |
| srt/model_executor/runner/decode_cuda_graph_runner.py | +1454,3 (L1454-1456) | G03 |
| srt/models/qwen4_exp.py | +71,1 (L71) | E01 |
| srt/models/qwen4_exp.py | +103,7 (L103-109) | E01 |
| srt/models/qwen4_exp.py | +517,15 (L517-531) | E03 |
| srt/models/qwen4_exp.py | +535,18 (L535-552) | E03 |
| srt/models/qwen4_exp.py | +769,1 (L769) | E04 |
| srt/models/qwen4_exp.py | +774,2 (L774-775) | E04 |
| srt/models/qwen4_exp.py | +786,2 (L786-787) | E04 |
| srt/models/qwen4_exp.py | +795,4 (L795-798) | E04 |
| srt/models/qwen4_exp.py | +811,3 (L811-813) | E05 |
| srt/models/qwen4_exp.py | +847,5 (L847-851) | E06 |
| srt/models/qwen4_exp.py | +853,2 (L853-854) | E06 |
| srt/models/qwen4_exp.py | +891,14 (L891-904) | E06 |
| srt/models/qwen4_exp.py | +950,1 (L950) | E07 |
| srt/models/qwen4_exp.py | +955,2 (L955-956) | E07 |
| srt/models/qwen4_exp.py | +1407,3 (L1407-1409) | E02 |
| srt/models/qwen4_exp.py | +1421,3 (L1421-1423) | E02 |
| srt/models/qwen4_exp.py | +1772,3 (L1772-1774) | E02 |
| srt/models/qwen4_exp.py | +1979,3 (L1979-1981) | E08 |
| srt/models/qwen4_exp.py | +1990,26 (L1990-2015) | E08 |
| srt/models/qwen4_exp.py | +2057,11 (L2057-2067) | E08 |
| srt/models/qwen4_exp.py | +2258,14 (L2258-2271) | E08 |
| srt/utils/common.py | +4001,1 (L4001) | U01 |

## Appendix B. Candidate `depends` for copied bodies (generated)

Callees of each copied fork body that resolve to a pinned definition (re-exports followed to the defining module; each entry verified with `fingerprint.resolve_file` + `_locate` at the pin). `self.*` calls are resolved against the same class only (inherited callees are not listed). "needs" lists names the copy uses that the pinned module does not define or import: fork-new definitions the plugin must provide, plus names the copy imports locally that exist elsewhere upstream (`get_parallel`, `get_context`, `get_exec` from `sglang.srt.runtime_context`; these only need an import). J03 lists the in-tree `moe_wna16_marlin_gemm` because the fork module imports that name; the plugin copy must call J01's plugin op instead. Curate before writing fingerprints: everything here is a candidate, the key ones are named in §1.

| Row | Copied target | Candidate depends (pinned, verified) | needs (new in fork) |
|---|---|---|---|
| S04 | `managers.scheduler.Scheduler.get_next_batch_to_run` | `managers.schedule_batch.NextBatchPlan`, `managers.schedule_batch.ScheduleBatch`, `managers.scheduler.Scheduler._arm_prefill_decode_interval`, `managers.scheduler.Scheduler._build_hisparse_decode_batch`, `managers.scheduler.Scheduler._process_hicache_events`, `managers.scheduler.Scheduler._should_defer_prefill`, `managers.scheduler.Scheduler.get_new_batch_prefill`, `managers.scheduler.Scheduler.process_pending_chunked_abort`, `managers.scheduler.Scheduler.stash_chunked_request`, `managers.scheduler.Scheduler.update_running_batch`, `observability.req_time_stats.set_schedule_time_batch`, `observability.scheduler_stage_metrics.scheduler_stage_method`, `runtime_context.get_spec` | none |
| S06 | `managers.scheduler.Scheduler._get_new_batch_prefill_raw` | `managers.schedule_batch.ScheduleBatch`, `managers.schedule_policy.PrefillAdder`, `managers.scheduler.Scheduler._add_request_to_queue`, `managers.scheduler.Scheduler._prefetch_after_device_hit_loss`, `managers.scheduler.Scheduler.can_schedule_lora_req`, `managers.scheduler.Scheduler.get_num_allocatable_reqs`, `observability.req_time_stats.set_time_batch`, `runtime_context.get_schedule` | none |
| S07 | `managers.scheduler.Scheduler.on_idle` | `managers.scheduler.Scheduler.is_fully_idle`, `managers.scheduler.Scheduler.maybe_send_health_check_signal`, `managers.scheduler.Scheduler.maybe_sleep_on_idle`, `managers.scheduler.Scheduler.publish_load_snapshot`, `observability.scheduler_stage_metrics.scheduler_stage_method` | none |
| B02 | `managers.scheduler_components.batch_result_processor.SchedulerBatchResultProcessor.process_batch_result_prebuilt` | `mem_cache.common.release_kv_cache`, `runtime_context.get_disagg` | none |
| B03 | `managers.scheduler_components.batch_result_processor.SchedulerBatchResultProcessor.process_batch_result_prefill` | `managers.scheduler_components.batch_result_processor.SchedulerBatchResultProcessor._append_prefill_hidden_states`, `managers.scheduler_components.batch_result_processor.SchedulerBatchResultProcessor._apply_chunked_prefill_logprobs`, `managers.scheduler_components.batch_result_processor.SchedulerBatchResultProcessor._apply_prefill_grammar`, `managers.scheduler_components.batch_result_processor.SchedulerBatchResultProcessor._apply_prefill_logprobs`, `managers.scheduler_components.batch_result_processor.SchedulerBatchResultProcessor._convert_embeddings`, `managers.scheduler_components.batch_result_processor.SchedulerBatchResultProcessor._get_prefill_hidden_capture_mode`, `managers.scheduler_components.batch_result_processor.SchedulerBatchResultProcessor._maybe_collect_customized_info`, `managers.scheduler_components.batch_result_processor.SchedulerBatchResultProcessor._maybe_collect_indexer_topk`, `managers.scheduler_components.batch_result_processor.SchedulerBatchResultProcessor._maybe_collect_routed_experts`, `managers.scheduler_components.batch_result_processor.SchedulerBatchResultProcessor._maybe_update_reasoning_tokens`, `managers.scheduler_components.batch_result_processor.SchedulerBatchResultProcessor._validate_pp_skip_output_comm`, `managers.scheduler_components.batch_result_processor.SchedulerBatchResultProcessor.add_sampling_mask_return_values`, `managers.scheduler_components.batch_result_processor.SchedulerBatchResultProcessor.consume_auxiliary_output`, `managers.scheduler_components.batch_result_processor.SchedulerBatchResultProcessor.get_sampling_mask_finish_reason`, `managers.scheduler_components.batch_result_processor.SchedulerBatchResultProcessor.materialize_sampling_mask_output`, `managers.scheduler_components.batch_result_processor.SchedulerBatchResultProcessor.move_logprobs_to_cpu`, `managers.scheduler_components.batch_result_processor.SchedulerBatchResultProcessor.snapshot_auxiliary_output_starts`, `mem_cache.common.maybe_cache_unfinished_req`, `mem_cache.common.release_kv_cache` | none |
| B04 | `managers.scheduler_components.batch_result_processor.SchedulerBatchResultProcessor._handle_sampling_mask_abort` | `mem_cache.common.release_kv_cache`, `runtime_context.get_disagg` | none |
| B05 | `managers.scheduler_components.batch_result_processor.SchedulerBatchResultProcessor._handle_finish_state_updated_req` | `managers.scheduler_components.batch_result_processor.SchedulerBatchResultProcessor._mamba_prefix_cache_update`, `managers.scheduler_components.batch_result_processor.SchedulerBatchResultProcessor._maybe_collect_customized_info`, `managers.scheduler_components.batch_result_processor.SchedulerBatchResultProcessor._maybe_collect_indexer_topk`, `managers.scheduler_components.batch_result_processor.SchedulerBatchResultProcessor._maybe_collect_routed_experts`, `mem_cache.common.release_kv_cache`, `runtime_context.get_disagg`, `runtime_context.get_exec` | none |
| P02 | `managers.schedule_policy.PrefillAdder.add_one_req` | `managers.schedule_policy.PrefillAdder._commit_prefill_admission`, `managers.schedule_policy.PrefillAdder._lock_node`, `managers.schedule_policy.PrefillAdder._mamba_gap_budget_for_req`, `managers.schedule_policy.PrefillAdder._select_prefill_admission`, `managers.schedule_policy.PrefillAdder.add_one_req_ignore_eos`, `managers.schedule_policy.PrefillAdder.budget_state`, `managers.schedule_policy.PrefillAdder._kv_shard_reserve_scratch`, `managers.schedule_policy._PrefillAdmission`, `mem_cache.base_prefix_cache.InitLoadBackParams` | `PrefillAdder._add_one_req` |
| M01 | `mem_cache.allocation.alloc_for_extend` | `hardware_backend.npu.dsv4.dsv4_common_hooks.maybe_write_dsv4_extend`, `mem_cache.allocation._alloc_extend_loc_with_kv_reuse`, `mem_cache.allocation._alloc_page_size`, `mem_cache.allocation.alloc_paged_token_slots_extend`, `mem_cache.allocation.alloc_req_slots`, `mem_cache.allocation.alloc_token_slots`, `mem_cache.allocation.write_cache_indices`, `utils.common.is_pin_memory_available` | `_alloc_for_extend` |
| M03 | `mem_cache.common.release_kv_cache` | `mem_cache.common._release_overallocated_kv_indices` | none |
| K02 | `mem_cache.kv_cache_configurator.KVCacheConfigurator._build_hybrid_linear_kv_pool` | `configs.model_config.dsa_layer_skips_topk`, `configs.model_config.get_dsa_index_head_dim`, `configs.model_config.get_dsa_index_kpool`, `configs.model_config.get_dsa_index_kpool_compress`, `configs.model_config.is_deepseek_dsa`, `layers.attention.qsa.config.parse_qsa_profile`, `mem_cache.kv_cache_configurator.KVCacheConfigurator._build_mha_quant_method`, `mem_cache.kv_cache_configurator.calculate_mla_kv_cache_dim`, `mem_cache.memory_pool.MHATokenToKVPool`, `runtime_context.get_exec`, `runtime_context.get_parallel`, `runtime_context.get_spec`, `runtime_context.max_speculative_num_draft_tokens` | `QSAHiSparseSlots` |
| F01 | `model_executor.forward_batch_info.ForwardBatch.init_new` (after, D3) | `managers.schedule_batch.ScheduleBatch` | none |
| R02 | `model_executor.model_runner.ModelRunner._forward_raw` | `model_executor.forward_context.ForwardContext`, `model_executor.forward_context.forward_context`, `model_executor.forward_context.has_forward_context`, `model_executor.model_runner.ModelRunner._extend_forward_kwargs`, `model_executor.model_runner.ModelRunner._maybe_execute_deferred_mamba_cow_and_clear`, `model_executor.model_runner.ModelRunner._prepare_eager_forward_batch`, `model_executor.model_runner.ModelRunner.forward_split_prefill`, `model_executor.model_runner.ModelRunnerOutput`, `model_executor.model_runner._prefill_cuda_graph_allows_context_parallel`, `runtime_context.get_global_dwdp_manager`, `utils.device_timer.device_timer_ctx` | none |
| G01 | `model_executor.runner.decode_cuda_graph_runner.DecodeCudaGraphRunner.capture_one_shape` | `layers.dp_attention.set_dp_buffer_len`, `layers.dp_attention.set_is_extend_in_batch`, `model_executor.forward_batch_info.PPProxyTensors`, `model_executor.forward_context.ForwardContext`, `model_executor.forward_context.forward_context`, `model_executor.runner.decode_cuda_graph_runner.DecodeCudaGraphRunner._capture_graph_size`, `model_executor.runner.decode_cuda_graph_runner.DecodeCudaGraphRunner._make_graph_key`, `model_executor.runner.decode_cuda_graph_runner.DecodeCudaGraphRunner._ragged_capture_slots`, `model_executor.runner.decode_cuda_graph_runner.DecodeCudaGraphRunner._record_in_graph_metadata_prep_done`, `model_executor.runner.decode_cuda_graph_runner.DecodeCudaGraphRunner.capture_prepare`, `model_executor.runner.flashinfer_autotune.maybe_flashinfer_autotune_speculative_draft`, `runtime_context.get_exec`, `utils.common.empty_context` | none |
| G02 | `model_executor.runner.decode_cuda_graph_runner.DecodeCudaGraphRunner.load_batch` | `model_executor.runner.decode_cuda_graph_runner.DecodeCudaGraphRunner._capture_graph_size`, `model_executor.runner.decode_cuda_graph_runner.DecodeCudaGraphRunner._make_graph_key`, `model_executor.runner.decode_cuda_graph_runner.DecodeCudaGraphRunner._max_dp_batch_size`, `model_executor.runner.decode_cuda_graph_runner.DecodeCudaGraphRunner._ragged_capture_slots`, `model_executor.runner.decode_cuda_graph_runner.DecodeCudaGraphRunner._ragged_graph_num_tokens`, `model_executor.runner.decode_cuda_graph_runner.DecodeCudaGraphRunner._replay_attn_backend`, `model_executor.runner.decode_cuda_graph_runner.DecodeCudaGraphRunner._resolve_attention_variant`, `model_executor.runner.decode_cuda_graph_runner.DecodeCudaGraphRunner._resolve_lora_variant`, `model_executor.runner.decode_cuda_graph_runner.DecodeCudaGraphRunner._stage_ragged_verify_layout`, `model_executor.runner.decode_cuda_graph_runner.DecodeCudaGraphRunner._validate_capture_hidden_mode`, `model_executor.runner.decode_cuda_graph_runner.build_replay_fb_view`, `multiplex.pdmux_context.get_current_stream_idx`, `speculative.ragged_verify.resolve_ragged_verify_layout` | none |
| G03 | `model_executor.runner.decode_cuda_graph_runner.DecodeCudaGraphRunner.execute` | `layers.logits_processor.LogitsProcessorOutput`, `model_executor.forward_batch_info.PPProxyTensors`, `model_executor.runner.decode_cuda_graph_runner.DecodeCudaGraphRunner._publish_read_done`, `model_executor.runner.decode_cuda_graph_runner.DecodeCudaGraphRunner._ragged_capture_slots`, `model_executor.runner.decode_cuda_graph_runner.DecodeCudaGraphRunner._replay_attn_backend`, `model_executor.runner.decode_cuda_graph_runner.DecodeCudaGraphRunner._resolve_shared_read_ends`, `model_executor.runner.decode_cuda_graph_runner.DecodeCudaGraphRunner.load_batch`, `utils.common.empty_context`, `utils.device_timer.device_timer_ctx` | none |
| Q01 | `layers.attention.qwen_sparse_attn_backend._resolve_flash_attn_varlen_func` | `utils.common.is_sm121` | none |
| Q04 | `layers.attention.qwen_sparse_attn_backend.QwenSparseAttnBackend` | none | `QwenSparseAttnBackend._kv_descales`, `is_fp8_kv_dtype` |
| Q08 | `layers.attention.qwen_sparse_attn_backend.QwenSparseAttnBackend.forward_extend` | `layers.attention.qsa.kernel.qsa_sparse_attention`, `layers.attention.qsa.sparse_attn.sparse_gqa_fwd_interface_triton`, `layers.attention.qsa.sparse_attn.sparse_gqa_fwd_interface_triton_ck`, `layers.attention.qwen_sparse_attn_backend.QwenSparseAttnBackend._forward_paged_attention`, `layers.attention.qwen_sparse_attn_backend.QwenSparseAttnBackend._is_speculative_paged_mode`, `layers.attention.qwen_sparse_attn_backend.QwenSparseAttnBackend._logical_to_physical`, `layers.attention.qwen_sparse_attn_backend.QwenSparseAttnBackend._pad_extend_output`, `layers.attention.qwen_sparse_attn_backend.QwenSparseAttnBackend._resolve_metadata` | `QwenSparseAttnBackend._store_kv` |
| Q09 | `layers.attention.qwen_sparse_attn_backend.QwenSparseAttnBackend` | none | `QwenSparseAttnBackend._qsa_local_head_shape`, `_resolve_flashinfer_qsa_ragged`, `get_parallel` |
| Q10 | `layers.attention.qwen_sparse_attn_backend.QwenSparseAttnBackend._forward_trtllm_sparse` | `layers.attention.qsa.sparse_attn.qwen_sparse_kv_extraction_compact_triton`, `layers.attention.qsa.sparse_attn.qwen_sparse_valid_counts_triton`, `layers.attention.qwen_sparse_attn_backend.QwenSparseAttnBackend._get_fa2_scratch`, `layers.attention.qwen_sparse_attn_backend.QwenSparseAttnBackend._get_trtllm_sparse_tables` | `QwenSparseAttnBackend._kv_descales` |
| Q11 | `layers.attention.qwen_sparse_attn_backend.QwenSparseAttnBackend.forward_decode` | `layers.attention.qwen_sparse_attn_backend.QwenSparseAttnBackend._forward_paged_attention`, `layers.attention.qwen_sparse_attn_backend.QwenSparseAttnBackend._resolve_metadata` | `QwenSparseAttnBackend._store_kv` |
| Q12 | `layers.attention.qwen_sparse_attn_backend.QwenSparseAttnBackend._forward_paged_attention` | `layers.attention.qsa.kernel.qsa_sparse_attention`, `layers.attention.qsa.sparse_attn.qwen_sparse_fa2_cu_seqlens_triton`, `layers.attention.qsa.sparse_attn.qwen_sparse_kv_extraction_compact_triton`, `layers.attention.qwen_sparse_attn_backend.QwenSparseAttnBackend._forward_trtllm_sparse`, `layers.attention.qwen_sparse_attn_backend.QwenSparseAttnBackend._get_fa2_scratch`, `layers.attention.qwen_sparse_attn_backend.QwenSparseAttnBackend._logical_to_physical`, `layers.attention.qwen_sparse_attn_backend.QwenSparseAttnBackend._resolve_metadata`, `layers.attention.qwen_sparse_attn_backend._resolve_flash_attn_varlen_func`, `layers.attention.qwen_sparse_attn_backend._resolve_trtllm_sparse_decode` | `QwenSparseAttnBackend._can_run_fa2_graph`, `QwenSparseAttnBackend._kv_descales`, `is_fp8_kv_dtype`, `operations_nvtx_range` |
| T03 | `layers.attention.qsa.qsa_indexer.QSAIndexer.select_decode_tokens` | `sglang.kernels.ops.elementwise.fast_topk.fast_topk`, `layers.attention.qsa.kernel.expand_qsa_block_indices`, `layers.attention.qsa.kernel.qsa_fast_topk`, `layers.attention.qsa.mqa.qsa_mqa_decode` | `get_context`, `get_exec` |
| T04 | `layers.attention.qsa.metadata.QSAIndexerMetadata` | `layers.attention.qsa.metadata.compressed_decode_view` | none |
| A03 | `layers.attention.qsa.sparse_attn.sparse_gqa_fwd_interface_triton` | `layers.attention.qsa.sparse_attn._get_best_config` | `_unit_scale`, `_validate_sparse_gqa_dtypes` |
| A05 | `layers.attention.qsa.sparse_attn.sparse_gqa_fwd_interface_triton_ck` | `layers.attention.qsa.sparse_attn._get_best_config` | `_unit_scale`, `_validate_sparse_gqa_dtypes` |
| A09 | `layers.attention.qsa.sparse_attn.qwen_sparse_kv_extraction_compact_triton` | none | `_unit_scale`, `is_fp8_kv_dtype` |
| J03 | `layers.moe.fused_moe_triton.fused_marlin_moe.fused_marlin_moe` | `sglang.kernels.ops.activation.activation.silu_and_mul`, `sglang.kernels.ops.moe.moe_align_single_token.moe_align_single_token`, `sglang.kernels.ops.moe.moe_topk_sum.moe_topk_sum`, `sglang.kernels.ops.moe.moe_wna16_marlin.moe_wna16_marlin_gemm`, `layers.moe.fused_moe_triton.fused_marlin_moe.get_scalar_type`, `layers.moe.fused_moe_triton.fused_marlin_moe.situ_and_mul`, `layers.moe.fused_moe_triton.fused_marlin_moe.swiglu_gpt_oss_sigmoid_alpha_contiguous`, `layers.moe.fused_moe_triton.fused_marlin_moe.swiglu_limit_func`, `layers.moe.moe_runner.triton_utils.moe_align_block_size.moe_align_block_size` | `get_context`, `get_exec` |
| Z01 | `hardware_backend.gpu.quantization.gptq_kernels.GPTQMarlinMoEKernel.process_weights_after_loading` | `hardware_backend.gpu.quantization.gptq_kernels.gptq_marlin_moe_repack`, `layers.quantization.marlin_utils.marlin_moe_permute_scales`, `layers.quantization.utils.replace_parameter` | none |
| Z02 | `layers.quantization.gptq.schemes.gptq_moe.GPTQMarlinMoEScheme.create_weights` | `utils.common.set_weight_attrs` | none |
| Z03 | `layers.quantization.auto_round.AutoRoundConfig.apply_gptq_quant_layer` | `layers.quantization.auto_round.AutoRoundConfig.check_cpu_support`, `layers.quantization.auto_round.AutoRoundConfig.check_quantized`, `layers.quantization.auto_round.AutoRoundConfig.get_gptq_config_kwargs`, `layers.quantization.auto_round.AutoRoundConfig.get_layer_config`, `layers.quantization.auto_round.AutoRoundConfig.log_gptq_default_assumptions_once`, `layers.quantization.marlin_utils.check_marlin_supported`, `layers.quantization.marlin_utils.check_moe_marlin_supports_layer`, `layers.quantization.unquant.UnquantizedLinearMethod` | none |
| H05 | `layers.hc_mix_triton.fused_hc_mix_supported` | `layers.hc_mix_triton._deterministic_inference` | none |
| H06 | `layers.hc_mix_triton.fused_hc_mix` | `layers.hc_mix_triton._get_counters` | none |
| H08 | `layers.hyperconnection.GatedResidual.mix` | `sglang.kernels.ops.elementwise.hc_mix.hc_mix`, `sglang.kernels.ops.elementwise.hc_mix.permute_pad_up_weight`, `layers.hc_mix_triton.fused_hc_mix`, `layers.hc_mix_triton.fused_hc_mix_supported` | none |
| E03 | `models.qwen4_exp.Qwen4ExpNGramEmbedding.__init__` | `layers.dp_attention.get_attention_dp_size`, `layers.dp_attention.is_dp_attention_enabled`, `layers.vocab_parallel_embedding.VocabParallelEmbedding`, `models.qwen4_exp.Qwen4ExpNGramEmbedding._build_head_vocab_and_offsets`, `models.qwen4_exp.Qwen4ExpNGramEmbedding._build_layer_multipliers`, `models.qwen4_exp._ple_table_is_fp8`, `models.qwen4_exp._use_attn_tp_ngram` | none |
| E06 | `models.qwen4_exp.Qwen4ExpPinnedHostEmbedding.__init__` | `models.qwen4_exp_ple_table.allocate_ple_host_table`, `models.qwen4_exp_ple_table.make_ple_file_prefetcher`, `models.qwen4_exp_ple_table.make_ple_file_rss_trimmer` | none |
| E07 | `models.qwen4_exp.Qwen4ExpPinnedHostEmbedding.gather` | `models.qwen4_exp.Qwen4ExpPinnedHostEmbedding.allocate_output` | none |
| E08 | `models.qwen4_exp.Qwen4ExpForConditionalGeneration.load_weights` | `layers.utils.common.get_layer_id`, `models.qwen4_exp.Qwen4ExpForConditionalGeneration._load_qwen4_exp_ple_buffer` | none |

Prefix `sglang.srt.` is omitted in this appendix; other prefixes (e.g. `sglang.kernels.`) are kept.
