# QSA host prefix cache: integration surface (W6)

Scope: `src/sglang_qsa_hisparse/hisparse/{prefix,prefix_cache}.py` (inventory
N02, moved verbatim from `fork:` `mem_cache/qsa_hisparse/`) and the scheduler
and allocation call sites it depends on. Line numbers are `pin:` (`76e06febab`)
unless marked `fork:`. Rows are from [patch-inventory.md](patch-inventory.md).

Objects and owners:
- `HostPrefixCache` (`runtime.prefix_cache`, one per TP rank): immutable
  `PrefixSnapshot`s at page64 lengths. A `PrefixReader` pins one snapshot; a
  `PrefixReservation` holds host bytes and one entry slot before a capture.
- `QSAHostPrefixCache` (the scheduler's `tree_cache`, a `ChunkCache`
  subclass): `matches[handle]` (reader of a matched, not yet restored
  attempt) and `restoring[handle]` (one record per request inside
  `alloc_for_extend`: reader, `prefix` pages, `suffix` pages, previous
  lengths). `handle` is the upstream `CacheRequestHandle(rid, attempt_id)`.
- Runtime (`kvcache.qsa_hisparse`, W1): per-request lease (physical staging
  slot + generation) and `_RequestCache` state (`seq_len`, prefix basis).

## 1. Call sites

| # | Upstream call site (pin) | Cache/runtime method | Row | Ported test (tests/prefix) |
|---|---|---|---|---|
| 1 | `Scheduler.__init__` 629, right after `init_hisparse_coordinator()` | wrap `ChunkCache` in `QSAHostPrefixCache(cache, runtime, tp_cpu_group)`; reject non-`ChunkCache`, hierarchical cache, Mamba extra buffer | S01 | none (tests construct the wrapper directly) |
| 2 | `Req.init_next_round_input(tree_cache)` from `_get_new_batch_prefill_raw` 3946 | `match_prefix(MatchPrefixParams(req=req))` | none (reached because of S01) | `test_tp_disagreement_*`, `test_real_request_*`, `test_queued_abort_*`, `test_old_abort_handle_*` |
| 3 | `PrefillAdder.add_one_req` 1156 | `pending_prefix_tokens` (budget charge, `ignore_eos` gate), `prefill_checkpoint_limit` (chunk cap) | P02 | `test_final_checkpoint_*`, `test_host_prefix_charge_*`, `test_ordinary_short_prompt_*`, `test_pending_host_hit_ignore_eos_*` |
| 4 | `PrefillAdder.add_chunked_req` 950 | `prefill_checkpoint_limit` | P01 | `test_chunk_continuation_*` |
| 5 | `alloc_for_extend` 344 (from `ScheduleBatch.prepare_for_extend` 2677) | `prepare_prefix_for_extend`, `note_extend_allocation`, `restore_prefix_for_extend`, `rollback_prefix_for_extend` | M01 | 8 tests, see section 4 |
| 6 | `maybe_cache_unfinished_req` (common.py 156) from `Scheduler.stash_chunked_request` 3483 (`chunked=True`) and `process_batch_result_prefill` 383 | `cache_unfinished_req` -> `_capture` | none; B03 keeps `admit_request_into_staging` after it | `test_ordinary_short_prompt_*`, `test_real_request_force_miss_*` |
| 7 | `release_kv_cache` (common.py 254) | `before_release` (-> `_capture` if inserting), runtime `release`, `cache_finished_req`, runtime `after_release` | M03 | none (W2 lifecycle ledger test) |
| 8 | `PagedTokenToKVPoolAllocator.free_group_end` 328 | runtime `after_logical_flush` / `after_release(pending_release)` | M02 | none (W2 lifecycle ledger test) |
| 9 | `Scheduler._release_aborted_request` 3281 -> `BasePrefixCache.finish(ABORT)` | `release_aborted_request` | none | `test_queued_abort_*`, `test_old_abort_handle_*` |
| 10 | `Scheduler.flush_cache` 5013, then `req_to_token_pool.clear()` | `reset` | none; M04 keeps generations monotonic across the clear | `test_pool_flush_*` (needs M04), `test_queued_abort_*` |
| 11 | `SchedulerWeightUpdaterManager._observe_weight_load` 89 | `invalidate_model` (= `reset`) | B06 | `test_weight_load_attempt_*` |
| 12 | `Scheduler.release_host_resources` 1844 (shutdown) | `release_host_resources` (= `reset`) | none | none |

Inherited `ChunkCache` traits the flow relies on: `disable` is True (FCFS
policy, so only site 2 matches; `zero_match_result` is a no-op for chunk
caches, hence the cache checks `SGLANG_RADIX_FORCE_MISS` itself);
`is_chunk_cache()` is True; `supports_mamba()` is False, so
`release_kv_cache` frees the Mamba/PLE slot with the request.

## 2. Call order and ownership transfer

At most one request is admitted per prefill batch (S05/S06) and overlap is
disabled, so the steps below run in this order for each request.

1. **Match** (site 2). `release_aborted_request(handle)` drops any reader
   left by an earlier match of the same attempt. Bypass inputs (multimodal,
   embeddings, explicit positions, M-RoPE, session, beam, hidden states,
   `skip_radix_cache_insert`, bigram keys, force-miss) converge with `None`
   and return the plain chunk-cache miss. Otherwise `host.acquire(namespace,
   tokens, len(key))` returns the longest page64 snapshot within the
   logprob/logits limit; after convergence the reader moves to
   `matches[handle]` and the result is a `-1` marker tensor of the snapshot
   length with `cache_protected_len=0`. Owner: the snapshot is pinned by
   `matches`; no device resource is held. Upstream reads only
   `len(prefix_indices)` until step 3 (`prepare_for_extend` 2641-2670).
2. **Admission charge** (P02/P01). `add_one_req` adds the pending snapshot
   length to `memory_budget.total_offset`/`current_offset` (the restore
   allocates fresh logical pages; upstream charges only the suffix), caps
   `rem_chunk_tokens` at `prefill_checkpoint_limit` (the last page64
   boundary of an unaligned prompt, so a checkpoint lands there and the
   logits tail runs in a later chunk), skips the `ignore_eos` fast path while
   a host prefix is pending, and refunds the charge if the request is not in
   `can_run_list`. `add_chunked_req` applies the same cap to continuing
   chunks. A rejected request keeps its reader in `matches` until its next
   match (step 1) or abort (site 9).
3. **Prepare** (M01, after `alloc_req_slots`, before `prefix_tensors` is
   read). Creates `restoring[handle]` (reader, `fresh`, previous lengths),
   converges, then for a hit checks that the marker is intact and the row is
   fresh, allocates `snapshot.length` logical pages and replaces
   `req.prefix_indices` with them. Owner: the prefix pages and the request
   row now belong to the `restoring` record. Because `prefix_tensors` is
   read after this, `last_loc` and `write_cache_indices` see real pages.
4. **Note allocation** (M01, after the suffix KV allocation). The record
   takes `out_cache_loc[offset:offset+extend_len]` as `suffix`. With the hook
   present, a later `alloc_aux_to_lengths` failure does not free
   `out_cache_loc` itself; rollback does.
5. **Restore** (M01, end of `alloc_for_extend`, after indices and aux
   lengths are written). Rechecks the namespace, records the restore stream,
   `runtime.restore_prefix` acquires a fresh lease and copies raw K/V into
   the staging slot, the compressed index into the new logical pages, the
   pending C4 ring and rope rows, and Mamba/short-conv/n-gram state, then
   `note_prefix_basis(entry_id)`; `host_hit_length = host_loaded_length =
   length`; the reader is closed and `matches`/`restoring` are cleared.
   Owner: the request owns its pages, row and lease; the snapshot is held only
   by the cache entry.
   **Rollback** (M01 wrapper, any exception inside `_alloc_for_extend`):
   drain the restore stream, free `prefix`+`suffix` pages, the Mamba slot and
   the row, `runtime.release` + `after_release` for a lease that exists, then
   close the reader. Later chunks (`fresh` false) free only suffix pages
   beyond the already owned partial page and restore the previous lengths.
   If rollback itself fails, the record stays (error note "retained
   ownership") and blocks rematch, prepare and release until a rollback
   succeeds; `finish(ABORT)` does not drop it.
6. **Forward.** A cold first chunk acquires its lease in
   `runtime.begin_batch`; a restored request already holds one.
7. **Capture** (site 6 after a chunk or a non-finished final prefill; site 7
   `before_release` for a request finishing at prefill with insertion). Only
   in phase `prefill` and at `seq_len % 64 == 0`. `runtime.capture_prefix`
   reserves the full new host footprint (shared basis segments count once),
   waits for the producer stream, and copies the new segments plus the full
   pending/rope/recurrent state into host tensors. Owner: the reservation.
8. **Publish.** After convergence, `host.publish` adds the snapshot (epoch
   and page64 checks, footprint within the reservation), `note_prefix_basis`
   records the new entry, and the reservation is closed in `finally`.
   `process_batch_result_prefill` 383-385 calls `admit_request_into_staging`
   only after this, so the phase is still `prefill` during capture.
9. **Release** (M03 then M02). `before_release(req, is_insert and not
   skip_radix_cache_insert)` (raises if a restore record is still open) ->
   `runtime.release` (drain: terminal event and copy events) ->
   `cache_finished_req` (drops any reader, frees the KV row) ->
   over-allocation release -> Mamba slot free -> `req_to_token_pool.free` ->
   `mark_kv_released` -> `runtime.after_release(lease)`. Inside a free group
   (`process_batch_result_*` brackets with `free_group_begin/end`) the lease
   is queued and committed by `after_logical_flush` at `free_group_end`; only
   then is the physical slot reusable.
10. **Abort, flush, weights.** `finish(ABORT)` closes a matched reader (not a
    restoring one). `reset` refuses to run during a restore, closes all
    matched readers and advances the epoch (pinned entries retire until their
    readers close; stale reservations cannot publish). `invalidate_model`
    resets at the start of every weight-update attempt, with or without a
    regular flush; a request admitted under the old epoch never publishes.

## 3. TP convergence points

`_converge` all-gathers a signature over `tp_cpu_group` and raises on any
difference; every rank must reach it in the same order.

| Where | Signature | Protects |
|---|---|---|
| `match_prefix`, undrained restore | `(handle, namespace, "undrained restore")` | all ranks raise together |
| `match_prefix`, bypass or force-miss | `(handle, namespace, None)` | same miss decision |
| `match_prefix`, after `acquire` | `(handle, namespace, snapshot.signature or None)` | chunk shape and admission charge (steps 2-3) |
| `prepare_prefix_for_extend`, per request | `(handle, snapshot.signature or None)` | logical page allocation and restore |
| `_capture`, before publish | `(namespace, length, sha256(tokens), captured)` | identical publication (a rank-local reservation failure becomes an error) |

`snapshot.signature` is `(namespace, length, sha256(tokens))`. The gates
before the `_capture` convergence (namespace, request state, phase, page64)
are derived from TP-identical scheduling; a divergence there would hang the
collective instead of raising (fork behavior).

## 4. W2 coverage check

The prefix cache uses rows S01, P01, P02, M01, M03, M02, B06 (W2) and,
through the flush test, M04. Method: run `tests/prefix` on the pristine pin
with fork definitions of selected rows transplanted in-process (scratch, not
committed). With M01, P01, P02, B06 and M04 all 31 tests pass; with none,
the 15 tests below fail; removing any single row fails exactly its tests.
The same file passes 31/31 against `fork:` SGLang. The 15 tests are marked
`integration`; the other 16 pass on the pin.

| Row | Integration tests |
|---|---|
| M01 | `test_active_request_shares_only_its_restored_and_published_segments`, `test_evicted_active_basis_falls_back_to_copying_all_current_bytes`, `test_restore_recycled_slot_exact_raw_index_pending_and_ple`, `test_first_restore_failure_releases_pages_rows_and_snapshot_pin`, `test_later_chunk_failure_preserves_previous_lease_and_frees_new_pages`, `test_failed_restore_drain_retains_ownership_until_explicit_cleanup`, `test_old_active_request_cannot_publish_after_model_invalidation`, `test_missing_recurrent_or_pending_state_fails_closed` |
| P02 | `test_final_checkpoint_page_boundaries_for_normal_and_ignore_eos_controls`, `test_ordinary_short_prompt_publishes_and_reuses_actual_checkpoint`, `test_pending_host_hit_ignore_eos_can_continue_in_chunks`, `test_host_prefix_charge_changes_admission_and_rejection_refund` |
| P01 | `test_chunk_continuation_captures_aligned_boundary_before_logits_tail` |
| B06 | `test_weight_load_attempt_invalidates_even_without_regular_flush` |
| M04 | `test_pool_flush_preserves_new_leases_and_rejects_stale_generations` |

No call site outside these rows is needed: every release path (finish,
sampling-mask abort, running or chunked abort, retraction at
`schedule_batch.py` 2191) goes through `release_kv_cache`, and queued
aborts use the upstream `finish(ABORT)`.

Not exercised by the ported file: S01 (wrap point, guards, TP group), M03
(order inside `release_kv_cache`: `before_release` and capture before the
lease drain, `after_release` after the row free) and M02 (deferred commit
at `free_group_end`). These need W2's lifecycle tests; S01's guards
(non-`ChunkCache`, hierarchical cache, Mamba extra buffer) have no test.

## 5. Evaluation: `--radix-cache-backend` instead of S01

**Feasible at the pin.** `create_tree_cache` (registry.py 228) runs from
`kv_cache_builder.build_kv_cache` in `Scheduler.__init__` after
`init_model_worker`, whose `init_all_attention_backends` already created the
runtime and attached `kvcache.qsa_hisparse` (fork `qwen_sparse_attn_backend.py`
254-257). The wrapper needs the runtime and the TP CPU group, not the
coordinator, so running before `init_hisparse_coordinator` is not a blocker.
`ctx.params.tp_cache_group` equals `tp_cpu_group` because V3 rejects DP
attention. A factory would be `QSAHostPrefixCache(default_radix_cache_factory(ctx),
runtime, ctx.params.tp_cache_group)` with S01's guards on the default result
(exactly `ChunkCache`, no hierarchical cache, no Mamba extra buffer);
`create_tree_cache` does not catch factory errors, and an unregistered
`--radix-cache-backend` name stops startup.

**What it removes:** only S01 (an `after` hook, not a REPLACE) and its
dependency on `Scheduler.__init__` ordering. P01, P02, M01, M03, M02 and
B06 are call-site hooks and stay until U23.

**Risks:** (a) selection needs the flag (launcher, W7); without it, V3 with a
prefix budget silently runs without host prefixes unless plugin code (for
example runtime construction) checks `get_memory().radix_cache_backend`;
(b) a registry entry is not a HookRegistry hook, so the framework needs a new
declaration kind, fingerprints for `mem_cache.registry` and
`kv_cache_builder.build_kv_cache`, and a final check of the private
`_RADIX_CACHE_REGISTRY` entry; (c) `canary_manager.attach_radix_cache`
(627) would see the wrapper, while the fork attaches the bare `ChunkCache`
(differs only with the env-enabled KV canary); (d) registration must happen
only when `hisparse` is active, to keep "off" pristine.

**Next pin (#42354, from P0-C; main not read here).** Hybrid-SSM models
(Qwen4-Exp via `hybrid_gdn_config`) get `UnifiedRadixCache` in disabled mode
under `--disable-radix-cache`, so S01's `type(...) is ChunkCache` guard stops
startup; and `create_tree_cache` rejects any cache, registered ones
included, whose `supports_mamba()` is False. `QSAHostPrefixCache` inherits
False. At the pin, True changes upstream behavior in many places
(`release_kv_cache` stops freeing the Mamba slot, `alloc_req_slots` reserves
3 Mamba slots per request and evicts, `PrefillAdder.is_hybrid_ssm_cache`,
`PrefillBudget` uses `full_evictable_size`, `cow_mamba` matching, invariant
checks). So at the next pin the host cache should be a registered factory
that wraps upstream's default cache by delegation (it owns the Mamba
lifecycle and answers `supports_mamba()`), not a `ChunkCache` subclass that
copies `__dict__`; the restore must then write into the Mamba slot that cache
allocates. The renamed lifecycle (#40988, #41520, #42202, #42270) and open
#42825/#42923 (no `prefix_indices`, length-only `match_prefix`) also replace
the `-1` marker hand-off of steps 1-3.

**Verdict:** keep S01 for Phase 1 (fork insertion point, no new framework
surface); move to a registered delegating factory in the next pin cycle,
where it is required.

## 6. Track I design note (no code)

- **Request identity.** Computed once per attempt in `QSAHostPrefixCache`
  from I1's transported fields: image records `(artifact_key, order, start,
  stop, grid)` sorted by offset, and a page-cumulative M-RoPE digest
  `d[k] = H(d[k-1] || mrope_positions[:, 64(k-1):64k])`. `records_at(L)` is
  every image with `start < L` (a straddling image keeps its full offsets,
  grid and key; images at or after `L` are excluded, so different suffixes
  still share). `_namespace` keeps model, epoch, extra key, salt and LoRA;
  it stops returning None for supported image requests and still does for
  I1's bypass set.
- **`PrefixSnapshot`.** Two new fields, `images = records_at(length)` and
  `mrope_digest = d[length / 64]`. `runtime.capture_prefix` receives them
  from `_capture` (the dataclass is frozen). `signature` appends a digest of
  both, so the match, prepare and capture convergences cover image identity
  without other changes; `_capture`'s own tuple gets the same field. Keys
  count toward the footprint.
- **`HostPrefixCache.acquire`.** Gains an `identity` argument: a candidate
  of length `L` matches only if the token bytes are a prefix and `images ==
  identity.records_at(L)` and `mrope_digest == identity.d[L / 64]`; the
  longest match wins as now. `capture_prefix`'s ancestor lookup passes the
  same identity. Text requests pass no identity and match only snapshots
  without images, so text behavior is unchanged.
- **Inside-image boundaries.** Nothing in the cache depends on image
  boundaries: checkpoints already land on page64 lengths (chunk ends and
  `prefill_checkpoint_limit`), and the straddling record makes an in-image
  hit exact. What changes is upstream-facing (I3): `prefill_checkpoint_limit`
  and capture must accept image requests, the suffix embeddings start at
  `extend_prefix_len` inside an image, the n-gram PLE history covers pad
  tokens (already in the captured recurrent state), and the restore leaves
  ViT work only for images that reach past `L`.
