# Phase 6 host prefix cache design (v0.5.22)

Orchestrator decisions (2026-10-18), based on the research brief below
(verified against the v0.5.22 checkout; [I] marks inference to test):

1. **Design (a1) + H.** `QSAHostPrefixCache` subclasses `UnifiedRadixCache`
   and is built by `create_unified_radix_cache(ctx, cache_class=...)` from
   M05's AROUND hook on `default_radix_cache_factory` (no CLI change). A hit
   is reported as `host_hit_length` and accepted in `init_load_back`;
   `load_back_is_cache_owned()` is False; `prefix_device_indices` supplies
   the restored pages; the restore runs in `maybe_hand_to_session`; capture
   stays in `checkpoint` before `super()`. S01's wrap and M06 go away.
2. **Adder.** S06/P02 (and P01 if simpler) become a `PrefillAdder`
   subclass returned by an AFTER hook on `Scheduler.get_prefill_adder_class`.
   No copied upstream body.
3. **Release (M03).** `free_kv_row` override drains the lease first; an AFTER
   hook on `release_kv_cache` runs `after_release`. M03's own mamba free is
   removed (the disabled URC frees the slot). If the release-order ledger
   test cannot be met this way, keep M03 as one rebased REPLACE and report.
4. **Production behavior kept.** (i) Allocation rollback stays, as a small
   AROUND hook on `alloc_for_extend` (no REPLACE). (ii) `ignore_eos`
   requests with a pending hit reuse the prefix as in production (the
   adder's toggle, section 3.1.2). (iii) `cached_tokens` and every other
   response field equal production; a difference visible only in
   Prometheus tier labels is reported for DEVIATIONS (owner decision), not
   silently accepted. (iv) The QSA coordinator gets a no-op
   `allocate_spec_scratch`; B03's new `request_finished` gate (#37771)
   follows production's semantics for QSA leases.
5. **M04** is dropped if #43439 makes it redundant (cite it, P3 "drop").

---

## Research brief: QSA host prefix cache at SGLang v0.5.22: design brief

Scope: port `QSAHostPrefixCache` (plugin `src/sglang_qsa_hisparse/hisparse/prefix_cache.py`) and rows
S01, S06, P01, P02, M01, M03, M05 and M06 from pin `35f3c96ff4` to `v0.5.22`. Production config:
`--disable-radix-cache --page-size 64 --mamba-radix-cache-strategy extra_buffer --chunked-prefill-size 2048`,
Qwen4Exp (GDN/mamba + QSA attention), `QSATokenToKVPool`, `HybridReqToTokenPool`, TP=2.

Unless stated otherwise, paths are relative to
`/home/zyk/projects/interests/ai-video/qwen/.worktrees/sglang-v0.5.22/python/sglang/srt/`.
"Plugin" paths are relative to `qsa-hisparse-plugin/src/sglang_qsa_hisparse/`.
Tags: **[V]** means I read it in code. **[I]** means inference that a test should confirm.

---

## 0. Recommendation (TL;DR)

Recommended design: **(a)**. Make the host cache a subclass of `UnifiedRadixCache`, built through
upstream's own `create_unified_radix_cache(ctx, cache_class=QSAHostPrefixCache)`
(registry.py:149-153, 191).

- **Selection.** Either keep M05 as an AROUND hook, or register a `--radix-cache-backend` (the
  backend needs a CLI flag).
- **How a hit is reported.** Report it as an L2-style host hit (`host_hit_length`). Accept it in
  `init_load_back`, and set `load_back_is_cache_owned()` to False.
- **Where pages come from.** Supply the fresh logical pages from `prefix_device_indices(req)`.
  That is upstream's own "indices off the node path" hook, and `alloc_for_extend` calls it
  (allocation.py:356-364, 381).
- **Where the restore runs.** In `maybe_hand_to_session(req)`. Upstream calls it once per request
  at the end of `alloc_for_extend` (allocation.py:479), after the row is written and
  `kv_allocated_len` is set. That is the same point as today's `restore_prefix_for_extend`.
- **Upstream precedent.** `LMCacheUnifiedRadixCache` already uses the same seams:
  - it reports external hits through host fields;
  - it allocates request-private slots in `init_load_back`;
  - its `prefix_device_indices` appends those private slots past the node path.

  See `mem_cache/storage/lmcache/lmcache_unified_radix_cache.py:90-101, 103-175, 356-377`. The
  external linker does the same through `private_load_indices` (unified_radix_cache.py:3816-3824;
  `unified_cache/unified_cache_linker.py:244, 360-437`).
- **Mamba.** It needs no special case. The disabled URC owns the mamba slot lifecycle and answers
  `supports_mamba()` True. M06 and M03's mamba free disappear, and the restore writes into the
  slot that `alloc_req_slots` allocated.
- **PrefillAdder rows.** P01, P02 and S06 move into a plugin `PrefillAdder` subclass. An AFTER hook
  on the new `Scheduler.get_prefill_adder_class()` returns it (scheduler.py:2362-2363, used at
  4034; added by 7e3da1b784, #43870).

REPLACE count for the rows in scope:

| | Pin | v0.5.22 as recommended |
|---|---|---|
| REPLACE rows | 5 (S06, P02, M01, M03, M06) | 0 |
| Non-replace hooks | — | 1 or 2 new after/around hooks, plus M05 (around) and M02 (after) unchanged |
| Fallback | — | Keep M03 as one rebased REPLACE if the release-order decomposition (section 3.1.6) is judged too implicit |

---

## 1. Upstream v0.5.22: one new request under `--disable-radix-cache` on a hybrid-SSM model

### 1.1 Which cache gets built [V]

**Construction path:**
1. `Scheduler.__init__` runs `init_model_worker()` (scheduler.py:1098-1111). That creates the pools
   (1107) and then the attention backends (1110). The plugin's Q03 hook attaches
   `kvcache.qsa_hisparse` during the attention-backend step.
2. `kv_cache_builder.build_kv_cache` (scheduler.py:570) builds
   `CacheInitParams(disable=disable_radix_cache, ..., tp_cache_group=get_dp_tp_group().cpu_group, enable_mamba_extra_buffer=...)`
   (kv_cache_builder.py:331-363).
3. It then calls `create_tree_cache` (kv_cache_builder.py:387 → registry.py:229).
4. The tree cache is assigned at scheduler.py:605. `canary_manager.attach_radix_cache` runs at
   620-621, and `init_hisparse_coordinator()` (S01/S02) runs at 624.

**Factory choice in `create_tree_cache`:**
- If `--radix-cache-backend` is set, its registered factory is used (registry.py:231-241).
- Otherwise `default_radix_cache_factory` runs. With `ctx.disable_radix_cache` set and a model that
  is not pure SWA, it returns `create_unified_radix_cache(ctx)` (registry.py:91-99).
- That builds `tree_components = (FULL, MAMBA)` for hybrid SSM (registry.py:165-169) and returns
  `(cache_class or UnifiedRadixCache)(params)` (191).
- **`create_unified_radix_cache` takes an optional `cache_class`.** This is the key seam for design (a).

**The cache it builds:**
- `UnifiedRadixCache.disable = params.disable` (unified_radix_cache.py:182).
- Components are constructed with `self` as their cache back-reference (201). A
  `StreamingSession(self)` is created (268), and `__init__` ends with `self.reset()` (321).
- `MambaComponent` skips its `page_size == 1` assert when the cache is disabled
  (unified_cache/components/mamba.py:73).
- `supports_mamba()` returns True, because MAMBA is in the tree components (unified_radix_cache.py:3638).
  So the hybrid-SSM guard in `create_tree_cache` passes (registry.py:290-295). Without the plugin,
  M06 has nothing to override.

**Policy and flags under a disabled cache:**
- The mamba extra buffer is off whenever the radix cache is disabled:
  `mamba_extra_buffer_of = disable_radix_cache is False and strategy in (...)`
  (arg_groups/model_override_base.py:260-265). Production's `--mamba-radix-cache-strategy extra_buffer`
  is therefore inert. `mamba_initial_tracking_slots == 0` (allocation_sizing.py:15-16), so there is
  one mamba slot per request.
- The schedule policy is forced to FCFS (managers/schedule_policy.py:328-333).
- The #41593 waiting-prefix refresh is skipped for non-sharing caches (schedule_policy.py:286-287).

### 1.2 Match (`match_kv_cache`) [V]

1. `_get_new_batch_prefill_raw` calls `req.init_next_round_input(self.tree_cache)` (scheduler.py:4136).
2. That passes `cow_mamba = tree_cache.supports_mamba()`, which is True, into `match_kv_cache`
   (managers/schedule_batch.py:1663-1672).
3. `match_kv_cache` (mem_cache/common.py:203-250) calls `tree_cache.match_prefix(MatchPrefixParams(key=RadixKey(..., limit=max_prefix_len), cow_mamba, req))`.
4. In a disabled URC, `_match_tree` returns `tree_core.empty_match_result` before any component
   finalizer or linker runs (unified_radix_cache.py:586-588). That result has
   `device_prefix_len=0`, root NodeIds for `last_device/last_host/best_match`, and
   `cache_protected_len=None` (unified_cache/unified_tree_core.py:535-541).
5. `SGLANG_RADIX_FORCE_MISS` goes through `zero_match_result`, which passes a non-sharing cache's
   result through unchanged (base_prefix_cache.py:320-325). A cache that wants force-miss must
   check the flag itself, as the plugin already does.

`match_kv_cache` writes these fields onto the `Req`:
- `prefix_len = device_prefix_len` (common.py:222).
- `last_node`, `last_host_node`, `best_match_node`, `host_hit_length`, `swa_host_hit_length` and
  `mamba_host_hit_length` (223-240).
- `num_matched_prefix_tokens = min(prefix_len + host_hit_length, max_len)` (243).
- `swa_branching_seqlen` and `mamba_branching_seqlen` (244-247).
- `kv.cache_protected_len`, but only if the result's value is not None (247-248). When it is None,
  `init_next_round_input` sets `kv.cache_protected_len = prefix_len` (schedule_batch.py:1676-1677).

`MatchResult` now carries `device_prefix_len: int` instead of a `device_indices` tensor
(base_prefix_cache.py:269-317, #42923 5a4d01a3ae). `Req.prefix_indices` is gone (#42825 5349d3e9d2).
The `Req` comment states the new contract: "Tokens [0, prefix_len) are already in the row, or,
before allocation, on the path to last_node" (schedule_batch.py:1185-1189).

### 1.3 Admission (PrefillAdder) [V]

`add_one_req` (schedule_policy.py:1336) proceeds as follows:

1. **Request cap.** `prefill_max_requests` gate (1339). The value is stored at 707 and read only here.
2. **ignore_eos fast path.** With `ignore_eos` set and `tree_cache.disable` True, it calls
   `add_one_req_ignore_eos` (1342-1343 → 1192). That path ignores host hits, never takes `req.lock`,
   and `assert req.prefix_len == 0` in its chunked branch (1318).
3. **Budget.**
   - `total_tokens = (len(fill) - prefix_len) + max_new + overhead + mamba_gap_reserve(include_host_hit=True) + request_slot_reserve`
     (1352-1371). `request_slot_reserve` was added by #37771.
   - `_lock_node(req.last_node)` (1372) is a no-op on a disabled URC: `inc_lock_ref`/`dec_lock_ref`
     return empty results (unified_radix_cache.py:993-1008).
4. **Shape selection.** `_select_prefill_admission(host_hit_length=req.host_hit_length)` sets
   `prefix_len = req.prefix_len + host_hit_length` (1514).
5. **Host load-back.** This step runs only if `needs_host_load_back()` (1396):
   - `full_load_tokens = host_hit_length`. Only `SharedSWAPrefillBudget` walks the FULL transfer
     spec (1405-1420; prefill_budget.py:66, 321). The QSA paged allocator uses the base
     `PrefillBudget` (allocator/base.py:75), whose `prepare_load_back` returns True
     (prefill_budget.py:79-89).
   - Then `init_load_back(InitLoadBackParams(best_match_node, host_hit_length, req))`, which now
     returns `(loaded_len, node)` (#43023 89b5935793).
   - Then `req.host_loaded_length, req.last_node = loaded` (1449) and
     `req.prefix_len += req.host_loaded_length` (1488).
   - `req.kv.cache_protected_len = req.prefix_len` is set only if
     `tree_cache.load_back_is_cache_owned()` (1489-1490).
6. **Commit.** `_commit_prefill_admission` (1586-1607):
   - asserts `prefix_len == admission.prefix_len`;
   - sets `extend_end = prefix_len + extend_len` (1590);
   - sets `req.lock = tree_cache.lock(req.last_node)` (1591);
   - reserves extend + max_new (`_update_prefill_budget`);
   - attributes the hit to device, host or storage tiers through `materialized_host_hit_len()`
     (schedule_batch.py:1473-1474; schedule_policy.py:910-950).

Chunked continuation goes through `add_chunked_req` (1092-1173):
- `chunked_req_limit` is still applied after the delayer (1116-1118). The scheduler sets it
  immediately before the call (scheduler.py:4054-4061).
- It sets `extend_end = prefix_len + new_len` (1155).

`extend_range` became `extend_end` plus the property `extend_len = extend_end - prefix_len`
(schedule_batch.py:1076, 1598-1599; #42822 1066decce6).

### 1.4 Allocation (`alloc_for_extend`) [V]

`ScheduleBatch.prepare_for_extend` (schedule_batch.py:2730) builds
`prefix_lens = [r.prefix_len]`, `extend_lens = [r.extend_len]` and `seq_lens = [r.extend_end]`
(2740-2744), then calls `alloc_for_extend` (2795). Inside `alloc_for_extend` (mem_cache/allocation.py:367-481):

1. **Prefix indices first.** `prefix_tensors = [_prefix_kv_indices(batch, r) ...]` is computed at
   the top, before any slot allocation (381). `_prefix_kv_indices` (356-364) works like this:
   - if the request already holds a row (a later chunk, or a session record), it returns
     `row[:prefix_len]`;
   - if `prefix_len == 0`, it returns an empty tensor;
   - otherwise it returns `batch.tree_cache.prefix_device_indices(req)`. This seam was added in
     #42825.

   Implementations of `prefix_device_indices`:
   - The base version reads the path to `req.last_node` (base_prefix_cache.py:541-546).
   - URC collects FULL indices from `last_node` up to the root, plus the linker's
     `private_load_indices[rid]` (unified_radix_cache.py:3816-3824).
   - LMCache's URC subclass appends privately loaded slots (lmcache_unified_radix_cache.py:90-101).
2. **Request and mamba slots.** `alloc_req_slots` (320-342) runs `ensure_mamba_capacity` (291-317).
   That calls `evict_for_alloc` only if `supports_mamba()`, and a disabled cache makes it a no-op
   (unified_radix_cache.py:654). Then `HybridReqToTokenPool.alloc` (mem_cache/memory_pool.py:1586-1640)
   gives each request without one a mamba slot and sets `mamba_needs_clear=True` (1610-1613). It
   adds ping-pong buffers only with the extra buffer enabled.
3. **New HiSparse call.** `batch.hisparse_coordinator.allocate_spec_scratch(req_pool_indices)` runs
   whenever the batch has a coordinator (408-409, #37771). The scheduler now sets
   `new_batch.hisparse_coordinator = self.hisparse_coordinator` on every prefill batch (scheduler.py:4225).
4. **Paged allocation.** `last_loc` is taken from `prefix_tensors[i][-1:]` (428-431), then
   `alloc_paged_token_slots_extend` runs (432-442).
5. **Row write.** `write_cache_indices` writes `prefix_tensors` into `row[0:prefix_len]` and
   `out_cache_loc` into `[prefix_len:seq_len]` (445-457). The Triton path takes `data_ptr()` of
   every prefix tensor, so these must be int64 device tensors (68-86).
6. `alloc_aux_to_lengths` runs (458-465).
7. **Per-request tail.** For each request it sets `kv_allocated_len = kv_committed_len = seq_len`
   and then calls `batch.tree_cache.maybe_hand_to_session(req)` (474-479). That call was added by
   #42537 c01d59eb90. URC forwards it to `StreamingSession.take`, which is a no-op for
   non-streaming requests (session/streaming_session.py:200-205).

`alloc_for_extend` has no try/except except the `alloc_aux_to_lengths` free (458-465), and
`prepare_for_extend` adds none. Any scheduler exception sends SIGQUIT to the parent
(scheduler.py:6136-6139), so allocation failures are fatal upstream.

### 1.5 Chunk end [V]

- `checkpoint_kv_cache` (common.py:253-262) calls `tree_cache.checkpoint(req, up_to=req.extend_end)`
  unless `skip_radix_cache_insert`, then sets `req.prefix_len = req.extend_end`. This is #42824's
  "progress as prefix_len".
- URC `checkpoint` (unified_radix_cache.py:1075-1078) calls `session.try_checkpoint`, which returns
  False for non-streaming requests, then `checkpoint_into_tree`. For a disabled cache with an
  unfinished request, that just returns (1088-1092).
- On the next round, `chunked_req.init_next_round_input()` is called without a tree cache
  (scheduler.py:4054), so no match happens. Allocation reads the prefix back from the row
  (allocation.py:358-361).

### 1.6 Release [V]

`release_kv_cache(req, tree_cache, *, checkpoint: bool)` (common.py:378-422) runs these steps in order:

1. **No row held.** If the request holds no row, it asserts `supports_mamba()` and frees an early
   mamba slot (382-392).
2. **Claim.** `tree_cache.claim_kv_row(req)` (394). In URC this is the PP-prefetch ticket plus the
   streaming claim (unified_radix_cache.py:1057-1066).
3. **Final checkpoint.** If `checkpoint and not req.skip_radix_cache_insert`:
   - it asserts `req.finished() or not supports_mamba()` (404-406);
   - calls `refresh_fill_ids()`;
   - calls `tree_cache.checkpoint(req, up_to=owned_kv_len)`.

   In a disabled URC with a finished request, that runs every component's
   `cleanup_after_caching_req(req, is_finished=True)` and returns (1088-1092).
   `MambaComponent.cleanup_after_caching_req` then frees the slot with `pool.free_mamba_cache(req)`
   (mamba.py:644-687). With the extra buffer it keeps no ping-pong slot, because nothing was inserted.
4. **Row free.** `tree_cache.free_kv_row(req.kv, [(cache_protected_len, owned_kv_len)])` (411;
   base_prefix_cache.py:487-502).
5. **Unlock.** `tree_cache.unlock(req.lock); req.lock = None` (412-413). This replaced `unpin(req)`
   (#42686), and `BasePrefixCache.unpin` no longer exists. The disabled URC's `dec_lock_ref`
   returns before validating the receipt (unified_radix_cache.py:1006-1007). #41452's anchored
   receipts (base_prefix_cache.py:172-240) are therefore inert here.
6. `_release_overallocated_kv_indices` (414-416, 425-454) frees the tail through `free_kv_row`.
7. **on_release.** `tree_cache.on_release(req, checkpointed=checkpoint)` (417). The keyword was
   renamed from `inserted=` by #42270. URC's version returns if `checkpointed`; otherwise it runs
   the component cleanup, which frees the mamba slot (unified_radix_cache.py:1068-1073).
8. `req_to_token_pool.free(req)` and `req.kv.mark_kv_released()` (421-422).

Neither step 7 nor step 8 frees the mamba slot for trees without mamba support any more; since
#42354 the mamba-capable tree owns that free.

### 1.7 What a disabled UnifiedRadixCache does [V]

| Operation | Behavior | Location |
|---|---|---|
| `match_prefix` | Returns the empty match at the root | 586-588 |
| `insert` | Returns `InsertResult(prefix_len=0)` | 618-620 |
| `checkpoint` | Inserts nothing and leaves `cache_protected_len` and `last_node` unchanged. For a finished request it only runs component cleanup, which frees the mamba slot | 1088-1092 |
| `evict`, `evict_for_alloc` | No-op | 654, 748 |
| Lock refs (normal and host) | No-op | 993-1052 |
| `touch_prefix` | No-op | 604-609 |
| `supports_prefix_sharing()` | False | 611-612 |
| Tree core | Built without KV events or an eviction policy | 223-232, #43253 |

---

## 2. Plugin reads that are gone or changed meaning

| Plugin use (file:line) | v0.5.22 status | Replacement |
|---|---|---|
| `req.prefix_indices`: prefix_cache.py:130, 151, 155-156, 167, 259; S04 copy scheduler.py:275-277; S06 copy 479; P02 copy 1425, 1524, 1553-1554 | **Removed** (#42825) | Length: `req.prefix_len` (schedule_batch.py:1187). Indices before allocation: `tree_cache.prefix_device_indices(req)` (base_prefix_cache.py:541). Indices after allocation: `req_to_token[req_pool_idx, :prefix_len]` (allocation.py:358-361). A placeholder tensor can no longer be carried on the Req. |
| `req.extend_range.length` / `.end`: prefix_cache.py:173; S04 copy 274-277; S06 copy 698, 707 | **Removed** (#42822) | `req.extend_len` (property, schedule_batch.py:1598-1599) and `req.extend_end` (1076). Upstream's stash check is now `extend_end > prefix_len` (scheduler.py:3792). |
| `MatchResult(device_indices=...)`: prefix_cache.py:103-109 | **Field removed** (#42923); the call raises TypeError | `device_prefix_len: int` (base_prefix_cache.py:305) plus a node, or `host_hit_length` plus `init_load_back`. |
| `init_load_back` returning `(indices, node)`: P02 copy 1504-1514, 1553-1554 | **Changed** (#43023) | Returns `Optional[(loaded_len:int, node)]` (base_prefix_cache.py:589-596). The caller does `prefix_len += loaded_len` (schedule_policy.py:1449, 1488). |
| `req.host_hit_length` / `req.host_loaded_length`: prefix_cache.py:196; P02 copy | **Still exist** (schedule_batch.py:1193, 1211) | `host_hit_length` now comes only from `MatchResult` in `match_kv_cache` (common.py:229-240). `host_loaded_length` is assigned from `init_load_back` at admission (schedule_policy.py:1449) and feeds tier metrics through `materialized_host_hit_len()` (schedule_batch.py:1473). The plugin's after-the-fact assignment in `restore_prefix_for_extend` comes after the metrics were already taken. Under design H, upstream sets both fields itself. |
| `cache_protected_len`: `MatchResult(cache_protected_len=0)` at prefix_cache.py:108; `req.kv.cache_protected_len = 0` at 168; M03 copy 339 | **Same meaning**, more writers | `MatchResult.cache_protected_len` stays Optional (base_prefix_cache.py:314). It is written by `match_kv_cache` (common.py:247-248); when None, `init_next_round_input` writes `prefix_len` (schedule_batch.py:1676-1677). PrefillAdder writes `prefix_len` after a load-back iff `load_back_is_cache_owned()` (schedule_policy.py:1489-1490). Release frees `[cache_protected_len, owned)` (common.py:411). |
| `last_node`, `last_host_node`, `best_match_node`: returned as None at prefix_cache.py:105-107; P02 copy 1437, 1466, 1513 | **Same fields**; URC stores NodeIds | The root is `root_node_handle()` (unified_radix_cache.py:3829). `last_node` is also where `prefix_device_indices` reads indices from, and it is what `lock`/`unlock` receive (schedule_policy.py:1591; common.py:412). Return the root ids from the empty match, not None. |
| `tree_cache.unpin(req)`: M03 copy lifecycle.py:340 | **Removed** (#42686) | `tree_cache.unlock(req.lock); req.lock = None` (common.py:412-413). |
| `on_release(req, *, inserted)`: prefix_cache.py:335-337; M03 copy 344 | **Renamed** (#42270) | `on_release(req, *, checkpointed)` (base_prefix_cache.py:557; called at common.py:417). Without the rename, upstream's call raises TypeError against the override. |
| `release_kv_cache(req, tree_cache, is_insert=True)`: M03 signature; B02-B05 call sites | **Changed** (#42270) | `release_kv_cache(req, tree_cache, *, checkpoint: bool)`, with `checkpoint` required and keyword-only (common.py:378). |
| `from sglang.srt.mem_cache.chunk_cache import ChunkCache`: prefix_cache.py:9; S01; M05 | **Module deleted** (#43201) | No replacement. The disabled `UnifiedRadixCache`, or `PureSWARadixCache` for pure-SWA models, serves `--disable-radix-cache` (registry.py:91-99). |
| `req.cache_request_handle` | Unchanged | `CacheRequestHandle(rid, attempt_id)` (schedule_batch.py:1059, 1430-1434). |
| `req.kv.kv_allocated_len` / `kv_committed_len` | Unchanged | Set to `seq_len` in the `alloc_for_extend` loop (allocation.py:474-477). `owned_kv_len()` is at schedule_batch.py:1491. |
| `match_prefix` hook | Unchanged signature | `match_prefix(MatchPrefixParams(key, cow_mamba, req))`. `RadixKey` (`limit`, `is_bigram`) is unchanged from the pin (radix_cache.py:58-113). URC's version is decorated with `rank_consensus` (debug only, utils/rank_consensus_checker.py:84-88). |
| `checkpoint(req, *, up_to, **kw)`, `claim_kv_row`, `free_kv_row(kv, ranges)` | Unchanged signatures | — |
| `req.mamba_lazy_is_insert` (B05 copy 1246, 1294) | **Renamed** | `req.mamba_lazy_checkpoint` (batch_result_processor.py:1404, 1448). |
| `batch.prefix_lens` / `batch.extend_lens` | Same fields | Built from `r.prefix_len` / `r.extend_len` (schedule_batch.py:2743-2744). |

New things the plugin must answer:
- `batch.hisparse_coordinator.allocate_spec_scratch(...)` on every prefill batch (allocation.py:408-409,
  scheduler.py:4225). `QSAHiSparseCoordinator` (plugin `hisparse/coordinator.py`) lacks it, so add a
  no-op. Retracted prefill batches also call `retract_req` (schedule_batch.py:2253-2254); that one
  exists.
- A force-miss under the disabled cache must still be checked inside the plugin (see 1.2).

---

## 3. Designs

### 3.0 Available seams (all [V])

| Need | Seam at v0.5.22 | Notes |
|---|---|---|
| Report a matched length without device nodes | (H) `MatchResult.host_hit_length` with `init_load_back` | LMCache precedent: lmcache_unified_radix_cache.py:146-160 sets `host_hit_length` and keeps `best_match_node = last_device_node`. |
| | (D) `MatchResult.device_prefix_len` with `last_device_node = root` | `prefix_device_indices` must then supply the indices. |
| Accept or decline the hit at admission | `init_load_back(params) -> (loaded_len, node)` | schedule_policy.py:1439-1449. Returning `(L, req.last_node)` makes `prefix_len += L`. Returning `0` re-selects the shape without the hit (1476-1487). |
| Keep the restored prefix request-owned | `load_back_is_cache_owned()` returns False | Leaves `cache_protected_len` at 0 (schedule_policy.py:1489-1490), so `release_kv_cache` frees the prefix pages (common.py:411). Same semantics as the linker (unified_radix_cache.py:3482-3485). |
| Supply prefix KV indices "off the node path" | `prefix_device_indices(req)` override | The only allocation-time source for a new request with `prefix_len > 0` (allocation.py:362-364). It runs before `alloc_req_slots`. It must return int64 device tensors (`last_loc`, `data_ptr()`). A synthetic node is **not** viable: a disabled URC never inserts (618-620), and URC reads indices from `tree_core` node values (unified_tree_core.py:1073-1090). |
| Restore after the row and mamba slot exist | `maybe_hand_to_session(req)` override | Called per request after `write_cache_indices`, `alloc_aux_to_lengths` and the `kv_allocated_len` update (allocation.py:474-479). That is the same state as the pin's `restore_prefix_for_extend`. QSA admits one request per prefill (S05/S06), so per-request versus per-batch does not matter. |
| Capture at checkpoints | `checkpoint` override (capture, then `super()`) | Capture must run before `super().checkpoint`, because the disabled URC frees the mamba slot inside `checkpoint` when the request is finished (1088-1092). |
| Refuse release while a restore is pending | `claim_kv_row` override | The first tree call in every release path (common.py:394). |
| Cache class construction | `create_unified_radix_cache(ctx, cache_class=...)` | registry.py:149-153, 191. No `__dict__` copying is needed. |
| Cache selection | `register_radix_cache_backend(name, factory)` plus `--radix-cache-backend name` (registry.py:58-72, 231-241), or an AROUND hook on `default_radix_cache_factory` (M05) | The flag is rejected only together with `--enable-int8-mamba-checkpoint` (arg_groups/mamba_hook.py:89-94) and KV sharding (kv_shard_hook.py:76-81). |
| Adder behavior (S06/P01/P02) | `Scheduler.get_prefill_adder_class()` (scheduler.py:2362, used at 4034) | An AFTER hook replaces the result when it is non-None (plugins/hook_registry.py:377-379). `prefill_max_requests` is stored at schedule_policy.py:707 and read only at 1339. `dllm/mixin/scheduler.py` and the disaggregation code build `PrefillAdder` directly and are unaffected, which is the same scope as S06. |

### 3.1 Design (a): plugin `UnifiedRadixCache` subclass (recommended)

#### 3.1.1 Construction

`QSAHostPrefixCache(UnifiedRadixCache)` is built by
`create_unified_radix_cache(ctx, cache_class=QSAHostPrefixCache)`.

There are two ways to select it:
- **(a1) M05 kept as an AROUND hook** on `default_radix_cache_factory`. It applies when
  `ctx.disable_radix_cache and qsa_private_host_prefix_active(ctx.params)`; otherwise the original
  runs.
- **(a2) A registered backend**, for example `qsa-host-prefix`, whose factory falls back to
  `default_radix_cache_factory(ctx)` when the predicate is False.

(a2) needs a launcher flag and a new declaration kind in the plugin framework; the plugin's own
`docs/prefix-cache.md` §5 lists those risks. (a1) changes nothing at the CLI. **Prefer (a1).**

At factory time the runtime already exists, because pools and attention backends are initialized
in `init_model_worker` (scheduler.py:1107-1110) before `build_kv_cache` (570). So
`params.token_to_kv_pool_allocator.get_kvcache().qsa_hisparse.prefix_cache` is available. [V for
the order; I that Q03 attaches during `init_all_attention_backends`, which the plugin doc also
states for the pin.]

The TP group comes from URC's own `self.tp_group = params.tp_cache_group`, the dp-tp CPU group
(unified_radix_cache.py:270; kv_cache_builder.py:344). V3 rejects DP attention, so this equals the
scheduler's `tp_cpu_group` [I, per plugin doc §5].

S01's guards move into the factory: no hierarchical cache, and `params.enable_mamba_extra_buffer`
False. The S01 wrap is deleted.

**Init order:** `URC.__init__` calls `self.reset()` (321). The plugin's `reset` override must
tolerate being called before the plugin attributes exist (set them before `super().__init__()`) and
must call `super().reset()`.

#### 3.1.2 Hit flow, design H (recommended over D)

1. **`match_prefix(params)`:** `result = super().match_prefix(params)`, which is the empty root
   match. Keep today's logic unchanged: namespace, bypass set, bigram, own FORCE_MISS check,
   `host.acquire`, TP convergence. On a hit, store the reader in `self.matches[handle]` and return
   `result._replace(host_hit_length=L)`. Leave `mamba_host_hit_length` at 0, because the restore
   writes into the request's own slot; then `_mamba_gap_budget_for_req` adds nothing. Optionally set
   `full_kv_hit_length=L`; outside the caches it is only read in the `rank_consensus` selector [V].
2. **`init_load_back(params)`:** if a reader is pending for `params.req`, mark it accepted and return
   `(L, params.req.last_node)`. Otherwise return `(0, last_node)`.
3. **`load_back_is_cache_owned()`:** return False.
4. **`prefix_device_indices(req)`:**
   - assert the request is accepted, `req.prefix_len == L` and not `holds_kv`;
   - `pages = token_to_kv_pool_allocator.alloc(L)` (raise if None, as today);
   - record the pages for rollback and return `pages.to(torch.int64)` on the device;
   - make it idempotent per request.

   This replaces `prepare_prefix_for_extend`. The pin allocated after `alloc_req_slots`; upstream
   now asks for prefix indices before it. Both happen before suffix allocation.
5. **`maybe_hand_to_session(req)`:**
   - call `super()`, which is the streaming no-op;
   - run `_converge((handle, signature or None))` for every request, matching the pin's
     per-request convergence in `prepare_prefix_for_extend`;
   - if the request is restoring: `runtime.restore_prefix(req, snapshot)`, `note_prefix_basis`,
     `host.reused_tokens += L`, close the reader, then pop `matches`/`restoring`.

   Upstream has already set `host_hit_length` and `host_loaded_length`.
6. **`checkpoint`:** call `_capture(req)`, then `super().checkpoint(...)`. The page-64 rule and the
   "prefill phase with a live lease" rule are unchanged.
7. **`prefill_checkpoint_limit(req)` / `pending_prefix_tokens(req)`:** use
   `req.prefix_len + (L if pending-and-not-accepted else 0)` instead of `len(req.prefix_indices)`.
8. **`claim_kv_row`, `on_release(req, *, checkpointed)`, `release_aborted_request`:** call `super()`
   in each. URC's `release_aborted_request` is harmless when disabled (2625-2663).
   `release_host_resources` and `invalidate_model` work as today plus `super()`.

**Budget equivalence of H (P02's charge).** At admission, `prefix_len` is 0, so `total_tokens`
already counts the L prefix tokens.

- Pin: `(F-L+X) < avail-(off+L)`.
- H: `(F+X) < avail-off`.

The two admission tests are algebraically identical. After a commit, upstream reserves only
`extend + max_new`. To reproduce the pin's post-admission offsets exactly (the pin left `charge = L`
in them), the adder subclass adds L to both offsets after a successful admission and returns
`self.budget_state()`. This only changes the verdict that latches `batch_is_full`.

**Metrics.** `_account_prefill_cache_admission` now books L as host-tier hits through
`materialized_host_hit_len()`. The pin booked them as device-tier, because `host_hit_length` was
0 at admission.

**ignore_eos.** Upstream sends `ignore_eos` requests on a disabled cache down the fast path
(schedule_policy.py:1342).
- Under H that path is a clean miss: `prefix_len` 0, no `init_load_back`, no allocation or restore.
  The reader is dropped at the next match or at `on_release` / `finish(ABORT)`.
- To keep the pin behavior (P02 routes ignore_eos hits through the normal path), the adder subclass
  can temporarily clear `req.sampling_params.ignore_eos` around `super().add_one_req(...)` when a
  host prefix is pending. That predicate is the only read of the candidate's `ignore_eos` within
  `add_one_req`'s extent [V]. The other reads are line 756 (running requests, in `__init__`) and
  1212 (inside `add_one_req_ignore_eos`).
- Choose one of the two. Both keep exact-match semantics.

**Design D instead (`device_prefix_len=L`).** This is simpler at match time but strictly worse:
- the budget lacks L, so P02's charge must be re-implemented;
- the ignore_eos fast path must be diverted, otherwise `assert req.prefix_len == 0` fires at
  schedule_policy.py:1318 in its chunked branch, and its non-chunked branch under-budgets L;
- metrics book the hit as a device hit.

#### 3.1.3 PrefillAdder subclass (replaces S06, P01, P02)

An AFTER hook on `Scheduler.get_prefill_adder_class` returns `QSAPrefillAdder` when
`hisparse_coordinator.uses_qsa_hisparse_leases`. The host cache exists only in P2 mode, so it
implies leases (plugin `hisparse/runtime.py:99, 167-184`).

| Method | Behavior | Replaces |
|---|---|---|
| `__init__` | `super()`, then `self.prefill_max_requests = 1` | S06 |
| `add_chunked_req` | Lower `chunked_req_limit` to `prefill_checkpoint_limit`, then `super()`. P01's equivalence argument still holds: the limit is read at 1116-1118, after the delayer, and set at scheduler.py:4055. | P01 (or keep P01 as its existing BEFORE hook) |
| `add_one_req` | Cap `rem_chunk_tokens` at the checkpoint limit; optional ignore_eos toggle; `super()`; restore `rem_chunk_tokens`; optional post-admission L charge | P02 (no copied body) |

#### 3.1.4 Mamba / PLE

- `alloc_req_slots` allocates the request's slot and sets `mamba_needs_clear=True`
  (memory_pool.py:1610-1613). The restore clears it and `mamba_cow_src_index`, as `restore_prefix`
  does today (plugin `hisparse/runtime.py:1220-1221`).
- PLE side state is addressed by the mamba slot ("registered on that pool, so every slot lifecycle
  event carries them along", mem_cache/ple_state_pool.py:1-3), so restoring into
  `req.kv.mamba_pool_idx` covers it. [I: the plugin's `_mamba_sibling_tensors`/`_restore_mamba`
  already include those tensors.]
- The disabled URC frees the slot in `checkpoint` (finished request) or `on_release` (not
  checkpointed) (1068-1073, 1088-1092; mamba.py:644-687). M03's explicit `free_mamba_cache` must go.
  Otherwise the second free trips "double free?" (memory_pool.py:1829).
- `cow_mamba=True` arrives in the match params, but the disabled match never stages a COW (586-588).
- `release_kv_cache` with `checkpoint=True` on an unfinished request now asserts (common.py:404-406).
  That is upstream's own contract for disabled hybrid-SSM.
- No extra buffer and no ping-pong (1.1).
- `translate_mamba_indices` is the identity for the static `HybridReqToTokenPool`. A unified mamba
  pool would need translation before the restore indexes physical slots (memory_pool.py:1668).

#### 3.1.5 Rollback (M01's wrapper)

- Upstream allocation failures are fatal (1.4), and none of the plugin's hook points needs a copied
  body.
- Option 1: keep a small AROUND hook on `alloc_for_extend` that, on exception, frees the
  plugin-owned prefix pages and reader and drains the lease, then re-raises.
- Option 2: drop it. This follows the "proportionate defenses" note, since the process exits on any
  scheduler exception (scheduler.py:6136-6139).
- `note_extend_allocation` is no longer needed. If a rollback wants the suffix, read it from the row.

#### 3.1.6 Release ordering (M03)

The pin's order is:
1. `claim_kv_row`;
2. `checkpoint` (capture with a live lease);
3. `qsa.release` (drain; fails closed);
4. `free_kv_row`;
5. `unlock`/overalloc;
6. `on_release`;
7. mamba free;
8. `req_to_token_pool.free`;
9. `mark_kv_released`;
10. `after_release(lease)`.

Without a REPLACE:
- `claim_kv_row` records the request being released.
- The cache's `free_kv_row` override drains first: `lease = runtime.release(req_pool_idx, rid)`, then
  `super()`. `free_kv_row` is the first logical free after `checkpoint` (common.py:411). Its other
  callers are streaming sessions and PD decode offload, which are out of scope.
- An AFTER hook on `mem_cache.common.release_kv_cache` calls `runtime.after_release(lease)` after
  `req_to_token_pool.free` and `mark_kv_released`. HookRegistry propagation rebinds the imported
  names, as M03 relies on today.
- That reproduces the pin's order. The one difference: the mamba free now happens inside
  `checkpoint` (before the drain) for finished requests. It still happens after the drain on the
  `checkpoint=False` path. [I] This is harmless, because the drain is CPU-synchronous
  (plugin `hisparse/slots.py:111-127`) and nothing allocates before `release_kv_cache` returns. It
  only weakens the "retain backing on a failed drain" property for the mamba slot.
- **Fallback:** keep M03 as one rebased REPLACE (`*, checkpoint`; `unlock(req.lock)`;
  `on_release(checkpointed=)`; no mamba free).

#### 3.1.7 Rows

| Row | Change |
|---|---|
| S01 | after → removed |
| S06 | REPLACE → after on `get_prefill_adder_class` |
| P01 | before → subclass method (or kept as is) |
| P02 | REPLACE → subclass method |
| M01 | REPLACE → none, or a small AROUND |
| M03 | REPLACE → `free_kv_row` override plus an AFTER hook (fallback: 1 REPLACE) |
| M05 | around kept (now `create_unified_radix_cache(..., cache_class=...)`) |
| M06 | REPLACE → removed |
| M04 | redundant (section 4) |

Net: **minus 5 REPLACE rows, plus 1 to 3 after/around hooks**.

#### 3.1.8 Risks

- `UnifiedRadixCache` is a 3834-line, fast-moving class (20 commits in range). The plugin overrides
  about 12 methods. Fingerprint at least these:
  - URC methods: `match_prefix`, `init_load_back`, `load_back_is_cache_owned`,
    `prefix_device_indices`, `maybe_hand_to_session`, `checkpoint`, `checkpoint_into_tree`,
    `on_release`, `claim_kv_row`, `reset`, `release_aborted_request`;
  - callers: `allocation._prefix_kv_indices`, `allocation.alloc_for_extend`,
    `PrefillAdder.add_one_req`, `common.release_kv_cache`, `common.match_kv_cache`.
- Using `maybe_hand_to_session` as the restore point borrows a hook named for streaming sessions. If
  upstream moves it, for example before the `kv_allocated_len` update, the fingerprint on
  `alloc_for_extend` must catch it.
- The host-load-back branch of `add_one_req` changes often (#43870, #43868, #36345, #40896 in range).
- `prefix_device_indices` now has side effects (it allocates). Keep it idempotent, because
  `buffer_mode/pipeline.py:1675` and `disaggregation/decode_hicache_mixin.py:72` also call it. Those
  paths are not used in production.
- TP collectives move:
  - the match converge stays where it is;
  - the per-request converge moves from prepare into `maybe_hand_to_session`;
  - `prefix_device_indices` runs only for hits, which ranks already agreed on at match time.

### 3.2 Design (b): wrap the disabled URC that upstream builds

There are two variants:
- **(b1)** Drop M05 and M06. Upstream already builds a disabled URC with MAMBA that passes the guard.
  S01 wraps it, either by `__dict__` copy into the subclass, as the pin does with ChunkCache, or by
  delegation.
- **(b2)** M05's AROUND hook wraps `original(ctx)`.

The hit, allocation, restore and release seams are the same cache methods as in (a), so the REPLACE
count equals (a). The differences:

- **`__dict__` copy:**
  - components are built with the original instance as their cache (unified_radix_cache.py:201),
    as is `StreamingSession(self)` (268);
  - `reset()` rebinds per-instance dicts only on the wrapper (393-424);
  - so internal callbacks through `component.cache` see the un-overridden original.

  In disabled mode those callbacks only read shared data today (`req_to_token_pool`,
  `enable_mamba_extra_buffer`) [V for mamba.py:644-687], so it works but is fragile.
- **Delegation proxy:** it fails the `isinstance(UnifiedRadixCache)` checks:
  - the PrefillAdder mamba-defer, host-lock and transfer branches, schedule_policy.py:1362, 1402, 1408;
  - `create_tree_cache` buffer_only/tlru/streaming, registry.py:250-288 — only if wrapped in M05.
- **S01 timing (b1):** the wrap still happens after `canary_manager.attach_radix_cache` (scheduler.py:620-621).
- **No advantage:** `create_unified_radix_cache` already takes `cache_class`, so (b) has nothing over
  (a).

### 3.3 Design (c): vendor a minimal ChunkCache-like `BasePrefixCache`

This means about 130 lines copied from the deleted `chunk_cache.py`, plus the same hit, allocation
and restore methods as (a).

**Mamba ownership has two options:**
- **`supports_mamba()` False**, as at the pin:
  - `create_tree_cache` raises for hybrid-SSM (registry.py:290-295), so **the M06 REPLACE stays**;
  - nothing frees the mamba slot. Either keep M03's free (REPLACE) or free it in the vendored
    `on_release`, which every release path reaches (common.py:417);
  - the not-holds-kv early path asserts `supports_mamba()` (common.py:384), which is reachable only
    for mamba-capable caches.
- **`supports_mamba()` True:** the vendored cache must own the mamba frees, copying URC's disabled
  branch. It must also stub `mamba_evictable_size`, `mamba_protected_size` and
  `all_mamba_values_flatten`, which the invariant checker and watchdog read
  (invariant_checker.py:204, 233, 267, 477; pool_stats_observer.py:262-280). It must also satisfy
  `ensure_mamba_capacity → evict_for_alloc` (allocation.py:311-316) and
  `PrefillAdder.is_hybrid_ssm_cache` (schedule_policy.py:657).

**Drawbacks:**
- Upstream no longer has any non-URC cache for disabled non-pure-SWA models. Every disabled-mode
  path in v0.5.22 is tested only against URC: root-handle match results, `last_node` NodeIds,
  `zero_match_result`, lock receipts. So the vendored cache drifts silently.
- It keeps 1-2 more REPLACE rows than (a): M06, and M03 unless its mamba free moves to `on_release`.

### 3.4 Comparison

| | (a) URC subclass via `cache_class` | (b) wrap upstream's URC | (c) vendored cache |
|---|---|---|---|
| Matched length without nodes | `host_hit_length` + `init_load_back` (or `device_prefix_len`) | same | same |
| Indices injection | `prefix_device_indices` override | same, on the wrapper | same |
| Restore point | `maybe_hand_to_session` | same | same |
| Mamba | upstream-owned; drop M03 free; no M06 | same | needs M06 (or stubs plus vendored frees) |
| REPLACE rows in scope | 0 (or 1 with M03 kept) | 0 (or 1) | 1-3 |
| Main risk | URC churn; overloaded hooks | back-reference / `isinstance` fragility | silent drift from upstream's URC-only disabled path |

**Recommendation: (a1) with design H**, the PrefillAdder subclass, and the M03 decomposition. Keep
M03 as the single fallback REPLACE if the owner prefers explicit release ordering over the implicit
"`free_kv_row` follows `checkpoint`" contract (base_prefix_cache.py:476-485 documents that order).

---

## 4. Other changes in `35f3c96ff4..v0.5.22` that touch rows S01-S09, P01-P02, B02-B06, M01-M06

Per-row status. Rows S02-S09 and B02-B06 come from a delegated diff of pin and NEW bodies; the
spot-checks I did myself are marked.

| Row | Status at v0.5.22 | Key facts |
|---|---|---|
| S01 | Breaks | Imports `ChunkCache` (deleted), and its `type(...) is ChunkCache` guard would reject the disabled URC. Under (a) it is removed. |
| S02 | Holds | `init_hisparse_coordinator` body is identical (scheduler.py:1271-1278). `ModelRunner.maybe_init_hisparse_coordinator` gained `num_draft_tokens=` (#37771). Coordinator ⇔ `enable_hisparse` still holds. |
| S03 | Holds | #37771 added a spec-only `make_staged_spec_input` hunk. `SamplingBatchInfo.from_schedule_batch` still reads only reqs and device. |
| S04 | REPLACE, rebase required | `filter_batch(chunked_req_to_exclude=)` is now `reqs_to_exclude`. `extend_range`/`prefix_indices` is now `extend_end > prefix_len` (scheduler.py:3792). `update_running_batch` returns Optional and mutates in place (e14ef4b658). The last-batch merge changed (0a21a3b798). The gates (`self.enable_hisparse` at 3796, 3809) are spelled the same. |
| S05 | Holds | — |
| S06 | Body changed | `adder = self.get_prefill_adder_class()(...)` (4034). `new_batch.hisparse_coordinator = ...` (4225) means the QSA coordinator needs `allocate_spec_scratch`. Other upstream edits too. Recommended: replace with the `get_prefill_adder_class` AFTER hook. |
| S07 | REPLACE, small rebase | New first statement `self.dp_attn_adapter.drop_sync_wait_carry()` (4506927635). The gate is unchanged. |
| S08 | Holds | New side-effect-free `ignore_waiting` branch (a1001bcf3a). |
| S09 | Holds | — |
| B02, B04 | REPLACE, rebase | `release_kv_cache(..., checkpoint=True/False)` keyword. `_prepare_kv_cache_release(req)` helper (46498bc8c5). |
| B03 | REPLACE, rebase | `discard_finish_reason` (f92199b569); `checkpoint=` keyword. #37771 adds a second `if get_memory().enable_hisparse: self.hisparse_coordinator.request_finished(req)` for decoding requests finishing in a prefill batch (batch_result_processor.py:383); the owner must decide whether QSA flips that gate too. The admit gate moves to 395. |
| B05 | REPLACE, rebase | `mamba_lazy_is_insert` is now `mamba_lazy_checkpoint`; `checkpoint=` keyword. |
| B06 | Holds | #43758 changes only the flush-failure return. |
| P01 | Holds as a before hook | The `chunked_req_limit` seam is unchanged (schedule_policy.py:1116-1118; scheduler.py:4055). Its callee `prefill_checkpoint_limit` must use `prefix_len`. |
| P02 | Copied body stale | Fixed by #42825, #42923, #43023, #37771 (`request_slot_reserve`), #36345 (`rem_mamba_slots`, `include_host_hit`), #43870, #40896. Replace with the subclass (3.1.3). |
| M01 | Copied body stale | `prefix_tensors` moved to the top via `_prefix_kv_indices` (381); `allocate_spec_scratch` (408); `_alloc_extend_loc_with_kv_reuse` now takes `prefix_tensors`; `maybe_hand_to_session` (479). Replaced by cache overrides. |
| M02 | Holds | Body identical (allocator/paged.py:386-393). |
| M03 | Copied body stale | Signature, `unlock`, `on_release(checkpointed=)` all changed. The mamba free must go under a URC-based cache. |
| M04 | Now redundant | #43439 (e8fb52c526) removed `req_generation.zero_()` from `ReqToTokenPool.clear` (memory_pool.py:465-471). Drop it, or gate it to pre-v0.5.22. |
| M05 | Around target behaves differently | The disabled branch now builds a URC (#43199-#43201), and `ChunkCache` cannot be returned. Under (a1) it returns `create_unified_radix_cache(ctx, cache_class=QSAHostPrefixCache)`. |
| M06 | Unnecessary under (a)/(b) | The disabled URC has `supports_mamba()` True. Needed only by (c) with `supports_mamba()` False. |

Specific commits called out in the task:
- **#41593 (20d1f3f56a) and #43479:** waiting-prefix LRU refresh. Inert here, because it returns
  early for `not supports_prefix_sharing()` (schedule_policy.py:286-287), and the policy is FCFS for
  disabled caches (328-333). [V]
- **#41452 (99d6f4e158):** anchored lock receipts. Inert for a disabled URC, which returns from
  `dec_lock_ref` before the tree core (unified_radix_cache.py:1006-1007). A vendored cache, as in
  (c), only needs no-op `inc_lock_ref`/`dec_lock_ref`, and the default `lock()`/`unlock()` wrap them
  (base_prefix_cache.py:528-534). [V]
- **#37771 (18210ed849):** HiSparse speculative decoding. Under QSA leases its new
  coordinator-presence call sites now reach the plugin coordinator: `allocate_spec_scratch` on every
  prefill batch. `prepare_spec_verify` and `commit_spec_accept_tokens` are spec-only. It also changes
  the B03 gate and adds `request_slot_reserve` to PrefillAdder budgets (the allocator default returns
  0, allocator/base.py:79-86). [V]
- **#42270 (ad7a9b53cc), #42686 (c7f3ae1dd5), #42537 (c01d59eb90):** the `checkpoint` flag,
  `TreeLock`/`unlock`, and `maybe_hand_to_session` respectively. [V]
- **#43870 (7e3da1b784):** added `get_prefill_adder_class`. [V]

---

## 5. To verify by test before committing to (a)

1. With `SGLANG_QSA_HISPARSE_V3=p2-offload`, `kvcache.qsa_hisparse.prefix_cache` exists when
   `create_tree_cache` runs (Q03 attaches in `init_all_attention_backends`).
2. Hit path end to end on one rank, then TP=2:
   - `init_load_back` returns L;
   - `prefix_len == L` and `cache_protected_len == 0` after admission;
   - `prefix_device_indices` returns pages that `write_cache_indices` lands in `row[0:L]`;
   - the restore runs once, with `mamba_needs_clear` False before the forward.
3. Release order ledger (`tests/lifecycle/test_release_order.py` adapted): capture, then drain, then
   `free_kv_row`, then `on_release`, then row free, then `mark_kv_released`, then `after_release`.
   The mamba slot is freed exactly once, on both the `checkpoint=True` and `checkpoint=False` paths.
4. The QSA coordinator no-op `allocate_spec_scratch`, and the B03 `request_finished` gate decision.
5. ignore_eos with a pending hit: decide between a miss and the toggle. Re-run `test_pending_host_hit_ignore_eos_*`.
6. Chunked hits: the chunk ends on the 64-page boundary; the next chunk reads its prefix from the row.
