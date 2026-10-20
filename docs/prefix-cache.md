# QSA host prefix cache: integration surface

Scope: `src/sglang_qsa_hisparse/hisparse/{prefix,prefix_cache}.py` and the
SGLang call sites they rely on, at pin v0.5.22 (`baa9befa38`; line numbers
are v0.5.22's). Rows are from [patch-inventory.md](patch-inventory.md); the
design and its alternatives are in
[phase6-prefix-design.md](phase6-prefix-design.md) (decisions 1-5). Before
v0.5.22 the cache wrapped a `ChunkCache` (S01, M05, M06 and a `-1` marker
tensor in `req.prefix_indices`); v0.5.22 deleted `ChunkCache` (#43201) and
`prefix_indices` (#42825), see this file's history for that design.

Objects and owners:
- `HostPrefixCache` (`runtime.prefix_cache`, one per TP rank): immutable
  `PrefixSnapshot`s at page64 lengths. A `PrefixReader` pins one snapshot; a
  `PrefixReservation` holds host bytes and one entry slot before a capture.
- `QSAHostPrefixCache` (the scheduler's `tree_cache`): a disabled
  `UnifiedRadixCache` subclass (plugin-owned since v0.5.22). It keeps
  `matches[handle]` (reader of a matched, not yet restored attempt) and
  `restoring[handle]` (one record per request inside `alloc_for_extend`:
  reader, `prefix` pages, `suffix`, previous lengths). `handle` is
  `CacheRequestHandle(rid, attempt_id)`. The disabled tree owns the request's
  mamba slot (frees it in `checkpoint` for a finished request, in
  `on_release` otherwise) and stores nothing.
- Runtime (`kvcache.qsa_hisparse`): per-request lease (physical staging slot
  and generation) and `_RequestCache` state (`seq_len`, prefix basis).

## 1. Call sites

| # | Upstream call site (v0.5.22) | Cache/runtime method | Row |
|---|---|---|---|
| 1 | `Scheduler.__init__` 570 `build_kv_cache` -> `create_tree_cache` (registry 229) -> `default_radix_cache_factory` (83) | `create_unified_radix_cache(ctx, cache_class=QSAHostPrefixCache)` when `qsa_private_host_prefix_active`; rejects an enabled radix cache, HiCache and the Mamba extra buffer. Runs after `init_model_worker` built the runtime (Q03) | M05 |
| 2 | `Req.init_next_round_input(tree_cache)` from `_get_new_batch_prefill_raw` 4136 -> `match_kv_cache` (common 203) | `match_prefix`: the disabled empty match at the root, with `host_hit_length = L` on a hit | none |
| 3 | `PrefillAdder.add_one_req` 1336 (adder from `get_prefill_adder_class` 2362) | `pending_prefix_tokens`, `prefill_checkpoint_limit`; `init_load_back` (1440) accepts the hit, `load_back_is_cache_owned()` is False | S06 (`QSAPrefillAdder`, P02) |
| 4 | `PrefillAdder.add_chunked_req` 1092 | `prefill_checkpoint_limit` | S06 (`QSAPrefillAdder`, P01) |
| 5 | `alloc_for_extend` 367 (from `ScheduleBatch.prepare_for_extend`) | `prepare_prefix_for_extend` and, on any exception, `rollback_prefix_for_extend` (hook); upstream calls `prefix_device_indices` (`_prefix_kv_indices` 356) and `maybe_hand_to_session` (479) | M01 |
| 6 | `checkpoint_kv_cache` (common 253) from `stash_chunked_request` 3648 and `process_batch_result_prefill` 394, 495 | `checkpoint(req, up_to=req.extend_end)`: `_capture`, then the disabled checkpoint | none; B03 admits staging after it |
| 7 | `release_kv_cache` (common 378) | `claim_kv_row` (refuses an open restore), `checkpoint` when checkpointing, runtime `release`, `free_kv_row`, `unlock`, `on_release` (drops a pending match), row free, runtime `after_release` | M03 |
| 8 | `PagedTokenToKVPoolAllocator.free_group_end` 386 | runtime `after_logical_flush` / `after_release(pending_release)` | M02 |
| 9 | `Scheduler._release_aborted_request` 3418 -> `finish(ABORT)` | `release_aborted_request` | none |
| 10 | `Scheduler.flush_cache` 5348 | `reset` (host prefixes, then the empty tree) | none |
| 11 | `SchedulerWeightUpdaterManager._observe_weight_load` 105 | `invalidate_model` (host prefixes only) | B06 |
| 12 | `Scheduler.release_host_resources` 1876 | `release_host_resources` | none |

`SGLANG_RADIX_FORCE_MISS` passes a non-sharing cache's match through
(`zero_match_result`), so the cache checks it itself. The policy is FCFS for
a disabled cache, so site 2 is the only match.

## 2. Call order and ownership

At most one request is admitted per prefill batch (S05, S06) and overlap is
disabled, so each request runs these steps in order.

1. **Match.** `release_aborted_request(handle)` drops a reader left by an
   earlier match of the attempt. Bypass inputs (unsupported multimodal input,
   embeddings, explicit positions, M-RoPE, session, beam, hidden states,
   `skip_radix_cache_insert`, bigram keys, force-miss) converge with `None`
   and miss. Otherwise `host.acquire(namespace, tokens, len(key), identity)`
   returns the longest page64 snapshot within the logprob/logits limit; after
   convergence the reader moves to `matches[handle]` and `match_kv_cache`
   records `prefix_len = 0`, `host_hit_length = L`, `cache_protected_len = 0`.
2. **Admission** (`QSAPrefillAdder`). `rem_chunk_tokens` is capped at
   `prefill_checkpoint_limit` (the last page64 boundary of an unaligned
   prompt, also inside an image). With a hit pending, an `ignore_eos` request
   takes the normal path; `total_tokens` counts all prompt tokens, which
   equals production's charge of the restored pages; `init_load_back`
   returns `(L, last_node)`, so `prefix_len = L` and `host_loaded_length = L`.
   After admission L is added to both budget offsets and booked as
   device-tier hits, as production did. A rejected request keeps its reader
   until its next match or abort.
3. **Allocation** (M01 around `alloc_for_extend`). `prepare_prefix_for_extend`
   records every request (`fresh`, previous lengths, reader) and converges;
   `prefix_device_indices` allocates L fresh logical pages for a hit (before
   the row); upstream writes them into `row[0:L]`, allocates and writes the
   suffix and sets `kv_allocated_len`; `maybe_hand_to_session` then runs
   `runtime.restore_prefix` (fresh lease; raw K/V, compressed index, pending
   C4 ring and rope rows, Mamba/short-conv/n-gram state), closes the reader
   and drops the record. On any exception `rollback_prefix_for_extend` frees
   the prefix pages, the suffix read from the row once `kv_allocated_len`
   moved, the Mamba slot and the row, drains and releases a lease, and closes
   the reader; a later chunk frees only pages past its owned partial page. A
   failed rollback keeps the record ("retained ownership"), which blocks
   rematch, allocation and release until a rollback succeeds.
4. **Forward.** A cold first chunk acquires its lease in `begin_batch`; a
   restored request already holds one.
5. **Capture** (sites 6 and 7). Only in phase `prefill` and at
   `seq_len % 64 == 0`: `runtime.capture_prefix` reserves the new host
   footprint, waits for the producer stream and copies the new segments and
   the pending/rope/recurrent state; after convergence `host.publish` adds
   the snapshot and the reservation closes. It runs before the disabled
   checkpoint frees a finished request's mamba slot.
6. **Release** (M03, then M02 at `free_group_end`); the order and the
   ownership of the physical slot are in M03's row and
   `tests/lifecycle/test_release_order.py`.
7. **Abort, flush, weights.** `finish(ABORT)` closes a matched reader (not a
   restoring one). `reset` refuses to run during a restore, closes matched
   readers and advances the epoch; `invalidate_model` does the same at the
   start of every weight-update attempt.

## 3. TP convergence points

`_converge` all-gathers a signature over the tree's `tp_group`
(`params.tp_cache_group`, the TP CPU group since V3 rejects DP attention)
and raises on any difference.

| Where | Signature |
|---|---|
| `match_prefix`, undrained restore | `(handle, namespace, "undrained restore")` |
| `match_prefix`, bypass or force-miss | `(handle, namespace, None)` |
| `match_prefix`, after `acquire` | `(handle, namespace, snapshot.signature or None)` |
| `prepare_prefix_for_extend`, per request | `(handle, snapshot.signature or None)` |
| `_capture`, before publish | `(namespace, length, sha256(tokens), image key digest, captured)` |

## 4. Tests

`tests/prefix` (activated with W2's rows: M01, M03, B06, S06's adder),
`tests/lifecycle` (release ledger, the factory and its rejections, the adder
gates), `tests/image/test_matching.py` and
`tests/image_boundaries/test_image_checkpoints.py` (image keys, hits at
every page64 boundary including inside an image, exact restored state).
`test_admission_tiers_and_cached_tokens_equal_production_for_hit_miss_and_chunked_hit`
pins the admission statistics and the response's `cached_tokens` against
production's values.

## 5. Image prefixes (Track I)

A supported image request carries an `ImagePrefixIdentity` (I1's artifact
keys, offsets, grids and a page-cumulative M-RoPE digest). A snapshot of
length L matches only if its tokens are a prefix and its image records equal
the identity's records intersecting `[0, L)` (a straddling image keeps its
full key) and its M-RoPE digest at L; images at or after L do not take part,
so different suffixes share. The bypass set misses.
