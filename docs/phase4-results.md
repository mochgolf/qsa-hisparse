# Phase 4 results: v0.5.21 (CPU part)

Branch `phase4` (merges of P4-A..D and follow-ups; `main` stays at the
76e06febab plugin until G4 passes). Pin `v0.5.21` (`e00930c548`), survey in
`phase4-survey.md`. Reports of the task agents are summarized below; the
GPU gate (G4-GPU) is pending an owner-approved window.

## REPLACE patches: 37 → 31

| Row | Before | After | How |
| --- | --- | --- | --- |
| S02 | replace | after | the original leaves the coordinator None when `enable_hisparse` is unset; the hook adopts the runner's coordinator under QSA leases (P4-A) |
| P01 | replace | before | v0.5.21's `PrefillAdder.chunked_req_limit`; the hook lowers it to the checkpoint limit (P4-A) |
| M04 | replace | around | snapshot `req_generation`, run the pinned `clear`, restore it (P4-A) |
| H05 | replace | removed | only H08's copy passes `stable=True`; H08 binds the plugin's predicate copy (P4-C) |
| G03 | replace | around + after | D6: v0.5.21's `_process_output_after_replay` hook point (P4-B) |
| R02 | replace | before | D7: hook on `_prepare_eager_forward_batch` (P4-B) |
| F01 | 2 × after | after | D3 reworded: upstream's `req_pool_indices_cpu` field; the eager-copy hook is gone (P4-B) |

Remaining 31 REPLACE rows (each with a keep reason in its inventory row):
A03 A05 A09 B02 B03 B04 B05 E03 E06 E07 E08 G01 G02 J03 K02 M01 M03 P02 Q01
Q08 Q10 Q11 Q12 S04 S06 S07 T03 T04 Z01 Z02 Z03. Ported to v0.5.21 bodies:
E03, E08, G01, G02, M03, P02 (`_add_one_req`), Q12, S04, S06, S07, T03.

## Port check

`tests/regression/test_replace_deltas.py`: every REPLACE row is registered;
each copy differs from the v0.5.21 definition by exactly the fork's edit
script (fork base → fork), except four hand merges in `RESOLVED`: E03
(upstream builds the table on meta and now wraps it in the constructor), Q12
(upstream's new ROCm branch inside the fork's NVTX rewrite; HiSparse decode
capture is not wired on ROCm, which is not a validated configuration), T04
(class REPLACE as a subclass), P02 (the fork split `add_one_req`;
`tests/lifecycle/test_renamed_copies.py` checks `_add_one_req` mechanically).

## Upstream changes that needed plugin edits

- `ModelRunner.ps` removed: the runtime reads `runner.tp_rank` (recorded pin
  edit in `tests/runtime/test_moved_sources.py`).
- `ChunkCache.cache_finished_req` removed; `release_kv_cache` frees the row
  and calls `tree_cache.on_release`: `QSAHostPrefixCache` drops a finished
  request's pending host match in `on_release` (after the logical free,
  before the row free, `after_release`, the free-group flush and physical
  reuse; it only closes a host-snapshot reader). Test `tests/prefix/test_release.py`.
- `configure_scheduler_process` no longer receives ranks: FW1 reads them from
  `get_parallel()`.
- `Scheduler.is_fully_idle(ignore_waiting=)`: S08's after hook accepts it.
- `hc_mix` moved to `sglang.kernels.ops.gemm.hc_mix`; `fast_topk` to
  `sglang.kernels.ops.attention.fast_topk`; `get_attention_dp_size` removed
  (E03 uses `get_parallel().attn_dp_size`).
- The hisparse CUDA kernel source counts misses directly; equal to the old
  formula for distinct top-k indices (`docs/runtime.md`).
- B06: v0.5.21's `begin/end_weight_update` session bypasses
  `_observe_weight_load` (as it would in the fork); outside the HiSparse
  contract, no code change.

## Tests (CPU, v0.5.21, approved interpreter)

`tools/run_cpu_tests.sh`: exit 0 — unit 569 passed, 11 skipped (GPU), 1
xfailed (known fork failure), 15 subtests; regression/model_compat
integration 4 passed, 1 skipped; full activation on `tests/runtime` and
`tests/prefix` 26 passed. `tools/fingerprint.py check`: 364 records match;
`tools/manifest.py --check` passes. `test_plugin_off` re-measured pristine
v0.5.21 outcome counts (upstream added 9 tests).
