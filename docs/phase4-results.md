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
each copy (mechanical edits reverted) equals `git merge-file` of the fork's
change (fork base → fork) into the v0.5.21 definition, so each changed line
is checked in place (G4-CPU finding: the first version compared edit text
without positions). Five hand merges are in `RESOLVED`: E03 (upstream builds
the table on meta and now wraps it in the constructor), Q12 (upstream's new
ROCm branch inside the fork's NVTX rewrite; HiSparse decode capture is not
wired on ROCm, which is not a validated configuration), T04 (class REPLACE as
a subclass), P02 (the fork split `add_one_req`;
`tests/lifecycle/test_renamed_copies.py` checks `_add_one_req` with the same
merge), M03 (upstream rewrote the cache handoff next to the fork's
insertion; the lease capture stays before the handoff and the logical free,
`after_release` right after `mark_kv_released`).

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

## G4-GPU (2026-10-08 window)

Production stopped 10:31 and restored 11:43 (ready, same profile hash,
`/health` 200, same model id and GPU memory). Plugin `phase4` `688ed2e`
(main checkout detached for the window), pin v0.5.21 clean, approved
interpreter. Plugin arm only; the reference is Phase 2's fork arm
(`plugin-g2-20261007/run2/F`). Raw evidence (private):
`qwen:results/plugin-g4-20261008/` (`window4.sh`, `window4b.sh`, `compare-*.txt`).

| Gate | Result |
| --- | --- |
| G2-1 | **PASS.** Probe comparator F = P on 5 reports (alignment 1, whole-K 1,187, native 1,187, graphs 217, top-k 110). Per-test outcomes (`pytest_outcomes.py`, step 8a plus step 8b on v0.5.21's moved files): 66 outcomes + 1 skip equal, the two declared known fork failures with the same exception line. The plugin run is `-v`, so it is checked against `tools/evidence/g21_step8_inventory_v0.5.21.txt` (the run2 inventory with the SM121 skip named) and the arms are compared by file, reason and count (`--plugin-inventory`, commit `5daef87`) |
| G2-2 | **PASS.** `--require-observer 2`: F = P on all 10 items — concurrency 79, lifecycle 89, qualification 1,181 (token IDs, cached tokens, logprobs to 262,016 tokens, concurrency 8), ledger 30, per-rank events 1,514, per-rank observer byte digests 1,505, server log 9, server info 4 |
| G2-2 compat-only | **PASS.** F (HiSparse unset) = P (`SGLANG_QSA_MODEL_COMPAT=1`) on 401 items, server log and server info |
| G2-4 | **PASS.** `fixed_bytes` 1,611,399,936, `logical_bytes_per_token` 768, equal measured weight (36.869), decode graph (0.504) and startup memory |
| I5 | **PASS** in both ViT-cache sessions, and equal to window 3 (old pin) in every case: hits 6144/2048/4096/4096/4352/3904/4096 with per-request restores on both ranks equal to their captures, negatives 0, token IDs and input logprobs, ViT encodes, batch-invariance digests, and the 10 text-control cases |

Window note: the first compat and I5 attempts failed their preflight at
11:11 because G2-2's server processes were still releasing the GPUs
(`window4.sh` had no wait between steps; logs in `failed-first/`); they were
rerun in the same window after an idle wait (`window4b.sh`), with the same
plugin commit.

Phase 4 at v0.5.21 is behavior-equal to the fork on every Phase 2 and
Track I evidence item, with REPLACE 37 → 31 and deviations D6, D7 in force.
