# Phase 5 results: upstream main 35f3c96ff4, reference = production (CPU part)

Branch `phase5` (P5-A..D merged). Pin `35f3c96ff4` (upstream main of
2026-10-04, production's base), reference `897286b12a` (production),
interpreter `results/dsh-maintenance-20261004/upstream-runtime-env`
(read-only). Survey: `phase5-survey.md`.

## Production coverage

- `test_inventory.py`: every one of production's 263 hunks against
  `35f3c96ff4` maps to exactly one inventory row; Appendix A has no `?`.
  187 hunks were carried by content (`tools/remap_hunks.py`), 76 mapped by
  the tasks (P5-C corrected two carried hunks of `sparse_attn.py`).
- `test_replace_deltas.py`: the pin is the reference's base, so every
  REPLACE copy equals production's definition (mechanical edits reverted);
  the only `RESOLVED` entry is T04 (class REPLACE as a subclass).
  `test_moved_sources.py`: the runtime package equals production's files
  with no pin edits. Fork digests of copied kernels are production's.
- Production-only changes now in the plugin: T05 (fast_topk keeps all
  candidates on radix overflow, `773f3c2d84`; production's `.cuh` shipped
  under the plugin's JIT name), A11/A12 + Q04 attach (FP8 KV descale in
  prefill and reference reads, `bdb935d70f`), Q13 (fused-KV guard for FP8
  with non-unit descales, production merge), U02–U04 (`SGLANG_NUMA_INTERLEAVE`,
  `eec9df4723`), M05 + M06 (`ChunkCache` for the QSA host prefix under
  #42354), R01 (live TP CPU group), runtime TP rank from the published
  parallel bundle, production's release order (M03) and prefix-cache API
  (`claim_kv_row`, `checkpoint`, `on_release`).

## Rows

69 rows (62 before). REPLACE 31 → 32: M06 (`create_tree_cache`, production's
mid-function change, no narrower seam). New: M05 (around), M06 (replace),
Q13 (around), T05 (around), U02 (attach), U03, U04 (after), A11, A12
(moved/owned), U05 (dropped comments). Q03 and Q04 gained attaches.

## Deviations against production

D1, D2 still apply (production logs as the fork). D3 now differs from
production only by not adding `kv_allocated_lens_cpu` (F01 fills
`req_pool_indices_cpu` exactly where production does). D5 still applies
(production keeps the whole-prompt `token_ids_logprobs`). D6, D7 still
apply (production's `execute` and `_forward_raw` are the fork's).

## Known gaps shared with production (no change, recorded)

- Attention data parallelism is configured with `--attn-dp-size` at the
  pin; the runtime's check reads only `enable_dp_attention`, so it no longer
  rejects attention DP. Production has the same gap; the HiSparse profiles do
  not enable attention DP.
- `--qsa-indexer-dtype fp8_e4m3` (SM90/SM100 only) would store the
  compressed index as FP8 while host checkpoints assume bf16; not usable on
  the SM89 host; same in production.
- Production's Q08 under a breakable prefill CUDA graph passes row indices
  to `prefill_slots`; reproduced exactly; production's profile disables the
  prefill graph (`--cuda-graph-backend-prefill disabled`).

## Tooling

`tools/qsa_service.py` accepts a wrapper before the launcher (production's
`numactl --interleave=all`). Evidence: `run_g2.py`/`run_compat.py` default to
production as the fork arm; new `run_g21.py` (G2-1 both arms, inventory
`g21_step8_inventory_35f3c96ff4.txt`) and `run_g23.py` (G2-3 native checks
with production's profile, port changed only; latency side by side).

## Tests (CPU, production interpreter)

`tools/run_cpu_tests.sh`: exit 0 — unit 686 passed, 54 skipped (GPU), 1
xfailed, 17 subtests; regression/model_compat integration 4 passed, 1
skipped; full activation on `tests/runtime` and `tests/prefix` 26 passed.
`tools/fingerprint.py check`: 390 records match; `tools/manifest.py --check`
passes.
