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

## G5-GPU (window 2026-10-09 00:35–02:53)

Production stopped 00:35 and restored 02:53 with its original profile
(ready, same profile hash, `/health` 200, same model id and GPU memory).
Plugin `phase5` `6a1c401` (main checkout detached for the window), pin
`35f3c96ff4` clean, production `897286b12a` clean, production interpreter.
F is a fresh run of production's code in the same window. Raw evidence
(private): `qwen:results/plugin-g5-20261009/` (`window5.sh`, `compare-*.txt`).

| Gate | Result |
| --- | --- |
| G2-1 | **PASS.** Probe comparator F = P on 5 reports (alignment 1, whole-K 1,187, native 1,187, graphs 217, top-k 110). Per-test outcomes against `g21_step8_inventory_35f3c96ff4.txt`: 91 outcomes + 2 skips equal, no failures on either arm (production fixed the two former known failures; the port follows it) |
| fast_topk (`773f3c2d84`) | **PASS.** Production's GPU test on production and the port on the plugin (T05): 29 passed each, including the overflow and graph-replay regressions |
| G2-2 | **PASS.** `--require-observer 2`: F = P on all 10 items — concurrency 79, lifecycle 89, qualification 1,181 (token IDs, cached tokens, logprobs to 262,016 tokens, concurrency 8), ledger 30, per-rank events 1,514, per-rank observer byte digests 1,505, server log 9, server info 4 |
| G2-2 compat-only | **PASS.** F = P on 401 items, server log and server info |
| G2-4 | **PASS.** `fixed_bytes` 1,611,399,936, `logical_bytes_per_token` 768 on both. Measured weight memory (reported, not compared) is 36.867 GB on F and 36.869 GB on P in both G2-2 and compat (about 2 MB), likely the plugin's own JIT kernel images; no other memory figure differs beyond 2 MB |
| I5 | **PASS** in both ViT-cache sessions (hits, per-request restores on both ranks equal to their captures, negatives, warm = cold token IDs, ViT checks), text control equal to this window's F. Two multi-token cases differ from window 3's token IDs (old pin and interpreter); not a criterion |
| G2-3 | **PASS** (production's profile and weights, port 8082): every check passed on both arms, token accounting equal for 31 requests, equal scheduler numactl (`--cpunodebind=2/3 --interleave=all`). Latency medians (2 samples each, s), F / P: 8192 cold 6.992 / 7.307 (first request after start 12.72 / 13.34, second 1.263 / 1.276), warm 0.126 / 0.128; 65536 cold 9.841 / 9.880, warm 0.203 / 0.205; 262016 cold 62.825 / 63.201, warm 0.515 / 0.514. No regression beyond about 1% after the first request |

Cutover (owner-approved if all passed): the release worktree
`qwen:.worktrees/qsa-plugin-6a1c401` and the profile
`qwen:service/profiles/qwen-local-plugin-20261009.json` (production's profile
with only the launcher, `PYTHONPATH`, `SGLANG_QSA_MODEL_COMPAT=1`, cwd and a
1200 s readiness timeout changed) are prepared; starting production with it
was refused by the session's permission check, so the original production was
restored and the switch is left to the owner.
