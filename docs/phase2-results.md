# Phase 2: GPU equivalence results

Date: 2026-10-07. Hardware: 2× RTX 4090 48 GB (SM89). Interpreter:
`service/runtime-env-sglang-20260923` (torch 2.13.0+cu130, sglang-kernel
0.4.7, flashinfer 0.6.18, triton 3.7.1). Fork arm F: `ee8fe158d6`
(`.worktrees/qsa-fork-ref-ee8fe158d6/python`). Plugin arm P: plugin `main`
at `333ad47` (`src/`) on pinned SGLang `76e06febab`, started through the
launcher. Raw evidence (private, not in this repository):
`qwen:results/plugin-g2-20261007/run2/`.

## Result

| Gate | Scope | Result |
| --- | --- | --- |
| G2-1 | Marlin GPU pytest (33), alignment, whole-K, CUDA graphs, native control, top-k probe, registered QSA/HC kernel tests | **PASS**: comparator F = P on 5 reports (whole-K 1,187, native 1,187, graphs 217, top-k 110, alignment 1 items). Kernel tests: both arms fail only the two known fork `__new__` tests (`test_qsa_paged_extend_trims_padding_rows_and_restores_output`, `test_qsa_cuda_extend_ignores_dp_attention_padding`) |
| G2-2 | Deterministic TP2 p2-offload server per arm, 8 GiB host prefixes, frozen 16-case fixtures: actual B2/B8 concurrency, lifecycle, qualification to 262,016 tokens with concurrency 8, ledger | **PASS**: all four fork harnesses passed on both arms; comparator with `--require-observer 2` F = P on 10 items: concurrency 79, lifecycle 89, qualification 1,181 (token IDs, cached tokens, logprobs), ledger summary 30, per-rank event sequences 1,514, **per-rank observer byte digests 1,505 (1,445 captures, 60 restores)**, server log figures, server info |
| G2-2 compat-only | Reduced deterministic profile (`--max-total-tokens 266304`), F with HiSparse unset vs P `SGLANG_QSA_MODEL_COMPAT=1`, 16 cases × 2 salted cold requests | **PASS**: F = P on 401 items, server log and server info |
| G2-4 | Memory | **PASS**: F = P `fixed_bytes` 1,611,399,936, `logical_bytes_per_token` 768, `max_total_num_tokens` 2,097,152 (equal to the historical record), equal measured weight/graph/startup memory |
| G2-3 | Native/light latency and smoke checks | Deferred: not equivalence evidence; run before any production cutover to the plugin |

## Window log

- 21:46 production stopped (`./qwen-service.sh stop`) after a ≥4 s idle check.
- First run failed fast and was discarded (kept for reference in
  `results/plugin-g2-20261007/`): the window script lacked `ninja` on PATH
  for the plugin's first Marlin JIT build, and leaked
  `CUDA_VISIBLE_DEVICES=0` from the single-GPU probes into the TP2 servers.
  Both were script defects, not plugin or fork behavior.
- run2 21:56–23:38: G2-1 F, G2-1 P (plugin Marlin JIT built on first use,
  about 3.5 min), G2-2 F (32 min), G2-2 P (30 min), compat F, compat P.
- 23:44 production restored with its running profile
  (`lab/results/dsh-local-promotion-20261004/promoted-8081-profile.json`,
  byte-identical to `service/current.json`); status ready, `/health` 200,
  same model id, same GPU memory.

## Notes

- The comparator skips run metadata (`run.json`, `harness-status.json`).
- The plugin arm activated `model_compat` and `hisparse` in both TP ranks;
  the launcher verified both activation records before readiness.
