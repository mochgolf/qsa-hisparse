# Baseline (P0-B)

Recorded 2026-10-07 for [GOAL.md](GOAL.md) / [PLAN.md](PLAN.md). This covers the
fork CPU suite at the reference commit, the pinned upstream subset (the
"plugin installed but off" baseline), and the GPU equivalence inventory and
plan for Phase 2. No GPU was used and no service was started or stopped by
this task (see the note on `test_service_control.py` below).

Path prefixes: `ref:` is `../.worktrees/qsa-fork-ref-ee8fe158d6` (new detached
worktree of the fork at the reference commit, created for this baseline);
`pin:` is `../.worktrees/sglang-pin-76e06febab`; `qwen:` is `..` (the
workspace that holds `results/`, `service/`, `lab/`). Private model paths,
served-model aliases and host details are not copied here; they are referenced
by the profile file that holds them.

## Environment

| Item | Value |
| --- | --- |
| Host | Linux 7.0.0-34-generic x86_64, 96 CPUs, 566 GiB RAM; CUDA hidden |
| Interpreter | `qwen:flash-next-env/bin/python`, CPython 3.12.3 (not modified) |
| torch / triton | 2.13.0+cu130 / 3.7.1 (`TRITON_INTERPRET=1`) |
| pytest | 9.1.1 |
| sglang-kernel | 0.4.6.post1 (pin and fork `python/pyproject.toml` require `==0.4.7`; the CPU suites do not need it) |
| Other | flashinfer-python 0.6.18, transformers 5.12.1, numpy 2.3.5 |
| Env's own sglang | editable install of `qwen:sources/sglang`; overridden by `PYTHONPATH` in every run below |
| Fork reference | `ee8fe158d64186b47236b007a299696030c372e8` ("Sanitize local deployment references and exclude private artifacts"); merge-base with the pin is `76e06febab`; 28 fork commits on top |
| Pin | `76e06febab732d28a61b75a61b7835284568cdfb` (2026-09-16) |
| Plugin repo | `f8f0958` (skeleton) |

Every run used `CUDA_VISIBLE_DEVICES=99`, `TRITON_INTERPRET=1`,
`PYTHONDONTWRITEBYTECODE=1` and `-p no:cacheprovider`; JUnit XML went to
`/tmp/p0b` (not retained). After the runs `git status --short --ignored` is
empty in `ref:`, `pin:` is unchanged, and the fork's main checkout is untouched.

## Fork CPU suite results

```bash
cd ref: && QSA_PYTHON=qwen:flash-next-env/bin/python PYTHONDONTWRITEBYTECODE=1 \
  bash scripts/test_qsa_hisparse_cpu.sh -rA -p no:cacheprovider --junitxml=/tmp/p0b/fork_cpu.xml --durations=25
```

Exit 0: **183 passed, 7 skipped, 0 failed, 20 subtests passed**; pytest 51.7 s,
wall 55 s. This equals the fork's own recorded final CPU suite
(`PREFIX_CACHE_VALIDATION.md`: 183 passed, 20 subtests, 7 CUDA-only skips).

| File | Passed | Failed | Skipped | Time |
| --- | ---: | ---: | ---: | ---: |
| `test/qsa_hisparse/test_deterministic_topk.py` | 7 | 0 | 0 | 0.1 s |
| `test/qsa_hisparse/test_flash_attention.py` | 1 | 0 | 0 | 0.0 s |
| `test/qsa_hisparse/test_gather.py` | 5 | 0 | 0 | 0.1 s |
| `test/qsa_hisparse/test_marlin_deterministic_alignment.py` | 29 | 0 | 4 | 2.1 s |
| `test/qsa_hisparse/test_model_compatibility.py` | 5 | 0 | 0 | 0.0 s |
| `test/qsa_hisparse/test_prefix_cache.py` | 31 | 0 | 0 | 1.5 s |
| `test/qsa_hisparse/test_runtime.py` | 9 | 0 | 0 | 22.3 s |
| `test/qsa_hisparse/test_service_control.py` | 14 | 0 | 0 | 12.7 s |
| `test/qsa_hisparse/test_single_request.py` | 8 | 0 | 0 | 0.7 s |
| `test/qsa_hisparse/test_slots.py` | 1 | 0 | 0 | 0.0 s |
| `test/registered/unit/model_executor/test_pool_configurator.py` | 44 | 0 | 0 | 0.2 s |
| `test/registered/unit/mem_cache/test_qsa_kv_pool.py` | 1 | 0 | 0 | 0.0 s |
| `test/registered/unit/models/test_qwen4_exp_ple_table.py` | 18 | 0 | 3 | 0.9 s |
| `test/registered/unit/managers/test_batch_result_processor_hidden_states.py` | 7 | 0 | 0 | 0.5 s |
| `test/registered/unit/managers/test_scheduler_chunked_req_gate.py` | 3 | 0 | 0 | 0.0 s |
| **Total** | **183** | **0** | **7** | |

Skips: four GPU tests gated by `SGLANG_TEST_MARLIN_GPU=1`
(`test_cuda_graph_replay_updates_integer_mapping[76|256|2048]`,
`test_whole_k_actual_group64_cuda_graph_target_batch_reference`), and three
PLE tests (two need CUDA pinned memory, one a device that reads pageable host
memory). Slowest test: `test_runtime.py::test_graph_byte_kernels_cpu_interpreter`
(20.4 s in the Triton interpreter).

**Side effect to note.** `test_service_control.py::ServiceLifecycleTests` runs
when cgroup v2 and `systemctl --user` are available. It drives the fork's
`scripts/qsa_service.py` against a fake HTTP fixture in a temporary directory
and creates/stops transient user units named
`qsa-cpu-test-<sha256(temp state dir)[:12]>.service`. No real profile or unit
is involved, and `systemctl --user list-units --all 'qsa-*'` lists no units
afterwards. Any later run of the fork suite (or a port of this file) does the
same; deselect that class with `-k 'not ServiceLifecycleTests'` where
starting transient units is not allowed.

### Supplementary: registered kernel tests at the fork (not in the fork runner)

Run as above with `PYTHONPATH=ref:python`, one file at a time.

| File | Passed | Failed | Skipped | Wall |
| --- | ---: | ---: | ---: | ---: |
| `test/registered/kernel/qsa/test_qsa.py` | 35 | 3 | 2 | 13 s |
| `test/registered/kernel/hyperconnection/test_hc_mix_triton.py` | 0 | 11 | 0 | 8 s |

`test_qsa.py` failures:

1. `test_qsa_paged_extend_trims_padding_rows_and_restores_output`:
   `AttributeError: 'QwenSparseAttnBackend' object has no attribute 'qsa_hisparse'`
   at `qwen_sparse_attn_backend.py` `_store_kv` (`if self.qsa_hisparse is not None:`),
   called from `forward_extend` line 1370. The test builds the backend with
   `QwenSparseAttnBackend.__new__` and never sets the attribute the fork added
   in `__init__`. It passes at the pin. This is a fork regression in an upstream
   test that the fork runner does not select. It was not fixed. A verbatim
   REPLACE of `_store_kv` reproduces it.
2. `test_qsa_graph_metadata_kernels_match_legacy_host_path`: `RuntimeError: No CUDA GPUs are available`
   (allocates on CUDA without a guard; same failure at the pin).
3. `test_qsa_graph_layout_covers_speculative_rows_and_padded_tail`: same as 2.

The fork's new `test_qsa_decode_score_width_matches_graph_without_padding_page_table`
passes on CPU. Skips: `test_qsa_chunk_prefill_accepts_fp8_cached_prefix`
(needs an FP8 GPU, SM89 or later) and the SM121-only kernel test. All 11 `hc_mix` failures are
`No CUDA GPUs are available` from `_make_inputs(device="cuda")`, including the
fork's `test_stable_fused_hc_mix_is_exact_across_batch_sizes`. That file is GPU-only.

## Pinned upstream subset results

These are the "plugin installed but off" reference. They were run at the pin root with
`PYTHONPATH=pin:python` and the settings above.

```bash
cd pin: && PYTHONPATH=pin:python PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=99 TRITON_INTERPRET=1 \
  qwen:flash-next-env/bin/python -m pytest -q -rA -p no:cacheprovider <files>
```

Unit subset (the five upstream files the fork runner selects): exit 0,
**72 passed, 3 skipped, 9 subtests passed**; pytest 10.7 s, wall 13 s.

| File | Pin result | Fork result | Fork's edit to the test |
| --- | --- | --- | --- |
| `unit/model_executor/test_pool_configurator.py` | 43 passed | 44 passed | adds `test_qsa_p2_offload_prices_fixed_raw_staging` |
| `unit/mem_cache/test_qsa_kv_pool.py` | 1 passed | 1 passed | none |
| `unit/models/test_qwen4_exp_ple_table.py` | 18 passed, 3 skipped | 18 passed, 3 skipped | none |
| `unit/managers/test_batch_result_processor_hidden_states.py` | 7 passed | 7 passed | drops the `batch_result_processor.get_memory` patch (the fork replaces `get_memory().enable_hisparse` with `self.hisparse_coordinator is not None` and removes the import) |
| `unit/managers/test_scheduler_chunked_req_gate.py` | 3 passed | 3 passed | sets `s.hisparse_coordinator = None` |

Kernel files at the pin:

| File | Passed | Failed | Skipped | Wall | CPU-runnable? |
| --- | ---: | ---: | ---: | ---: | --- |
| `kernel/qsa/test_qsa.py` | 35 | 2 | 2 | 12 s | Partly. Failures 2 and 3 above are CUDA-only. |
| `kernel/hyperconnection/test_hc_mix_triton.py` | 0 | 10 | 0 | 7 s | No. All tests need CUDA. |

Two of the 35 `test_qsa.py` passes are vacuous on CPU:
`test_qsa_cuda_extend_ignores_dp_attention_padding` and
`test_qsa_triton_block_expansion_matches_torch_reference` `return` early when
CUDA is unavailable. W7's CPU "off" baseline is therefore the five unit files
(72/3, 9 subtests) plus `test_qsa.py` with exactly the two CUDA failures
listed. `test_hc_mix_triton.py` belongs to Phase 2.

## GPU equivalence plan

### Inventory

**Target configuration** (fork `PREFIX_CACHE_VALIDATION.md`): two 48 GiB RTX 4090
(SM89), TP2, local W4A16 AutoRound Qwen4Exp with INT8-row PLE, FP8 E4M3 KV,
C4/page64, 262,144-token context, 8 active requests, 2,048-token chunked
prefill, full decode CUDA graphs B1–B8. The host prefix budget is 8 GiB per TP
rank, with up to 256 entries.

**Frozen fixtures:** `qwen:results/prefix-cache-service-20260916/fixtures.json`
(19.3 MB). Its SHA256 is `171231bd51a27bc106e66905c1a59136326cfddf927467d7167e0a48764216d8`,
which matches the hash frozen in the fork's `PREFIX_CACHE_VALIDATION.md`.
- Format 1. `tokenizer_sha256` is `0997f410c57a1f4e…`. The tokenizer of the model in
  `service/candidate-acceptance.json` still hashes to this value (checked 2026-10-07).
- 16 cases: prefix lengths 64, 128, 2,048, 4,096, 8,192, 32,768, 65,536 and 262,016,
  each with a `copy` suffix ("Reply with the single word READY") and an
  `arithmetic` suffix ("What is 17 plus 25?"). Each case has `prefix_ids`, `input_ids` and
  `max_new_tokens=32`. The prefix is a header, a repeated record and a footer, cut
  to the exact token length.
- Generated with `test/manual/qsa_hisparse_prefix_acceptance.py freeze --model <tokenizer dir> --fixtures <path>`,
  which uses only the tokenizer, offline, and refuses to overwrite an existing file.
- Embedded acceptance: "Exact output token IDs; nonzero warm cached tokens; zero
  salted cold cached tokens."

**Historical goldens, same Python tree as the reference:**
`qwen:results/prefix-cache-service-20260916/candidate-7-*`. Snapshot 7 is
runtime `6bfed25cb0` with harnesses `07e1d774bf`. Both
`git diff 6bfed25cb0 ee8fe158d6 -- python/` and
`git diff 07e1d774bf ee8fe158d6 -- python/` are empty, and the 19-file runtime
manifest `candidate-7-source.json` matches `ref:` byte for byte. The harnesses
in `ref:` differ from `candidate-7-harnesses.json` only in the default `--url`
(8081/8082 → 30000). The strengthened ledger harness hash `9402c0f3…` matches `ref:`.

| Artifact | Content |
| --- | --- |
| `candidate-7-qualification.json` | `qualify` run, 16 cases. Each case records `cold_0`, `cold_1`, `seed`, `warm` and `repeated_warm` (`output_ids`, `text`, `cached_tokens`, `meta_info` with top-5 logprobs). A `concurrent` section holds 8× `prefix-65536-copy`, 128 tokens cold and warm. `passed: true`, no mismatches. Runtime 07:56:58–08:20:22 (23.4 min). |
| `candidate-7-actual-concurrency.json` | 2 cold controls plus warm B2 and B8 (128 tokens, `ignore_eos`). Graph evidence on both ranks: {2, 8}. |
| `candidate-7-lifecycle.json` | Flush, miss and refill; abort; reuse after abort; input logprobs at starts 0, 64, 2048 and 4096 |
| `candidate-7-ledger.json`, `candidate-7-events/rank-{0,1}.jsonl` (552 MB) | 42,682 events and 60 restores per rank. `raw_bytes` 1,611,251,712; `index_bytes` 1,610,661,888. 8 `graph_capture_complete` events per rank (B1–B8). |
| `service/acceptance-state/service.log` (private, launch at 07:50) | `QSA P2 offload pool budget: fixed_bytes=1611399936, logical_bytes_per_token=768`; `max_total_num_tokens=2097152 … available_gpu_mem=3.30 GB` |
| `qualification-summary.json` | 8 checks true, latency medians |
| `marlin-accepted-source-gpu.json` | group128/M76: `fixed_alignment_repeatable: true` |
| `marlin-whole-k-group64.json`, `marlin-whole-k-graphs.json` | 237 cases with 0 differing; 43 graph cases with 0 differing |
| `marlin-native-control-final.json`, `native-control-comparison.json` | 237 native stage SHA256 values equal to `repro-marlin-batch-invariance.json` |
| `marlin-whole-k-gpu-tests-enabled.log` | `SGLANG_TEST_MARLIN_GPU=1`: 33 passed |
| `qsa-topk-tiled-gpu.json` | 36 cases exact, 504 graph replays, 36 dynamic-length checks, empty shapes exact. 2048×65536 large prefill exact (0.103 s, 21,633,536 B extra). |
| `final-native-{lifecycle,concurrency,latency}.json`, `final-openai-smoke.json`, `final-native-concurrency.py`, `final-openai-smoke.py` | Native/light profile checks (hardcoded `127.0.0.1:8081`) |
| Retained failures | `original-cold.json` (native nondeterminism), `fork-deterministic-cold.json` (clean `e6e90a9` under deterministic mode), `fixed-align-baseline-cold.json`, `candidate-5-*` (2/8 concurrent mismatches before whole-K) |

These goldens were produced with the interpreter `service/runtime-env`, which
**no longer exists**. They are a historical cross-check, not the Phase 2 oracle.

**Launch profiles** (private JSON under `qwen:service/`; they hold the model
path, served name and interpreter):

- **Deterministic comparison profile:** `service/candidate-acceptance.json`
  (port 8082, state `service/acceptance-state`). It contains the flags below plus
  `--model-path` and `--served-model-name` from the file:
  ```text
  -m sglang.launch_server --model-impl sglang --host 127.0.0.1 --port 8082 --enable-p2p-check
  --dtype bfloat16 --ple-offload-embedding --json-model-override-args '{"text_config":{"ple_embedding_dtype":"int8_row"}}'
  --enable-metrics --kv-cache-dtype fp8_e4m3 --linear-attn-backend triton --mamba-ssm-dtype float32
  --mamba-radix-cache-strategy extra_buffer --mamba-track-interval 64 --max-mamba-cache-size 40
  --mem-fraction-static 0.96 --image-processor-backend pil --reasoning-parser qwen3 --tool-call-parser qwen3_coder
  --tp-size 2 --pp-size 1 --numa-node 3 2 --context-length 262144 --random-seed 147342228
  --disable-radix-cache --disable-overlap-schedule --skip-server-warmup --page-size 64
  --cuda-graph-backend-decode full --cuda-graph-backend-prefill disabled --disable-cuda-graph-padding
  --prefill-decode-interval 1 --max-running-requests 8 --max-total-tokens 2097152 --chunked-prefill-size 2048
  --cuda-graph-max-bs-decode 8 --cuda-graph-bs-decode 1 2 3 4 5 6 7 8 --enable-deterministic-inference
  ```
  Environment: `SGLANG_QWEN38_GDN_QKVZ_WNA16=0 SGLANG_QSA_HISPARSE_V3=p2-offload
  SGLANG_QSA_HISPARSE_V3_OBSERVE=strict SGLANG_QSA_HISPARSE_PREFIX_CACHE_MB=8192
  SGLANG_QSA_HISPARSE_PREFIX_CACHE_MAX_ENTRIES=256 SGLANG_QSA_HISPARSE_V3_EVENTS=<arm events dir>
  SGLANG_LOGPROB_CHUNK_SIZE=256 PYTHONPATH=<source>/python`. Strict observation
  requires deterministic inference and enables in-run exact byte checks:
  restored PLE, recurrent, pending-C4, compressed-index and FP8 K/V tensors, plus
  `selected_check` and `graph_selected_check`.
- **Native/light profile:** `service/qualified-candidate.json`. It uses the same
  flags without `--enable-deterministic-inference` and adds `--enable-cache-report`
  (port 8081 in the file). Environment: `OBSERVE=light`, no events, otherwise as above.
  The latency, native lifecycle, native concurrency and OpenAI smoke checks use this profile.
- `service/baseline-deterministic.json`: deterministic, prefix cache 0 (no-cache control).
- `service/current.json` (production) uses a different checkpoint and source,
  with torch 2.14.1 and sglang-kernel 0.4.9. It is not usable for this pin.
- `ref:examples/qsa_hisparse/serve.sh`, `QSA_HISPARSE.md` and `PREFIX_CACHE.md` hold the
  sanitized generic profile and the INT8-row PLE flags.

**Manual HTTP harnesses** (`ref:test/manual/`). Run them with that directory
as the working directory, because they import `qsa_hisparse_prefix_acceptance`.

| Script | What it does |
| --- | --- |
| `qsa_hisparse_prefix_acceptance.py {freeze,baseline,qualify}` | `baseline`: two salted cold runs per case. `qualify`: adds seed, warm and repeated warm, plus an optional `--concurrency {2,8}` cold/warm batch on the longest prefix whose input is under 260,000 tokens. Defaults: `--max-prefix 8192`, `--url :30000`. `--continue-on-mismatch` keeps a failed report and a failed exit status. |
| `qsa_hisparse_prefix_concurrency.py` | Fixed 128-token continuations (`ignore_eos`) on `prefix-<N>-copy` (default 65,536). Two cold oracles, then warm B2 and B8 behind a barrier. Requires per-rank `graph_replay` events whose leases all carry this run's B2 and B8 rids. The events dir must hold exactly `rank-0.jsonl` and `rank-1.jsonl`. |
| `qsa_hisparse_prefix_lifecycle.py` | Uses `prefix-8192-copy`: seed, warm (8192), `/flush_cache`, miss (0), refill (8192), abort of a 4096-token request, reuse (8192), then input logprobs at starts 0, 64, 2048 and 4096 |
| `qsa_hisparse_prefix_ledger.py <events>` | Final record complete; one raw/index pointer set per rank; 12 fixed pool fields constant; host bytes ≤ budget; final event idle with all logical pages free; TP prefix sequences equal |
| `qsa_hisparse_prefix_latency.py` | Paired one-token cold/warm requests at 8,192, 65,536 and 262,016 tokens (2 repeats); medians and ratio |
| `qsa_deterministic_topk_probe.py [--large-prefill]` | Top-k k ∈ {512, 2048} × 3 widths × 3 tie modes × strided, against an independent Python oracle, with CUDA graph replay |
| `marlin_deterministic_alignment.py` | group128/M76 alignment repeatability; no model load |
| `marlin_batch_invariance.py [--cuda-graphs] [--native]` | Actual group64 geometry, whole-K; no model load |

**GPU pytest:** `test/qsa_hisparse/test_marlin_deterministic_alignment.py` with
`SGLANG_TEST_MARLIN_GPU=1`; `test/registered/kernel/qsa/test_qsa.py` (FP8
cached-prefix extraction on SM89, graph metadata kernels, Triton block
expansion, DP padding); `test_qsa_indexer.py` and `test_qsa_strided_zero_fill.py`.
The last two are unmodified but exercise the fork-modified `qsa_indexer.py` and
`sparse_attn.py`. `test/registered/kernel/hyperconnection/test_hc_mix_triton.py`
includes the fork's stable test. The fork's validation documents record no GPU
run of these registered kernel files.

**Not reusable:** `ref:archive/experiments/*` (revision `5f8ae436`, private
placeholders) is historical only. `qwen:lab/results/sglang-fix-merge-20260923/{GPU-PLAN.md,gpu_window.py}`
and `qwen:lab/results/dsh-qsa-production-20261004/` document the owner-approved
window procedure used before: save `service/current.json` bytes, require ≥3 s
idle, stop 8081 through its controller, run the candidate on 8082 with
isolated state, then restore and verify. They also hold a native
seven-case semantic protocol (`qwen:lab/qualify_qwen38_flash_next_cases.py`,
which includes one image case). That protocol is not bitwise evidence.

### Arms and invariants

- **F** (fork reference): `PYTHONPATH=ref:python`.
- **P** (plugin `compat+hisparse`): `PYTHONPATH=<plugin entry-point dir>:<plugin>/src:pin:python`,
  `SGLANG_QSA_MODEL_COMPAT=1`, `SGLANG_QSA_HISPARSE_V3=p2-offload`, `SGLANG_PLUGINS=qsa_hisparse`.
- Both arms use the same interpreter, model, flags, seed, fixtures, harness
  files (from `ref:`) and job order. Each arm gets its own `SGLANG_CACHE_DIR`,
  events dir, controller state dir and output dir. Jobs run strictly one at a time. Record
  `git rev-parse`, the plugin tree hash, the SHA256 of the 19 manifest files
  (F) and of the plugin package (P), interpreter package versions, the
  `nvidia-smi` snapshot and `/get_server_info` for every arm.

### Ordered serial GPU jobs

Run F then P for each step. Commands are shown for F; P differs only in the
environment above (and in plugin-arm probe wrappers, see Gaps).

| # | Job | Command | Artifacts | Pass criteria (exact, no tolerance) |
| --- | --- | --- | --- | --- |
| 0 | Pre-window, CPU only | Fork CPU suite and plugin `tools/run_cpu_tests.sh` in the chosen GPU interpreter; `tools/fingerprint.py check`; `sha256sum fixtures.json` | logs | Fork: 183/7/0, 20 subtests. Plugin: green. Fingerprints OK. Fixture hash `171231bd…`. |
| 1 | Window open (owner) | Owner's procedure (see above); no other GPU process | `nvidia-smi`, saved production profile bytes | GPUs idle |
| **G2-1** | | *single GPU `cuda:0`, no model load* | | |
| 2 | Marlin GPU pytest | `SGLANG_TEST_MARLIN_GPU=1 python -m pytest -q test/qsa_hisparse/test_marlin_deterministic_alignment.py` (P: ported `tests/model_compat` equivalent) | log | 33 passed, 0 skipped; P has the same test IDs and outcomes |
| 3 | Marlin alignment | `python test/manual/marlin_deterministic_alignment.py --output <arm>/marlin-align.json` | JSON + `.pt` | `fixed_alignment_repeatable: true`; `frozen_native` and `stable` have 1 distinct output each |
| 4 | Marlin whole-K | `python test/manual/marlin_batch_invariance.py --output <arm>/marlin-whole-k.json` | JSON + `.pt` | `cases` 237, `differing_cases` 0, `fixed_case_repeatable` true; per-case output SHA256 F = P |
| 5 | Marlin whole-K graphs | `... --cuda-graphs --batch-sizes 1 8 96 2048 --patterns identical spread_routes --output <arm>/marlin-graphs.json` | JSON + `.pt` | 43 cases, 0 differing; F = P |
| 6 | Marlin native control | `... --native --output <arm>/marlin-native.json` | JSON + `.pt` | F = P: all 237 per-stage SHA256 equal between the fresh fork and plugin runs in the same window. Comparison with the historical `repro-marlin-batch-invariance.json` is informational only (different interpreter; not an acceptance criterion) |
| 7 | QSA top-k probe | `python test/manual/qsa_deterministic_topk_probe.py --large-prefill --output <arm>/topk.json` | JSON | 36/36 `deterministic_exact`; Σ`cuda_graph_exact_replays` = 504; 36 `cuda_graph_dynamic_threshold_exact`; empty shapes exact; `large_prefill.deterministic_exact`. Time and peak are reported only. |
| 8 | Registered GPU kernels | `python -m pytest -q -rA test/registered/kernel/qsa/test_qsa.py test/registered/kernel/qsa/test_qsa_indexer.py test/registered/kernel/qsa/test_qsa_strided_zero_fill.py test/registered/kernel/hyperconnection/test_hc_mix_triton.py` | log | F = P per test ID. All pass except the SM121 skip and the known fork failure `test_qsa_paged_extend_trims_padding_rows_and_restores_output`, which is reported, not waived. `hc_mix` and other tests keep their own upstream assertions. |
| **G2-2** | | *TP2, deterministic profile, one server per arm* | | |
| 9 | Server start | Deterministic profile, fresh events dir | server log, `/get_server_info` | Ready. Ledger shows `graph_capture_complete` for bs 1–8 on both ranks. |
| 10 | Actual concurrency | `python qsa_hisparse_prefix_concurrency.py --url http://127.0.0.1:8082 --fixtures <fixtures.json> --events <arm>/events --output <arm>/actual-concurrency.json` | JSON | `passed`: cold₀ = cold₁; every B2/B8 `output_ids` = cold reference; `cached_tokens` = 65,536; graph evidence {2,8} on both ranks |
| 11 | Lifecycle | `python qsa_hisparse_prefix_lifecycle.py --url … --fixtures … --output <arm>/lifecycle.json` | JSON | `passed`: cached 8192 → flush → 0 → 8192; abort `finish_reason` `abort`, then 8192; logprob `cached ≤ start` and length = `len(input_ids) − start` |
| 12 | Qualification | `python qsa_hisparse_prefix_acceptance.py qualify --url … --fixtures … --max-prefix 262016 --concurrency 8 --continue-on-mismatch --output <arm>/qualification.json` | JSON | `passed` with `output_mismatches == []`. All 16 cases: cold₀ = cold₁ = warm = repeated-warm `output_ids`; cold `cached_tokens` 0; warm > 0 (snapshot 7: equal to the prefix length). Concurrent 8: cold 0 / warm > 0 cached, pairwise equal IDs. |
| 13 | Ledger | `python qsa_hisparse_prefix_ledger.py <arm>/events --output <arm>/ledger.json` (after the last request) | JSON | `passed`, `tp_prefix_sequence_equal`, static pools unchanged, host bytes ≤ budget, final idle. The server log has no strict-check failure. |
| 14 | Stop arm server | Owner-approved controller | logs | Clean stop, no listener |
| 15 | Cross-arm comparison (CPU, offline) | Comparator (see Gaps) over steps 9–13 | report | For every request in every report, F = P on `output_ids`, `cached_tokens` and `prompt_tokens`. F = P on ledger summaries (restores, max restored tokens, evictions, max entries, max host bytes, fixed pool fields), on the normalized per-rank prefix event sequence and on the B1–B8 capture set. F = P on per-checkpoint byte digests (exact cached bytes/state). Logprobs are also compared bitwise and reported. F against snapshot-7 goldens is reported as an environment cross-check. |
| **G2-3** | | *TP2, native/light profile, one server per arm* | | |
| 16 | Native checks | Start native/light profile; `qsa_hisparse_prefix_latency.py --url … --fixtures … --output <arm>/latency.json`, `qsa_hisparse_prefix_lifecycle.py`, the 8×128-token native concurrency (`final-native-concurrency.py` logic, URL set to the arm), the OpenAI smoke (`final-openai-smoke.py` logic); stop | JSONs | Latency: cold cached 0, warm cached = length, one token. Times reported side by side, no threshold (the contract measures latency and promises no speedup). Lifecycle as in step 11. 8/8 requests complete 128 tokens with 65,536 cached. Smoke: both answers `42`; warm cached 2,048 of 2,089 tokens; cold `prompt_tokens_details` null. Native outputs are not compared bitwise. |
| **G2-4** | | *offline, from steps 9, 13, 16* | | |
| 17 | Memory | Compare logs, ledgers and `/get_server_info` | report | F = P on `fixed_bytes` and `logical_bytes_per_token` (historical 1,611,399,936 / 768), `max_total_num_tokens` (2,097,152), all 12 ledger `FIXED_POOL_FIELDS` (historical raw 1,611,251,712, index 1,610,661,888), `lease_capacity` 8, logical capacity 2,097,152 and `mamba_bytes`. GPU used memory before and after is reported. |
| 18 | Window close (owner) | Restore production by the owner's procedure | status | Production ready, profile bytes unchanged |

Expected duration, from snapshot 7: model load about 2 min (113 s), decode graph
capture about 3 min, and steps 10–13 about 25 min (qualification 23.4 min), so
about 35 min per deterministic arm. A native/light arm takes about 13 min, and
G2-1 about 15 min per arm. Allow about 3 h for F+P, plus production stop and
restore.

## Gaps

1. **No fork goldens exist for the Phase 2 environment.** Snapshot 7 used the
   same Python tree but an interpreter (`service/runtime-env`) that has since
   been deleted. Arm F must be generated from `ref:` in the same window and
   environment as arm P. Snapshot-7 outputs serve only as a cross-check; an
   F-vs-snapshot-7 mismatch is environment drift and does not count against
   the plugin.
2. **Interpreter for GPU.** `flash-next-env` has sglang-kernel 0.4.6.post1, but
   the pin and fork require 0.4.7. `qwen:service/runtime-env-sglang-20260923`
   matches: torch 2.13.0+cu130, triton 3.7.1, sglang-kernel 0.4.7, flashinfer
   0.6.18, transformers 5.12.1, pytest 9.1.1, and no sglang distribution.
   Using it read-only needs owner approval, or the owner provides a private
   environment. The production interpreter (torch 2.14.1, kernel 0.4.9) does
   not match the pin.
3. **Plugin activation without installing.** `PYTHONPATH` alone does not expose
   the `sglang.srt.plugins` entry point: `entry_points()` returns `[]` in both
   interpreters. Phase 2 needs a private directory with the plugin's
   `.dist-info`, for example `pip install --no-deps --target <private dir>`,
   which writes only there. That directory must be on `PYTHONPATH` for the
   spawned TP schedulers, which call `load_plugins()` at `scheduler.py:5833`.
   Activation must be shown in each rank's log.
4. **Plugin-arm probe wrappers.** `marlin_deterministic_alignment.py` and
   `marlin_batch_invariance.py` load `stable_align.py` from the fork path
   (`python/sglang/srt/layers/moe/fused_moe_triton/`, `--source`). They and
   `qsa_deterministic_topk_probe.py` import `sglang` kernels directly and never
   call `load_plugins()`. Arm P needs wrappers that set the switches, activate
   the plugin before importing, and resolve `sglang_qsa_hisparse.kernels.*`,
   with oracles and assertions unchanged (W4/W5). The `SGLANG_TEST_MARLIN_GPU`
   tests need the same in the ported `tests/model_compat`.
5. **No tooling for cross-arm exact cached bytes.** The fork's
   exact-bytes checks run within one process (strict mode), and the ledger has
   no tensor digests. Proving that F and P cache the same bytes and state
   needs an observer. It would hash every published checkpoint (FP8 K/V and
   scales, compressed index, pending C4 and positions, recurrent conv/temporal
   state, PLE histories) keyed by token length, without changing the
   computation. The observer must be identical in both arms; the sitecustomize
   hook in `qwen:results/prefix-cache-service-20260916/cold-trace/` is a
   precedent. It must be built and CPU-tested before the window.
6. **No cross-arm comparator yet.** An offline comparator for steps 9–17 is
   needed. Harness namespaces include `time_ns`, so rids and salts differ
   between arms and ledger sequences must be compared after normalizing rids.
7. **Compat-only arm (decided after G0: reduced profile, owned by W8).** The fork has no qualified profile or fixture with
   `SGLANG_QSA_HISPARSE_V3` unset; at TP2/256K/B8 the raw KV would not fit
   without offload. Two options:
   - The owner picks a reduced deterministic profile (smaller context and
     `--max-total-tokens`) and runs `acceptance.py baseline` (cold only; prefix
     reuse requires `p2-offload`) with F (V3 unset) vs P (`SGLANG_QSA_MODEL_COMPAT=1` only).
   - Phase 2 is scoped to `compat+hisparse`, and compat-only is covered by CPU
     and kernel tests.
8. **GPU window.** The owner approved GPU validation with
   `service/runtime-env-sglang-20260923` on 2026-10-07. The 46.7 GB figure
   comes from recorded profiles; a live check on 2026-10-07 showed the GPUs
   nearly idle. Each job re-checks GPU memory and ports first and never
   stops a process it did not start. `--numa-node 3 2` in
   the profiles is specific to this host.
9. **Fork baseline failure.** `test_qsa_paged_extend_trims_padding_rows_and_restores_output`
   fails at the fork (AttributeError above). W4 should port it unchanged and
   record the expected failure. It must not be "fixed" silently in the port.
10. **Unowned files.** `test_service_control.py` (14 tests) and
    `scripts/qsa_service.py` are not assigned to any workstream in PLAN.md.
    The fork suite starts transient user units through them (see above).
11. **Non-target models on GPU.** GOAL.md requires that non-target models are
    unaffected, but no model or fixture is identified for a GPU check. Today
    only W7's CPU checks cover this.
12. **Images (Phase 3).** No frozen bitwise image fixture exists. The only
    image request is in the native seven-case protocol.
13. **JIT caches.** Build keys are content-hashed
    (`kernels/jit/utils/compile/loader.py`), but separate `SGLANG_CACHE_DIR`s
    per arm keep evidence attributable. The first-call JIT time is not part of
    any pass criterion.

## Acceptance basis (after G0)

Every GPU criterion compares fresh fork (F) and plugin (P) runs from the same
window, interpreter and fixtures. Historical goldens and hashes from earlier
environments are reported for information only and never decide a gate.
