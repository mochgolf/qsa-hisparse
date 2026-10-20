# Phase 6 results: v0.5.22, reference = production

Owner decisions and rules: PLAN.md "Phase 6"; survey `docs/phase6-survey.md`;
host prefix design `docs/phase6-prefix-design.md`. Tasks P6-A..D ran in
parallel worktrees and were merged into `phase6` (P6-B `586be8c`, P6-D
`80da09e`, P6-C `c618102`, P6-A `2559a47`, P6-A follow-up `b30818a`).

## CPU (G6-CPU input)

Interpreter `../results/phase6-20261018/runtime-env` (torch 2.14.1,
sglang-kernel 0.4.9, flashinfer-python 0.7.1, triton 3.8.0, transformers
5.19.0; `pip freeze` equals production's except those three bumps:
`../results/phase6-20261018/runtime-requirements.txt`).

| Check | Result |
|---|---|
| `tools/fingerprint.py check` | 402 pinned fingerprints match v0.5.22 |
| `tools/manifest.py --check` | current (64 rows) |
| `tools/run_cpu_tests.sh` | unit 723 passed, 54 skipped, 1 xfailed; subprocess integration 4 passed, 1 skipped; activated integration 11 passed; 0 failed |
| Previous state (`main`, pin 35f3c96ff4, production interpreter) | 685 + 3 + 26 passed |

The activated-integration pass shrank because the prefix tests now need only
P6-A's rows and run in the unit pass with those rows activated.

### REPLACE rows: 32 → 26

| Row | Phase 5 | Phase 6 | How |
|---|---|---|---|
| S06 | replace (332 lines) | after on `Scheduler.get_prefill_adder_class` | #43870 seam; `QSAPrefillAdder` (`prefill_max_requests = 1`) |
| P02 | replace + attach | none (in `QSAPrefillAdder.add_one_req`) | no copied body; production's budget charge and `ignore_eos` reuse kept |
| P01 | before | none (in `QSAPrefillAdder.add_chunked_req`) | |
| M01 | replace (133 lines) | around on `alloc_for_extend` | pages via `prefix_device_indices`, restore via `maybe_hand_to_session` (#42825, #42537) |
| M06 | replace | dropped | the disabled `UnifiedRadixCache` supports mamba (#43198–#43201) |
| B04, B05 | replace ×2 | one before on `_prepare_kv_cache_release` (B04) | `46498bc8c5` seam; equivalence for every caller in the reason |
| S01 | after | none | M05 builds `QSAHostPrefixCache` itself; rejections raise in the factory |
| M04 | around | dropped | #43439 (`e8fb52c526`) no longer zeroes `req_generation` |

Kept REPLACE (26): A03 A05 A09 B02 B03 E03 E06 E07 E08 G01 G02 J03 K02 M03
Q01 Q08 Q10 Q11 Q12 S04 S07 T03 T04 Z01 Z02 Z03. Hand merges (`RESOLVED`):
M03 (production's mamba free dropped: the cache owns the slot), Q08 (#41933
pack without `output_dtype`, so the FP8 descale is applied once), T03
(#43463's bound only with `fast_topk`), T04 (unchanged).

### Host prefix cache

`QSAHostPrefixCache` is a `UnifiedRadixCache` subclass built by M05 through
`create_unified_radix_cache(ctx, cache_class=...)`: a hit is reported as
`host_hit_length`, accepted in `init_load_back`, pages come from
`prefix_device_indices`, the restore runs in `maybe_hand_to_session`,
capture in `checkpoint` before `super()`. `cached_tokens`, its tier
breakdown and the admission tier statistics equal production's for a hit,
a miss and a chunked hit (`tests/prefix/test_prefix_cache.py`). Docs:
`docs/prefix-cache.md`.

### Upstream overlaps (R2)

- #41933 vs the FP8 descale (Q08): FP8 bytes reach the chunk kernel raw; the
  scale is applied once; bitwise equal to production's gather and far from
  "scale lost"/"scale doubled" (`tests/qsa/test_prefill_pack_descale.py`).
- #43463 vs T03: the context-length bound applies only with `fast_topk`;
  deterministic decode scores every row as in production.
- #42175 vs H08: new `mix` signature and 3-tuple residuals adapted; the
  stable path keeps production's combine.
- C02 vs #40102: the KV-shard scratch and production's bias are each
  subtracted once (scratch is 0 unless KV sharding is on).
- #37771: QSA coordinator gets a no-op `allocate_spec_scratch`; B03's new
  `request_finished` gate stays on `enable_hisparse` (production semantics).

Deviations D1–D3, D5–D7 still apply unchanged; D4 no longer arises (the
new seam affects only the scheduler's own adder).

### Differences from production (for owner review; none on the served path's results)

Error or unreachable paths:
1. Mamba slot release order: on the checkpointed release the disabled tree
   frees the slot in `checkpoint`, before the lease drain (production: after
   `on_release`); still freed exactly once. Only a failed drain (which kills
   the scheduler) is affected.
2. Rollback after an `alloc_aux_to_lengths` failure (impossible: QSA pools
   have no aux cache) or a `write_cache_indices` failure (process exits).
3. S01's rejections (radix cache/HiCache enabled, mamba extra buffer) raise
   while the tree cache is built, with new wording; still before serving.

Upstream v0.5.22 behavior now reaching the plugin (inert or unobservable for
the production profile; GPU confirms):
4. Per-request TP convergence and prefix page allocation happen at the start
   of `alloc_for_extend`, before the request row is allocated.
5. `_get_new_batch_prefill_raw` is upstream's again (e.g. `batch_is_full` is
   not forced on `NO_TOKEN`; S04 resets it every round under QSA).
6. Q08 without a runtime gathers with upstream's Triton pack (same bytes).
7. T03 non-deterministic decode inherits #43463 (`min_scored_len=512`) and
   the restructured TileLang MQA kernel (same selection contract).
8. The detokenizer process now loads plugins (v0.5.22 calls
   `load_plugins()` there): ~5 s of fingerprint checks at startup, no hooks
   reached, no activation record.
9. transformers 5.19.0 enters the image artifact key, so image item hashes
   and pad values differ from production's; keys stay exact within a server.

Other platforms or options only (not production): K02 on NPU/out-of-tree
platforms (p2-offload raises), C02 with KV sharding, Q10's SM100 fused
branch, H08 on SM100 in deterministic mode (keeps production's combine).

### For G6-GPU

FP8 prefix-hit prefill vs production; deterministic decode after the
TileLang MQA change; non-deterministic decode with `min_scored_len` through
the plugin's `fast_topk`; image prefix hits (I5) with the v0.5.22 image
keys; the Rust tree core (`UnifiedRadixCache`) is compiled on first use into
`~/.cache/sglang/rust_extensions` (already built by the CPU tests for this
user). Evidence tools: fork arm = production code on production's
interpreter, plugin arm = v0.5.22 on the Phase 6 interpreter
(`docs/baseline.md` "Phase 6").

## G6-GPU (window 2026-10-19 09:29–11:55)

Owner-approved (manual permission mode). Production was idle (0 running, 0
queued) and stopped 09:29:32; restored 11:52 with its unchanged profile
(`service/current.json` sha256 `99752c69…`, the profile it was running),
ready by 11:55: `/health` 200, same model id, launcher verified 2 scheduler
activation records, GPU memory 45,358 MiB per GPU (fresh-start value; 47 GB
before the stop included served long-context caches). Plugin `phase6`
`a61c64d` clean, pin v0.5.22 `baa9befa38` clean, production `897286b12a`
clean; F on production's interpreter, P on the Phase 6 interpreter. Raw
evidence (private): `qwen:results/plugin-g6-20261019/` (`window6.sh`,
`status.txt`, `provenance.txt`, `compare-*.txt`).

| Gate | Result |
| --- | --- |
| G2-1 | **PASS.** Probe comparator F = P on 5 reports (alignment 1, whole-K 1,187, native 1,187, graphs 217, top-k 110); `steps.json` is runner metadata (every step exit 0 in both arms; now in the comparator's `RUN_METADATA`). Step 8: 91 outcomes + 2 skips equal per arm, the 4 v0.5.22-only indexer tests pass in P, no failures |
| fast_topk (`773f3c2d84`) | **PASS.** 29 passed on each arm |
| FP8 descale on GPU | Covered by step 8a (production's FP8 descale GPU tests in `tests/qsa/test_qsa.py`, passed) and G2-2 (FP8 KV, deterministic, F = P). The window's extra `descale-plugin` step was invalid: `test_prefill_pack_descale.py`/`test_fp8_descale.py` are Triton-interpreter CPU tests (CPU tensors, faked `q.is_cuda`) and fail on a real GPU without `TRITON_INTERPRET` ("Pointer argument cannot be accessed from Triton"); they pass in the CPU suite |
| G2-2 | **PASS.** `--require-observer 2`: F = P on all 10 items — concurrency 79, lifecycle 89, qualification 1,181 (token IDs, cached tokens, logprobs to 262,016 tokens, concurrency 8), ledger 30, per-rank events 1,514, per-rank observer byte digests 1,505 (captured and restored prefix bytes), server log 9, server info 4 |
| G2-2 compat-only | **PASS.** F = P on 401 items, server log and server info |
| G2-4 | **PASS.** `fixed_bytes` 1,611,399,936, `logical_bytes_per_token` 768, `max_total_num_tokens` 2,097,152, `lease_capacity` 8, `mamba_bytes` 2,374,447,760 on both; measured weight 36.869 GB both (G2-2), 36.867 / 36.869 (compat) |
| I5 | **PASS** in both ViT-cache sessions (10 image cases: hits at page-64 boundaries incl. inside images, negatives, swapped/different preprocessing; text control equal to this window's F) |
| G2-3 | **PASS** (production's profile and weights, port 8082): every check passed on both arms, token accounting equal for 31 requests, equal scheduler numactl (`--cpunodebind=2/3 --interleave=all`) |

**Warm-hit latency (diagnosed below).** G2-3 medians (s), F / P — cold: 8192 7.189 /
7.221, 65536 9.936 / 9.917, 262016 63.305 / 63.450 (within 0.5%); warm:
8192 0.125 / 0.138, 65536 0.204 / 0.240, 262016 0.516 / 0.564. Every warm
sample is slower in P (+10.0/+14.9 ms, +35.5/+36.4 ms, +45.9/+51.6 ms),
growing with prefix length; F reproduces its Phase 5 warm times (0.1248 vs
0.1248 s at 8192), and Phase 5's P was within ~2% of F. Outputs, token
counts and restored bytes are equal, so this is a performance difference in
the hit path at v0.5.22 (plugin seams or upstream allocation/preparation),
not a correctness one. Cause not yet identified (needs a GPU profile).

### Warm-hit latency diagnosis (profiling window 2026-10-20 05:54–06:17)

Owner-approved second window (production idle, stopped 05:54:52, restored
06:14–06:17 with the unchanged profile, `/health` 200, launcher verified both
ranks). `qwen:results/plugin-g6-20261019/profile/` (`profile_warm.py`,
`analyze.py`, `spans.py`, `inside.py`, `children.py`, traces): both arms with
production's profile and weights, latency-harness request shape (65,548
tokens, 65,536 cached), 5 unprofiled warm hits, 2 under the torch profiler
(CPU + GPU, stacks, shapes), 5 more unprofiled. Unprofiled warm medians:
F 0.2001 s, P 0.2218 s (+21.7 ms; G2-3 measured +36 ms at this length).

Attribution (TP0 traces, second profiled request unless noted):
- **Scheduling, match, admission, allocation and restore** (the redesigned
  host prefix cache): `scheduler.get_next_batch_to_run` 142.8 ms F /
  140.6 ms P (profiled), and the restore itself 106.7 / 105.1 ms
  (`restore_prefix_for_extend` / `maybe_hand_to_session`): no cost.
- **Plugin hooks in the forward**: HC mix +0.1 ms, fused HC +0.8, Marlin MoE
  +0.1, QSA `forward_extend` +0.7 (profiled) — the wrapper overhead Phase 5
  already had.
- **Upstream decoder stages** (`srt/layers/layer_boundary/`: #42478, #42481,
  #42936, #43460): Python scaffolding per layer in every eager forward
  (`stage.prepare`, `prepare._run_entry`, `gated.read/update_and_read`,
  `exit.finish`); the model forward is +12 ms under the profiler for a
  12-token extend. Constant per forward: invisible in GPU-bound prefill
  chunks, visible in short extends; decode runs CUDA graphs.
- **Upstream `/generate` contract check** (`entrypoints/api_contract.py`,
  `http_server.py:921`): schema validation of the request body, O(prompt
  length) — measured on CPU 1.2 / 9.1 / 35.7 ms at 8,192 / 65,536 / 262,016
  tokens (including a `json.loads` of 0.3 / 2.5 / 9.9 ms that Starlette
  caches in the server). Only `/generate` runs it (the latency harness uses
  `/generate` with `input_ids`); the OpenAI routes production's clients use
  do not.
- A 6.75 ms `aten::repeat` in the first profiled request's
  `_compute_mrope_positions_extend` (shape [1, 12]; 0.08 ms in the second
  request, 0.18 ms in F's first) is a first-call stall, not steady state.

Conclusion: the gap comes from upstream v0.5.22 (contract check growing with
prompt length; per-forward decoder-stage overhead), not from the plugin's
port; the plugin's own hit path costs the same as production's. Nothing to
fix in the plugin; the cutover decision weighs these upstream costs
(production's chat traffic pays only the per-forward stage overhead on short
extends).
