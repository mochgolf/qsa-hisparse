# Validation of the SGLang fork

## Upstream update, 2026-10-04

Status: **FROZEN_SOURCE_INDEPENDENTLY_REVIEWED_CPU_GPU_SERVICE; PRODUCTION RESTORED**.
Frozen tested source: `2fe0731e03c42f79092aa6894b7d81674c7a6fdd`.

Integration: upstream main `35f3c96ff4794a4de15daf12caad371084a037ee` is merged
over the real common ancestor `32290dda2cea4bb95274b3d08e43d4dad74e9676`
(561 commits) by the two-parent merge commit
`80dc48ddfc10162f9735f65b6c8f6238267b4eb0`. Follow-ups are `2d66964226`
(docs), `00fe7d9c49` (style), `857aee8910` (doc scoping), `0c0e30dcdd`
(TP-rank migration), `0ffe23cc63` (TP CPU-group migration) and `2fe0731e03`
(fused-fallback fixture completion). The stale `172b1b4825` record is outdated;
[PROVENANCE.json](PROVENANCE.json) lists the verified ancestry facts.

CPU, worker-run with CUDA hidden in two environments — the isolated upstream
dependency environment (Torch 2.14.1+cu130, Triton 3.8.0, FlashInfer
0.7.0.post1, Transformers 5.17.0, sglang-kernel 0.4.9, CUTLASS DSL 4.8.0,
tokenizers 0.23.2, xgrammar 0.2.7) and the existing `flash-next-env`
(Torch 2.13.0, FlashInfer 0.6.18, sglang-kernel 0.4.6.post1, xgrammar 0.2.1):
both passed **221 tests, 7 skipped and 24 subtests**; the JUnit reports record
252 cases with 0 failures and 0 errors.

GPU kernel suite, independent reviewer, RTX 4090 (SM89) with the isolated
upstream dependency environment:
`test_qsa`, `test_fast_topk`, `test_qsa_fused_kv_prepare` and
`test_qsa_strided_zero_fill` passed **240 tests, 1 skipped** (the skip is the
SM121-only kernel) with **0 failures**.

Service window, independent reviewer, TP2/PP1, 262144 context, eight running
slots, full decode CUDA graphs B1–8, FP8 KV and the QSA P2 host-prefix cache:
the frozen seven cases passed **7/7**, including the exact 262016-token input
and English, Chinese, code, reasoning, tool-call and image requests. The
lifecycle window completed B1–8 with **36 requests of 128 output tokens each
and 65536 cached tokens each**; OpenAI usage reported 0 cached tokens on the
cold request and 2112 on the warm one; an explicit abort returned
`finish_reason=abort`; after-abort recovery, idle flush, reseed from native 0 to
65536 and the final health check passed. Graph evidence shows CUDA graph replay
for every B1–8 batch, no non-graph decode batches and no error lines. The
baseline comparison matched recorded request previews, image bytes and prompt
token counts for every case; the reasoning case differs only in wording and
length (125 versus 126 output tokens) while both sources pass the frozen oracle.
No bitwise numerical equivalence and no performance improvement are claimed.

The two real startup failures on the merged tree are retained as history: the
removed `ModelRunner.tp_rank` (fixed in `0c0e30dcdd`) and the removed
`ModelRunner.tp_group` (fixed in `0ffe23cc63`). The GPU review also had two
earlier windows whose failures must not be conflated:

- the first window (`results/dsh-maintenance-20261004/phase2-final-gpu-tests.txt`)
  reported **68 failed, 172 passed, 1 skipped** because the PATH lacked `ninja`,
  the running production service left insufficient free GPU memory, and one
  upstream fused test's `__new__` fixture was not initialized;
- the second window
  (`lab/results/dsh-qsa-production-20261004/phase2-group/gpu-preflight.stdout`
  and `.xml`) reported **34 failed, 206 passed, 1 skipped**: 33 of those
  failures came from the harness inheriting the production
  `SGLANG_QSA_HISPARSE_V3=p2-offload` mode, plus the same single fixture gap.
  This was a corrected environment run of the same window, not a second
  independent environment.

The fixture gap is a test-fixture migration gap, not a product runtime API bug:
the fixture built the backend with `__new__` and never set
`req_to_token_pool`. It was fixed in `2fe0731e03`. Only the final window, with
the correct environment and that fix, produced **240 passed, 1 skipped and 0
failures**; the earlier windows are not erased by it.

Latest upstream main re-check: `affa261e3d289fe4f907c9b2e8d773fef0d36dba` is
four commits ahead of the tested `35f3c96ff4`. They touch
`multimodal_gen`/diffusion documentation and tests plus AMD CI, with no QSA
runtime or dependency overlap. They are **not** merged and **not** covered by
the frozen tests; the integration remains on `35f3c96ff4`.

Limitations of this record:

- the production profile runs `SGLANG_QSA_HISPARSE_V3_OBSERVE=light`, so no
  full-model strict bitwise/determinism validation exists for this revision;
- the no-host-prefix configuration (`SGLANG_QSA_HISPARSE_PREFIX_CACHE_MB=0`,
  which selects the disabled-radix `UnifiedRadixCache`) is covered only by CPU
  factory/registry tests and was never exercised as a service;
- the SM121 kernel was skipped and no AMD or other-GPU execution was performed;
  the open HIP RoPE follow-ups (#42483, #42494) are not incorporated;
- no wheel was built, packaged, pushed or deployed, and no performance
  certification was produced;
- the production profile and environment were not modified.

Raw evidence: `lab/results/dsh-qsa-production-20261004/phase2-final/`
(`gpu-preflight.xml`, `gpu-preflight.stdout`, `seven-cases.jsonl`,
`lifecycle-summary.json`, `lifecycle-requests.jsonl`, `graph-log-evidence.json`,
`baseline-comparison.json`, `service-test-window.log`, `production-*.json`),
plus `phase2-service-state/` and `phase2-fixed-service-state/` for the startup
failures, and `results/upstream-qsa-analysis-20261004/latest-main-delta.json`
with `latest-amd-qsa-details.json`. All of these evidence paths are relative to
the maintenance workspace root `/home/zyk/projects/interests/ai-video/qwen`, not
to this checkout, and the raw artifacts themselves are not committed
(`raw_artifacts_included: false`). `phase2-final/run-summary.json` records
`candidate_passed=true` and `production_restored=true` for the frozen source,
with the `production-restore` and `production-after-status` commands both
exiting 0; the candidate was stopped and the original 8081 service was restored
from its preserved profile.

## Upstream follow-up, 2026-09-23

Status: **GPU_FUNCTIONAL_TESTS_PASSED_BASELINE_RESTORED**. Merge commit
`2f06478454256c2f6e3666858d00b37c5ae9132e` integrates upstream main at
`32290dda2cea4bb95274b3d08e43d4dad74e9676` (52 new upstream commits).
The QSA conflict resolution retains local CUDA/HiSparse FP8 descale, graph and
capture paths while including upstream's ROCm packed decode. A companion fix
passes the existing K/V scales and FP8 dtype flag through the new packed
wrapper to its Triton kernel. AMD execution and numerical agreement were not
tested because no AMD GPU was available.

The maintained CPU runner passed **191 tests, 7 skipped and 20 subtests**;
three upstream communicator test files passed **14 tests and 42 subtests**.
On isolated TP2 port 8082 with the same model and precision profile, the
frozen seven cases passed **7/7**, including 262016 input tokens. Two short
non-greedy streams returned native output IDs matching usage; both reached
their intentional eight-token length limit, so they validate sampling/SSE
transport, not final-answer quality. The first generated delta took 59.17 s
on the initial FlashInfer JIT path, versus 0.132 s on the next request; this
does not establish a TTFT improvement. A separate prefix-only window completed
**10/10** requests after a test-wrapper fixture path error prevented these
requests in the first window. The warm and eight burst requests each reused
2048 tokens; both TP ranks recorded actual B8 graph replay. At final idle,
running/queued/used-token/Mamba-used gauges were zero, and each rank had zero
active QSA leases, eight free leases, 40 available Mamba slots and no pending
release.

Both 8082 startups logged a non-blocking `/freeze_gc` loopback connection
refusal during readiness; health and all subsequently sent requests passed.
The original baseline service was restored on 8081 after each window, with
health/model checks and the original profile bytes verified. No candidate
was promoted to production. These functional checks do not show that the
previous long-conversation “你好” user-event hallucination is fixed; its
quality cause remains unresolved. Raw service profiles, request outputs and
event logs remain in the private local lab records, outside this repository.

## Upstream update, 2026-09-23

Status: **GPU_FUNCTIONAL_TESTS_PASSED_PRODUCTION_ROLLED_BACK**. This update merges SGLang main at
`172b1b4825ac9865076b78ef2c0daa66c7cc39dd`, 421 commits after the
previous upstream revision. The merge was based on QSA HiSparse remote main
`83bee0adf5`, which already includes the prefix-cache service branch. The
integration merge is `06f19bb21a8a24e5ed66496c9ad3657763e2b4f3`.
The GPU-tested runtime commit is `f8a69629b09a403ed8375703d96440f66013a6ba`.

The maintained CPU runner (`QSA_PYTHON=python bash scripts/test_qsa_hisparse_cpu.sh`)
uses Python 3.12.3, Torch 2.13.0, Triton 3.7.1, FlashInfer 0.6.18,
Transformers 5.12.1 and Triton's interpreter with CUDA hidden. It passed:
**187 passed, 7 skipped, 20 subtests passed**. The new release test
checks that the QSA host-prefix adapter accepts `owned_kv_len` and frees the
owned pages. The existing suite covers INT8-row PLE construction and prefix
checkpoint boundaries without changing their numerical expectations.

The source still requires `sglang-kernel==0.4.7`; the local CPU test
environment has `0.4.6.post1`. Upstream also raises `xgrammar` from 0.2.1 to
0.2.7, `cache-dit` from 1.3.0 to 1.5.1, and `sgl-eval` from 0.1.0 to 0.1.2.
The CPU environment still has xgrammar 0.2.1. The isolated GPU service
environment has `sglang-kernel 0.4.7` and `xgrammar 0.2.7`.

The first TP2 startup exposed the removed `ModelRunner.ps` field. Both QSA
adapters now read `runner.tp_rank`; the complete CPU runner passed again with
**187 passed, 7 skipped, 20 subtests passed**. The next startup exposed a local
CUDA compiler/header mismatch during first-time TileLang compilation. The
isolated service profile now selects the installed CUDA 12.8 compiler and
headers; this is a deployment setting, not a change to the model or QSA
numeric contract.

With two RTX 4090s, the candidate service reached ready on port 8082 using
TP2, PP1, 262144 context, eight running slots, full decode CUDA graphs for
batch sizes 1–8, FP8 KV, pinned INT8-row PLE offload and the QSA P2 host
prefix cache. The frozen seven-case qualification passed **7/7**, including
the exact 262016-token input, code, reasoning, tool call and image request.
A cold seed followed by a warm request and eight concurrent 128-token requests
passed **10/10**; the warm requests each reported 2048 cached prefix tokens.
Both TP ranks recorded actual B8 graph replay. After the requests, running and
queued counts were zero, with zero active QSA leases, eight free leases and 40
available Mamba slots on each rank. The successful log window had no scheduler
traceback or compilation failure.

The same tested source and environment reached ready on production port 8081.
`/health`, model alias, TP2/B8/256K server configuration and a short generation
request passed. Real traffic then exposed a long first-token outlier on an
approximately 28K-token prompt and repeated misinterpretation of an earlier
user greeting during a longer conversation. The session export shows no new
greeting at those later steps. The first-token outlier appears in the

The service was rolled back to the preserved prior source and environment after
the required idle gate. The default profile again points to that baseline;
`/health`, model listing and an exact short generation check passed after the
rollback. Treat this branch as a candidate requiring long-conversation and
cold long-prefill diagnosis before another production promotion. Per-request
JSONL, graph events, log windows, the original promotion decision and the
rollback analysis are recorded in the local lab results. The earlier checks
establish service function and resource release for this configuration; they
do not establish production latency, long-conversation correctness, long soak
stability, or complete token-for-token equivalence with the previous source.
No Python wheel was built or published for this update.

## Previous source-fork migration, 2026-09-16

Date: 2026-09-16. Status: **CPU_VALIDATED_GPU_NOT_RUN**.

The tested runtime and packaging sources are committed as
`2da3ca5a09b422a17643f8755854a04d2aaf663c`. This integrates upstream
`76e06febab732d28a61b75a61b7835284568cdfb` through merge commit `59ad842ebc`.
The original QSA runtime was `5f8ae43640404eaee4c645d8cad2d0ef6e7dc6b0`.

### Executed checks

| Check | Result |
| --- | --- |
| Original QSA CPU baseline before the upstream merge | 72 passed, 1 skipped, 9 subtests passed |
| Final `scripts/test_qsa_hisparse_cpu.sh` | **102 passed, 3 skipped, 9 subtests passed** |
| Triton CPU interpreter | Included in the final suite; C4/tail movement and scaled compact/strided gather passed |
| Startup example | Parsed by the current `ServerArgs` parser and accepted by the QSA configuration guard; no model loaded |
| Compatibility imports | Old module names resolve to the new runtime classes |
| Refactor equivalence | All 10 moved class/function definitions retain their executable AST after documented identifier changes |
| Python wheel | Built with Rust extensions disabled; metadata, license and packaged QSA source contents checked |
| Isolated wheel import | Installed without dependencies into a temporary target; imported from that target and exercised a lease |
| Static checks | Selected Ruff F401/F821/UP037 checks, changed Python parsing, shell syntax and maintained Markdown links passed |
| Git/archive integrity | Both ancestries retained; no Git alternates; all 69 archived original payloads are byte-identical |

The three skips are upstream PLE checks requiring CUDA pinned memory or a GPU
that reads pageable host memory. The CPU runner hides CUDA and explicitly
enables Triton's interpreter. It does not start or replace a serving process.

The merge regression oracle compares gathered K/V to independent Torch indexing
with exact equality. Non-unit FP8 scales catch lost descale operations; NaN
scratch exposes missing zero-fill stores. Existing runtime tests cover slot
generations, short prompts, request routing, graph failure cleanup, copy/event
ordering, and logical free-group completion with CPU substitutes. No numerical
tolerance or supported model geometry was relaxed.

### Reproduce

In an environment with the runtime dependencies and pytest:

```bash
QSA_PYTHON=python bash scripts/test_qsa_hisparse_cpu.sh
```

Packaging was checked in a separate temporary environment with setuptools,
setuptools-scm, setuptools-rust, wheel, build, and the existing runtime
dependencies available:

```bash
SGLANG_BUILD_RUST_EXTS=none python -m build --wheel --no-isolation \
  --outdir /tmp/qsa-hisparse-wheel python
```

This checks Python packaging, not compilation or qualification of native Rust
extensions. The wheel was built from the staged sources subsequently committed
as `2da3ca5a09`; its development-version suffix names the pre-commit base
`59ad842ebc`. It is a local validation artifact, not a published release.

### Environment and remaining scope

CPU/interpreter tests used Python 3.12.3, Torch 2.13.0, Triton 3.7.1,
FlashInfer 0.6.18, Transformers 5.12.1, and the existing `sglang-kernel 0.4.6.post1`.
The merged source now requires **`sglang-kernel 0.4.7`**. Native dependencies in
the serving environment were not upgraded by this maintenance task.

Both GPUs were occupied by the existing service. CUDA kernel execution,
native-extension builds, TP2 service output comparisons, graph capture/replay
on hardware, lifecycle recovery under live traffic, and performance measurements
were not run for this revision. Validate those paths with the new dependency
set before deploying the merged source.

The throughput, latency and 8×256K capacity measurements under `archive/` remain
historical evidence for their recorded revisions. They are not validation of
this merge or of the source-fork refactor.

Before publication, remote commit `33828617caff4a58de7cc32544b42ca4de948f56`
(blog prose edits) was merged into the relocated archive article. The runtime,
tests and packaging sources remain identical to `2da3ca5a09`; the original
69-file archive equality check above describes the migration snapshot, before
that editorial update.
