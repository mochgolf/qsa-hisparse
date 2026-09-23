# Validation of the SGLang fork

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
prefill-forward stage, but the exact cause of either issue is not established.
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
