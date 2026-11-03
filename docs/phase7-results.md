# Phase 7: main snapshot maintenance (2026-11-02)

The maintenance candidate targets unmodified SGLang main
`582168dd7c3821198c92766f0319b9e1c8c17bd3`, selected after fetching upstream
on 2026-11-02. The previous plugin is `983603dc7c`, pinned to v0.5.22
(`baa9befa38`). Branch: `maintenance/sglang-main-20261102`. Upstream changes
and PR decisions: [phase7-survey.md](phase7-survey.md).

This update preserves the Phase 6 behavior contract. CPU reviews and the
complete GPU matrix pass: kernels, deterministic TP2 text/bytes, compat-only,
both image sessions, fixed allocations and native checks. The original
v0.5.22 production release is restored healthy, including cold/warm smoke.
The prescribed final read-only GPU review clears without findings.
No publication or production promotion is part of this maintenance candidate.

## Behavioral contract and changes

The existing production reference `897286b12a` and base `35f3c96ff4` remain
unchanged. Preserve page-64 exact text/image prefix reuse, FP8 descales applied
once, complete recurrent/PLE byte snapshots, stable selection, deterministic
Marlin reduction, request-generation protection and release ordering. Existing
numerical expectations and tolerances are retained. Production output is a
compatibility oracle; it does not independently establish numerical validity.

- Scheduler and result-processing copies carry the current upstream bodies,
  including running-request ownership, KV weight-version tracking, deferred
  prefill and typed customized information. S05 accepts `num_held_rows`.
- Allocation, rollback and release use upstream `StaticKVStorage` and
  `StateStorage`; cache construction supplies the new typed `tree_slots`.
  Request rows and state slots are allocated separately upstream. CPU fixture
  ports use the real acquisition/release interfaces, retaining the exact
  byte, one-free and ordering assertions.
- Graph copies retain upstream capture hooks, live forward contexts, early
  verification metadata and revised replay preparation. K02 forwards the
  explicit `state_pool` argument. Removed PD-multiplex/ragged helper
  dependencies are replaced with their current definitions.
- QSA backend copies retain current routed layer reads and kernel signatures.
  The plugin's FP8 scaling, deterministic selection and offload branches are
  retained. Hyperconnection dependencies follow their upstream relocation.
- Drop Z02: merged upstream [#35955](https://github.com/sgl-project/sglang/pull/35955)
  now owns GPTQ MoE BF16 weight metadata and TP/act-order scale sizing. Replacing
  it with the old copy would also lose upstream's corrected act-order layout.
  REPLACE hooks decrease from 26 to 25; the new S10 ownership hook brings
  the manifest to 63 rows.
- Selectively incorporate the loader closure change from open upstream
  [#38900](https://github.com/sgl-project/sglang/pull/38900), head
  `9f0056b7889362b66d899c5797bf527f034d85a0`, into E08. A `nonlocal` boolean
  replaces the helper's self-reference, releasing the parameter snapshot
  before quantization post-processing without changing tensor calculations.
  A weak-reference regression with cyclic GC disabled checks the lifetime.
- The new scheduler collaborator requires `coordinator.request_lists()`. Add
  its ordered staging projection and S10, which delays the QSA running-set
  join until host-ready collection. Real scheduler/invariant-checker
  counterexamples fail before these fixes and pass afterwards; requests never
  simultaneously belong to staging and running.
- Compat-only QSA reads require typed KV router dependencies as well as the
  HiSparse runtime. Protect those reads at activation and test source tampering
  against an isolated copy of the pinned modules.
- Runtime numerical code remains unchanged. Its original buffer access and
  Mamba properties still resolve to the current storage. Fingerprints cover
  the new storage/router definitions; tests exercise real proxies and literal
  recurrent/PLE state bytes.
- Image artifact transport remains necessary: upstream's wider sentinel and
  `MultimodalDataItem.identity` do not populate the Qwen-VL item's full key.
  NUMA tests follow upstream's new CPU-only binding semantics and verify that
  an unavailable memory policy strips the plugin's interleave flag as well.

Manual source merges are explained per row in
`tests/regression/test_replace_deltas.py::RESOLVED`. Fingerprints are refreshed
from the pristine snapshot, independently of the plugin copies; they establish
source identity, not correctness of the port.

## Environment and evidence

Workspace-relative paths below are private local evidence, not repository
assets. `results/qsa-maintenance-20261102/runtime-env` is a separate venv using
Python 3.12.3. A `.pth` reads the existing Phase 6 packages; the new environment
locally overrides `sglang-kernel==0.4.10`, `flashinfer-python==0.7.2`,
`sgl-deep-gemm==0.2.3.post2` and `humming-kernels==0.1.18`. Torch 2.14.1,
Triton 3.8.0 and Transformers 5.19.0 remain unchanged from Phase 6. The measured native
versions are recorded in `environment.lock.json`. No shared interpreter was
modified. These installed versions are qualified by the GPU evidence below
for the declared profiles on 2× RTX 4090 48 GB (SM89).

The pristine snapshot is `sources/sglang-main-582168dd7c`; its standalone Git
metadata shares read-only historical objects with the legacy repository for
source-delta checks. Tests use `PYTHONPATH`, `CUDA_VISIBLE_DEVICES=99`,
`TRITON_INTERPRET=1` and no bytecode writes. GPU and service-lifecycle tests
are excluded by the existing runner.

| Check | Result |
| --- | --- |
| Frozen pre-change plugin (`git archive 983603dc7c`), old pin/env, complete CPU runner | 725 passed / 54 skipped / 29 deselected / 1 xfailed; subprocess integration 4 passed / 1 skipped; activated runtime integration 11 passed |
| Scheduler/lifecycle/prefix after ownership review fixes, maintenance env | 103 passed, 12 subtests passed |
| Graph/pools | 51 passed; G01/G02/K02 source-delta checks pass |
| Image/page-boundary/NUMA, maintenance env | 96 passed / 2 skipped / 1 xfailed, 6 subtests passed |
| Runtime source/dependency/new storage checks | 18 passed; unchanged runtime unit subset 17 passed |
| Framework/evidence/launcher/source-delta/plugin-off regression | 235 passed (maintenance env) |
| Complete final ordinary CPU suite after numerical remediation (reference checks enabled) | **787 passed / 54 skipped / 29 deselected / 1 xfailed**, 24 subtests passed; exit 0 |
| Final subprocess integration | **4 passed / 1 skipped**, 166 deselected |
| Final activated-runtime integration | **11 passed**, 72 deselected |
| Complete affected evidence suite after the optional-field schema adapter | **141 passed**; separate from the earlier complete 787-pass unit run |
| Fingerprints and generated manifest | 457 fingerprints match the pristine pin; 65 manifest rows, 25 REPLACE hooks; manifest current |
| Independent read-only CPU review | G7-CPU and both numerical-remediation CPU reviews cleared; `reviews/G7CPU*.md` |
| New-pin GPU qualification | Complete declared matrix passes (2026-11-02); G7-GPU cleared with no actionable findings |
| Original production restoration | Exact profile restored healthy at 12:59 America/Los_Angeles; text/image cold/warm smoke passes with prefix hits |
| Production transition | Not authorized; original release is retained |

The full pre-change baseline is in
`results/qsa-maintenance-20261102/baseline.log`. The first baseline attempt
used the working plugin while parallel edits began and was discarded; the
reported baseline comes from a separate immutable archive. Intermediate port
runs exposed changed fixture APIs, stale dependencies/manifest and a removed
upstream test; review also found the missing coordinator protocol, premature
running-set join and a compat-only fingerprint gap; these were fixed and rerun, not counted as passing evidence.

For plugin-off regression, measure pristine upstream without the plugin on
the path first (`pristine-upstream.json`/`.log`): 106 unit tests passed,
3 skipped; QSA 36 passed, 252 skipped and the same 13 CUDA-unavailable
failures as the previous pin; 34 subtests passed. Upstream #45672 deleted
`test_qsa_kv_pool.py`; the replacement subset uses its CPU
`test_request_table.py`. The plugin-off check requires this exact outcome
set, empty hooks and no activation record. The 13 expected failures are
hardware exclusions, not successful kernel tests.

Review corrections have before/after evidence in `scheduler-ownership-before.log`,
`scheduler-ownership-after.log`, `scheduler-ownership-suite.log` and the
`qsa-maintenance-20261102-request-lists-{before,after}.log` files. The router
source-tamper check and non-target scope checks pass (46 tests); their retained
logs are `router-regression.log` and `router-scope-check.log`. The selective
loader change also has an old-fails/new-passes witness in `loader-regression.log`.
After the code fixes, the full unit run passed 742 cases and found one
inventory classification error: S10 is an adaptation of main, not a hunk
from the reference fork. Move it to the plugin-only section 9; the manifest
and exact Appendix A assertions stay unchanged. All five inventory checks
then pass (`inventory-final.log`). The final clean complete unit rerun passed all 743 cases
(`cpu-unit-final.log`, exit 0); the same source's integration stages are in
`cpu-final.log` and remain valid after this documentation-only move.

Final commands supply `QSA_FORK_ROOT` explicitly, so optional
reference-digest and production CPU-probe checks are exercised rather than
skipped for an unset environment variable.

## Qualification scope

The owner authorized stop/regression/restore windows. The existing G2-1,
G2-2, compatibility-only, G2-4, image and G2-3 matrix uses the fixed legacy
reference, frozen inputs and exact observer/token/cache contracts. The main
GPU inventory was frozen before outcomes were run. Completed evidence is
reused when its source and environment remain unchanged; failed runs remain
separate. All declared GPU stages, production restoration and its cold/warm
smoke pass; the final read-only GPU review is cleared.

Open PRs that alter precision, Marlin launch behavior or prefix/cache
architecture are deferred with reasons in the survey. A future production
transition is a separate decision after the GPU evidence is available.

G7-CPU is cleared by the prescribed `gpt-6.1-sol` / `xhigh` read-only
review (`reviews/G7CPU.md`, exit 0). The reviewer observed the 742-pass run
plus the successful inventory correction; the later clean 743-pass run
confirms the same source. Only documentation/result records changed after
clearance. That CPU gate did not establish GPU qualification; the subsequent
GPU evidence is recorded below. Production promotion remains separate.

## Authorized GPU windows and numerical diagnosis

Owner approved the stop/regression/restore window on 2026-11-02. Production
was healthy and idle for three consecutive observations, then stopped at
08:45 America/Los_Angeles. The exact original profile is saved privately as
`results/qsa-maintenance-20261102/gpu/production-profile.json` (SHA-256
`821b87c9a9d4ffa4663f3dcadcb3a6d9b8ff001a6299844cd55c7bafb84e1231`).
The window orchestrator restores that profile in its finalization path,
tracks only processes it launched, and never adopts unrelated GPU jobs.
No publication or production promotion is included in the authorization.

Main kernel inventory was frozen before outcome runs: 105 required outcomes
and the existing SM121-only skip. Preserve all v0.5.22 outcomes and add ten
new main cases. #44114 renamed two BF16/FP8 strided-gather parameter IDs;
two exact source-derived aliases retain their cross-arm comparison, while
the new FP8-to-FP8 case remains plugin-only and must pass. Collection and
comparator CPU checks: 26 passed (`inventory-cpu-tests.log`).

| GPU check | Recorded result |
| --- | --- |
| G2-1 per-arm stages | All exit 0; step 2: 33 passed on each arm |
| G2-1 exact probe comparison | 5 reports equal: alignment 1, graphs 217, native 1,187, whole-K 1,187, top-k 110 items |
| G2-1 registered GPU outcomes | 91 common outcomes equal; 14 plugin-only outcomes pass; one SM121 skip on each arm; no declared known failures |
| fast_topk overflow/graph cases | 29 passed on each arm |
| G2-2 reference arm | Completed: concurrency, lifecycle, all 16 frozen text cases through 262,016 tokens and ledger checks pass |
| G2-2 initial candidate | Cross-arm token/logprob and first-prefill state failures retained; not counted as passing evidence |
| G2-2 corrected candidate | All four harnesses exit 0; 16 cases through 262,016 tokens and B2/B8 concurrency pass. Complete comparison: 10 reports equal; each rank has 1,505 exact capture/restore records and 1,514 exact ledger items after the source-backed optional-field normalization below |
| G2-4 fixed allocations | All 12 fields present/non-null and equal on both ranks; actual Mamba pool 2,374,447,760 bytes per rank |
| Final-source G2-1 rerun | All stages exit 0; five exact reports and frozen outcome inventory match |
| Compat-only | All 16 fixed cold cases pass per arm; complete comparison passes all 3 files (401 exact qualification items) |
| I5 images | Both ViT-cache-on/off sessions exit 0: 10 image cases and 10 text controls per session; both ranks' capture/restore checks pass |
| Native G2-3 | Both arms pass latency, lifecycle, 8×128-token concurrency and OpenAI smoke; all 31 requests' prompt/cache accounting and NUMA arguments match |

Evidence is private under `results/qsa-maintenance-20261102/gpu/`, including
`window.py`, `status.json`, per-stage logs, per-rank observer records and
`plugin-source-sha256.json`. Source hashes match the frozen package at this
point. F is the accepted legacy numerical reference `897286b12a`, not the
currently deployed v0.5.22 plugin; the original deployed profile is retained
for restoration. Recorded package sets:

| Package | F reference | P candidate |
| --- | --- | --- |
| torch | 2.14.1 | 2.14.1 |
| sglang-kernel | 0.4.9 | 0.4.10 |
| flashinfer-python | 0.7.0.post1 | 0.7.2 |
| triton | 3.8.0 | 3.8.0 |
| transformers | 5.17.0 | 5.19.0 |
| sgl-deep-gemm | 0.2.1.post1 | 0.2.3.post2 |
| humming-kernels | 0.1.12 | 0.1.18 |

Both GPUs are RTX 4090 48 GB (SM89), driver 595.84. Results remain bounded
to these profiles and hardware. They establish compatibility with the
frozen reference, not scientific validity or all-hardware certification.

The first window ended after a numerical mismatch. Original production was
restored healthy at 09:40 America/Los_Angeles with the exact saved profile.
The initial candidate differs on the very first cold 2,048-token prefill:
input tokens, M-RoPE and NGram context match, while KV/index and recurrent
state digests differ on both ranks. This predates prefix restoration and
decode. `gpu/numerical-divergence.json` retains the cross-arm differences;
failed candidate reports remain in place. Upstream #42087 changed the QSA
configuration selector under global batch-invariant mode. A bounded,
one-factor diagnostic (`selector-diag/`) confirmed that the selector changes
actual model calculations, but restoring the historical selector alone does
not restore F parity. The unmodified-selector control reproduces the failed
P KV/index and recurrent state exactly. Its pending-ring/RoPE digests differ
after a preceding request; these include inactive storage and cannot serve
as a fresh-process control. Both complete diagnostic reports are retained;
the proposed selector source fix was held back while diagnosis moved to
intermediate first-prefill tensors. No expected outputs, tolerances or
fixture selection have been changed.

Fresh F/P processes in `tensor-diag/` each ran the same 2,060-token/max-one
request. Both ranks reproduce every original first-checkpoint state field
exactly in their respective arms, validating the diagnostic instrumentation.
Of 709 common tensor/parameter points per rank, the first activation
difference is the first layer's HC mixed output. Its normalized input and
both projection weights are byte-identical. The original F/P generated HC
call bodies and two pointwise Triton kernels have identical ASTs, while both
GEMMs use `extern_kernels.mm(..., out=...)`, which dispatches to
`aten::mm.out`. Main #42087 newly registers that overload globally.

The bounded `hc-op-diag/` executes the original F cached HC function with
the saved, verified input/weights in two fresh candidate-interpreter
processes. Native global batch-invariant registration reproduces P's saved
HC output exactly on both ranks; skipping **only** `aten::mm.out`
registration restores F's saved HC output exactly on both ranks. This
establishes the first divergence's cause without changing model parameters,
tolerances or expected results. A scoped target-model CUDA compatibility
adapter (D01) and historical QSA launch selector are implemented. D01's
native namespace, dispatcher library and optional bmm fallback retain
upstream ownership and teardown; non-target and non-NVIDIA platforms
delegate. Targeted checks pass 89 cases with one skip; the native-registration
counterexample fails both bmm settings before adaptation. The selector's
global-invariant regression fails five cases before restoration. All 455
fingerprints match main, with 64 manifest rows and 25 REPLACE hooks. The
complete remediation unit run passes 772 cases (54 skipped, 29 deselected,
one xfailed); subprocess integration passes four cases (one skipped), and
activated-runtime integration passes 11. The complete runner exits zero.
Prescribed read-only review clears G7-CPU-remediation after two documentation
corrections: move D01 to the main-only inventory and correct the selector
ownership description (`reviews/G7CPU-remediation*.md`). Product source
remained unchanged during these corrections. Full GPU continuation starts
with the corrected candidate; the completed F reference arm is retained.
The operator diagnostic is not a full-model qualification result.

The first corrected full-model continuation is retained separately in
`gpu-continuation/` and is **not passing evidence**. All 96 captured KV/index
and recurrent checkpoints through 65,536 tokens match F on both ranks;
the first request's pending/RoPE fields also match. Its 65,548-token prompt
still differs at generated token 17, so the run was stopped instead of
continuing the remaining stages. Later inactive pending bytes diverge after
that decode history; the full-byte comparator remains unchanged. Diagnosis
now isolates the final 12-token prefill and first graph decode. Original
production was restored healthy at 10:38 America/Los_Angeles with the exact
saved profile; this continuation does not clear the GPU gate.

`decode-diag/` then captured the unchanged 65,548-token/max32 request's tail12
and first decode. Both arms reproduce their original first32 token IDs and
first-checkpoint fields. Early MoE-return tensors are not equivalent
boundaries: F has already applied the TP sum, while P defers it to the layer
boundary. Summing the saved rank-local gates and rounding to BF16 reproduces
F's return through layer3 exactly; these raw-return differences are not a
new numerical defect.

Main #42087 also caps the shared-expert gate at eight warps whenever global
invariant mode is on. At width 2,560, F uses 16 warps for rows below 1,024 and 8
otherwise. The layer0 tail gate is a retained negative control: all inputs
and outputs match despite different launch geometry (`gate-op-diag/`). The
first observed local gate delta is layer3/rank1: all four inputs match, but
three active BF16 values differ. Rank0 remains equal. A separately frozen
layer3 experiment reproduces each original output, restores F by changing
only that warp-cap condition, and preserves the rank0 negative control
(`gate-op-diag-layer3/`). Subsequent TP rounding masks this particular local
delta; it does not establish sole cause or full-model sufficiency.

D02 now preserves the exact reference launcher only for target NVIDIA CUDA
calls with global invariant mode on. It retains the native kernel and
delegates other calls. Independent source review passes; targeted CPU checks
pass 109 cases with one expected skip. The source is frozen at 65 manifest
rows, 25 REPLACE hooks and 457 unique protected definitions. Final CPU checks
pass 787 cases, then 4 subprocess integration cases (one skip) and 11 activated
integration cases; the complete runner exits zero. The prescribed D02 review
clears without findings (`reviews/G7CPU-D02.md`). The subsequent complete
GPU matrix passes; no tolerances or expected numerical/byte outputs changed.

The corrected G22 run completed every harness with exit zero. Its first
offline comparison accepted all numerical, byte and restore-consistency
reports but rejected two ledger files: F serializes
`graph_key.stream_idx: null`, while main omits it. Upstream
`a076243ceacf376b9632d8da37dc8e17005595bd` (#43898) removes exactly that optional
PDMux field, whose default `None` denotes one stream. Both recorded profiles
disable PDMux and overlap; all other fields in all 16 graph-capture records
are identical. The source/blob/profile audit is retained in
`graph-key-schema-audit.{md,json}`.

The evidence reader now normalizes only missing
`graph_capture_complete.graph_key.stream_idx` to `None`. Non-null values
remain compared, and other events, nested fields, shapes, dtypes and bytes
remain strict. Two old-reader counterexamples fail before this schema fix;
16 new regressions cover equivalence, immutable raw files and preserved
rejection paths. The entire evidence CPU suite passes 141 tests. No raw
report was rewritten. The failed `gpu-final/compare-g22.log` remains; the
complete offline rerun passes all 10 report files in
`gpu-final/compare-g22-schema.log`. Both ranks have 1,505 exact observer
records, including capture/restore consistency. Original production was
restored healthy at 11:52 before the remaining window began.

The remaining GPU stages are under `gpu-remainder/`. Its complete 78-file package
hash map equals the one used for the corrected G22 run. Final-source G21
passes again: five reports equal, 91 common outcomes plus 14 required
plugin-only outcomes pass and the existing SM121 skip is unchanged.
Compat-only then passes all 16 frozen cases in each arm and all three
comparison files (401 qualification items, server log and server info).
Both I5 sessions pass their 10 image cases and 10 text-reference controls,
with no missing-rank or capture/restore consistency problems.

TileLang emits static race-check warnings for QSA Logits stores during JIT.
The checks stayed enabled. A bounded source/generated-CUDA audit of the
seven cached specializations proves unique physical store ownership and
disjoint CTA intervals for these write sites under the declared valid
metadata. The warning witnesses violate their defining position/column
relations. Standard-library enumeration passes; evidence is in
`tilelang-warning-audit/{proof.md,ownership.json,check_ownership.py}`.
This closes those reported write witnesses for this profile; no GPU race
sanitizer, all-kernel guarantee or universal compiler diagnosis is claimed.

Native G23 passes all four harnesses in both arms. All 31 requests have
equal prompt/cache token counts; outputs are not required to be identical
under this native profile. Both ranks retain their NUMA CPU binding
(`--cpunodebind=2` / `3`) and `--interleave=all`. Each length has two
cold/warm pairs; measured medians below have no acceptance threshold and
include cold-start compilation effects. They are not a performance guarantee.

| Prefix tokens | F cold (s) | P cold (s) | F warm (s) | P warm (s) |
| --- | --- | --- | --- | --- |
| 8,192 | 7.084 | 9.576 | 0.126 | 0.132 |
| 65,536 | 9.292 | 9.246 | 0.198 | 0.220 |
| 262,016 | 59.272 | 64.578 | 0.505 | 0.534 |

The remaining window stopped idle production at 11:57:35 and restored the
exact original profile at 12:59:35 America/Los_Angeles. All ten remaining
stages exit zero, and the orchestrator exits zero with no restoration error
(`gpu-remainder/status.json`). Production still runs the v0.5.22 release
checkout `qsa-plugin-d9c2339`; its `src/` equals the previous qualified plugin
revision `983603dc7c`. The saved/current profile SHA-256 is unchanged:
`821b87c9a9d4ffa4663f3dcadcb3a6d9b8ff001a6299844cd55c7bafb84e1231`.
Health, ready state and model alias `local-qwen3.8-flash-next-v1` pass.
Synthetic text cold/warm requests both answer `42`; two procedural images
produce nonempty cold/warm answers with equal prompt counts. Warm hits are
2,048 text tokens and 4,352 image tokens. Native image answer identity is
not required. Evidence: `gpu-remainder/production-smoke{,-check}.json`.

`qualification-summary.json` gathers the corrected gates without rewriting
earlier failures. `source-hash-audit.json` checks the complete 78-file package
set: corrected G22, remaining-window before/after, and final source all
match. After that check, only `environment.lock.json`'s freeform `source`
note was updated from pending to qualified. The audit records both hashes
and confirms unchanged interpreter, package versions, runtime code and every
other package file. This metadata edit did not alter the tested computation.

The prescribed read-only `gpt-6.1-sol` / `xhigh` review exits zero and reports
no actionable findings or missing obligations within the declared scope
(`reviews/G7GPU.md`, G7-GPU cleared). Only this clearance record changed
after review. The maintenance candidate is complete; production promotion
and publication remain separate owner decisions.
