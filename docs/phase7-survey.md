# Phase 7 survey: v0.5.22 to upstream main

Routine maintenance, 2026-11-02. This survey fixes the upstream comparison at
`582168dd7c3821198c92766f0319b9e1c8c17bd3` (main, committed
2026-11-02 07:10:40 America/Los_Angeles). It does not track a moving branch
during the port. The previous plugin release is `983603dc7c`, with SGLang
`baa9befa381511cd4d56f83980951e5a45fffc6d` (v0.5.22).

Evidence: GitHub plugin metadata and patches retrieved on 2026-11-02,
`git log`/`git merge-base` in the fork's upstream object store, and source
inspection of the immutable main snapshot. No upstream comments, pushes,
issues, or pull requests were created. GitHub's `merge_commit_sha` for an
**open** PR is a temporary test merge, not evidence that it was merged.

## Source and dependency changes

Main has **1,528 commits** not reachable from v0.5.22. The merge base is
`684fea8af3b11a5b51ec9966c02959a32ad3421c`; the release has its own
cherry-picks, so this is not a fast-forward of the release branch.

| Requirement | v0.5.22 / current lock | Main snapshot |
| --- | --- | --- |
| torch | 2.14.1 | 2.14.1 |
| transformers | 5.19.0 | 5.19.0 |
| sglang-kernel | 0.4.9 | **0.4.10** |
| flashinfer-python | 0.7.1 | **0.7.2**, cu13 extra |
| sgl-deep-gemm | 0.2.2 | **0.2.3.post2** |
| Python requirement | plugin >=3.10 | SGLang >=3.11 |

The old validation interpreter can provide CPU source-port diagnostics,
but its native packages do not establish that this dependency set starts or
runs on a GPU. A launcher lock must describe measured installed packages,
not pretend that a dependency bump has already been exercised.

The 405 unique fingerprints in plugin `983603dc7c` compare with main as
follows: **53 unchanged, 176 module-only changes, 167 changed definitions,
9 missing dependencies**. No hook target itself is missing. The removed
dependencies are the old decode-graph ragged helpers, PD multiplex context,
ragged-verify resolver, `HybridReqToTokenPool.alloc` and
`free_mamba_cache`, two old HC elementwise symbols, and the old contiguous
GPT-OSS sigmoid helper. These counts were computed from the old committed
fingerprint records, before regeneration; refreshing hashes is not a
semantic compatibility check.

## Merged changes to retain

Each merge below is an ancestor of the frozen main snapshot and is not an
ancestor of v0.5.22 (checked with `git merge-base --is-ancestor`). Adoption
means retaining the upstream implementation while carrying only the
plugin-specific delta, rather than cherry-picking the PR a second time.

| PR | Merge | Port consequence |
| --- | --- | --- |
| [#35955](https://github.com/sgl-project/sglang/pull/35955), GPTQ Marlin BF16 and TP act-order scales | `3256c33825739d12261572f97437300ddef62950`, 2026-10-29 | Upstream now allocates scale tensors with `params_dtype`, sizes the w2 table correctly, and honors full w2 loading. Reassess Z02/Z04 against this; the plugin's separate w13 scale permutation and AutoRound g128-to-g64 split are different changes. |
| [#42087](https://github.com/sgl-project/sglang/pull/42087), Qwen batch-invariant kernels | `e3205080ce400eb8ef3d5261fe1ef8c122548ac6`, 2026-10-20 | Retain upstream improvements with three scoped compatibility adaptations: D01 omits only the new target NVIDIA CUDA `mm.out` registration (same-input HC controls isolate its numerical change); plugin QSA keeps the reference launch selector; D02 keeps the exact reference shared-gate launcher for target NVIDIA CUDA/global-ON calls. Non-target paths delegate. Shared-gate controls include a negative layer and a TP-masked local delta, so they do not alone prove full-model sufficiency. The complete declared GPU matrix subsequently passes with these adaptations; no tolerance or golden changes. |
| [#43043](https://github.com/sgl-project/sglang/pull/43043), image identity | `3b932cde262945fb489e187f658fde7bbecd31a2`, 2026-10-18 | Generic multimodal items gain a 32-byte identity and wide sentinels. The full-identity producer is Kimi-specific; Qwen artifact-key transport remains necessary. Port tests that depend on the old 30-bit padding format. |
| [#44114](https://github.com/sgl-project/sglang/pull/44114), FP8 sparse decode scratch | `3b34d9f8b6edc23216f29c1edf02d1ff0306f9de`, 2026-10-17 | Q10/Q11 copies must retain the upstream FP8 scratch change and apply checkpoint descales once. |
| [#46543](https://github.com/sgl-project/sglang/pull/46543), caller-held QSA top-k/PLE outputs | `5e420a1258679719f88bee2b2cd5d19cde2da8dc`, 2026-10-26 | Preserve output-buffer ownership and live metadata when rebasing indexer hooks. |
| [#45791](https://github.com/sgl-project/sglang/pull/45791), long cached-prefix indexer | `642e9c1448ccb1f95c3d0fd1153e4f3b8cb478f1`, 2026-10-26 | Relative prefill scoring changes the metadata and kernel path used on long host-prefix hits. |
| [#45753](https://github.com/sgl-project/sglang/pull/45753), generated-KV gather | `148bea73026df8780c393860a9acaacb0317ccfd`, 2026-10-27 | Preserve the upstream removal of the extra D2D copy in non-offloaded prefill. |
| [#46980](https://github.com/sgl-project/sglang/pull/46980), recurrent-state ownership | `306a699a6096e761485053af1c7d3e9113f64e1f`, 2026-10-28 | The host-prefix implementation must use the new recurrent-state owner and release APIs; old pool-method names are insufficient. |
| [#46990](https://github.com/sgl-project/sglang/pull/46990), typed KV views | `d5bc823ad58ad73f280d87303410c20824a7f717`, 2026-10-28 | Attention, graph and pool code must follow routed KV access. The surrounding #46977–#46991 refactor also separates storage, allocation identities and request-row lifecycle. |

The scheduler ownership series visible in main (#43895–#43902) changes
request KV-length setters, finish-state setters, running-request ownership,
step-launch accounting and transfer queues. In particular a replacement of
`get_next_batch_to_run` must preserve upstream launch and response ordering;
a clean textual merge alone is insufficient.

The Qwen-VL processor still sets `uses_media_artifacts_without_cache` only
for `qwen3_vl` and `qwen3_vl_moe`; `qwen4_exp` in the nearby worker-concurrency
list does not enable that feature. The plugin's positive preprocessing-cache
requirement and full artifact-key transport therefore remain applicable.

## Selected narrow open-PR candidate

[#38900](https://github.com/sgl-project/sglang/pull/38900), head
`9f0056b7889362b66d899c5797bf527f034d85a0`, is open and mergeable on the
survey date. Its production change removes the nested PLE loader helper's
self-reference (`load_qwen4_exp_ple_shard._warned_downcast`) in favor of a
`nonlocal` boolean. That cycle otherwise retains `params_dict` and old
loader-format parameters until cyclic GC, extending memory lifetime into
quantization repacking. The plugin's E08 replacement has the same cycle.

**Recommendation: incorporate this lifetime fix into E08**, with credit and
the frozen PR head, and add a CPU weak-reference test with cyclic GC disabled
plus warn-once/value-preservation checks. It does not change loaded values,
the supported PLE formats, or kernel precision. Record the implementation and
actual test result in the Phase 7 result report; this survey alone does not
claim the candidate is implemented or tested.

## Open PRs deliberately deferred

PR state and heads below are observed on 2026-11-02. An open PR's green
mergeability is a textual GitHub result, not runtime validation.

| PR | State; frozen head | Reason to defer |
| --- | --- | --- |
| [#47800](https://github.com/sgl-project/sglang/pull/47800), FP32 deterministic matmul precision | open, mergeable; `ad6a401974a028fd4afedc7cc4194fa62168540c` | Changes global deterministic FP32 `tl.dot` from implicit TF32 to IEEE unless TF32 is explicitly enabled. This is a meaningful precision/output change with GPU validation requirements; BF16/FP16 calls are unchanged. Do not hide it inside a source-compatibility refresh. |
| [#47188](https://github.com/sgl-project/sglang/pull/47188), Marlin one block per SM | open, mergeable; `9a62e6ce64cbe46f971080c27edcba9fd88700cb` | Relevant SM86/SM89 split-K corruption fix, but requires the repeated-launch GPU reproducer. The plugin's deterministic whole-K branch already fixes `blocks_per_sm=1`; its ordinary auto-config branch still merits a separate GPU check. |
| [#47878](https://github.com/sgl-project/sglang/pull/47878), exhaustive QSA dense prefill | draft, mergeable; `fdd4371ba2991bd25068f7ab1bcb231117c5b2a1` | Changes attention reduction order and dispatch, including FP8 prefix reads; requires numerical and performance evidence on the plugin's hardware/profile. |
| [#45865](https://github.com/sgl-project/sglang/pull/45865), cpu_tensor QSA retraction | open, mergeable; `012e853665c55a437515bc8acb2c97d872592b18` | Adds compressed keys and ring state to ordinary pool CPU backups. The offload runtime rejects retraction and uses its own complete snapshots; importing this does not add supported runtime retraction. Revisit for compat-only retraction support. |
| [#47051](https://github.com/sgl-project/sglang/pull/47051), two-group pending ring | open, conflicting; `f78ee79014c5a5eb490e338a7d2e1ced2e340145` | Fixes speculative-verify ring overwrites. The P2 runtime requires speculation off; changing its ring layout now would expand the supported contract without addressing an active path. |
| [#46352](https://github.com/sgl-project/sglang/pull/46352), remaining decode reservation | open, mergeable; `83fcc7d1c815229b17293f35fc2a144fafdbf2ed` | Fixes readmission after decode retraction and related shared scheduler paths. Runtime retraction is rejected; avoid a global admission-policy change solely for this port. |
| [#47895](https://github.com/sgl-project/sglang/pull/47895), bounded allocations and image fixes | open, mergeable; `a72059600b5bb5124d6a4c94e7eb62693a73bc67` | Bundles empty-grid, grayscale GPU JPEG, GPU image chunking and bounded PLE host registrations. The supported profile uses PIL preprocessing; bounded host-registration ownership is a separate platform-sensitive implementation. |
| [#46926](https://github.com/sgl-project/sglang/pull/46926), FlashInfer post release | open, mergeable; `003c85cca261ecd692783ca320ce248b90e43fc7` | Keep the snapshot's 0.7.2 requirement until a separately tested dependency bump; no reason to silently change the native package set twice. |
| [#44444](https://github.com/sgl-project/sglang/pull/44444), required plugins | open, mergeable; `27c2e8010e5355ddfde5e964c1f110c5a3247140` | Useful upstream fail-fast API, but modifies the general loader/hook contract across eight files. The plugin launcher already verifies activation per rank; retain that path until upstream accepts the API. |
| [#38464](https://github.com/sgl-project/sglang/pull/38464), early plugin loading | open, mergeable; `8e610cb4135120c036c996d8ba55dbfc690465a4` | Plugin process loading remains the launcher's responsibility in this cycle. |
| [#36968](https://github.com/sgl-project/sglang/pull/36968), SM89 decode preflight | open, mergeable; `0570f5e17dbfe1aecea74a7d24d16733237d3eb8` | Its backend resolver/fail-early policy differs from the plugin's qualified vendored FA3 fallback on SM89; wholesale adoption could reject that supported configuration. |
| [#36644](https://github.com/sgl-project/sglang/pull/36644), QSA FP8 KV support | open, mergeable; `3df8e1e7dbc5807696622afe2929b6c33c185ca3` | The plugin already carries its own FP8 descale implementation and upstream has fused/typed-view changes. Avoid a second, overlapping scale path. |
| [#41461](https://github.com/sgl-project/sglang/pull/41461), deferred FP8 widening | open, conflicting; `191da9f3573bef0fd466a0f4a1e3fc5e37f6718a` | Performance kernel change overlaps the plugin FP8-aware sparse kernels; needs a dedicated GPU comparison. |
| [#43032](https://github.com/sgl-project/sglang/pull/43032), packed PLE prefill | open, conflicting; `aa4bdc5d63951597d45bababa480a3284ccce3c0` | Adds a different convolution/state-writeback kernel path. Its H20 evidence is not plugin SM89 prefix-state parity evidence. |
| [#35485](https://github.com/sgl-project/sglang/pull/35485), M-RoPE continuation | open, conflicting; `85c95172af16a3cb30c88c6d7fae2800460ac05e` | Existing stale patch is not a clean current-main fix; normal page-aligned host hits remain within the prompt's precomputed positions. |
| [#35488](https://github.com/sgl-project/sglang/pull/35488), pluggable HiSparse backing | open, conflicting; `a9b734d1d94368fc39d1e44af84b6160e3c0cb57` | Broad new HiCache backing/protocol (27 files, +4,265/-131), not a compatible replacement for QSA host prefixes and full recurrent/image identity snapshots. |
| [#26161](https://github.com/sgl-project/sglang/pull/26161), unified tree/HiCache | open, conflicting; `12f320f2d52a04e8fb21900668c5dd4067b28cdb` | Stale DSA-oriented path, no demonstrated QSA/multimodal equivalence. |
| [#44345](https://github.com/sgl-project/sglang/pull/44345), unified compressed QSA pool | draft, conflicting; `bc949179bad66d9598700f0b9e6076ca999285a1` | New pool layout, not a necessary TP2 private-host runtime fix. |
| [#43527](https://github.com/sgl-project/sglang/pull/43527) / [#44447](https://github.com/sgl-project/sglang/pull/44447), QSA KV sharding | open, conflicting; `4f0a18fdd1aa1f307ebdda15c6a6645a22c56007` / `880f5125e40108b55abeb5cc393bbaa51ba406ec` | Adds sharded prefill/PD topology outside the supported TP2 runtime. |

The AutoRound source has no commits in the old-pin-to-main range; the
plugin's TP group split still needs its own delta. Searches for INT8-row PLE
did not identify an upstream replacement for the plugin's storage/gather
contract. This is a scoped search result, not a claim that no relevant PR
exists anywhere.

## Evidence boundary

This document is a source/API and PR-selection survey. PR authors' reported
GPU results are upstream context, not tests executed here. CPU port results,
replacement-delta checks, activation fingerprints and review findings belong
in the result report. Existing v0.5.22 production qualification remains tied
to its original code and native environment; it cannot qualify this main
snapshot automatically.
