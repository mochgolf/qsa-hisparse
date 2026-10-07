# Parallel migration plan

Goal and owner decisions: [GOAL.md](GOAL.md). This file assigns parallel
tasks, file ownership, interfaces, and gates. Paths without a prefix are in
this repository; `fork:` is `../qsa-hisparse` at `ee8fe158d6`; `pin:` is
`../.worktrees/sglang-pin-76e06febab` (pristine SGLang `76e06febab`).

## Shared rules (every agent)

1. **Feature placement.** A fork change that can alter behavior while
   `SGLANG_QSA_HISPARSE_V3` is unset belongs to `model_compat`; a change that
   is behavior-identical to upstream whenever the HiSparse runtime is absent
   belongs to `hisparse`. Record the argument in the patch's `reason`.
2. **Same target in both features.** `model_compat` owns any REPLACE of a
   target both features touch, copying the fork body verbatim (including its
   `qsa_hisparse` branches, which are inert when the runtime is absent).
   `hisparse` may add only BEFORE/AFTER/AROUND hooks on such targets.
   Duplicate REPLACE fails activation.
3. **REPLACE bodies are copied from the fork**, not rewritten. Allowed
   mechanical edits only: explicit `super(Class, self)` for zero-argument
   `super()`, plugin-owned names for custom ops/JIT modules, imports rewritten
   to plugin modules, and preserved decorators (`lru_cache` etc.). List every
   edit in the patch `reason` and cover it with a test. Members the fork
   *added* (absent at the pin) use `attach`/`attach_value`, which fail if
   upstream defines the name. Name every other upstream definition whose
   behavior the copy assumes in `depends` (definitions with `@overload`
   variants cannot be fingerprinted; name their callers instead).
   Fingerprints: `tools/fingerprint.py write <module-name> <targets...>`,
   always against `pin:`. Never hand-edit hashes.
4. **Prefer the narrowest hook.** AFTER/AROUND at a function boundary first;
   REPLACE only for mid-function insertion. Each REPLACE is listed in
   `docs/patch-inventory.md` with the upstream interface that would remove it.
5. **Tests.** CPU only (`tools/run_cpu_tests.sh`, which hides CUDA and uses
   the Triton interpreter). Port the fork's tests with their assertions
   intact; do not loosen or delete assertions to make a port pass. New
   ordering/ownership tests use independent reference construction.
6. **Prohibited** unless the task card says otherwise: GPU use, starting or
   stopping services, installing into or modifying any Python environment
   (`flash-next-env`, `quant-env`, `../service/runtime-env-sglang-20260923`,
   which is the default interpreter of `tools/run_cpu_tests.sh` and is used
   read-only), editing `fork:` or `pin:`, pushing, opening or commenting on
   PRs/issues. Write only inside your assigned paths.
7. **Intentional deviations** from the fork are allowed only when listed in
   [DEVIATIONS.md](DEVIATIONS.md) by the orchestrator.
8. The fork's `test_service_control.py::ServiceLifecycleTests` starts systemd
   user units; deselect it (`-k 'not ServiceLifecycleTests'`).
9. **Target-model scope.** `model_compat` hooks on generic paths (GPTQ /
   AutoRound MoE loading, Marlin MoE, request-row generations, QSA kernels
   shared with other QSA models) call the original unless
   `sglang_qsa_hisparse.scope.target_model_active()` is true. The target set
   is the fork's validated model (`Qwen4ExpForConditionalGeneration` and the
   text model it builds). The predicate reads SGLang's published model
   configuration and raises if it cannot decide; it never guesses. Each
   row's scope decision goes in the patch `reason`.
10. **Report** in your final message: files changed, tests run with results,
   unresolved items, and any deviation from this plan.

## Layout

| Path | Content | Owner |
| --- | --- | --- |
| `src/sglang_qsa_hisparse/{plugin,features,patching,fingerprint,errors}.py`, `patches/framework.py` | Framework (fixed contract) | orchestrator |
| `src/sglang_qsa_hisparse/{launch,scope}.py` | Launcher preflight/activation records; target-model scope (rule 9) | W7 |
| `tools/evidence/` | Byte observer, F-vs-P comparator, compat-only reduced profile | W8 |
| `src/sglang_qsa_hisparse/hisparse/` | Fork `mem_cache/qsa_hisparse/*` and `hisparse_graph.py` as `graph.py` (moved verbatim, imports rewritten) | W1 |
| `src/sglang_qsa_hisparse/kernels/` | Plugin-owned kernels (`stable_align.py`, Marlin JIT copy, PLE gather, stable HC) | W4/W5 by file |
| `src/sglang_qsa_hisparse/patches/model_compat/*.py` | Shared-path patches | W4, W5 by file |
| `src/sglang_qsa_hisparse/patches/hisparse/*.py` | Runtime integration patches | W2, W3, W4 by file |
| `src/sglang_qsa_hisparse/fingerprints/<module>.json` | One file per patch module | that module's owner |
| `tests/<workstream>/` | Ported and new tests | that workstream |

Import mapping: `sglang.srt.mem_cache.qsa_hisparse.X` →
`sglang_qsa_hisparse.hisparse.X`; `sglang.srt.layers.attention.qsa.hisparse_graph`
→ `sglang_qsa_hisparse.hisparse.graph`; `...fused_moe_triton.stable_align` →
`sglang_qsa_hisparse.kernels.stable_align`.

## Phase 0: baseline and inventory (parallel)

| Task | Output | Notes |
| --- | --- | --- |
| P0-A inventory | `docs/patch-inventory.md` | Every fork change (28 modified srt files, Marlin JIT/op files, new files) → target, feature (rule 1), hook type, workstream, fingerprint targets, `depends`, upstream interface that would remove a REPLACE |
| P0-B baseline | `docs/baseline.md` | Fork CPU suite at `ee8fe158d6` in a new detached worktree; pinned upstream subset of the same upstream tests; GPU equivalence fixture inventory (locations, frozen inputs, commands) without running GPU |
| P0-C upstream status | `docs/upstream-status.md` | Re-verify #26161, #34398, #22038, #38855, #35485, #39862; upstream `main` changes since the pin to each planned PR area; contribution requirements |
| P0-D framework | done by orchestrator | Plugin skeleton, fail-closed activation, fingerprints, `attach`, CPU runner, moved runtime package |

Gate G0: Codex review of framework + three documents.

## Phase 1: plugin parity on the pin (parallel workstreams)

| Workstream | Fork scope | Plugin files |
| --- | --- | --- |
| W1 runtime | `mem_cache/qsa_hisparse/*` minus prefix modules' tests, `hisparse_graph.py` | `hisparse/` (except `prefix*.py` behavior), `tests/runtime/` ← fork `test_runtime.py`, `test_slots.py`, `test_single_request.py`, `test_gather.py` |
| W2 scheduler/lifecycle | `managers/scheduler.py`, `scheduler_components/{batch_result_processor,weight_updater}.py`, `schedule_policy.py`, `mem_cache/{allocation,common,memory_pool}.py`, `allocator/paged.py` | `patches/*/scheduler.py`, `patches/*/lifecycle.py`, `tests/lifecycle/` incl. release-ordering ledger test |
| W3 pools/graph | `pool_configurator.py`, `kv_cache_configurator.py`, `qsa_kv_pool.py`, `model_runner.py`, `decode_cuda_graph_runner.py`, `forward_batch_info.py` | `patches/*/pools.py`, `patches/*/graph.py`, `tests/pools/` ← fork pool configurator tests |
| W4 QSA attention | `qwen_sparse_attn_backend.py`, `qsa/{kernel,metadata,qsa_indexer,sparse_attn}.py` | `patches/model_compat/qsa_attention.py`, `patches/hisparse/qsa_backend.py`, `kernels/qsa_*.py`, `tests/qsa/` ← `test_deterministic_topk.py`, `test_flash_attention.py`, registered `kernel/qsa/test_qsa.py` additions |
| W5 model compat | `quantization/{auto_round,gptq/schemes/gptq_moe}.py`, `gptq_kernels.py`, `fused_marlin_moe.py`, `moe/.../layer.py`, Marlin JIT `.cuh/.h` + op wrapper, `hc_mix_triton.py`, `hyperconnection.py`, `models/qwen4_exp.py`, `utils/common.py` | `patches/model_compat/{quantization,marlin,qwen4_exp,hyperconnection}.py`, `kernels/csrc/marlin_moe/`, `kernels/{marlin_moe,hc_mix,ple_gather}.py`, `tests/model_compat/` ← `test_marlin_deterministic_alignment.py`, `test_model_compatibility.py`, hc mix test additions |
| W6 prefix cache | `qsa_hisparse/{prefix,prefix_cache}.py` integration surface | `tests/prefix/` ← `test_prefix_cache.py`; `docs/prefix-cache.md`; verify every scheduler/allocation call site the prefix cache relies on is covered by W2 patches; evaluate building the host cache through the pin's `register_radix_cache_backend`/`--radix-cache-backend` instead of a scheduler patch (needed anyway at the next pin, see Phase 4) |
| W7 launch/regression | fork `scripts/qsa_service.py`, `test/qsa_hisparse/test_service_control.py` | `src/sglang_qsa_hisparse/launch.py`: preflight (entry point discoverable from a private dist-info path that spawned processes inherit, allowed by `SGLANG_PLUGINS`), sets `SGLANG_QSA_ACTIVATION_DIR`, starts the server, and fails unless every scheduler/TP rank wrote an activation record before readiness; `src/sglang_qsa_hisparse/scope.py` (rule 10) with tests; service-control port adapted to the launcher (lifecycle tests deselected per rule 8); `tests/regression/`: off ⇒ zero hooks and pinned upstream tests pass; non-target model ⇒ scoped hooks delegate; inventory completeness (every fork hunk mapped) |
| W8 evidence tooling | fork `test/manual/qsa_hisparse_prefix_*.py`, baseline gaps 5–7 | `tools/evidence/`: a neutral checkpoint-byte observer (hashes of restored/captured raw K/V, index, pending ring, Mamba/PLE state per request and rank, enabled by env, identical in F and P), an offline F-vs-P comparator (rid-normalized ledgers, token IDs, hashes), and the reduced deterministic compat-only profile (context, `--max-total-tokens`, fixtures) with its run script; all CPU-tested |

Row IDs in `docs/patch-inventory.md` assign each fork hunk to a workstream;
a workstream implements exactly its rows. Use the inventory's equivalence
arguments for narrow hooks; if an argument fails, fall back to the listed
REPLACE and report it.

Interfaces between workstreams: W2/W3/W4 patches call the runtime only through
the attributes the fork already uses (`kvcache.qsa_hisparse`,
`uses_qsa_hisparse_leases`, `graph_enabled`, coordinator methods). W1 keeps
those names unchanged. Integration (orchestrator): merge W1–W7, run the full
CPU suite and `tools/fingerprint.py check`.

Gate G1: Codex review of the merged Phase 1.

## Phase 2: GPU equivalence (serial, owner-approved window)

Approved by the owner on 2026-10-07 (interpreter
`../service/runtime-env-sglang-20260923`). All criteria compare fresh fork and
plugin runs from the same window; historical goldens are informational.
G2-1 kernel and graph checks (Marlin deterministic, QSA FP8 extraction,
deterministic top-k probe, B1–B8 capture). G2-2 frozen deterministic HTTP
fixtures: fork vs plugin `compat+hisparse` token IDs and cached bytes/state
(W8 observer), and fork (V3 unset) vs plugin `compat` on W8's reduced
profile.
G2-3 prefix acceptance, concurrency, latency, lifecycle scripts. G2-4 memory:
identical `fixed_bytes` and static pools. Gate G2: Codex review of evidence.

## Phase 3: two parallel tracks

**Track I, image prefix reuse (in the plugin).**
I1 identity: carry the full artifact key, content digest and grid to the
scheduler for Qwen-VL image items; define the bypass set (video/audio,
precomputed embeddings, skipped hashing, multi-span items, unscoped hashes).
I2 matching: snapshot schema with image records intersecting `[0, L)` and a
page-cumulative M-RoPE digest; `HostPrefixCache.acquire` and TP signatures.
I3 inside-image boundaries: suffix embedding slicing via `extend_prefix_len`,
per-image ViT cache reuse, PLE n-gram history over pad tokens, checkpoint
capture at every page64 boundary. I4 CPU counterexamples. I5 GPU validation
(window). I1 and I2 run in parallel against an `ImagePrefixIdentity` contract
fixed first by the orchestrator; I3 follows I1. Gate G3-I.

**Track U, upstream PRs (branches from fresh `upstream/main`, local only).**
Upstream moved 1201 commits past the pin and is actively reworking several
target areas (`docs/upstream-status.md`). Each U task first decides between
contributing to an existing upstream PR, rebasing onto it, or a new PR, and
follows upstream style (no defensive `getattr` probing, `msgspec.Struct`
containers, `model_runner.py` orchestration only, tests under
`test/registered/unit/<module>` with `register_cpu_ci`). Engaging on an
existing upstream PR (comments, reviews) is publication and needs owner
confirmation.
U1 HiSparse coordinator protocol and scheduler gating (align with open
#35488); U23 (one owner) prefix-cache lifecycle hooks on the reworked
upstream API (`checkpoint`, `on_release`, `claim_kv_row`; track
#42823/#42824/#42825/#42923) together with allocator free-group/release
callbacks and KV pool runtime attachment, because both meet in
`release_kv_cache` (inventory M03); its lifecycle contract (checkpoint, drain,
logical release, free-group flush, physical reuse) is frozen in the task
before code;
U4 pool fixed reservation + `full_kv_pool` factory; U5 decode CUDA graph
backend lifecycle hooks, widening the existing extend-only
`ForwardBatch.req_pool_indices_cpu`; U6 QSA backend extension points, FP8
descales, SM86/SM89 flash-attention fallback, `decode_score_width`;
U7 GPTQ MoE scale sizing/dtype incl. w13 `size_k` (align with open #35955),
AutoRound g64 split, INT8-row PLE (meta-device table is already upstream,
#39928); U8 deterministic Marlin whole-K, QSA stable top-k and stable HC
(align with open #42087); U9 monotonic `req_generation`, HiSparse decode batch
multimodal inputs. Track I's identity transport aligns with draft #41792
(`MultimodalDataItem.identity`). Each task works on its own branch from the
same `upstream/main` commit; U1 and U23 both touch scheduler release paths,
so U23 rebases onto U1 if both are prepared. The rest are independent. Gate G3-U per batch, then
owner confirmation per PR.

## Phase 4: shrink REPLACE (after upstream merges)

Each merged PR: new pin cycle, convert the matching REPLACE into a protocol
implementation, re-run Phase 1 CPU and Phase 2 GPU gates. Known break at the
next pin: upstream #42354 gives hybrid-SSM models (Qwen4-Exp) a
`UnifiedRadixCache` under `--disable-radix-cache` and rejects caches without
`supports_mamba()`, so the host prefix cache must become a registered radix
cache backend. Upstream test paths also moved (QSA tests to
`test/registered/kernels/ops/attention/qsa/`, `hc_mix_triton.py` to
`kernels/ops/gemm/hc_mix.py`).

## Review protocol

After each gate's work is merged, run from this repository:

```bash
codex exec -m gpt-6.1-sol -c model_reasoning_effort='"xhigh"' -s read-only \
  -o reviews/<gate>.md - < reviews/<gate>.prompt.md
```

The prompt names the gate scope, GOAL.md, PLAN.md, and the evidence files.
Findings are fixed or answered in `reviews/<gate>-response.md` before the
next phase starts.
