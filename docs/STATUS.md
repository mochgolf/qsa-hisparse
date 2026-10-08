# Status

Progress log for [PLAN.md](PLAN.md). Newest entries last.

| Gate | State | Evidence |
| --- | --- | --- |
| P0-D framework | done (`f8f0958`) | `tests/test_framework.py` 15 passed on the pin |
| P0-A inventory | done | `docs/patch-inventory.md`: 196 hunks → 89 rows, 39 REPLACE (~3,900 copied lines), 63 targets resolve at the pin |
| P0-B baseline | done | `docs/baseline.md`: fork CPU 183 passed/7 skipped; pin subset 72 passed/3 skipped |
| P0-C upstream status | done | `docs/upstream-status.md`: main `0b635266d4`, 1201 commits after pin |
| G0 review | cleared (`reviews/G0final.md`) | `reviews/G0*.md` |
| Phase 1 W1–W8 | done; G1 cleared | `reviews/G1*.md` |
| G1 review | cleared (`reviews/G1r.md`) | |
| Phase 2 GPU | done; G2 cleared (orchestrator closure after 4 review rounds) | `docs/phase2-results.md`, `reviews/G2*.md` |
| Phase 3 tracks I/U | not started | |

## Log

- 2026-10-07: Goal and decisions recorded. Pinned worktree
  `../.worktrees/sglang-pin-76e06febab` created. Found that SGLang's
  `load_plugins`/`apply_hooks` swallow `Exception`; activation errors derive
  from `BaseException` and every applied target is verified. Fork runtime
  package (9 modules) and `stable_align.py` import cleanly on the pin.
- Open item (W7): if `SGLANG_PLUGINS` excludes `qsa_hisparse`, or the
  dist-info is not on the path of a spawned process, switches are ignored and
  SGLang serves pristine upstream silently. Add a launcher that verifies the
  entry point is discoverable and allowed before starting, and evidence of
  activation in every process (main, scheduler/TP).
- P0-B: fork suite's `test_service_control.py::ServiceLifecycleTests` starts
  and stops short-lived systemd *user* units `qsa-cpu-test-<hash>` against a
  fake server (none remain). Plugin runs must deselect it unless the owner
  allows it. Fork bug found: registered `test_qsa.py` paged-extend test fails
  (`QwenSparseAttnBackend` built via `__new__` lacks `qsa_hisparse`).
  Phase 2 needs: an interpreter matching the pin (sglang-kernel 0.4.7; the
  candidate is `service/runtime-env-sglang-20260923`, owner approval), a
  dist-info launch path visible to spawned processes, fork golden outputs
  regenerated in the same window, and a neutral cached-byte hashing tool.
- P0-C plan impacts (to fold into PLAN.md after P0-A): Track U must align
  with moving upstream work instead of duplicating it: U1 vs open #35488
  (HiSparseCoordinator protocol), U2 vs reworked prefix-cache lifecycle
  (`checkpoint`, `on_release`, `claim_kv_row`; open #42823/4/5, #42923 drop
  `Req.prefix_indices`), U5 widen existing `req_pool_indices_cpu`, U7 vs open
  #35955 (GPTQ scales) and merged #39928 (meta PLE), U8 vs open #42087
  (stable top-k, batch-invariant HC), I1 vs draft #41792
  (`MultimodalDataItem.identity`). #42354 (main only) replaces ChunkCache with
  UnifiedRadixCache for hybrid-SSM models under `--disable-radix-cache`: the
  host prefix cache must become a radix-cache backend with `supports_mamba()`
  at the next pin. Evaluate `register_radix_cache_backend` (exists at the
  pin) for W2/W6. Plugins are not loaded in tokenizer-worker/detokenizer
  subprocesses at the pin (open #38464), relevant for Track I processor hooks.
- P0-A done. Framework gained `attach`/`attach_value` (fork-added members,
  fail if upstream defines the name); 19 framework tests pass. Deviations
  D1–D3 recorded in DEVIATIONS.md. PLAN.md updated with rule 3 mechanical
  edits, inventory-row ownership, Track U alignment, next-pin break (#42354).
- Owner (2026-10-07): GPU validation and `service/runtime-env-sglang-20260923`
  approved; deviations D1–D3 accepted.
- G0 (gpt-6.1-sol xhigh): 2 blockers, 9 majors, 1 minor, all accepted and
  fixed (see `reviews/G0-response.md`); framework tests 38 passed.
- G0 round 2 (gpt-6.1-sol xhigh): 6 resolved, 6 partial, 9 new (2 blockers).
  All fixed (`reviews/G0r-response.md`): module-level verifier, binding
  chains per mode, own-target apply with frozen hooks, manifest from the
  inventory, property rejection, full binding analysis. 56 framework tests.
- G0 round 3: 4 must-fix (2 blockers) fixed: dependency protection,
  identity-complete binding chains (`Undescribable` otherwise), duplicate
  declaration rejection, wheel manifest. 68 framework tests. CUDA chains added.
- 2026-10-07: production QSA service observed running (scheduler TP0/TP1,
  ~45 GB per GPU), started outside this work. A CUDA-mode fingerprint refresh
  (imports only, no model) ran while it was up. GPU validation must wait for
  an agreed window; never stop or reuse its GPUs without the owner.
- G0 round 4: 2 blockers (shallow behavior records; unmanifested same-source
  hooks) and 2 minor items fixed; threat model documented in PLAN.md. CUDA
  chains must be regenerated in a GPU window (chain format 2). 80 tests.
- Owner (2026-10-07): guard against over-engineering. Round-5 review stopped.
  Removed binding chains (recursive closure/class/operator descriptions,
  CPU/CUDA modes) and in-place mutation snapshots; activation now relies on
  module bytes, manifest, exact registry entries, identity checks and the
  launcher's exclusive plugin loading and version lock. PLAN rule 10 (keep it
  simple) added. 62 framework tests.
- G0 cleared by the proportionate final review; its one minor item (binding
  analysis redundant with module hashes) applied: `fingerprint.py` 146 lines,
  42 framework tests. Phase 1 started with eight parallel workstreams.
- W6 done (branch `worktree-agent-a4324b8a18d2c2856`, `fcc029d`): 31 prefix
  tests ported (16 pass alone, 15 `integration` need W2 rows M01/P01/P02/B06/M04;
  all 31 pass with those rows applied). Radix-cache backend: keep S01 now,
  switch at the next pin. Integration-time decisions: activate features for
  `integration` tests in a separate pytest process (activation is
  process-global); W2 asked to add S01 guard and M02/M03 ordering tests.
- W1 done (branch `worktree-agent-aca35e5bc131c2060`, `5a015f1`): runtime tests
  ported (71 pass; 11 `integration` need Q04/Q08/Q09/Q10/S03/K03/M02/A01/A07/A09),
  moved sources proven byte-identical modulo import rewrites, 86
  `RUNTIME_DEPENDS` pinned. Fork defects kept verbatim: runtime check of the
  nonexistent `enable_priority_preemption` field (preemption still fails via
  retract); `_namespace` `position_ids`/`mrope_positions` terms inert.
  Integration: activate before collection; Q10 must reach A09's replacement.
  `fingerprint.verify` now hashes each module file once (no AST parse on
  match).
- W3 done (branch `worktree-agent-a6df9f10f64ad2d88`, `c13258b`): all rows, no
  fallback REPLACE, 79 tests, pool sizing checked independently (B2/B4/B8).
  D3 gap found: `EagerRunner.load_batch` copies `ForwardBatch` with
  `dataclasses.replace`, dropping the instance attribute; fixed with a second
  F01 `after` hook on `EagerRunner.load_batch`. At merge: add it to the
  inventory F01 row and manifest (OVERRIDES), empty W3's `PENDING_MANIFEST`,
  and extend D3's note.
- W8 done (branch `worktree-agent-a0b7d8c70addb6962`, `37fc222`): observer
  (sitecustomize import hook on capture/restore in either runtime module),
  comparator, reduced compat profile (`--max-total-tokens 266304`, all 16
  fixtures kept), 67 tests. At merge: report scheduling-dependent ledger
  fields (`forward_id`, `req_pool_idx`, `generation`, `lease_slot`) without
  gating on them; plugin-arm readiness must use the W7 launcher's
  activation-record readiness, not only `/health`.
- W5 done (branch `worktree-agent-a8a160f9d844c003c`, `3c46570`): all rows,
  no fallback, 119 passed / 7 skipped (GPU), scoped REPLACE dispatchers,
  Marlin JIT copy isolated (distinct module, cache, export; RTLD_LOCAL).
  Phase 2 prep: port fork `test/manual/marlin_batch_invariance.py` to the
  plugin op (orchestrator); seed the plugin Marlin JIT cache if
  `SGLANG_CRASH_ON_JIT_COMPILE` is set. Accepted: in-tree `.cuh` headers not
  fingerprinted (GPU gate covers them); two unreachable `stable=` edge cases.
- W7 done (branch `worktree-agent-ac2436d649c5a1466`, `1a48f04`): scope via
  SGLang `get_config`, launcher (preflight, exclusive plugins, version lock,
  per-rank activation records before readiness), service controller ported,
  plugin-off and inventory regressions; 104 passed. Open: lock file is not
  package data (wheel installs would refuse to start; Phase 2 uses
  `PYTHONPATH=src`); systemd lifecycle tests ported but not yet run (needs
  owner OK).
- Phase 1 integrated on main (all eight branches merged). Merge-time fixes:
  F01 `EagerRunner.load_batch` hook and Q01 `cache_clear` / P02
  `_add_one_req` attaches added to the manifest; `RUNTIME_DEPENDS` guard
  removed; HiSparse with a non-target model rejected (Q03); QSA backend scope
  decided on first use for `__new__`-built backends; W2 test fixture removes
  attached members; comparator ignores scheduling-dependent ledger fields.
  Real loader activates `compat` (28 targets + 6 attaches) and
  `compat+hisparse` (60 + 6). `tools/run_cpu_tests.sh` runs three passes:
  396 unit passed (12 GPU skipped, 1 known fork xfail), 1 subprocess
  integration passed, 26 activated integration passed. 358 fingerprints match.
- G1 (gpt-6.1-sol xhigh): no parity, ordering or plugin-off defects; 3 major
  and 2 minor evidence-tooling gaps. Lock file packaged (finding 5). W5 resumed
  for plugin-arm GPU probe runner (finding 1); W8 resumed for comparator
  coverage, required observer evidence and launcher-ready gating (2–4).
- G1 findings fixed (W5 `46669a3`, W8 `af67564`/`1f4299d`, lock `fb4f627`);
  CPU runner 408 + 3 + 26 passed. G1 re-check running.
- G1 cleared (re-check: all five findings resolved, no new findings).
- Phase 3 started (owner, 2026-10-07): Track I agents I-A/I-B and Track U
  agents UA/UB/UC (CPU only; upstream worktrees `.worktrees/upstream-U{A,B,C}`
  at upstream main `b7b2975b57`). Contract: `hisparse/image_identity.py`.
- Phase 2 window (owner authorized the orchestrator to stop and restore
  production): production stopped 21:46 via `./qwen-service.sh stop` after a
  ≥4 s idle check; pre-stop state in
  `qwen:results/plugin-g2-20261007/production-before.txt` (running profile
  `lab/results/dsh-local-promotion-20261004/promoted-8081-profile.json`,
  byte-identical to `service/current.json`). Restore with
  `./qwen-service.sh start --profile <that path>`, then check 8081 health and
  models. Window script runs G2-1 (F, P), G2-2 deterministic (F, P, observer),
  compat-only (F, P); G2-3 native checks deferred to a production cutover.
- UB done: U1 local branch `qsa/U1-hisparse-coordinator-gating` in
  `.worktrees/upstream-UB` (2 commits, incl. a staging-abort bug fix on
  upstream main); would remove S02, S04, S07, B02–B05 and S08/S09. U23
  premature (upstream #42923/#43023 rewriting match/load-back); design doc
  only. Pending owner decision after Phase 2: drop S06 by requiring
  `--prefill-max-requests 1` in the launcher (would be deviation D4).
- I-B done (branch `worktree-agent-abe5dca0d700d6dac`, `a53192c`): the pin
  already embeds suffixes, M-RoPE and PLE history correctly at in-image
  boundaries; only `_namespace` admits supported image requests; 18 tests.
  I5 GPU evidence must include a per-image ViT cache-miss case (ViT batch
  invariance is unverified).
- G2-2 fork arm (run2) passed all four harnesses; observer 1,445 captures
  and 60 restores per rank.
- I-A done (branch `worktree-agent-a02098c91d10c7b42`, `1c040c0`): row I1
  after-hook on `QwenVLImageProcessor.compose_image_artifacts` carries the
  artifact key to the scheduler; `identity_for`/`BYPASS` in
  `hisparse/image_request.py`; matching on the image key in `acquire`,
  ancestor lookup and TP signatures; 30 image tests. Merge after the Phase 2
  window (the window runs from this checkout's `src/`): replace I-B's
  `identity_for` stub with I-A's import, re-record one Track I diff for
  `prefix.py`/`prefix_cache.py`/`runtime.py`. Decisions: image reuse needs
  `--mm-preprocess-cache-size-mb > 0` (documented, not forced); image keys
  not counted in the host footprint (~100 B each); text and image
  checkpoints never match each other.
- UA done: U9a `qsa/U9-monotonic-req-generation` (M04) and U9b
  `qsa/U9-hisparse-decode-mm-inputs` (S03), both real upstream bugs; U7: Z02
  is open #35955 (support needs owner OK), `qsa/U7-gptq-moe-w13-scale-k` and
  `qsa/U7-autoround-moe-marlin-group-split` stacked on it (split needs GPU
  speed/GSM8K numbers); INT8-row PLE dropped this cycle (#41624 overlaps).
- UC done: `qsa/U6-qsa-sm8x-varlen-fallback` (Q01; upstream bug hit by the
  validation interpreter's FA4-without-FA2 setup), `qsa/U6-qsa-fp8-kv-scales`
  (no REPLACE removed without a recorded deviation), `qsa/U8-marlin-moe-batch-invariant`
  (J01–J04, not compiled; needs GPU matrix); stable top-k → #42087; stable HC
  dropped; U4 dropped, U5 deferred (no in-tree consumer).
- Phase 2 run2: **G2-1 PASS** (Marlin align 1, whole-K 1,187, graphs 217,
  native 1,187, top-k 110 items equal F=P; registered kernel tests: same two
  known fork `__new__` failures on both arms). **G2-2 PASS** (10 evidence
  items equal incl. observer byte digests 1,505/rank, qualification 1,181,
  ledgers, lifecycle, concurrency, memory figures; all four harnesses passed
  on both arms). Comparator skips run metadata `harness-status.json`.
- Phase 2 complete (`docs/phase2-results.md`); production restored 23:44 and
  verified. G2 review running.
- G3-U (gpt-6.1-sol xhigh): ready after owner confirmation: U1, U9a, U7 w13
  (as a contribution to #35955 only). Needs changes: U9b (CustomTestCase),
  U7 group split (doc widths; GPU accuracy/perf before opening), U6a/U6b (CPU
  test placement; GPU smoke/regression before opening), U8 (numerical
  reference; compile/run/accuracy before opening); doc corrections for U4,
  U5, U23 (`on_release` precedes row free), stable HC scope. UA/UB/UC
  resumed for the CPU-side fixes.
- G2 closed by the orchestrator after the third re-check: no parity issue;
  comparator hardened (exact signatures, inventory, verbose skip IDs); see
  `reviews/G2-response.md`.
- G3-U follow-ups done (CPU side): U9b `445e71bb34` (CustomTestCase), U7 split
  message `5b20ecdda8`, U1 `b8d2e16836` (gate tests pin the flag), U6a
  `6fa6778afc`, U6b `20509a88ca`, U8 `5e9921bba7` (CPU tests moved to CPU-
  registered unit files; GPU tests written, not run; numerical reference in
  U8). Docs corrected (U4: K02/K03 can use `_resolve_kv_pool_class` at the
  next pin; U5, U7, U8, U23). Ready for owner decision: U1, U9a, U9b, U7 w13
  (contribution to #35955). Need GPU validation before opening: U6a, U6b
  (with U6a on SM89), U8, U7 group split.
- G3-U re-check: U1, U9a, U9b, U7 w13 ready for owner publication decision;
  U6a, U6b, U8 ready after their documented GPU validation; U7 group split
  deferred until #35955 merges (rebase and GPU plan then). G3-U closed by the
  orchestrator: every branch classified, none needs further CPU changes.
- G3-I (CPU) findings addressed (`reviews/G3I-response.md`); I-C harness
  merged (`6279fcf`): 8 frozen image cases (fixture sha256 `39913ab9…`),
  ViT observer, text control vs Phase 2 fork arm; CPU 505 + 3 + 26 passed.
  Accepted I-C deviations: `/generate` prompts, `detail: "high"` as the
  preprocessing miss, ViT observer added to W8's site hook. Next: G3-I
  re-check, then the I5 GPU window (needs owner approval).
- Owner (2026-10-08): **no PRs to sgl-project/sglang for now** (branches stay
  local; no pushes, PRs or upstream comments). D4 not adopted (S06 stays
  fork-faithful). GPU window 2 approved (orchestrator stops/restores
  production): I5 image evidence, then U6a/U6b/U8 GPU validation in
  `results/dsh-maintenance-20261004/upstream-runtime-env` (torch 2.14.1,
  sglang-kernel 0.4.9, FA4 without FA2), evidence kept locally.
- G3-I re-check: original findings resolved; two harness gaps (a page64 hit
  that is not a multiple of 2048; per-request image restore evidence) sent to
  I-C before the window.
- G3-I (CPU) closed by the orchestrator after two re-checks (`reviews/G3I-response.md`):
  image fixtures sha `59f72d85…` (10 cases), per-request restores on every
  configured TP rank. Window 2 next: I5 (`results/plugin-window2-20261008/`),
  then U6a/U6b/U8 upstream GPU checks.
