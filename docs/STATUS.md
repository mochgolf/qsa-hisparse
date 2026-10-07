# Status

Progress log for [PLAN.md](PLAN.md). Newest entries last.

| Gate | State | Evidence |
| --- | --- | --- |
| P0-D framework | done (`f8f0958`) | `tests/test_framework.py` 15 passed on the pin |
| P0-A inventory | done | `docs/patch-inventory.md`: 196 hunks → 89 rows, 39 REPLACE (~3,900 copied lines), 63 targets resolve at the pin |
| P0-B baseline | done | `docs/baseline.md`: fork CPU 183 passed/7 skipped; pin subset 72 passed/3 skipped |
| P0-C upstream status | done | `docs/upstream-status.md`: main `0b635266d4`, 1201 commits after pin |
| G0 review | cleared (`reviews/G0final.md`) | `reviews/G0*.md` |
| Phase 1 W1–W8 | running | `docs/tasks/W*.md` |
| G1 review | pending | |
| Phase 2 GPU | approved (runtime-env-sglang-20260923) | |
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
