You are an independent reviewer for gate G6-CPU of the QSA HiSparse SGLang
plugin: the move from upstream `35f3c96ff4` to the SGLang release v0.5.22
(`baa9befa38`), keeping production `897286b12a` as the behavioral
reference. Work read-only. Be proportionate (owner's instruction against
over-engineering): report only defects that break production-equivalent
behavior, false-failure risks of the activation checks, and removable
complexity; not style, and not threats PLAN.md excludes.

Repository: this checkout, branch `phase6` (previous state: `main`, pin
35f3c96ff4). Read-only checkouts (`qwen:` = the parent directory of this
repository): new pin `qwen:.worktrees/sglang-v0.5.22`, previous pin
`qwen:.worktrees/sglang-main-35f3c96ff4`, production
`qwen:.worktrees/sglang-dsh-production-20261004`. History via
`git -C qwen:qsa-hisparse diff <a> <b> -- <path>` or `git show <commit>`
(a blob-less partial clone: do not run `git log -S/-G/-p` or `git blame`).

Read: `docs/PLAN.md` ("Phase 6", and Phase 4 P1–P4 / Phase 5 Q2–Q3),
`docs/phase6-survey.md`, `docs/phase6-prefix-design.md` (decisions at the
top), `docs/phase6-results.md`, `docs/prefix-cache.md`, `docs/DEVIATIONS.md`,
`docs/patch-inventory.md`; `git diff main phase6 -- src tests tools`.

Check, in priority order:
1. Host prefix cache (`src/sglang_qsa_hisparse/hisparse/prefix_cache.py`,
   `patches/hisparse/{scheduler,lifecycle}.py`): against v0.5.22's
   `UnifiedRadixCache`, `alloc_for_extend`, `PrefillAdder.add_one_req`,
   `release_kv_cache` and `match_kv_cache`, does an exact host hit restore
   the same KV and mamba/PLE state into the same row positions as
   production, with the same admission budget, chunk boundaries (page 64,
   including inside images), `cached_tokens`, TP convergence and release
   order (lease drain before the logical free, pages and mamba slot freed
   exactly once, no leak on miss/abort/retract)? Is any upstream path that
   production's cache relied on now bypassed (e.g. a caller of
   `prefix_device_indices`, `init_load_back` or `maybe_hand_to_session` the
   design did not consider, `ignore_eos`, retraction, abort before
   allocation)?
2. The new and narrowed hooks' equivalence arguments (S06 after on
   `get_prefill_adder_class`, M01 around, B04 before on
   `_prepare_kv_cache_release`, H08 around) and the hand merges in
   `tests/regression/test_replace_deltas.py` `RESOLVED` (M03, Q08, T03),
   especially Q08's FP8 descale (applied exactly once) and T03's bound.
3. Do the checks prove what they claim at v0.5.22: `test_replace_deltas.py`,
   `test_inventory.py`, `test_moved_sources.py` (with `prefix_cache.py` now
   plugin-owned), fingerprints covering every upstream definition the new
   cache relies on (activation must fail closed if one changes)?
4. Test changes: assertions weakened, or ported tests edited beyond what the
   move requires? Removed tests (`tests/lifecycle/test_renamed_copies.py`,
   three upstream NUMA tests) justified?
5. The "Differences from production" list in `docs/phase6-results.md`: is
   any item wrong, missing, or reachable in production's profile
   (`qwen:service/profiles/qwen-local-plugin-mm-20261009.json`)?

You may run CPU tests if your sandbox allows:
`QSA_PIN_ROOT=qwen:.worktrees/sglang-v0.5.22 QSA_PYTHON=qwen:results/phase6-20261018/runtime-env/bin/python QSA_FORK_ROOT=qwen:qsa-hisparse tools/run_cpu_tests.sh <paths>`.
For each finding: severity, file:line, a concrete failure scenario, the
smallest fix. Final line "G6-CPU: cleared" or "G6-CPU: not cleared".
