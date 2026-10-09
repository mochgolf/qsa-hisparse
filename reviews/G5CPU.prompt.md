You are an independent reviewer for gate G5-CPU of the QSA HiSparse SGLang
plugin: the move to upstream main `35f3c96ff4` with the production build
`897286b12a` as the reference. The owner's requirement: no production
feature may be lost. Work read-only. Be proportionate (owner's instruction
against over-engineering): report real correctness gaps, not style.

Repository: this worktree, branch `phase5` at `df1c718`; previous state:
`main` (v0.5.21 pin, fork `ee8fe158d6` reference). Read-only checkouts:
pin `qwen:.worktrees/sglang-main-35f3c96ff4`,
production `qwen:.worktrees/sglang-dsh-production-20261004`;
history via `git -C qwen:qsa-hisparse` (log/show/diff only).

Read: `docs/PLAN.md` ("Phase 4" P1–P4, "Phase 5" Q1–Q3), `docs/phase5-survey.md`,
`docs/phase5-results.md`, `docs/DEVIATIONS.md`, `docs/patch-inventory.md`
(new rows M05, M06, Q13, T05, U02–U05, A11, A12; Appendix A).

Check, in priority order:
1. Production coverage: does any production behavior in `git diff 35f3c96ff4
   897286b12a -- python/sglang` fail to reach the running plugin, despite
   its hunk being mapped (a row mapped but not implemented, implemented
   only for some callers, or scoped off where production applies it)? Pay
   attention to the new production-only rows (T05 fast_topk JIT copy and its
   scope, A11/A12/Q04/Q13 FP8 descale and the fused-KV guard, U02–U04 NUMA
   interleave in the main process, M05/M06 tree-cache selection) and to
   production's release order and prefix-cache API (M03, `hisparse/prefix_cache.py`).
2. Do the checks prove what they claim: `test_inventory.py` (hunk coverage),
   `test_replace_deltas.py` (copies equal production at pin = base),
   `test_moved_sources.py` (runtime equals production)?
3. New hooks' equivalence arguments (M05 around, Q13 around, T05 around,
   U03/U04 after, Q03 after + attach) against production's code.
4. Test changes: assertions weakened, or ported tests edited beyond what
   the move requires?

You may run CPU tests (`QSA_PIN_ROOT=<pin> QSA_PYTHON=qwen:results/dsh-maintenance-20261004/upstream-runtime-env/bin/python QSA_FORK_ROOT=qwen:qsa-hisparse tools/run_cpu_tests.sh <paths>`)
if your sandbox allows. For each finding: severity, file:line, a concrete
failure scenario, the smallest fix. Final line "G5-CPU: cleared" or
"G5-CPU: not cleared".
