You are an independent reviewer for gate G4-CPU of the QSA HiSparse SGLang
plugin: the move from SGLang pin `76e06febab` to `v0.5.21` (`e00930c548`)
and the reduction of whole-function REPLACE patches. Work read-only. Be
proportionate (owner's instruction against over-engineering): report real
correctness problems, not style or new infrastructure.

Repository: this worktree, branch `phase4` at `e8e377f`; compare with the
previous state at tag `pin-76e06febab-final` (`git diff pin-76e06febab-final..phase4`).
Pristine checkouts (read-only): new pin
`qwen:.worktrees/sglang-v0.5.21`, old
pin `qwen:.worktrees/sglang-pin-76e06febab`,
fork `ee8fe158d6` via `git -C qwen:qsa-hisparse show ee8fe158d6:<path>`.

Read: `docs/PLAN.md` (shared rules, "Activation guarantees", "Phase 4"),
`docs/phase4-survey.md`, `docs/phase4-results.md`, `docs/DEVIATIONS.md`
(D3 reworded, D6, D7 new), `docs/tasks/P4-*.md`, and the changed rows in
`docs/patch-inventory.md`.

Check, in priority order:
1. Each ported REPLACE copy (E03, E08, G01, G02, M03, P02/`_add_one_req`,
   Q12, S04, S06, S07, T03) equals the v0.5.21 definition plus the fork's
   change; in particular the four hand merges in `RESOLVED` of
   `tests/regression/test_replace_deltas.py` (E03, Q12, T04, P02). Does the
   test itself prove what it claims?
2. The narrowings S02, P01, M04, H05 claim to be behavior-identical to the
   fork at v0.5.21: is each argument correct (callers, ordering, state that
   the hook cannot see)? D6 (G03) and D7 (R02) are accepted deviations:
   check that the implementation matches their stated effects and nothing
   more.
3. Upstream changes the plugin now relies on: `on_release` replacing
   `cache_finished_req` (release order: restore → async copy → logical
   release → free-group flush → physical reuse), `runner.tp_rank`, FW1 ranks
   via `get_parallel()`, S08 `ignore_waiting`, D3's upstream field and the
   dropped eager hook. Any changed upstream definition whose new behavior
   breaks an assumption in a hook's `reason` or the runtime?
4. Test changes: were any assertions weakened, or ported tests edited beyond
   what the pin move requires (`tests/prefix/conftest.py`, pool tests'
   `override`, `test_plugin_off` counts, graph tests)?

You may run CPU tests (`QSA_PIN_ROOT=<new pin> QSA_PYTHON=qwen:service/runtime-env-sglang-20260923/bin/python QSA_FORK_ROOT=qwen:qsa-hisparse tools/run_cpu_tests.sh <paths>`)
if your sandbox allows. For each finding: severity, file:line, a concrete
failure scenario, and the smallest fix. Final line "G4-CPU: cleared" or
"G4-CPU: not cleared".
