You are an independent reviewer re-checking gate G1. Work read-only. Do not
modify files. Be proportionate (owner's instruction; PLAN.md rule 10 and the
"Activation guarantees" threat model apply).

Repository: current directory. Read `docs/GOAL.md`, `docs/PLAN.md`,
`docs/baseline.md` (G2 job list and plugin-arm commands), `reviews/G1.md`
and `reviews/G1-response.md`. Fork: `../qsa-hisparse` at `ee8fe158d6`
(read-only copy `../.worktrees/qsa-fork-ref-ee8fe158d6`); pinned SGLang:
`../.worktrees/sglang-pin-76e06febab`.

Tasks:
1. For each G1 finding, verify the fix in the current tree: resolved /
   partially / not, with evidence.
2. Review the new or changed code for defects that would make the Phase 2
   GPU gate pass when parity does not hold, or fail when it does:
   `tools/evidence/{plugin_probe,compare,run_compat}.py`,
   `tests/model_compat/test_plugin_probe.py`,
   `tests/model_compat/test_marlin_deterministic_alignment.py`,
   `tests/evidence/`, `src/sglang_qsa_hisparse/launch.py`, `pyproject.toml`.
3. Do not re-raise items G1 did not raise unless they are blockers for the
   GPU gate under the threat model.

Output: per-finding verdicts; new findings by severity (at most ten) with
file:line, failure scenario and smallest fix; final line "G1: cleared" or
"G1: not cleared".
