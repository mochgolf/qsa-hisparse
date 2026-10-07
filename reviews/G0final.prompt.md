You are an independent reviewer doing the final check of gate G0. Work
read-only. Do not modify files. Be proportionate: the owner has asked to
avoid over-engineering.

Repository: current directory (SGLang general plugin `sglang_qsa_hisparse`).
Read `docs/GOAL.md`, `docs/PLAN.md` (especially "Activation guarantees
(threat model)" and rule 10), `docs/DEVIATIONS.md`, `docs/tasks/*.md`.
Prior rounds: `reviews/G0*.md` and `reviews/*-response.md`. After round 4 the
owner warned against over-engineering; the binding-chain and in-place
mutation machinery was removed (commit a6db554) in favor of the threat model
in PLAN.md. Fork: `../qsa-hisparse` at `ee8fe158d6`; pinned SGLang:
`../.worktrees/sglang-pin-76e06febab`.

Report only:
1. Defects that break a GOAL.md obligation under the stated threat model
   (e.g. a requested feature that can end up partially or not applied
   without the process stopping; pin drift in a patched or depended module
   that is not detected; a scheduler/TP process that can serve without
   activation evidence once the W7 launcher exists).
2. False-failure risks that would block real activation on the pinned
   SGLang with the validation interpreter.
3. Complexity that can be removed without weakening item 1.
Do not propose defenses against threats PLAN.md excludes. Do not re-raise
items earlier rounds settled unless the simplification reopened them.

Scope: `src/sglang_qsa_hisparse/*.py`, `src/sglang_qsa_hisparse/patches/framework.py`,
`src/sglang_qsa_hisparse/{manifest.json,fingerprints/framework.json}`,
`tools/{fingerprint,manifest,run_cpu_tests}.*`, `tests/`, `docs/PLAN.md`,
`docs/tasks/*.md`.

Output: findings by severity (blocker, major, minor) with file:line, failure
scenario and smallest fix, at most ten; then a final line "G0: cleared" or
"G0: not cleared". "Cleared" means Phase 1 can start; minor items may be
assigned to workstreams.
