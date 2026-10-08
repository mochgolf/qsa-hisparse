You are an independent reviewer for gate G3-U (Track U: upstream PR
preparation). Work read-only. Do not modify files, do not push, do not use
GitHub write operations. Be proportionate (owner's instruction).

Read `docs/GOAL.md`, `docs/PLAN.md` ("Track U"), `docs/upstream-status.md`,
`docs/tasks/U.md` and every `docs/upstream/U*.md`. The prepared branches are
local, in the fork repository `../qsa-hisparse` (shared object store; inspect
with `git -C ../qsa-hisparse log/show/diff upstream/main..<branch>`):
- `qsa/U1-hisparse-coordinator-gating`
- `qsa/U9-monotonic-req-generation`, `qsa/U9-hisparse-decode-mm-inputs`
- `qsa/U7-gptq-moe-w13-scale-k`, `qsa/U7-autoround-moe-marlin-group-split`
  (stacked on open PR #35955's head)
- `qsa/U6-qsa-sm8x-varlen-fallback`, `qsa/U6-qsa-fp8-kv-scales`
- `qsa/U8-marlin-moe-batch-invariant`
Base: upstream main `b7b2975b57`.

For each branch: is the change correct, minimal and in upstream style
(upstream rules in upstream-status.md section 4)? Do the tests demonstrate
the bug/behavior and would they pass upstream CI on CPU? Is the PR
description in the doc accurate (claims backed by the diff and tests)? Is
the decision (new PR / contribute / drop / defer) sound given the related
upstream PRs? What GPU validation is genuinely required before opening?
Also check the dropped/deferred items (U4, U5, U23, INT8-row PLE, stable HC,
stable top-k via #42087) for wrong reasoning.

Output: per branch a verdict (ready after owner confirmation / needs changes
/ do not open) with concrete findings (file:line, issue, smallest fix); then
findings on the dropped/deferred decisions; final line "G3-U: cleared" or
"G3-U: not cleared" ("cleared" means the ready branches can be offered to the
owner for publication decisions).
