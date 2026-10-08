You are an independent reviewer doing the scoped re-check of gate G4-CPU.
Work read-only. Be proportionate (owner's instruction).

Scope: only the finding of `reviews/G4CPU.md` and its fix at `phase4`
`bd67144` (`reviews/G4CPU-response.md`). Read
`tests/regression/test_replace_deltas.py`, `tests/lifecycle/test_renamed_copies.py`,
the M03 copy (`release_kv_cache` in `src/sglang_qsa_hisparse/patches/hisparse/lifecycle.py`),
its new `RESOLVED` entry, and `docs/phase4-results.md` "Port check".
Pristine checkouts: new pin `/home/zyk/projects/interests/ai-video/qwen/.worktrees/sglang-v0.5.21`,
fork via `git -C /home/zyk/projects/interests/ai-video/qwen/qsa-hisparse show <commit>:<path>`
(fork base `76e06febab`, fork `ee8fe158d6`).

Questions: is the finding resolved (does a moved or misplaced fork line now
fail the check)? Is the M03 hand merge correct against the fork's release
order? Report a new finding only if a REPLACE copy can now pass while not
being the fork's change on the v0.5.21 body, with a concrete reproduction.
Final line "G4-CPU: cleared" or "G4-CPU: not cleared".
