# G4-CPU response

Review: `reviews/G4CPU.md` (one P2 finding, no implementation defects).

| # | Resolution |
| --- | --- |
| P2 `changes()` loses edit locations | Fixed: `test_replace_deltas.py` now requires each copy to equal `git merge-file` of the fork's change into the pinned definition (and `test_renamed_copies.py` uses the same `carried()`), so a moved insertion fails. `test_carried_checks_each_change_in_place` holds the counterexamples (fork line without upstream's edit; fork line moved before the release it must follow; an adjacent upstream edit conflicts). The stricter check found that M03 is a hand merge (upstream rewrote the handoff next to the fork's insertion); its resolution is recorded in `RESOLVED` after checking the order (lease capture before `claim_kv_row` and the logical free; `after_release` after `mark_kv_released`; skipped when a streaming session keeps the row). Full CPU suite: exit 0 (569 + 4 + 26). |

**Re-check (`reviews/G4CPUr.md`):** resolved, no new findings; M03's hand
merge preserves the fork's release order. G4-CPU cleared (2026-10-08).
