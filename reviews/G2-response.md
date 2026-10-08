# G2 response

Review: `reviews/G2.md`. All three findings accepted.

| # | Resolution |
| --- | --- |
| 1 (high) kernel criterion | The second failure has the same root cause as the first (tests build the backend with `__new__`; F fails both with `AttributeError: ... 'qsa_hisparse'`). It is declared in `docs/baseline.md` step 8 as a second known inherited fork failure, ported as a strict xfail, and `docs/phase2-results.md` reports exact counts per arm. No assertion was changed; no GPU rerun is needed because the per-test outcomes already agree. |
| 2 (medium) masked statuses | `tools/evidence/pytest_outcomes.py` compares per-test outcomes by file::test across arms and fails on any undeclared failure; `run2/compare-g21-tests.txt`: 66 tests per arm, equal outcomes, two declared known failures, PASS. The window script now aggregates every command's exit status. phase2-results.md states that run2's `status.txt` G2-1 exits are unreliable and what acceptance used instead. |
| 3 (medium) provenance | `run2/provenance.json`: fork and pin python trees (clean), plugin `src/` tree unchanged during the window, launcher activation records with native versions, and the arms' effective environment reconstructed retrospectively (labelled as such). `run_g2.py` now records argv, relevant environment and source trees in `run.json`. |

## Re-check (`reviews/G2r.md`)

Findings 1 and 3 resolved. Finding 2's remaining items and the three new
tool defects are fixed: `pytest_outcomes.py` takes one log per pytest run,
requires exactly one final summary per log with counts equal to the parsed
lines and `--expect-tests` tests per arm, rejects conflicting duplicate IDs,
ERROR, XPASS and `[XPASS(strict)]`, and accepts a failure only when declared
with its exception type (read from the FAILURES section when the short
summary carries no message). Nine counterexample tests cover the reviewer's
false-pass cases. The window wrapper propagates step statuses and exits with
the aggregate. Run2 re-evaluated: PASS (67 tests per arm).

## Second re-check (`reviews/G2r2.md`)

Both residual items fixed: known failures are matched on the full exception
line (`AttributeError: 'QwenSparseAttnBackend' object has no attribute
'qsa_hisparse'`), and an XFAIL counts only with `--runxfail` evidence of the
same signature (run2: CPU rerun of the plugin test, which fails in
`_store_kv` before any GPU work; later windows run P step 8a with
`--runxfail`). IDs and skips must equal the frozen inventory
`tools/evidence/g21_step8_inventory.txt` (from run2's fork arm; skips are
keyed by file and reason because run2 used `-rA` without `-v`). 13
counterexample tests, including the reviewer's two. Run2: PASS.

## Third re-check (`reviews/G2r3.md`) and closure

No parity issue was found; both items were fixed: known failures must equal
the whole exception line, and with `-v` logs (future windows run step 8 with
`-v -rA`) skipped tests are matched by node ID against the inventory. Run2
was recorded without `-v`; its single skip is identified by source line:
fork `test/registered/kernel/qsa/test_qsa.py:174` and plugin
`tests/qsa/test_qsa.py:180` are both inside
`test_qsa_sm121_compaction_and_attention_match_sparse_reference`. 17
counterexample tests in `tests/evidence/test_pytest_outcomes.py`.

**Closure (orchestrator, 2026-10-08):** G2 is cleared. Every finding from
four review rounds is resolved, the reviewer reported the raw evidence
supports parity in each round, and the remaining rounds only hardened this
comparator against hand-edited logs. Per the owner's instruction to avoid
over-engineering, no further review round is spent on this tool; the owner
may overrule.
