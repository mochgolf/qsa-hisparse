# G2 response

Review: `reviews/G2.md`. All three findings accepted.

| # | Resolution |
| --- | --- |
| 1 (high) kernel criterion | The second failure has the same root cause as the first (tests build the backend with `__new__`; F fails both with `AttributeError: ... 'qsa_hisparse'`). It is declared in `docs/baseline.md` step 8 as a second known inherited fork failure, ported as a strict xfail, and `docs/phase2-results.md` reports exact counts per arm. No assertion was changed; no GPU rerun is needed because the per-test outcomes already agree. |
| 2 (medium) masked statuses | `tools/evidence/pytest_outcomes.py` compares per-test outcomes by file::test across arms and fails on any undeclared failure; `run2/compare-g21-tests.txt`: 66 tests per arm, equal outcomes, two declared known failures, PASS. The window script now aggregates every command's exit status. phase2-results.md states that run2's `status.txt` G2-1 exits are unreliable and what acceptance used instead. |
| 3 (medium) provenance | `run2/provenance.json`: fork and pin python trees (clean), plugin `src/` tree unchanged during the window, launcher activation records with native versions, and the arms' effective environment reconstructed retrospectively (labelled as such). `run_g2.py` now records argv, relevant environment and source trees in `run.json`. |
