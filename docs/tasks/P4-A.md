# P4-A scheduler and lifecycle at v0.5.21

PLAN.md "Phase 4" (rules P1–P4) and the common rules in this folder's
README apply; survey: `docs/phase4-survey.md`. CPU only.

Rows: S01–S09, P01–P02, B02–B06, M01–M04 (`patches/hisparse/{scheduler,lifecycle}.py`,
`patches/model_compat/{scheduler,lifecycle}.py`).

1. Port the REPLACE rows whose target changed: S04, S06, S07, P01, P02, M03.
   Register all 13 REPLACE rows of this area (B02–B05, M01, M03, M04, P01,
   P02, S02, S04, S06, S07) in `test_replace_deltas.py`.
2. Re-check the hooks on changed targets (S08, S09 AFTER) and every changed
   `depends` (scheduler, schedule_policy, batch_result_processor,
   schedule_batch, chunk_cache, `mem_cache.common`); the S01 wrapping of the
   chunk cache and the release order (restore → async copy → logical release
   → free-group flush → physical reuse) must hold at the new pin.
3. P3 for each of the 13 REPLACE rows. S06 stays fork-faithful (owner
   declined deviation D4); S03's D5 hook stays.
4. Fingerprints: `hisparse_scheduler.json`, `hisparse_lifecycle.json`,
   `model_compat_scheduler.json`, `model_compat_lifecycle.json`.
5. Tests: `tools/run_cpu_tests.sh tests/lifecycle tests/regression` plus your
   rows' integration tests; report anything that fails only because another
   task's rows are not ported yet.

Report: per REPLACE row keep/narrow/drop with one-line reason, files
changed, tests run, open questions.
