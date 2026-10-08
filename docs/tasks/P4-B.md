# P4-B decode CUDA graph and pools at v0.5.21

PLAN.md "Phase 4" (rules P1–P4) and the common rules in this folder's
README apply; survey: `docs/phase4-survey.md`. CPU only.

Rows: G01–G03, R01–R02, F01, C01–C02, K01–K03 (`patches/hisparse/{graph,pools}.py`).

1. Port the REPLACE rows whose target changed: G01, G02, G03 (decode CUDA
   graph runner: `capture_one_shape`, `load_batch`, `execute`), R02
   (`ModelRunner._forward_raw`). Register all 5 REPLACE rows of this area
   (G01–G03, K02, R02) in `test_replace_deltas.py`.
2. Re-check the hooks on changed targets (F01 `ForwardBatch.init_new`
   AFTER, including deviation D3's instance attribute and EagerRunner hook;
   C01 `DefaultPoolConfigurator.__init__` AFTER; K01, K03 AROUND) and every
   changed `depends` (`capture_prepare`, `build_replay_fb_view`,
   `FullCudaGraphBackend`, `ShapeKey`, `PhaseConfig`, memory pools,
   `QSATokenToKVPool`, `MambaPool`).
3. P3 for each of the 5 REPLACE rows. The CUDA graph runner was reworked
   upstream; check whether it now offers a seam (backend or shape hooks)
   that replaces a copy.
4. Fingerprints: `hisparse_graph.json`, `hisparse_pools.json`.
5. Tests: `tools/run_cpu_tests.sh tests/graph tests/pools` plus your rows'
   integration tests; report anything that fails only because another
   task's rows are not ported yet.

Report: per REPLACE row keep/narrow/drop with one-line reason, files
changed, tests run, open questions.
