# P4-D runtime, image identity, activation and evidence at v0.5.21

PLAN.md "Phase 4" (rules P1–P4) and the common rules in this folder's
README apply; survey: `docs/phase4-survey.md`. CPU only.

Paths: `hisparse/` (moved runtime N01–N03, `depends.py`, image identity),
`patches/hisparse/image_identity.py` (I1), `patches/framework.py` (FW1),
`launch.py`, `fingerprints/{runtime,hisparse_image_identity,framework}.json`,
`tests/{runtime,image,image_boundaries,launch,prefix,scope,service}/`,
`tests/test_framework.py`, `tools/evidence/`.

1. Runtime: 34 changed and 1 missing definitions in `runtime.json`
   (`get_attention_dp_size`, `ParallelState`, `ScheduleBatch`, `Req`,
   memory pools, `ServerArgs`, `Envs`, hisparse JIT kernels, ...). Decide for
   each whether the moved runtime still holds; where it must change, the
   change is a recorded edit in `tests/runtime/test_moved_sources.py` (the
   modules otherwise stay fork copies).
2. FW1: `configure_scheduler_process` changed; `verify_final` must still run
   in every scheduler/TP process after `load_plugins()`. Check `launch.py`
   and `environment.lock.json` (expected unchanged; the survey lists the
   dependency pins).
3. Image identity: re-check I1 and `MultimodalInputs.from_processor_output`
   (changed), and that the preprocess-cache path still carries artifact keys
   for `qwen4_exp`.
4. Evidence tooling for G4-GPU (CPU-prepared, not run): pin paths are set;
   make `run_g2.py`, `run_compat.py`, `pytest_outcomes.py` work against
   v0.5.21 (upstream moved test files, e.g. QSA tests, between the pins),
   keeping the Phase 2 fork arm results as the reference.
5. Tests: `tools/run_cpu_tests.sh` on your test folders and `tests/evidence`;
   report anything that fails only because another task's rows are not
   ported yet.

Report: per changed definition the decision (still holds / edited, why),
files changed, tests run, open questions.
