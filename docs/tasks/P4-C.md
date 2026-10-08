# P4-C model, QSA attention and quantization at v0.5.21

PLAN.md "Phase 4" (rules P1–P4) and the common rules in this folder's
README apply; survey: `docs/phase4-survey.md`. CPU only.

Rows: E*, H*, Q*, T*, A*, J03, Z* (`patches/model_compat/{qwen4_exp,hyperconnection,qsa_attention,quantization,marlin}.py`,
`patches/hisparse/qsa_backend.py`, `kernels/`).

1. Port the REPLACE rows whose target changed: E03, E08 (`qwen4_exp`), Q12
   (`_forward_paged_attention`), T03 (`select_decode_tokens`), and H05/H06,
   whose module moved to `sglang.kernels.ops.gemm.hc_mix` (#41243; give
   `COPIES` the fork's old module file). Register all 19 REPLACE rows of
   this area in `test_replace_deltas.py`.
2. Re-check T02 (AROUND) and every changed or missing `depends`
   (`fast_topk` moved to `sglang/kernels/ops/attention/fast_topk.py`,
   `qwen3_5`, `GatedResidual`, `load_jit`, the pinned-host embedding).
   `tests/model_compat/test_model_compat_copies.py` stays the fork-fidelity
   check for kernels, Marlin sources and copies of unchanged targets; ported
   copies move to the delta test.
3. P3 for each of the 19 REPLACE rows. Scope rule 9 is unchanged.
4. Fingerprints: `model_compat_{qwen4_exp,hyperconnection,quantization,marlin}.json`,
   `qsa_attention.json`, `qsa_backend.json`.
5. Tests: `tools/run_cpu_tests.sh tests/model_compat tests/qsa` plus your
   rows' integration tests; report anything that fails only because another
   task's rows are not ported yet.

Report: per REPLACE row keep/narrow/drop with one-line reason, files
changed, tests run, open questions.
