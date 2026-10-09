You are an independent reviewer for gate G3-I (GPU part): Track I image
host prefix reuse, as evidenced by GPU window 3. Work read-only. Be
proportionate (owner's instruction): judge whether the evidence supports the
Track I acceptance criteria; do not ask for new infrastructure.

Read:
- `docs/PLAN.md` "Track I acceptance" (criteria 1–7), `docs/DEVIATIONS.md` D5;
- `docs/window3-results.md` (claims) and `docs/window2-results.md` (D5's
  failing control);
- raw evidence (private, read-only) in
  `qwen:results/plugin-window3-20261008/`:
  `window3.sh`, `status.txt`, `i5/summary.json`,
  `i5/{vit-cache-on,vit-cache-off}/{image-prefix.json,run.json,server_info.json}`,
  observer and ViT logs under each session, `d5_check.py`, `d5.log`,
  `d5/plugin/results.json`, `image-fixtures.json`;
- the harness that produced the verdicts (`tools/evidence/{run_image,image_prefix_harness,image_fixtures,compare}.py`)
  and the D5 change (`src/sglang_qsa_hisparse/patches/model_compat/scheduler.py`,
  `tests/lifecycle/test_d5_decode_batch_logprobs.py`) at main `dfc68ac`.

Check that the run used the reviewed code and frozen fixtures (sha
`59f72d85…`), that each claim in `window3-results.md` follows from the raw
evidence, and that each criterion 1–7 is covered by what was actually run.

Report a finding only if a criterion is not supported by the evidence, or a
claim in the results doc is wrong, with the concrete evidence. Final line
"G3-I (GPU): cleared" or "G3-I (GPU): not cleared".
