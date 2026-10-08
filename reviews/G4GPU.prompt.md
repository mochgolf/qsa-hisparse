You are an independent reviewer for gate G4-GPU of the QSA HiSparse SGLang
plugin at SGLang v0.5.21 (Phase 4). Work read-only. Be proportionate
(owner's instruction): judge whether the GPU evidence supports the claims;
do not ask for new infrastructure.

Read `docs/phase4-results.md` (both sections), `docs/PLAN.md` "Phase 4"
(gates), `docs/phase2-results.md` (what Phase 2 compared), and the raw
evidence (private, read-only):
- this window: `/home/zyk/projects/interests/ai-video/qwen/results/plugin-g4-20261008/`
  (`window4.sh`, `window4b.sh`, `status.txt`, `provenance.txt`,
  `compare-g21.txt`, `compare-g21-tests.txt`, `compare-g22.txt`,
  `compare-compat.txt`, `g21-plugin*.log`, `P/`, `i5/`, `failed-first/`);
- reference fork arm: `/home/zyk/projects/interests/ai-video/qwen/results/plugin-g2-20261007/run2/`
  (`F/`, `g21-fork-step8.log`);
- window 3 I5 (old pin): `/home/zyk/projects/interests/ai-video/qwen/results/plugin-window3-20261008/i5/`.
Tools at `phase4` `70fbab1`: `tools/evidence/{compare,pytest_outcomes,run_g2,run_compat,run_image}.py`,
`tools/evidence/g21_step8_inventory{,_v0.5.21}.txt`, and commit `5daef87`
(the `--plugin-inventory` change made during the window).

Check: the run used the reviewed plugin commit and the pristine v0.5.21
checkout; each claim in the "G4-GPU" section follows from the raw evidence;
the `--plugin-inventory` change does not let a real outcome difference pass;
the failed-first attempts were preflight failures only and the reruns are
valid evidence. Report a finding only if a claim is unsupported or wrong,
with the concrete evidence. Final line "G4-GPU: cleared" or
"G4-GPU: not cleared".
