You are an independent reviewer for gate G5-GPU of the QSA HiSparse SGLang
plugin at upstream 35f3c96ff4 with production (`897286b12a`) as the
reference. Work read-only. Be proportionate (owner's instruction).

Read `docs/phase5-results.md` (both sections) and `docs/PLAN.md` "Phase 5",
and the raw evidence (private, read-only):
`/home/zyk/projects/interests/ai-video/qwen/results/plugin-g5-20261009/`
(`window5.sh`, `status.txt`, `provenance.txt`, `compare-g21.txt`,
`compare-g21-tests.txt`, `compare-g22.txt`, `compare-compat.txt`,
`compare-g23.txt`, `topk-*.log`, `F/`, `P/`, `i5/`, `make_plugin_profile.py`),
the prepared cutover profile
`/home/zyk/projects/interests/ai-video/qwen/service/profiles/qwen-local-plugin-20261009.json`
against production's
`/home/zyk/projects/interests/ai-video/qwen/lab/results/dsh-local-promotion-20261004/promoted-8081-profile.json`,
and the tools at `phase5` (`tools/evidence/{compare,pytest_outcomes,run_g2,run_g21,run_g23,run_compat,run_image}.py`).

Check: the run used the reviewed plugin commit and clean pin/production
checkouts; each claim in the "G5-GPU" section follows from the raw evidence
(including the latency samples behind the G2-3 medians and the memory
figures); the prepared cutover profile differs from production's only as
stated and would run the tested plugin commit. Report a finding only if a
claim is unsupported or wrong, or the cutover profile would not run what was
tested, with concrete evidence. Final line "G5-GPU: cleared" or
"G5-GPU: not cleared".
