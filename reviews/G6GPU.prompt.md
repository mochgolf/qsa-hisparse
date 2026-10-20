You are an independent reviewer for gate G6-GPU of the QSA HiSparse SGLang
plugin (Phase 6: SGLang v0.5.22, reference = production `897286b12a`). Work
read-only and proportionately (owner's instruction against over-engineering):
report only conclusions the evidence does not support, missing gate
criteria, and errors in the results document; not style.

Read `docs/PLAN.md` ("Phase 6" gates), `docs/baseline.md` ("Phase 5" and
"Phase 6" GPU steps, and the step table's pass criteria),
`docs/phase6-results.md` ("G6-GPU" and "Warm-hit latency diagnosis"), and the
private evidence in `qwen:results/plugin-g6-20261019/` (`qwen:` = the parent
directory of this repository): `window6.sh`, `status.txt`, `provenance.txt`,
`production-before.txt`, `production-after.txt`, `compare-g21.txt`,
`compare-g21-tests.txt`, `compare-g22.txt`, `compare-compat.txt`,
`compare-g23.txt`, `i5.log`, `descale-plugin.log`, the per-arm `F/` and `P/`
directories, and `profile/` (`profile_warm.py`, `analyze.py`, `spans.py`,
`inside.py`, `children.py`, `F/timings.json`, `P/timings.json`,
`production-before.txt`, `production-after.txt`; the gz traces are large —
sample them only if needed).

Check:
1. Does each gate in the results table meet its pass criteria from
   `docs/baseline.md`, with the arms, interpreters and commits stated
   (provenance)? Is anything required by PLAN.md "Phase 6" G6-GPU missing?
2. Are the explanations of the two irregularities sound: the invalid
   `descale-plugin` step (CPU-interpreter tests run on CUDA) and the
   `steps.json` comparator item?
3. Is the warm-hit latency attribution supported by the profiling evidence
   (plugin hit path and hooks ≈ production; upstream `/generate` contract
   check and `layer_boundary` stage scaffolding account for the gap), or is
   a plugin-caused cost still plausible? Point to the evidence.
4. Was production restored with its original profile both times?

For each finding: severity, location, why, the smallest fix. Final line
"G6-GPU: cleared" or "G6-GPU: not cleared".
