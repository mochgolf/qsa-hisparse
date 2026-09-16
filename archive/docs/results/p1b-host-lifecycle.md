# P1b execution-01 results

Driver rc1. Baseline source `4b0c246e41cd336121dcd965ac73c70aca6e271f`; experimental source `d46f4b83078f8916b82e66e4d3f748f0d107cfba`; preparation lab `a25c1f8`, launch `19467a7`.

Driver status: `FAIL_RESTORED`. Its sole error is `P1b steady performance gate did not pass`. Component, strict service, lifecycle, workspace and diagnostic profile gates all passed. Performance is inconclusive under the frozen acceptance screen; this is not a crash, output mismatch or demonstrated performance regression. Independent result reviews accompany the final main decision below.

## Why the run did not qualify

| Pair | OLD mean TPOT | NEW mean TPOT | OLD−NEW mean gain | OLD−NEW P95 gain |
|---|---:|---:|---:|---:|
| 1 | 96.802183 ms | 96.635626 ms | 0.166557 ms | 0.186278 ms |
| 2 | 97.291468 ms | 96.527663 ms | 0.763805 ms | 0.675961 ms |
| 3 | 97.032126 ms | 96.604624 ms | 0.427502 ms | 0.447374 ms |

All three mean gains are positive. Their median is **0.427502 ms** (~0.44% of baseline), but the required gain is **0.459886 ms**, from max(2×OLD MAD, first-to-last OLD drift). OLD MAD is0.229943ms; drift is0.229943ms. The median falls short by0.032384ms. The predeclared strict `>` test correctly does not pass. Neither rounding the gap away nor substituting the P95 criterion is permitted.

P95 improves in every pair, with median gain0.447374ms, above its0.382423ms noise threshold. That passes the P95 gate but cannot replace the failed primary mean gate. The result suggests a small potential benefit; it does not establish a retained steady-performance optimization under this contract.

Medians of per-request metrics (not pooled gap percentiles): steady mean97.032126→96.604624ms, steady P9598.243550→97.796176ms, TTFT43.161718→43.134201s, handoff154.934514→154.688806ms, E2E117.858049→117.503977s. Paired median E2E gain is0.354073s. Handoff, TTFT, E2E and startup remain diagnostics; P1a's earlier allocation gain is not reused to qualify P1b. All six primary streams have complete uncoalesced token-gap coverage.

## What passed

- Both device component runs cover8 short/near256K tail cases with per-rank12Q/1KV/head_dim256, FP8 K/V and distinct1.3/0.7 descales. Packed BF16 K/V match the independent reference bytewise and old/new FA2 outputs match bytewise. All three local-index/table/scale negatives per device are detected. This is per-device component evidence, not TP2 or concurrency qualification.
- Strict long output, six primary long outputs and two diagnostic-server long outputs pass the full768 golden checks. The separate cancellation and nine sentinels also pass their gates;19 request ledger rows are present.
- Strict long has767 decode steps and2340 selected-byte/mapping checks per rank; strict sentinel has48. Cancellation again reaches completed handoff and two decode steps before cleanup, then its sentinel succeeds. This proves deferred cleanup, not interruption of in-flight copies.
- Candidate workspace is3,221,584B/rank and zero after release, a1,007,556B payload reduction relative to old4,229,140B. The host slab remains1,610,612,736B/rank reserved while active/free and pointer reuse satisfy P1a's lifecycle contract.
- The separate profile has768 `qsa.compact_mapping` CPU scopes/rank,1920 retained `aten::copy_` spans/rank, no scope missing copy and no forbidden arange/scatter/index-put operations. Each profiled handoff has797 CUDA memcpy submissions and zero `cudaHostAlloc`. The intended operations were removed; this diagnostic cannot overrule the service-performance gate.

## Restoration and candidate disposition

Service restoration and the independent follow-up passed the health, launch-argument, working-tree, source and environment identity checks against the pre-test snapshot. Supervisor queued main once after driver rc1. No extra GPU run was launched for this result review.

The unaccepted local-mapping optimization was reverted in source commit `695366c9a8c24fd18ab6c02e36b25f2e99668259`. A Git diff against accepted P1a source4b0 is empty for `python/` and `test/`; P1a startup slab reuse remains intact.

P1b is not accepted as a performance optimization. No threshold is loosened and no automatic rerun-to-pass is scheduled. Further planning should prioritize the actual deployment objective and use this measured ~0.4ms scale when weighing additional B1 tuning against B2 ownership work. No P2/B8 or production-performance qualification follows from this run.

## Final decision and P1 closure

**INCONCLUSIVE_NOT_ACCEPTED; correctness/lifecycle/profile PASS; baseline restored.** Independent measurement review recomputed the six primary streams and confirmed the frozen gate. Independent lifecycle review reread all nine long outputs, cancellation/sentinels and both raw traces; it found no additional blocker. See `P1B-RESULT-MEASUREMENT-REVIEW.md` and `P1B-RESULT-LIFECYCLE-REVIEW.md`.

Rollback CPU8 passed with CUDA disabled; `ROLLBACK-CPU.json` binds the reverted source and empty Python/test diff against P1a. The original driver status remains `FAIL_RESTORED` and its raw evidence is not rewritten.

Main closes P1 with the accepted P1a startup slab allocation and its measured approximately 1.09 s first-handoff improvement. The approximately 97 ms B1 steady TPOT remains unresolved. P1b's small favorable direction is preserved as unaccepted evidence, not a demonstrated absence of benefit. Further copy fusion is deferred until the batch path provides evidence for it; another B1 repetition is not scheduled. Proceed in the accepted order to P2 logical/physical separation and request ownership, with B2 eager correctness before P3 graph work. This phase transition does not claim a steady-performance or online-service GO.
