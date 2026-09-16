# P3a execution-01 main audit

Main verdict: **PASS_P3A_B2_EAGER_FUNCTIONAL; PERFORMANCE_UNRESOLVED_GC_ASYMMETRY**.
This supersedes the automatic `PASS_P3A_FUNCTIONAL_SPEED_SCREEN` classification
in the preserved manifest/reducer. It does not change those original artifacts.
P3a is accepted as a functional prerequisite for P3b CPU preparation. There is
no accepted batching speedup, graph support, overall P3 PASS, 256K speed/SLO or
eight-request qualification.

## Evidence and correctness

Execution used candidate `e54706c6d3b78e1b169fbddcd38a98a2007c6a2f`, common
measurement baseline `873f82bc43a6a2e425de17b619124d71194f67fe`, lab driver
`9a0110a5d32c809a34d78774ad1d33f2f1c43870`, and frozen component/imported
helpers `d31a0e243816d2f713f8bd4a0ae195635d67dcbb`. Both source worktrees
remain clean. The functional P2 prerequisite is lab `db79e85`.

Main replayed every frozen reducer offline: component, strict short, strict long,
all six raw-SSE light arms and performance aggregation match saved results
exactly (`execution-01/main-reducer-recheck.json`). Separate decode, ownership
and integer-nanosecond measurement reviews inspected raw evidence; their reports
are `ANALYSIS-EXECUTION-01-{DECODE,OWNERSHIP,MEASUREMENT}-REVIEW.md`.

- Component: each actual GPU has 28 B2 cases plus three B1 lifecycle calls.
  Coverage includes all 16 tail pairs, all 12 actual layers on the short fixture,
  two-layer near-context forced misses `[511,511]`, independent K/V bytes and
  non-unit scales, same-forward C4 close, native resolver oracle, FA2 storage
  path, inactive-slot isolation and six negative controls. Component evidence
  alone is not TP2 service or full-model numerical qualification.
- Short service: resident and offload both match all 32 A and 128 B output IDs
  from the frozen P2 golden. Both ranks agree on all 127 forwards: 31 ordered
  A+B forwards followed by 96 B forwards. The 4095/4094-token prompts exercise
  mixed tails and single-row closes.
- Long offload: 261119/261118-token A/B prompts, A release with B continuation,
  C reuse of A's physical slot/native request index at generation +1, C abort,
  and B completion pass. B continues for 96 forwards after A release and 93
  after C release. Sampled host/compact/mapping checks cover every actual layer.
  There is no long resident output golden; long evidence qualifies the tested
  storage and lifecycle behavior, not full long-context numerical equivalence.
- Both ranks close all leases, logical pages, Mamba resources, host/hot active
  usage, pending releases and staging ownership. Persistent offload reservation
  remains 3 GiB pinned host, 99 MiB hot (`103809024` bytes) and `21577780` bytes
  workspace per rank. Raw/index backing pointers remain stable. Reservations
  are not leaks or newly freed memory, and allocator totals are not additive to
  these component inventories.

## Performance observation and rejection of causal acceptance

The frozen sequence is B,N,N,B,B,N with three fresh pairs. The workload is short
B2 eager only; the primary window has 30 synchronized gaps ending at output
tokens 3–32. Independent arithmetic uses integer timestamp differences:

| Pair, baseline→candidate arms | Mean gain ms | Mean reduction | P95 change ms |
|---|---:|---:|---:|
| 1→2 | 1.249128 | 1.196997% | -0.387072 |
| 4→3 | 1.312862 | 1.263805% | -1.080235 |
| 5→6 | 2.278737 | 2.183014% | -2.207363 |

The observed median gain is 1.312862 ms (1.263805%), exceeding the frozen
0.059435 ms baseline MAD/drift threshold. The median P95 change is -1.080235 ms,
within the 0.209027 ms envelope. Candidate B2 means remain approximately
102–103 ms. B1 is diagnostic and one pair regresses slightly. Minor differences
from the driver are sub-microsecond epoch-float rounding, not gate changes.

However, `light-6-candidate/server.log` records `post-warmup freeze_gc failed`
at 21:09:12 with an 30001 self-connection refusal. Unlike the other five light
arms, this arm has no successful `/freeze_gc` or Scheduler TP0/TP1, Tokenizer
Manager and Detokenizer Manager freeze records. Main traced the actual source:
`http_server._wait_and_warmup` calls the endpoint once, logs and continues on
failure; the endpoint dispatches to those processes and calls `gc.freeze()`.
The driver supplies no later retry. A successful health check cannot repair or
prove equal GC state. The impact on these measured windows was not recorded.

Therefore the numerical screen passes, but controlled attribution to batching
is **unresolved**. We neither assume a direction/magnitude for GC's effect nor
drop arm 6 and redefine the three-pair rule after seeing results. Earlier
preliminary wording of “about 1.26% faster” is an observation, not acceptance.
The two strict-short arms also show the startup race; their output/storage
oracles and complete lifecycle evidence remain valid.

## Restore and next action

The pre-test service was restored with health, launch arguments, working tree,
source and complete environment matching its snapshot. Post-idle SIGTERM NCCL
destruction warnings and ASGI cancellation messages occur after completed requests.

Proceed with P3b CPU design using the accepted functional P3a source and native
graph runner; this document authorizes no unreviewed graph/GPU execution. Before
any subsequent performance arm, normalize GC after health and require all four
process success records before sending measured requests. Performance-only
remeasurement can reuse immutable accepted correctness evidence if its source
and helper bindings remain exact; retain all three fresh pairs. Do not repeat
component/long correctness solely to repair this measurement setup.
