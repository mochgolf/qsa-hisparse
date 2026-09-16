# QSA HiSparse P0-P5 closeout

Overall verdict: **CLOSED_NO_PRODUCTION_GO.** The ordered development campaign
is complete at its authorized boundary. The candidate demonstrates bounded B4
capacity, ownership, graph execution and a passing final 256K/768-token B1
regression. Production remains on source `bd83f02`; candidate `7c37f8e` is not
deployed because B8 and a fixed long-context SLO were not qualified.

| Phase | Accepted result | Practical conclusion |
|---|---|---|
| P0 | Correctness/E2E and component attribution accepted | B1 offload preserved the 768-token oracle and released resources. Its light steady mean was 97.036 ms versus 93.820 ms resident, so no decode-speed gain was established. |
| P1 | P1a startup host slab accepted; P1b steady optimization rejected | Moving the slab allocation to startup reduced median handoff from 1247.478 to 155.368 ms and first output gap from 1.383 to 0.293 s. The attempted steady-path optimization did not meet its acceptance gate. |
| P2 | `PASS_P2_B2_EAGER_BOUNDED` | Two near-context requests exercised independent ownership, refetch, release, generation reuse and cancellation. Prefills remained serialized and existing decode paused during a new prefill. |
| P3 | Exact B1/B2 full-graph functionality and short-B2 speed screen accepted | The bounded short-B2 screen reduced median mean decode gap from 102.335 to 26.888 ms. It was not a 256K, mixed-arrival or fixed-SLO qualification. |
| P4 | Functional resident/offload bridge accepted; three-arm acceptance incomplete | Interleaved prefill/decode semantics and final-source functional oracle were established. The planned light comparison arms were not completed; the selected interval-1 behavior was observed later in P5b. |
| P5 | B4 function, descriptive interval-1 measurement and corrected P5c final regression accepted | B4 full-graph steady decode measured 29.853 ms mean and 30.464 ms P95, or 133.988 token/s aggregate. During a 261,119-token replacement prefill, survivor P95 gap was 815.666 ms. B8 was not run. |

## Final regression decision

P5c used the final source for one 261,120-token prompt followed by 768 greedy
tokens. Its original comparison failed because it retained a resident-eager
oracle produced at source `f04c0d7`, before score-width alignment `114ac4b`.
The aligned implementation deliberately gives eager and full graph the same
deterministic top-k collection order, so that older vector no longer describes
the accepted semantics.

Matched final-source isolation closed the cause. Fresh resident-eager equals
the P5c offload-full graph output at all 768 positions, and fresh offload-eager
equals both at all 16 collected positions around the first old mismatch. The
old vector differs at 752 positions beginning at zero-based index 9. Rebinding
P5c to the independently collected aligned resident vector makes the preserved
P5c evidence pass exact output, SSE, TP2 graph replay, ownership, GC and final
idle reduction. No candidate numerical patch or GPU retry was required.

## Deployable scope

The retained engineering evidence supports continued development from the
candidate branch: startup allocation is fixed, B2 and B4 ownership work, full
graphs materially improve the bounded short decode path, the final long-output
regression passes, and terminal resource closure is sound.

No eight-request result exists. No B8, simultaneous multi-prefill, fixed SLO,
long-context performance qualification or production GO is claimed. The pre-test
service was restored and revalidated; no additional GPU execution is scheduled.
