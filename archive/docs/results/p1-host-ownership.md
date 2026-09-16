# P1a execution-01 results

Driver rc=0. New source `4b0c246e41cd336121dcd965ac73c70aca6e271f`; old source `3d155d30c13a7ce4024a8f8a7482696f7d4a0dfb`. Driver preparation `c02af3c`, launch record `b725fe0`.

Final verdict: `PASS_P1A`, errors empty. Arithmetic, strict/diagnostic summaries, supervisor completion and baseline restoration all passed review. Retain the startup slab change and proceed to one measured P1b steady-mapping change; no additional P1a GPU run is required.

## Measured effect

Three fresh-server pairs ran OLD→NEW, NEW→OLD, OLD→NEW with identical 261120-token input, 768 greedy output, TP2 eager B1 configuration. The following are medians of the three per-request measurements, not pooled gap percentiles.

| Metric | Old | New |
|---|---:|---:|
| Handoff wall, slower rank | 1247.478 ms | 155.368 ms |
| First output-to-output gap | 1.383298 s | 0.293259 s |
| TTFT | 43.138175 s | 43.156817 s |
| Steady mean inter-token gap | 97.230577 ms | 97.439769 ms |
| Steady P95 inter-token gap | 98.503838 ms | 98.716029 ms |
| Request E2E | 119.060374 s | 118.148548 s |

Paired handoff improvements are 1105.864 / 1092.110 / 1081.745 ms; their median exceeds the frozen 32.061 ms baseline-noise threshold. Paired first-gap improvements are 1.105395 / 1.090039 / 1.081567 s, exceeding the 0.033368 s threshold. D2H event totals remain about 66.8–67.0 ms/rank. This is the measured removal of request-time allocation, not a faster transfer path.

Steady P95 NEW−OLD is −0.181958 / +1.249018 / +0.212191 ms. The +0.212191 ms median remains within the predeclared 0.257305 ms baseline-noise allowance, so the no-regression screen passes. Pair 2 individually regressed by 1.249 ms: the result does not establish uniform or statistically proven non-regression, and no steady-speed improvement is claimed. Paired median E2E reduction is 0.911826 s; E2E is diagnostic under this contract.

All six primary SSE streams have 768 single-token outputs, 767 observed gaps and 766 steady gaps, with no coalescing. Diagnostic profile requests are excluded from these statistics.

## Allocation cost and correctness

New-source primary startup slab allocation takes 1082.382–1103.122 ms/rank. It moved before readiness; it did not disappear. Process-launch-to-observed-health times are about 151–153 seconds in both arms, with model-load and health-observation noise. Those measurements do not resolve the exact startup increment.

The adapter keeps 1,610,612,736 bytes (1.5 GiB) of pinned tensor backing per rank, 3 GiB TP2, even while idle. Active lease bytes return to zero after release; reserved backing and its pointer remain. This ledger measures adapter-owned tensor payload, not every internal pinned-allocator reservation or bytes returned to the OS.

Nine completed long requests match all 768 immutable golden IDs: one strict new-source request, six primary light requests and two diagnostic-server requests. The cancelled request is separate. The request ledger contains 19 rows including nine sentinels and the cancelled long request.

Strict long-request evidence contains 767 decode steps and 2340 selected K/V checks per rank, including tail/page/checkpoint coverage. Each strict sentinel has 48 checks/rank. Cancellation was requested at `handoff_begin`; the request completed handoff and reached two decode steps before draining and releasing. Its same-process sentinel passed. This proves deferred cancellation cleanup, not interruption of in-flight handoff chunks. Logical capacity returns to 262144, index/Mamba identity is preserved, and raw backing restoration/re-admission checks pass.

In the diagnostic server, generations 3/4/5 (first long, sentinel, second long) use the same slab address on each rank. Each rank has one init event; handoff rows report full active bytes, and release rows report zero active bytes with reserved backing retained. The profiled second-long handoff has zero `cudaHostAlloc`, 797 `cudaMemcpyAsync` spans/rank and the expected 64-decode annotation inventory. This zero-allocation trace finding is limited to the captured second-long handoff; no first-long trace was collected.

All primary sampled NVML peaks are 45454 MiB. These samples include request reuse and do not establish a lower GPU peak or a continuous-time maximum. No new concurrency capacity is inferred.

## Restoration and scope

Driver restoration and the independent follow-up both passed the health, launch-argument, working-tree, source and environment identity checks against the pre-test snapshot. Supervisor exited after driver rc0 and queued main once.

P1a concerns startup slab allocation and B1 request latency only. P1 steady-decode optimization remains incomplete; there is no P2 multi-owner, B8 concurrency or production-performance GO. Subsequent P1 work must target the measured steady path and keep the existing numerical/lifecycle gates.
