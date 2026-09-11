# P2 final main audit: bounded B2 eager acceptance

Main verdict: **PASS_P2_B2_EAGER_BOUNDED**, based on execution-04, both independent raw-evidence reviews, and main's exact offline reducer recheck. Service candidate is clean `26b29ebcaf0a6d4adf3dea674d7bf1e7f27e5351`; driver lab `a0a2868`. Earlier failed attempts remain preserved and are not promoted.

## Accepted service evidence

| Gate | Observed result |
|---|---|
| Short resident/offload oracle | A4095/B4094 input; complete A32/B128 output token arrays identical; all127 normalized decode batches identical |
| Real TP2 B2 | Both ranks agree on every ordered decode row; 31 consecutive A+B forwards in each arm |
| Two near-context requests | Long A261119 and B261118 retain independent logical/index, host, hot and ring ownership and decode together |
| Long close/refetch | Mixed tails and individual-row C4 closes; A/B strict checks cover all12 layers on both ranks, with positive misses at every checked long A/B layer |
| A release, B continuation | A finishes32 outputs; B executes96 more decode forwards and finishes128 outputs |
| Generation reuse and cancellation | C4096 reuses A's req row2/physical slot0, generation3→4; terminal SSE explicitly reports abort after3 outputs; B executes93 more forwards after C release |
| Resource closure | Every arm ends logical available524288, Mamba available40, physical active/free0/2, no pending release, no staging owner, host/hot active0; final idle passes |
| Service restoration | Health, launch arguments, working tree, source and captured environment matched the pre-test snapshot |

The initial request token is produced by prefill, so 32 A outputs correspond to31 joint decode forwards. The first B-only transition and B+C pair are retained in the raw lifecycle evidence, not counted as steady B1 timing.

## Memory and timing

Each offload rank has one stable raw backing of1,611,067,392 bytes (about1.500 GiB), versus3,221,618,688 bytes in the resident control (about3.000 GiB). Separate index backing remains402,702,336 bytes; logical capacity is524288 tokens. The observed long logical peak is522368 tokens. Each rank reserves3 GiB pinned host backing, has a99 MiB peak of separately allocated hot storage, and about6.03 MiB shared workspace. Raw ring views are already included in raw bytes. Reserved backing persists for reuse; active resource closure does not mean total CUDA allocation returns to zero.

Boundary-excluded **strict-observe diagnostic** long B2 TPOT is121.008 ms mean /146.015 ms P95, with16.528 output tokens/s aggregate. Post-C B1 is105.866/125.997 ms. These single-run values include validation and synchronization and are not latency qualification or a stable optimization gain. The reproducible extractor, exact windows, per-arm timing, handoff, memory and sampled NVML peaks are in ANALYSIS-EXECUTION-04-MEASUREMENT.md and execution-04/timing-memory-analysis.json.

## Limits and phase transition

This accepts the planned P2 bounded eager functionality and capacity contract. The implementation still allows only two active leases, serializes prefills, and pauses existing decode while a new prefill runs. It does not establish B4/B8, simultaneous prefill, interleaving quality, CUDA graph support, latency SLOs, production GO or long-context resident/offload output equality. The exact model-output oracle is the short same-schedule control; long evidence establishes the tested data paths, capacity and lifecycle.

The inherited PASS96 component was executed at source739edcd71bfaaeec914cf12a81de2a2fe1e9ed99. It remains byte-identical and retains that identity; the candidate's two-file scheduler-metadata/CPU-test delta has explicit reviewed compatibility. This is not exact-candidate component execution. Startup freeze_gc self-call and intentional shutdown tracebacks are recorded; none coincides with a serving failure or failed closure.

Main now advances to P3 under the accepted P0–P5 plan: obtain two independent interface recommendations, implement the minimum stable batched eager path personally, then evaluate native CUDA graph integration. First preserve an appropriate light B1/B2 control; strict P2 timing does not set the improvement threshold. No new GPU window is authorized by this report alone; P3 must freeze its concrete source, correctness and measurement gates before guarded execution.
