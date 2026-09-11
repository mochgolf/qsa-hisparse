# B4 production-readiness contract

Run two fresh TP2 services from clean source
`7c37f8ee93ed988bfcad82bff077255431f63bc2`, in this order:

1. `baseline-chunk4096`
2. `treatment-chunk2048`, only after the baseline reducer passes

The arms must otherwise have identical launch arguments and public environment.
Both use `p2-offload`, light observation, full decode graphs for exact batch sizes
1, 2, 3 and 4, disabled graph padding, eager prefill, interval 1, context 262144,
four running requests, 1048576 total tokens, TP2, no MTP/speculation, radix,
overlap, mixed chunks, priority scheduling/preemption, or deterministic inference.
`SGLANG_QSA_HISPARSE_V3_EVENTS` and capture variables must be absent.

Each arm concurrently submits four frozen near-256K requests: two copies each of
request A and B with unique RIDs and 768 greedy output tokens. Require every SSE
stream to finish with exact prompt/completion metadata and 768 output IDs. No
token-vector oracle or cross-arm equality is an acceptance gate.

Persist per-request TTFT, E2E and inter-token gaps; a concurrency-intersection
steady window and aggregate output rate; survivor gaps overlapping each later
request's submit-to-first-token window; health, four-process post-health GC gate,
NVML, final idle, candidate cleanup, and exact production restoration. The
submit-to-first-token window is an external observer boundary and is not called
an exact internal prefill interval because source EVENTS are disabled.

Any operational, configuration, completion, timing, health, cleanup, source, or
restore failure stops the sequence. Preserve completed evidence, restore the baseline
once from the outer `finally`, and do not retry. A passing run is bounded B4
production-readiness evidence for these two chunk settings; it is not a token
correctness oracle, fixed SLO, deployment, or production-GO decision.
