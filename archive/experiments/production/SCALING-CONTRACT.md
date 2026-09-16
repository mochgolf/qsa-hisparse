# Latest-upstream HiSparse 256K scaling contract

## Frozen source and service

- Candidate: clean `8606d4b2df088512157c8df9615cff10e11028ae`, rebased on upstream `4309c7ce19dc42fb42cc9e7d883691c8dd8bda10`.
- The baseline is snapshotted, stopped only after a three-second idle gate, and restored exactly once after the candidate stops.
- Candidate uses TP2, a separate test endpoint, no MTP, `p2-offload/light`, 2,097,152 total tokens, chunk 2048, and full decode CUDA graphs B1–B8. Radix cache, overlap scheduling, mixed chunk, priority scheduling, deterministic inference, speculative decoding, prefill graphs, and diagnostic source events remain disabled.

## Workload

- Reuse frozen request-A's exact 261,120 input IDs and authoritative nine-token chat tail.
- Use greedy `ignore_eos` generation with 1,022 output tokens. This is the largest legal output under the upstream request-length guard: `261120 + 1022 = 262142`; B8 reserves at most `2,097,136` tokens.
- In one candidate service lifetime, execute B1, B2, B4, then B8 once each. A barrier releases all clients in a group together. Wait for three seconds of idle and full KV/Mamba capacity recovery between groups.
- This test submits concurrent prefills but does not claim GPU-parallel multi-prefill. It measures the scheduler's actual prefill/decode interleave and requires a later interval where every request decodes concurrently.

## Required evidence

- Every request completes exactly 1,022 one-token SSE events with strictly increasing timestamps and no error.
- For each group, define the common steady window as `max(request token 65 timestamp)` through `min(request final-token timestamp)`. Every request must contribute at least 32 tokens in the window; at least 64 is recorded as the preferred-quality flag.
- Report common-window aggregate decode tok/s, per-request rate, speedup and scaling efficiency relative to B1, per-request TPOT statistics, TTFT, model-reported prefill throughput, and total group wall time.
- Each group must increment TP0 and TP1 `decode_cuda_graph`, leave `decode_none` at zero, use non-graph prefill, and recover KV=2,097,152, Mamba=40, running=0, queue=0.
- The candidate log must contain B1/B2/B4/B8 HiSparse ragged-FA2 active markers on both TP ranks and no OOM, graph fallback/capture failure, device assertion, unexpected worker exit, or unexpected EOF.
- Continuous NVML sampling must be error-free and end with an eight-sample per-device memory plateau within 16 MiB.
- Final status is descriptive throughput plus correctness/routing/resource acceptance. One run does not establish a fixed SLO.
