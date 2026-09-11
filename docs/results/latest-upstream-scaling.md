# Latest-upstream 8x256K scaling result

> These measurements predate the TP2 AutoRound Marlin recovery in source
> `9f8016f5aac9c87d08d01f51761b3874127856a0`. The capacity result remains
> valid; see [Marlin g64 recovery](marlin-g64-recovery.md) for the matched B1
> decode correction.

## Verdict

`PASS_LATEST_UPSTREAM_REPLAY_AND_8X256K_PATH_PRESSURE`

The HiSparse stack was replayed cleanly on upstream `4309c7ce19dc42fb42cc9e7d883691c8dd8bda10`; the tested clean source is `8606d4b2df088512157c8df9615cff10e11028ae`. One service lifetime completed B1, B2, B4, and B8 in order. Every request used 261,120 prompt tokens and produced 1,022 output tokens.

This proves the latest-upstream candidate can admit, decode, finish, and release 8x256K-class requests on this server. It is a single-run capacity and scaling result, not a fixed SLO or production deployment decision.

## Performance

Decode rate uses the common pure-decode interval from the latest request's token 65 to the earliest request completion, so all requests in a group contribute during the same interval.

| Concurrent requests | Completed | Common interval | Min tokens/request in interval | Aggregate decode tok/s | Per-request decode tok/s | Mean TPOT | Aggregate prefill tok/s | Group wall time |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1/1 | 14.845 s | 957 | 64.466 | 64.466 | 15.512 ms | 4141.962 | 79.088 s |
| 2 | 2/2 | 17.144 s | 828 | 96.653 | 48.327 | 20.680 ms | 4080.786 | 148.647 s |
| 4 | 4/4 | 15.428 s | 573 | 148.757 | 37.189 | 26.922 ms | 4020.123 | 284.987 s |
| 8 | 8/8 | 2.281 s | 61 | 214.852 | 26.857 | 37.364 ms | 3952.726 | 557.507 s |

Relative to B1, aggregate decode throughput scales to 1.499x at B2, 2.308x at B4, and 3.333x at B8. Scaling efficiency is 75.0%, 57.7%, and 41.7%, respectively. Per-request decode throughput falls by 25.0%, 42.3%, and 58.3%.

Aggregate prefill throughput changes only from 4141.962 tok/s at B1 to 3952.726 tok/s at B8, a 4.57% reduction. Requests were submitted together, but the current scheduler serialized/interleaved their chunked prefills. The measured TTFT ranges therefore grow from 63.229 s at B1 to 63.453-528.858 s at B8; this run does not demonstrate simultaneous GPU prefill.

The B8 interval passed the contract's minimum of 32 common tokens per request, with 61-62 observed. It missed the preferred 64-token quality target by three tokens and spans only 2.281 seconds, so its 214.852 tok/s aggregate rate is descriptive rather than a stable SLO estimate.

## Route, capacity, and restore checks

- CUDA graph decode deltas were 2,042 / 2,298 / 2,810 / 3,834 for B1 / B2 / B4 / B8, on both TP ranks, with zero eager decode delta.
- Server samples observed graph batches through the requested group size, including B1-B8 during B8. HiSparse active markers appeared on TP0 and TP1 for every batch size.
- All 15 requests returned exactly 1,022 monotonic single-token SSE events, with no request or server fatal error.
- After every group, available KV tokens returned to 2,097,152, Mamba slots returned to 40, and running/queued requests returned to zero.
- NVML collected 7,574 samples without error. Both GPUs peaked at 46,136 MiB and reached a flat terminal plateau; peak utilization was 99% in every group.
- The pre-test service health, launch settings, source and environment were restored; the test endpoint was released.

The independent evidence audit passed the frozen contract. The candidate was not deployed by this run and `production_go` remains false.

## Evidence

- `execution-01/manifest.json`: source, contract, launch, and exact restore gates
- `execution-01/reduced.json`: consolidated performance, route, capacity, NVML, and GC result
- `execution-01/candidate/b{1,2,4,8}/reduced.json`: per-group reductions
- `requests.jsonl`: compact per-request output and timing records
- `routes.jsonl`: compact per-group graph and capacity records

Large raw SSE, server log, metrics, and NVML traces remain in `execution-01` locally and were intentionally not added to Git.
