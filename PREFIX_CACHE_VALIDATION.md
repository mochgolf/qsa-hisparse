# Host prefix qualification, 2026-09-16

Status: passed for the configuration and scope below; native/light service deployed.

## Configuration and evidence

The target is two 48 GiB RTX 4090 cards, TP2, the local W4A16 Qwen4Exp model,
FP8 E4M3 KV, C4/page64, a 262,144-token context limit, eight active requests,
2,048-token chunked prefill and full decode CUDA graphs for batches 1–8.
The cache budget is 8 GiB per TP rank, with at most 256 retained/pending entries.
The checkpoint budget is additional CPU memory; active request backing and
model/PLE host allocations are separate.

The source baseline is `e6e90a9c66`. Service control is `f9731024aa`,
deterministic comparison corrections are `5c11ef5dc4`, and host prefix reuse is
`0f0d27abce`. The deterministic whole-K Marlin correction is `6bfed25cb0`.
The final live concurrency, latency and strengthened ledger harnesses are
`07e1d774bf`; the 19-file runtime manifest remained unchanged during qualification.
Machine-specific profiles, frozen inputs, raw HTTP responses,
source hashes and per-rank ledgers are retained outside the repository in
`../results/prefix-cache-service-20260916/`. Model paths and raw logs are not
part of this source distribution.

The frozen input manifest has SHA256
`171231bd51a27bc106e66905c1a59136326cfddf927467d7167e0a48764216d8`.
It predates candidate measurements. No failing fixture was removed and no
candidate-dependent numerical tolerance was introduced.

## Independent checks completed

| Check | Result |
| --- | --- |
| Final CPU regression suite | 183 passed, 20 subtests passed, 7 CUDA-only skips |
| Marlin stable placement | Independent W4/group128 GPU counterexample and reference probe passed; CUDA graphs passed for 76, 256 and 2,048 routing rows |
| Marlin whole-K reduction | Actual group64 geometry: 237 placement/background cases exact; 43 graph cases at M1/8/96/2048 exact, including changed routing on replay; 33 GPU-enabled tests passed |
| Marlin native regression | All stage SHA256 values in 237 cases equal the pre-correction native control |
| QSA deterministic top-k | 36 exact independent-oracle cases, 504 graph replays and 36 dynamic-length graph checks passed on SM89 |
| Top-k wide prefill | 2,048 × 65,536 scores: 0.103 s in the standalone probe; 20.63 MiB peak additional allocation including the 4 MiB output |
| Cache lifecycle on snapshot 7 | Live flush/miss/refill, cancellation, subsequent row reuse and requested input logprobs passed |
| Actual concurrent decode on snapshot 7 | 65,536-token prefix; two cold 128-token controls exact; every warm B2/B8 output ID exact; both ranks recorded the corresponding B2/B8 graph replays |
| Frozen single-request HTTP cases on snapshot 7 | All 16 passed at prefix lengths 64, 128, 2,048, 4,096, 8,192, 32,768, 65,536 and 262,016; exact cold/repeated-cold/warm/repeated-warm token IDs for both suffix branches |
| Service lifecycle | Owned cgroup start/stop/restart, readiness, idempotency, profile drift, failure cleanup and unrelated-listener refusal passed |

The seven CPU-suite skips include four Marlin CUDA tests run separately
above and three existing PLE CUDA tests. They are not reported as CPU passes.
Real model strict restoration checks also compare every saved PLE, recurrent,
pending-C4, compressed-index and FP8 raw-KV tensor exactly after transfer.

The unchanged original service and the clean fork both failed identical-cold
greedy repeatability. Independent traces located native quantized expert row
placement and QSA top-k ordering as sources of nondeterminism. Exact cold/warm
HTTP comparisons therefore use deterministic math on both arms. Native/light
serving performance is qualified separately and does not claim bitwise output
repeatability. See [the frozen plan](PREFIX_CACHE_PLAN.md) for the counterexamples
and the rejected FlashInfer implementation that exceeded SM89 shared memory.

Snapshot 5 passed all 16 single-request cases, but failed two of eight long
concurrent output comparisons. Stable placement alone left Marlin's K-reduction
partition dependent on batch shape. Snapshot 7 adds whole-K reduction in
deterministic mode, force-miss handling and retry-safe exceptional restore
cleanup. The model comparisons and lifecycle checks are rerun on that source;
the earlier failed concurrent run is retained in the evidence directory.

The ledger checker also rejects incomplete final records, requires the final
event to have no active leases or pending releases and all logical pages free,
and compares request ID, row, generation, lease slot and forward ID across TP.
It checks twelve fixed pool capacities/byte sizes as well as raw/index pointers
within each rank. Twenty-one synthetic false-pass counterexamples were rejected.

## Long-context and deployment results

All 16 frozen cases passed on snapshot 7, with actual reuse equal to the selected
prefix length. The original eight-way cold/warm regression also passed with no
output mismatches. The separate fixed-128-token test proves actual B2/B8 graph
execution and exact output agreement with repeated cold controls. These exact
numerical comparisons use the deterministic configuration on both arms.

Each TP rank recorded 42,682 events and 60 restores, up to 262,016 tokens.
Their prefix event sequences, including request identities, matched exactly.
The maximum accounted checkpoint storage was 8,589,913,568 bytes, below the
8,589,934,592-byte per-rank budget; maximum retained entries were 120 and
1,229 evictions occurred. All 2,097,152 logical tokens were available at the
final idle event, with no active leases or pending releases.

The raw-KV pool remained 1,611,251,712 bytes and compressed index pool remained
1,610,661,888 bytes per GPU, with fixed addresses and capacities. All twelve
recorded fixed pool fields stayed constant. Idle CUDA allocated memory ranged
from 45,375,525,888 to 45,437,541,888 bytes, a 59.14 MiB working variation;
maximum idle reserved memory was 47,796,191,232 bytes. The claim is bounded
historical cache storage and fixed device pools, not invariant CUDA allocator
reservations or zero restoration workspace.

The native/light service is running on its original endpoint and served-model
name. Its live flush/refill, cancellation, row reuse and requested input-logprob
checks passed. Eight simultaneous native/light requests also completed 128
tokens each and each reused 65,536 tokens. This native check asserts completion
and cache accounting, not bitwise numerical repeatability.

The final native/light profile keeps the original model, TP2/B8/256K configuration
and local API endpoint. It enables 8 GiB per-rank host checkpoints, 256-row
logprob chunks, and OpenAI cache reporting. Readiness passed, and an idempotent
start retained the same process. The original source/environment remain
available through the separate rollback profile.

Two paired one-token HTTP measurements per length, using distinct cold salts
and identical warm inputs, gave these medians. Timing includes checkpoint
restoration and the complete HTTP response; these are local observations, not
a general throughput or streaming-latency guarantee.

| Prefix tokens | Cold request | Warm request | Observed ratio |
| ---: | ---: | ---: | ---: |
| 8,192 | 1.416 s | 0.131 s | 10.8× |
| 65,536 | 10.889 s | 0.210 s | 51.8× |
| 262,016 | 60.514 s | 0.510 s | 118.7× |

Every warm request reported the complete listed prefix as reused. Device usage
was 46,740 MiB per GPU before and after this series, following lifecycle and
concurrency warmup. The cache does not remove model weights or fixed pools.

The OpenAI chat cold and warm requests both answered `42`; the warm response
reported 2,048 cached tokens for a 2,089-token prompt. The initial smoke helper
incorrectly required a details object on the cold response. Upstream
`UsageProcessor._details_if_cached` intentionally returns `None` for zero hits;
the helper was corrected to interpret that as zero. The original failed helper
and response are retained, with the parser correction recorded separately.
Request text and warm-hit requirements were unchanged; runtime code was not
modified, and the corrected check passed after an explicit cache flush.

## Reproduction entry points

- `test/manual/qsa_hisparse_prefix_acceptance.py`: frozen token fixtures, two
  salted cold controls, seed, warm/repeated warm, branching suffix and concurrency.
- `test/manual/qsa_hisparse_prefix_lifecycle.py`: real cache flush, cancellation,
  row reuse and input-logprob preservation.
- `test/manual/qsa_hisparse_prefix_concurrency.py`: fixed 128-token continuations
  and per-rank proof of actual B2/B8 graph execution against cold controls.
- `test/manual/qsa_hisparse_prefix_ledger.py`: TP event agreement, bounded CPU
  accounting, pool identity and logical-page recovery.
- `test/manual/qsa_hisparse_prefix_latency.py`: complete one-token HTTP latency,
  including restoration, for paired cold/warm requests.
- `scripts/test_qsa_hisparse_cpu.sh`: focused and upstream integration regressions.

The initial cache reuses ordinary text only, at captured page64 boundaries.
Unsupported request state bypasses reuse or is rejected at startup as documented
in [PREFIX_CACHE.md](PREFIX_CACHE.md). Cold computation, prefix transfers and
attention for new tokens remain. A service restart discards the in-memory cache.
Image/video inference and other model or hardware configurations are outside
this qualification. Native batch-dependent arithmetic remains as characterized
in the baseline; exact HTTP output comparisons above use deterministic mode.
