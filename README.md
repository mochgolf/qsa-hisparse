# QSA HiSparse

Experimental integration of HiSparse CPU KV offload with Qwen Sparse Attention
(QSA) in [SGLang](https://github.com/sgl-project/sglang).

This repository keeps the reviewable experiment assets: the upstream patch
series, design notes, bounded validation harnesses, profiles, and reduced
results. Sanitized SVG figures derived from NSYS are included; raw service logs,
SSE traces, NSYS reports/SQLite exports, model weights, and other machine data
are intentionally excluded. Public artifacts omit local paths, process IDs,
served-model aliases, private email addresses, and live endpoint state.

Read the full Chinese engineering write-up:
[在双 RTX 4090 48GB 上把 HiSparse 移植到 QSA](docs/blog/qsa-hisparse-dual-4090-48gb.md).

## Tested revision

- SGLang upstream base: `4309c7ce19dc42fb42cc9e7d883691c8dd8bda10`
- QSA HiSparse head: `5f8ae43640404eaee4c645d8cad2d0ef6e7dc6b0`
- Fork branch: [`mochgolf/qwen38-hisparse-upstream-latest-20260911`](https://github.com/mochgolf/sglang/tree/qwen38-hisparse-upstream-latest-20260911)
- Hardware: 2x SM89 GPUs, tensor parallelism 2
- Model setup: Qwen3.8 Flash Next, BF16 compute, FP8 E4M3 KV, no MTP

## Measured results

The short-context fast-path A/B used the same service configuration for each
pair. A separate matched latest-upstream check restored B1 from 66.77 to 86.37
tok/s (+29.36%) after correcting its MoE backend. The 256K run used 261,120
prompt tokens per request and one pre-correction latest-upstream service lifetime.

| Workload | Before | QSA HiSparse fast path | Change |
| --- | ---: | ---: | ---: |
| B1, 38K | 69.38 tok/s | 86.07 tok/s | +24.05% |
| B2, 4K | 130.50 tok/s aggregate | 158.34 tok/s aggregate | +21.33% |
| B8, 4K | 400.16 tok/s aggregate | 464.15 tok/s aggregate | +15.99% |

| 256K concurrency | Aggregate decode | Per request | Mean TPOT |
| ---: | ---: | ---: | ---: |
| 1 | 64.47 tok/s | 64.47 tok/s | 15.51 ms |
| 2 | 96.65 tok/s | 48.33 tok/s | 20.68 ms |
| 4 | 148.76 tok/s | 37.19 tok/s | 26.92 ms |
| 8 | 214.85 tok/s | 26.86 tok/s | 37.36 ms |

The B8 result proves admission, decode, completion, and release of eight
256K-class requests on the tested server. Prefill remained serialized or
interleaved by the scheduler, and these single-run numbers are not a fixed SLO.

## Repository layout

- `patches/`: ordered 13-commit patch series against the pinned upstream base
- `docs/design/`: memory model, interface contract, and integration plan
- `docs/results/`: phase conclusions and independent review summaries
- `experiments/`: frozen microbenchmark, profile, fast-path, and service drivers
- `results/`: compact manifests and reduced measurements

## Apply the patch series

```bash
git clone https://github.com/sgl-project/sglang.git
git -C sglang checkout 4309c7ce19dc42fb42cc9e7d883691c8dd8bda10
git -C sglang am ../qsa-hisparse/patches/*.patch
```

The frozen drivers document the acceptance logic used on the test server. They
use placeholders for private artifacts and require equivalent model, NUMA,
CUDA, and SGLang settings before execution. Start with
[`docs/design/deployment-assessment.md`](docs/design/deployment-assessment.md)
and [`docs/results/latest-upstream-scaling.md`](docs/results/latest-upstream-scaling.md).

## Scope

Validated behavior includes TP2 ownership, lifecycle and generation reuse,
FP8 K/V payloads and scales, host writeback/refetch, CUDA Graph decode from B1
through B8, resource recovery, and 8x256K-class capacity. Concurrent GPU
prefill and a portable production SLO are outside the current evidence.

## License

Apache-2.0. The patch series targets SGLang and retains its original file
headers and commit attribution.
