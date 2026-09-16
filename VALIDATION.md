# Validation of the source-fork migration

Date: 2026-09-16. Status: **CPU_VALIDATED_GPU_NOT_RUN**.

The tested runtime and packaging sources are committed as
`2da3ca5a09b422a17643f8755854a04d2aaf663c`. This integrates upstream
`76e06febab732d28a61b75a61b7835284568cdfb` through merge commit `59ad842ebc`.
The original QSA runtime was `5f8ae43640404eaee4c645d8cad2d0ef6e7dc6b0`.

## Executed checks

| Check | Result |
| --- | --- |
| Original QSA CPU baseline before the upstream merge | 72 passed, 1 skipped, 9 subtests passed |
| Final `scripts/test_qsa_hisparse_cpu.sh` | **102 passed, 3 skipped, 9 subtests passed** |
| Triton CPU interpreter | Included in the final suite; C4/tail movement and scaled compact/strided gather passed |
| Startup example | Parsed by the current `ServerArgs` parser and accepted by the QSA configuration guard; no model loaded |
| Compatibility imports | Old module names resolve to the new runtime classes |
| Refactor equivalence | All 10 moved class/function definitions retain their executable AST after documented identifier changes |
| Python wheel | Built with Rust extensions disabled; metadata, license and packaged QSA source contents checked |
| Isolated wheel import | Installed without dependencies into a temporary target; imported from that target and exercised a lease |
| Static checks | Selected Ruff F401/F821/UP037 checks, changed Python parsing, shell syntax and maintained Markdown links passed |
| Git/archive integrity | Both ancestries retained; no Git alternates; all 69 archived original payloads are byte-identical |

The three skips are upstream PLE checks requiring CUDA pinned memory or a GPU
that reads pageable host memory. The CPU runner hides CUDA and explicitly
enables Triton's interpreter. It does not start or replace a serving process.

The merge regression oracle compares gathered K/V to independent Torch indexing
with exact equality. Non-unit FP8 scales catch lost descale operations; NaN
scratch exposes missing zero-fill stores. Existing runtime tests cover slot
generations, short prompts, request routing, graph failure cleanup, copy/event
ordering, and logical free-group completion with CPU substitutes. No numerical
tolerance or supported model geometry was relaxed.

## Reproduce

In an environment with the runtime dependencies and pytest:

```bash
QSA_PYTHON=python bash scripts/test_qsa_hisparse_cpu.sh
```

Packaging was checked in a separate temporary environment with setuptools,
setuptools-scm, setuptools-rust, wheel, build, and the existing runtime
dependencies available:

```bash
SGLANG_BUILD_RUST_EXTS=none python -m build --wheel --no-isolation \
  --outdir /tmp/qsa-hisparse-wheel python
```

This checks Python packaging, not compilation or qualification of native Rust
extensions. The wheel was built from the staged sources subsequently committed
as `2da3ca5a09`; its development-version suffix names the pre-commit base
`59ad842ebc`. It is a local validation artifact, not a published release.

## Environment and remaining scope

CPU/interpreter tests used Python 3.12.3, Torch 2.13.0, Triton 3.7.1,
FlashInfer 0.6.18, Transformers 5.12.1, and the existing `sglang-kernel 0.4.6.post1`.
The merged source now requires **`sglang-kernel 0.4.7`**. Native dependencies in
the serving environment were not upgraded by this maintenance task.

Both GPUs were occupied by the existing service. CUDA kernel execution,
native-extension builds, TP2 service output comparisons, graph capture/replay
on hardware, lifecycle recovery under live traffic, and performance measurements
were not run for this revision. Validate those paths with the new dependency
set before deploying the merged source.

The throughput, latency and 8×256K capacity measurements under `archive/` remain
historical evidence for their recorded revisions. They are not validation of
this merge or of the source-fork refactor.
