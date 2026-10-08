# sglang-qsa-hisparse

QSA HiSparse (request-scoped CPU KV offload and host prefix reuse for Qwen
Sparse Attention) as an SGLang general plugin, pinned to SGLang
`v0.5.21` (`e00930c548`; previously `76e06febab`, tag `pin-76e06febab-final`). See [docs/GOAL.md](docs/GOAL.md) and [docs/PLAN.md](docs/PLAN.md).

With no switch set the plugin registers nothing. `SGLANG_QSA_MODEL_COMPAT=1`
enables the reference fork's shared-path behavior; `SGLANG_QSA_HISPARSE_V3`
additionally enables the offload runtime. Activation verifies source
fingerprints of every patched SGLang definition and stops the process on any
failure.

CPU tests: `tools/run_cpu_tests.sh`.
