# QSA HiSparse

QSA HiSparse is a maintained SGLang source fork for Qwen Sparse Attention with
CPU KV offload. The runtime, kernels, scheduler integration, tests, and build
files are all in this repository. Clone it and install the source directly.

The fork retains SGLang's Git ancestry and the original QSA commits. The latest
integrated upstream revision is `76e06febab` (2026-09-16); see
[provenance](PROVENANCE.json) and the [upstream maintenance guide](UPSTREAM.md).

## Install

Use a separate environment on Linux with NVIDIA CUDA. Python 3.10+ is required;
the exact Torch, CUDA-related packages, and other dependencies are declared in
[`python/pyproject.toml`](python/pyproject.toml). Building native extensions
also requires the upstream Rust/CUDA build toolchain.

```bash
git clone https://github.com/mochgolf/qsa-hisparse.git
cd qsa-hisparse
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ./python
```

The Python distribution and imports retain the name `sglang` for compatibility.
This checkout supplies SGLang itself; use one SGLang checkout per environment.
See the [upstream README](README.sglang.md) for general SGLang setup.

## Run

The current offload contract targets Qwen3.8 Flash Next, TP2, FP8 E4M3 KV,
C4 compression, page size 64, and a 262,144-token per-request capacity. The
multi-request runtime permits 2, 4, or 8 request slots, including B1 decode
within that capacity. Unsupported configurations fail during startup.

After installing the runtime and preparing a compatible model:

```bash
MODEL_PATH=/path/to/model bash examples/qsa_hisparse/serve.sh
```

This launches the bounded B8 configuration on two GPUs. Model weights, host RAM,
and GPU capacity must fit the checkpoint. Read the [runtime guide](QSA_HISPARSE.md)
for memory ownership, mode selection, optional INT8-row PLE offload, and limits.
The [service controller](examples/qsa_hisparse/SERVICE.md) provides start, stop,
restart, status, and logs. [Host prefix reuse](PREFIX_CACHE.md) keeps reusable
state in CPU memory while preserving the existing GPU staging and decode pools.

## Read and develop

| Entry point | Purpose |
| --- | --- |
| [`qsa_hisparse/`](python/sglang/srt/mem_cache/qsa_hisparse/) | Configuration, C4 layout, leases, runtime, scheduler coordinator |
| [`qwen_sparse_attn_backend.py`](python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py) | Prefill writes, sparse selection, FA2 decode and graph integration |
| [`qsa/`](python/sglang/srt/layers/attention/qsa/) | Metadata, indexer, gather and graph byte-movement kernels |
| [`test/qsa_hisparse/`](test/qsa_hisparse/) | Focused CPU and Triton-interpreter regression suite |
| [`QSA_HISPARSE.md`](QSA_HISPARSE.md) | Architecture, invariants, configuration and source map |
| [`UPSTREAM.md`](UPSTREAM.md) | Merge workflow and compatibility review points |
| [`VALIDATION.md`](VALIDATION.md) | Current integration evidence and remaining GPU checks |
| [`archive/`](archive/README.md) | Frozen 2026-09-11 experiments, engineering write-up and measurements |

Run the CPU suite in an environment with the runtime dependencies and pytest:

```bash
python -m pip install pytest
QSA_PYTHON=python bash scripts/test_qsa_hisparse_cpu.sh
```

Historical throughput and 8×256K service results apply to their recorded source
revisions. They are not performance claims for the current upstream merge.

## License

Apache-2.0. SGLang's source headers, license, and commit attribution are retained.
