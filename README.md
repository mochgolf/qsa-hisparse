# QSA HiSparse — SGLang plugin

QSA HiSparse adds request-scoped CPU KV offload and host prefix reuse (text
and images) for Qwen Sparse Attention to an unmodified SGLang, as an
out-of-tree SGLang general plugin (entry point group `sglang.srt.plugins`).

> **Migration.** This repository used to hold a source fork of SGLang. That
> code is preserved unchanged in the branch
> [`legacy/sglang-fork`](../../tree/legacy/sglang-fork) and the tag `fork-final`.
> The plugin reproduces the fork's behavior; see [Validation](#validation).

## Supported versions

The plugin is pinned to one SGLang revision and verifies it at startup:

- SGLang upstream `35f3c96ff4` (main, 2026-10-04), unmodified;
- torch 2.14.1, sglang-kernel 0.4.9, flashinfer-python 0.7.0.post1, triton
  3.8.0 ([`environment.lock.json`](src/sglang_qsa_hisparse/environment.lock.json));
- model: Qwen3.8 Flash Next (`Qwen4ExpForConditionalGeneration`), TP2, FP8
  E4M3 KV cache, page size 64, up to 262,144 tokens per request, up to 8
  running requests; tested on 2× RTX 4090 48 GB (SM89).

Activation compares the source bytes of every patched SGLang module with
pinned fingerprints and stops the process on any mismatch, so a different
SGLang revision fails at startup instead of misbehaving.

## Install

```bash
# SGLang at the pinned revision, in its own environment
git clone https://github.com/sgl-project/sglang.git
git -C sglang checkout 35f3c96ff4
python -m pip install -e sglang/python

# the plugin
python -m pip install git+https://github.com/mochgolf/qsa-hisparse.git
```

## Run

Start the server through the plugin's launcher. It enables the plugin in
every scheduler process, checks the native library versions, and reports
ready only after each TP rank recorded a verified activation:

```bash
export SGLANG_QSA_MODEL_COMPAT=1            # the model's shared-path changes
export SGLANG_QSA_HISPARSE_V3=p2-offload    # the offload runtime (needs the line above)
export SGLANG_QSA_HISPARSE_PREFIX_CACHE_MB=32768        # host prefix cache per TP rank
export SGLANG_QSA_HISPARSE_PREFIX_CACHE_MAX_ENTRIES=256
python -m sglang_qsa_hisparse.launch -- \
  --model-path <Qwen3.8 Flash Next W4A16 AutoRound INT8PLE checkpoint> \
  --model-impl sglang --dtype bfloat16 --tp-size 2 \
  --ple-offload-embedding --json-model-override-args '{"text_config":{"ple_embedding_dtype":"int8_row"}}' \
  --kv-cache-dtype fp8_e4m3 --page-size 64 --context-length 262144 \
  --linear-attn-backend triton --mamba-ssm-dtype float32 \
  --mamba-radix-cache-strategy extra_buffer --mamba-track-interval 64 --max-mamba-cache-size 40 \
  --mem-fraction-static 0.96 --max-running-requests 8 --max-total-tokens 2097152 \
  --chunked-prefill-size 2048 --prefill-decode-interval 1 \
  --disable-radix-cache --disable-overlap-schedule \
  --cuda-graph-backend-decode full --cuda-graph-backend-prefill disabled \
  --disable-cuda-graph-padding --cuda-graph-max-bs-decode 8 --cuda-graph-bs-decode 1 2 3 4 5 6 7 8 \
  --image-processor-backend pil --mm-preprocess-cache-size-mb 2048 \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder --enable-cache-report
```

This is the maintainer's production configuration (optionally under
`numactl --interleave=all` with `SGLANG_NUMA_INTERLEAVE=1`).

- With no switch set the plugin registers nothing. `SGLANG_QSA_MODEL_COMPAT=1`
  alone applies only the model-compatibility changes.
- Keep `--disable-radix-cache`: prefix reuse is the plugin's CPU host prefix
  cache. Repeated prefixes are reused at 64-token page boundaries;
  `usage.prompt_tokens_details.cached_tokens` reports the hit.
- Image prompts are reused too, including at boundaries inside an image.
  Images are identified by content, preprocessing, order, position and grid.
  This needs `--mm-preprocess-cache-size-mb > 0` (a CPU-only cache, no GPU
  memory). Video, audio and caller-supplied features bypass reuse.
- Unsupported configurations fail at startup.

## Validation

The plugin is checked against the code it replaces, in the same GPU window
and environment, on GPU-kernel probes and tests, a deterministic TP2 server
qualification to 262,016 tokens (token IDs, cached tokens, logprobs and
byte digests of every restored state), a compatibility-only profile, memory
figures, image prefix evidence, and native latency/smoke checks. Results:
[`docs/phase5-results.md`](docs/phase5-results.md) (current pin, reference =
the fork's production build), [`docs/phase4-results.md`](docs/phase4-results.md),
[`docs/phase2-results.md`](docs/phase2-results.md),
[`docs/window3-results.md`](docs/window3-results.md) (images). Intentional
differences from the fork are listed in [`docs/DEVIATIONS.md`](docs/DEVIATIONS.md).

## How it works

Patches are declared per inventory row ([`docs/patch-inventory.md`](docs/patch-inventory.md))
and applied through SGLang's `HookRegistry` (before/after/around hooks, and
whole-definition replacement where no narrower seam exists). Activation is
fail-closed: the declared patches must equal [`manifest.json`](src/sglang_qsa_hisparse/manifest.json),
every target and dependency must match its pinned fingerprint, and no other
plugin may hook the same targets. The plan, rules and history are in
[`docs/GOAL.md`](docs/GOAL.md), [`docs/PLAN.md`](docs/PLAN.md),
[`docs/STATUS.md`](docs/STATUS.md) and `reviews/` (independent reviews per gate).
Paths written `qwen:…` in the docs refer to the maintainer's private
workspace (raw GPU evidence, service profiles) and are not part of this
repository.

## Development

CPU tests (no GPU; select SGLang sources with `QSA_PIN_ROOT`, an interpreter
with `QSA_PYTHON`):

```bash
QSA_PIN_ROOT=<sglang checkout at 35f3c96ff4> QSA_PYTHON=<python> tools/run_cpu_tests.sh
```

## License

Apache-2.0 ([LICENSE](LICENSE)). Parts of the plugin are copied from SGLang
(Apache-2.0) and from the QSA HiSparse fork of it.
