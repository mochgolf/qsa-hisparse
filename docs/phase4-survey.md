# Phase 4 survey: pin 76e06febab → v0.5.21

Orchestrator, 2026-10-08. Read-only comparison of the pristine checkouts
`../.worktrees/sglang-pin-76e06febab` and `../.worktrees/sglang-v0.5.21`
(`e00930c548`, tag `v0.5.21`, a release branch cut from main at
`bd78095030` on 2026-09-29 plus cherry-picks; 730 commits past the old pin).

## Why v0.5.21

- It is the first SGLang release after the old pin (v0.5.20's branch does not
  contain it), so the plugin maps to a published version.
- Native dependencies are unchanged: torch 2.13.0, sglang-kernel 0.4.7,
  transformers 5.12.1 (`python/pyproject.toml`). The other pins that moved
  (xgrammar 0.2.7, `regex`) are already what the approved interpreter
  `../service/runtime-env-sglang-20260923` has; sentencepiece is pinned to
  0.2.1 upstream for InternVL only (the interpreter has 0.2.2; not used by
  Qwen). So CPU tests and GPU validation keep the same interpreter and select
  SGLang sources with `PYTHONPATH`, as before; `environment.lock.json` is
  unchanged.
- Upstream main (1268 commits past the old pin) moved to torch 2.14.1 /
  sglang-kernel 0.4.9 and contains #42354 (hybrid-SSM models get a
  `UnifiedRadixCache`), which requires the host prefix cache to become a
  registered radix-cache backend. Neither #42354 nor #39862/#39893 is in
  v0.5.21; that is the next cycle.

## What changed under the plugin

Patch targets whose definition changed (`moved`: the module moved, #41243
`layers/hc_mix_triton.py` → `sglang/kernels/ops/gemm/hc_mix.py`):

| Row | Hook | Patch module | Upstream change (lines -/+) | Target |
|---|---|---|---|---|
| C01 | after | hisparse/pools.py | -10 +5 | `model_executor.pool_configurator.DefaultPoolConfigurator.__init__` |
| E03 | replace | model_compat/qwen4_exp.py | -13 +25 | `models.qwen4_exp.Qwen4ExpNGramEmbedding.__init__` |
| E08 | replace | model_compat/qwen4_exp.py | -6 +32 | `models.qwen4_exp.Qwen4ExpForConditionalGeneration.load_weights` |
| F01 | after | hisparse/graph.py | -11 +41 | `model_executor.forward_batch_info.ForwardBatch.init_new` |
| FW1 | before | framework.py | -31 +16 | `managers.scheduler.configure_scheduler_process` |
| G01 | replace | hisparse/graph.py | -2 +7 | `model_executor.runner.decode_cuda_graph_runner.DecodeCudaGraphRunner.capture_one_shape` |
| G02 | replace | hisparse/graph.py | -4 +39 | `model_executor.runner.decode_cuda_graph_runner.DecodeCudaGraphRunner.load_batch` |
| G03 | replace | hisparse/graph.py | -3 +14 | `model_executor.runner.decode_cuda_graph_runner.DecodeCudaGraphRunner.execute` |
| H05 | replace | model_compat/hyperconnection.py | moved | `layers.hc_mix_triton.fused_hc_mix_supported` |
| H06 | around | model_compat/hyperconnection.py | moved | `layers.hc_mix_triton.fused_hc_mix` |
| K01 | around | hisparse/pools.py | -0 +1 | `mem_cache.kv_cache_configurator.KVCacheConfigurator._build_token_to_kv_pool` |
| K03 | around | hisparse/pools.py | -2 +2 | `mem_cache.qsa_kv_pool.QSATokenToKVPool.__init__` |
| M03 | replace | hisparse/lifecycle.py | -16 +18 | `mem_cache.common.release_kv_cache` |
| P01 | replace | hisparse/scheduler.py | -1 +19 | `managers.schedule_policy.PrefillAdder.add_chunked_req` |
| P02 | replace | hisparse/scheduler.py | -2 +9 | `managers.schedule_policy.PrefillAdder.add_one_req` |
| Q12 | replace | model_compat/qsa_attention.py | -1 +20 | `layers.attention.qwen_sparse_attn_backend.QwenSparseAttnBackend._forward_paged_attention` |
| R02 | replace | hisparse/graph.py | -0 +1 | `model_executor.model_runner.ModelRunner._forward_raw` |
| S04 | replace | hisparse/scheduler.py | -7 +2 | `managers.scheduler.Scheduler.get_next_batch_to_run` |
| S06 | replace | hisparse/scheduler.py | -1 +17 | `managers.scheduler.Scheduler._get_new_batch_prefill_raw` |
| S07 | replace | hisparse/scheduler.py | -1 +9 | `managers.scheduler.Scheduler.on_idle` |
| S08 | after | hisparse/scheduler.py | -2 +5 | `managers.scheduler.Scheduler.is_fully_idle` |
| S09 | after | hisparse/scheduler.py | -1 +1 | `managers.scheduler.Scheduler.collect_inflight_reqs` |
| T02 | around | model_compat/qsa_attention.py | -1 +1 | `layers.attention.qsa.kernel.qsa_fast_topk` |
| T03 | replace | model_compat/qsa_attention.py | -1 +1 | `layers.attention.qsa.qsa_indexer.QSAIndexer.select_decode_tokens` |

Unchanged REPLACE targets (22) need only new fingerprints. Fingerprinted
definitions (targets and `depends`) per file; 63 of 102 fingerprinted modules
changed bytes:

| Fingerprint file | same | changed | missing |
|---|---|---|---|
| framework.json | 6 | 2 | 0 |
| hisparse_graph.json | 38 | 11 | 0 |
| hisparse_image_identity.json | 4 | 1 | 0 |
| hisparse_lifecycle.json | 14 | 6 | 0 |
| hisparse_pools.json | 17 | 8 | 0 |
| hisparse_scheduler.json | 49 | 28 | 0 |
| model_compat_hyperconnection.json | 5 | 1 | 6 |
|  missing: `sglang.srt.layers.hc_mix_triton._deterministic_inference` | | | |
|  missing: `sglang.srt.layers.hc_mix_triton._get_counters` | | | |
|  missing: `sglang.srt.layers.hc_mix_triton._grid_barrier` | | | |
|  missing: `sglang.srt.layers.hc_mix_triton._hc_mix_persistent_kernel` | | | |
|  missing: `sglang.srt.layers.hc_mix_triton.fused_hc_mix` | | | |
|  missing: `sglang.srt.layers.hc_mix_triton.fused_hc_mix_supported` | | | |
| model_compat_lifecycle.json | 2 | 0 | 0 |
| model_compat_marlin.json | 18 | 1 | 0 |
| model_compat_quantization.json | 17 | 0 | 0 |
| model_compat_qwen4_exp.json | 16 | 4 | 1 |
|  missing: `sglang.srt.layers.dp_attention.get_attention_dp_size` | | | |
| model_compat_scheduler.json | 2 | 1 | 0 |
| qsa_attention.json | 40 | 3 | 1 |
|  missing: `sglang.kernels.ops.elementwise.fast_topk.fast_topk` | | | |
| qsa_backend.json | 4 | 0 | 0 |
| runtime.json | 51 | 34 | 1 |
|  missing: `sglang.srt.distributed.parallel_state_wrapper.ParallelState` | | | |

Elsewhere: `fast_topk` is at `sglang/kernels/ops/attention/fast_topk.py`;
`get_attention_dp_size` and `ParallelState` were removed or renamed (the
owning task finds the replacement).
