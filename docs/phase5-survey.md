# Phase 5 survey: pin v0.5.21 → upstream main 35f3c96ff4, reference = production

Orchestrator, 2026-10-08. Owner: upgrade to upstream main without losing any
production feature; pin `35f3c96ff4` (upstream main of 2026-10-04, the base
of production) was chosen over the latest main (`1638c5d699` needs
transformers 5.19 and the prefix-cache lifecycle refactor #42823/#42923/#43023).

## Reference and environment

- Production on 8081 runs `.worktrees/sglang-dsh-production-20261004`
  (`897286b12a`): fork `ee8fe158d6` merged with upstream `35f3c96ff4`, plus
  production-only changes `773f3c2d84` (fast_topk keeps all candidates on
  radix overflow), `bdb935d70f` (FP8 KV descale in prefill and reference
  reads), `eec9df4723` (`SGLANG_NUMA_INTERLEAVE`), and the merge's
  adaptations (TP rank from the published parallel bundle, live TP CPU
  group, `registry.py` keeping `ChunkCache` for the QSA host prefix under
  #42354).
- New reference: `REFERENCE_FORK_COMMIT = 897286b12a`, `FORK_BASE_COMMIT =
  PINNED_SGLANG_COMMIT = 35f3c96ff4`. Because the pin is the reference's
  base, `test_replace_deltas.py` now requires every REPLACE copy to equal
  production's definition (mechanical edits reverted).
- Interpreter: production's `results/dsh-maintenance-20261004/upstream-runtime-env`
  (torch 2.14.1, sglang-kernel 0.4.9, flashinfer 0.7.0.post1, triton 3.8.0,
  transformers 5.17.0 = the pin's requirements; no sglang installed; pytest
  9.1.1), used read-only with `PYTHONPATH`.

## Production delta vs the previous reference

`git diff 35f3c96ff4 897286b12a -- python/sglang`: 35 modified files, 248
hunks, 15 new files (previous: 31/196/15). `tools/remap_hunks.py` carried 187
hunks to their rows by identical content; 76 are `?` in Appendix A (new
production changes, or fork changes production re-merged) — by file:

| File (python/sglang/) | `?` hunks | Area |
|---|---|---|
| kernels/jit/csrc/elementwise/fast_topk.cuh | 12 | C (production fix `773f3c2d84`) |
| srt/layers/attention/qwen_sparse_attn_backend.py | 24 | C |
| srt/layers/attention/qsa/kernel.py | 7 | C (`bdb935d70f`) |
| srt/layers/attention/qsa/sparse_attn.py | 4 | C |
| srt/layers/moe/fused_moe_triton/fused_marlin_moe.py | 1 | C |
| srt/models/qwen4_exp.py | 6 | C |
| srt/mem_cache/common.py | 4 | A |
| srt/mem_cache/registry.py | 4 | A (#42354 adaptation) |
| srt/model_executor/forward_batch_info.py | 4 | B |
| srt/model_executor/model_runner.py | 1 | B |
| srt/model_executor/pool_configurator.py | 1 | B |
| srt/environ.py, srt/utils/numa_utils.py | 1 + 7 | D (production `eec9df4723`) |

The 15 new files (runtime package, QSA aliases) map to their rows; their
production content may differ from `ee8fe158d6` (`tests/runtime/test_moved_sources.py`
now compares with production).

## Plugin targets changed v0.5.21 → 35f3c96ff4

| Row | Hook | Patch module | Upstream change (lines -/+) | Target |
|---|---|---|---|---|
| B03 | replace | hisparse/scheduler.py | -2 +2 | `managers.scheduler_components.batch_result_processor.SchedulerBatchResultProcessor.process_batch_result_prefill` |
| B05 | replace | hisparse/scheduler.py | -0 +3 | `managers.scheduler_components.batch_result_processor.SchedulerBatchResultProcessor._handle_finish_state_updated_req` |
| C01 | after | hisparse/pools.py | -0 +8 | `model_executor.pool_configurator.DefaultPoolConfigurator.__init__` |
| F01 | after | hisparse/graph.py | -14 +31 | `model_executor.forward_batch_info.ForwardBatch.init_new` |
| G02 | replace | hisparse/graph.py | -1 +1 | `model_executor.runner.decode_cuda_graph_runner.DecodeCudaGraphRunner.load_batch` |
| K02 | replace | hisparse/pools.py | -0 +8 | `mem_cache.kv_cache_configurator.KVCacheConfigurator._build_hybrid_linear_kv_pool` |
| K03 | around | hisparse/pools.py | -1 +14 | `mem_cache.qsa_kv_pool.QSATokenToKVPool.__init__` |
| M03 | replace | hisparse/lifecycle.py | -9 +8 | `mem_cache.common.release_kv_cache` |
| P02 | replace | hisparse/scheduler.py | -7 +48 | `managers.schedule_policy.PrefillAdder.add_one_req` |
| Q01 | replace | model_compat/qsa_attention.py | -3 +19 | `layers.attention.qwen_sparse_attn_backend._resolve_flash_attn_varlen_func` |
| Q02 | after | model_compat/qsa_attention.py | -0 +5 | `layers.attention.qwen_sparse_attn_backend.QwenSparseAttnBackend.__init__` |
| Q03 | after | hisparse/qsa_backend.py | -0 +5 | `layers.attention.qwen_sparse_attn_backend.QwenSparseAttnBackend.__init__` |
| Q05 | after | model_compat/qsa_attention.py | -0 +4 | `layers.attention.qwen_sparse_attn_backend.QwenSparseAttnBackend._metadata_from_forward_batch` |
| Q07 | after | hisparse/qsa_backend.py | -0 +4 | `layers.attention.qwen_sparse_attn_backend.QwenSparseAttnBackend._capture_cuda_graph_metadata` |
| Q08 | replace | model_compat/qsa_attention.py | -1 +17 | `layers.attention.qwen_sparse_attn_backend.QwenSparseAttnBackend.forward_extend` |
| Q10 | replace | model_compat/qsa_attention.py | -21 +67 | `layers.attention.qwen_sparse_attn_backend.QwenSparseAttnBackend._forward_trtllm_sparse` |
| Q11 | replace | model_compat/qsa_attention.py | -0 +5 | `layers.attention.qwen_sparse_attn_backend.QwenSparseAttnBackend.forward_decode` |
| Q12 | replace | model_compat/qsa_attention.py | -2 +2 | `layers.attention.qwen_sparse_attn_backend.QwenSparseAttnBackend._forward_paged_attention` |
| R02 | before | hisparse/graph.py | -1 +1 | `model_executor.model_runner.ModelRunner._prepare_eager_forward_batch` |
| S07 | replace | hisparse/scheduler.py | -1 +1 | `managers.scheduler.Scheduler.on_idle` |
| S09 | after | hisparse/scheduler.py | -5 +4 | `managers.scheduler.Scheduler.collect_inflight_reqs` |
| T03 | replace | model_compat/qsa_attention.py | -0 +3 | `layers.attention.qsa.qsa_indexer.QSAIndexer.select_decode_tokens` |
| T04 | replace | model_compat/qsa_attention.py | -4 +19 | `layers.attention.qsa.metadata.QSAIndexerMetadata` |

Fingerprinted definitions per file:

| Fingerprint file | same | changed | missing |
|---|---|---|---|
| framework.json | 10 | 0 | 0 |
| hisparse_graph.json | 35 | 14 | 0 |
| hisparse_image_identity.json | 5 | 0 | 0 |
| hisparse_lifecycle.json | 19 | 3 | 0 |
| hisparse_pools.json | 17 | 8 | 0 |
| hisparse_scheduler.json | 54 | 20 | 1 |
|  missing: `sglang.srt.mem_cache.common.maybe_cache_unfinished_req` | | | |
| model_compat_hyperconnection.json | 11 | 0 | 0 |
| model_compat_lifecycle.json | 2 | 0 | 0 |
| model_compat_marlin.json | 17 | 2 | 0 |
| model_compat_quantization.json | 17 | 0 | 0 |
| model_compat_qwen4_exp.json | 20 | 1 | 0 |
| model_compat_scheduler.json | 2 | 1 | 0 |
| qsa_attention.json | 34 | 12 | 0 |
| qsa_backend.json | 2 | 2 | 0 |
| runtime.json | 61 | 26 | 0 |
