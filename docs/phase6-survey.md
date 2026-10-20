# Phase 6 survey: pin 35f3c96ff4 → v0.5.22 / upstream main

Routine maintenance, 2026-10-18. Read-only comparison of the pristine
checkouts `../.worktrees/sglang-main-35f3c96ff4` (current pin),
`../.worktrees/sglang-v0.5.22` (`baa9befa38`, tag `v0.5.22`, release branch
cut from main at `684fea8af3` on 2026-10-13) and
`../.worktrees/sglang-main-7c9a78efbe` (upstream main, 2026-10-19 13:27 +08:00).
Nothing was pushed or commented upstream; PR states are from `gh` (GET only).

## Candidates

| | v0.5.22 | main `7c9a78efbe` |
|---|---|---|
| Commits past the pin | 646 | 996 |
| torch | 2.14.1 (same) | 2.14.1 (same) |
| sglang-kernel | 0.4.9 (same) | **0.4.10** (#43922) |
| flashinfer_python | 0.7.1 (#43494) | 0.7.2 (#44591) |
| transformers | 5.19.0 (#42784) | 5.19.0 |
| Patch targets with changed definitions | 27 | 28 (+S05) |
| REPLACE rows among them | 16 of 32 | 16 of 32 |
| Fingerprinted definitions: same / module only / changed / missing | 90 / 203 / 95 / 2 | 69 / 208 / 111 / 2 |

Every fingerprinted module except a few changed bytes, so all fingerprints
must be regenerated whichever target is chosen. Production's interpreter
(`results/dsh-maintenance-20261004/upstream-runtime-env`: torch 2.14.1,
sglang-kernel 0.4.9, transformers 5.17.0) matches neither; a new interpreter
is needed (production's must not change).

## Blocking upstream change: prefix-cache lifecycle refactor

Both candidates contain the refactor that Phase 5 deferred:

- #42822 `extend_range` → `extend_end` from `prefix_len`; #42824 prefill
  progress as `prefix_len`; #42825 `Req.prefix_indices` removed (KV indices
  derived from `last_node` at allocation); #42823 match write-back folded into
  `match_kv_cache`; #42923 `match_prefix` returns only the matched length;
  #43023 `init_load_back` returns only the loaded length; #41452 anchored lock
  receipt for non-root releases; #41593 waiting requests refresh cached
  prefixes.
- #43198/#43199/#43200 serve `--disable-radix-cache` with
  `UnifiedRadixCache`/`PureSWARadixCache`; **#43201 deletes `ChunkCache`**.

Plugin impact: `QSAHostPrefixCache` subclasses and wraps `ChunkCache`
(`hisparse/prefix_cache.py`), S01 checks `type(tree_cache) is ChunkCache`, and
M05/M06 force a `ChunkCache` under `--disable-radix-cache`. The plugin reads
`prefix_indices`/`extend_range` 19 times (`hisparse/prefix_cache.py`,
`patches/hisparse/scheduler.py`, `patches/hisparse/lifecycle.py`).
`claim_kv_row`, `free_kv_row`, `checkpoint` and `on_release` still exist on
`BasePrefixCache`. `register_radix_cache_backend` + `--radix-cache-backend`
is the supported extension point; a registered backend would replace M05/M06
(two patches, one REPLACE) instead of porting them.

## Plugin targets with changed definitions (v0.5.22; main adds S05)

| Rows | Lines -/+ (v0.5.22) | Target (`sglang.srt.`) |
|---|---|---|
| B02 replace | -1 +1 | `managers.scheduler_components.batch_result_processor.SchedulerBatchResultProcessor.process_batch_result_prebuilt` |
| B03 replace | -16 +36 | `….process_batch_result_prefill` |
| B04 replace | -6 +2 | `….\_handle_sampling_mask_abort` |
| B05 replace | -9 +5 | `….\_handle_finish_state_updated_req` |
| C01 after | -13 +31 | `model_executor.pool_configurator.DefaultPoolConfigurator.__init__` |
| C02 before | -1 +1 | `….calculate_pool_sizes` |
| F01 after | -8 +35 | `model_executor.forward_batch_info.ForwardBatch.init_new` |
| G01 replace | -4 +9 (main -11 +11) | `model_executor.runner.decode_cuda_graph_runner.DecodeCudaGraphRunner.capture_one_shape` |
| G02 replace | -3 +16 | `….load_batch` |
| H08 around | -4 +28 | `layers.hyperconnection.GatedResidual.mix` |
| K01 around | -21 +31 | `mem_cache.kv_cache_configurator.KVCacheConfigurator._build_token_to_kv_pool` |
| K02 replace | -6 +6 | `….\_build_hybrid_linear_kv_pool` |
| K03 before | -38 +18 | `mem_cache.memory_pool.HybridLinearKVPool.__init__` |
| M01 replace | -1 +8 | `mem_cache.allocation.alloc_for_extend` |
| M03 replace | -6 +7 | `mem_cache.common.release_kv_cache` |
| M04 around | -1 +3 | `mem_cache.memory_pool.ReqToTokenPool.clear` |
| M05 around | -24 +13 | `mem_cache.registry.default_radix_cache_factory` |
| P01 before | -8 +21 | `managers.schedule_policy.PrefillAdder.add_chunked_req` |
| P02 replace | -14 +22 (main -15 +32) | `….add_one_req` |
| Q08 replace | -25 +13 | `layers.attention.qwen_sparse_attn_backend.QwenSparseAttnBackend.forward_extend` |
| Q10 replace | -0 +2 (main -3 +10) | `….\_forward_trtllm_sparse` |
| S03 after | -0 +5 | `managers.scheduler.Scheduler._build_hisparse_decode_batch` |
| S04 replace | -24 +22 (main -34 +28) | `….get_next_batch_to_run` |
| S05 after (main only) | -6 +11 | `….get_num_allocatable_reqs` |
| S06 replace | -25 +31 (main -35 +40) | `….\_get_new_batch_prefill_raw` |
| S07 replace | -0 +1 | `….on_idle` |
| S08 after | -0 +4 | `….is_fully_idle` |
| T03 replace | -1 +5 | `layers.attention.qsa.qsa_indexer.QSAIndexer.select_decode_tokens` |

Missing dependencies: `mem_cache.chunk_cache.ChunkCache` (#43201) and
`utils.common.Range`.

Fingerprinted definitions per file (same / module only / changed / missing):

| File | v0.5.22 | main |
|---|---|---|
| framework.json | 6/3/1/0 | 6/2/2/0 |
| hisparse_graph.json | 11/30/11/0 | 8/32/12/0 |
| hisparse_image_identity.json | 3/2/0/0 | 2/2/1/0 |
| hisparse_lifecycle.json | 2/18/10/1 | 2/16/12/1 |
| hisparse_pools.json | 5/14/8/0 | 5/13/9/0 |
| hisparse_scheduler.json | 2/46/26/1 | 2/44/28/1 |
| model_compat_hyperconnection.json | 7/2/2/0 | 5/4/2/0 |
| model_compat_lifecycle.json | 0/1/1/0 | 0/1/1/0 |
| model_compat_marlin.json | 15/3/1/0 | 8/9/2/0 |
| model_compat_numa.json | 0/7/1/0 | 0/6/2/0 |
| model_compat_quantization.json | 13/3/1/0 | 7/8/2/0 |
| model_compat_qwen4_exp.json | 5/15/1/0 | 4/16/1/0 |
| model_compat_scheduler.json | 0/1/2/0 | 0/1/2/0 |
| qsa_attention.json | 12/39/8/0 | 11/39/9/0 |
| qsa_backend.json | 0/6/0/0 | 0/6/0/0 |
| runtime.json | 17/37/33/2 | 16/34/37/2 |

## QSA / Qwen3.8 upstream changes since the pin

| PR | In v0.5.22 | Relevance |
|---|---|---|
| #41933 fuse QSA prefill KV gathering and dtype conversion | yes | Q08 (`forward_extend`, REPLACE) changed; must re-check the production FP8 descale in prefill reads (A11/A12, `bdb935d70f`) against the fused gather |
| #43463 bound the decode indexer's work by the context length | yes | T03 (`select_decode_tokens`, REPLACE) |
| #37771 HiSparse speculative decoding (multi-step KV swapping) | yes | scheduler/coordinator code under S03/S04/S06 |
| #41446 HiSparse: no host sync in eager backup | yes | coordinator (DSA path) |
| #42175 Qwen3.8-Flash-Next BF16 decode on Blackwell | yes | model/decode path; check against compat rows |
| #44114 keep sparse decode scratch in FP8 for FP8 KV pools | **no** (main) | production uses `fp8_e4m3` KV; touches the decode paths Q10/Q11 replace |
| #39724 Qwen3.8 prefill context parallel | no (main) | not used at TP2 |
| #40917 build KV host pools from buffer info | no (main) | host pools |

## Tracked and new PRs (state on 2026-10-18)

| PR | State | Note |
|---|---|---|
| #35955 GPTQ Marlin MoE BF16/act-order scales | open (upd. 10-16) | U7 |
| #42087 batch-invariant kernels for Qwen3.8 | open (upd. 10-14) | U8 |
| #35488 HiCache as HiSparse logical KV pool | open (upd. 09-22) | U1 |
| #36644 FP8 KV cache in QSA | open | U6 |
| #36968 QSA fail early on SM89 without FA2 | open (upd. 10-13) | U6b |
| #41792 Kimi image identity (draft) | closed; superseded by #43043 (merged 10-18) | I1 contract |
| #38464 load plugins when constructing ServerArgs | open (upd. 10-09) | W7 |
| #35485 M-RoPE extend past mm table | open, stale | I4 |
| #26161 HiSparse unified tree & HiCache | open, stale | none |
| #44444 `SGLANG_REQUIRED_PLUGINS` fail-fast loading | **new, open** | overlaps the launcher's exclusive-plugin check (W7) |
| #41461 defer FP8 KV widening in sparse chunk-prefill | new, open | FP8 prefill reads (A11/A12) |
| #44345 compressed QSA with unified hybrid memory pool (draft) | new | future pool layout (K01–K03) |
| #43527 / #44447 QSA / Qwen3.8 KV cache sharding | new, open | not used at TP2 |
| #43032 avoid padded PLE prefill activations | new, open | model compat |
