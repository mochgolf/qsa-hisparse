# Upstream status (P0-C)

Task P0-C, 2026-10-07. Read-only survey of `sgl-project/sglang` relative to the
pin `76e06febab`. Nothing was pushed, commented, or committed. Sources: the fork
repository's `upstream` remote (blob:none partial clone), the pristine pin
worktree `../.worktrees/sglang-pin-76e06febab`, and `gh` (GET only). Items marked
**unverified** were not established by reading code or by a command result.

## 1. Upstream `main` snapshot

| Item | Value |
| --- | --- |
| Fetch | `git fetch upstream main` at about 2026-10-07 16:41 UTC; `upstream/main` moved `35f3c96ff4..0b635266d4` |
| `upstream/main` HEAD | `0b635266d4a09f8db2d12bdcb793b085199faa8f`, committed 2026-10-07T23:26:36+08:00, "[diffusion] CI: warm up CI cases at the request's quality level (#42958)" |
| Pin | `76e06febab732d28a61b75a61b7835284568cdfb`, committed 2026-09-16T18:11:40+08:00; ancestor of `upstream/main`: yes |
| Commits since pin | **1201** (`git rev-list --count 76e06febab..upstream/main`; first-parent count is also 1201, i.e. linear squash history) |
| Fork reference | `ee8fe158d6`; merge-base with the pin is the pin itself |

Trial merge of the fork onto current main (`git merge-tree --write-tree
--merge-base=76e06febab ee8fe158d6 upstream/main`; writes objects only, no refs
or checkouts) reports textual conflicts in 11 files: 8 source files
(`qwen_sparse_attn_backend.py`, `fused_marlin_moe.py`,
`scheduler_components/batch_result_processor.py`, `mem_cache/common.py`,
`mem_cache/qsa_kv_pool.py`, `forward_batch_info.py`, `pool_configurator.py`,
`models/qwen4_exp.py`), one test
(`test/registered/unit/managers/test_batch_result_processor_hidden_states.py`),
and `README.md`/`python/pyproject.toml`. All other fork files auto-merged
textually (not semantically checked; several call renamed APIs, see section 3).
`python/sglang/srt/layers/hc_mix_triton.py` was renamed upstream to
`python/sglang/kernels/ops/gemm/hc_mix.py` (100% similarity, #41243); the trial
merge carried the fork's edits to the new path.

## 2. Tracked PRs

| PR | Title | State | Merge commit, merged at (UTC) | In pin `76e06febab` | In `main` `0b635266d4` |
| --- | --- | --- | --- | --- | --- |
| [#26161](https://github.com/sgl-project/sglang/pull/26161) | [HiSparse] Support Unified tree & HiCache | OPEN, not draft; labels `hicache`, `hisparse`; review required; REST `mergeable=false`, `mergeable_state=dirty`; last head commit 2026-06-14, last update 2026-06-29; latest base and extra CI runs failed (per PR body) | - | no | no |
| [#34398](https://github.com/sgl-project/sglang/pull/34398) | [VLM] Add content-addressed preprocessing cache infrastructure | MERGED | `e8c7dddfa0`, 2026-08-13 06:09 | yes | yes |
| [#22038](https://github.com/sgl-project/sglang/pull/22038) | [VLM] Chunk-aware ViT encoding with per-image cache and lazy device transfer | MERGED | `34d5765e2f`, 2026-04-04 08:55 | yes | yes |
| [#38855](https://github.com/sgl-project/sglang/pull/38855) | fix(qsa): dequantize FP8 cached prefixes in the sparse prefill kernels | MERGED | `55b45cb45a`, 2026-09-10 21:07 | yes | yes |
| [#35485](https://github.com/sgl-project/sglang/pull/35485) | Fix M-RoPE positions when an extend reaches past the precomputed mm table | OPEN, not draft; no labels; review required; last update 2026-08-19; fixes issue #35482 (OPEN) | - | no | no |
| [#39862](https://github.com/sgl-project/sglang/pull/39862) | [Fix] HiCache: carry registered Mamba slot side states (Qwen4-Exp PLE) through the host tier | MERGED | `5606fb8592`, 2026-10-04 02:15 | **no** | yes |

Containment was checked with `git merge-base --is-ancestor <merge commit> <ref>`.

### #26161 HiSparse unified tree / HiCache

- (a) Plugin: no overlap with the QSA runtime. It is DSA-only: `init_hisparse_radix_cache` requires `compress_ratio == 1`, a DSA `index_k_with_scale_buffer` sidecar and `DSAIndexerPoolHost`; DeepSeek V4 falls back to the old path. It edits `scheduler.py`/`hisparse_coordinator.py`, which the plugin patches, and is stale: a patch dry-run against `main` fails (scheduler.py 6/6 hunks, hisparse_coordinator.py 5/12, registry.py 1/1), and it implements the removed `cache_finished_req`/`cache_unfinished_req`/`req.prefix_indices` API.
- (b) Image host-prefix reuse: not usable as-is. Host prefixes are matched on `RadixKey(token_ids, extra_key)`, page-aligned; image identity is only whatever the pad tokens encode (30-bit pad value). Load-back restores full KV plus DSA indexer pages only; no recurrent/Mamba/PLE state, no M-RoPE, no image-boundary handling.
- **QSA or multimodal support: none demonstrated.** No QSA code, no multimodal code, no tests added (6 files, none under `test/`); the only validation in the PR body is DeepSeek-V3.2-Exp TP8 GSM8K with an L3 (file backend) round trip.

### #34398 multimodal preprocess identity

- (a) Plugin: in the pin; adds `srt/multimodal/cache/{identity,preprocess_cache}.py` (`build_artifact_key`, `build_processor_fingerprint`, `parse_content_hash`, `resolve_multimodal_item_hash(namespace=...)`). The plugin does not patch it; 0 upstream commits to `multimodal/cache`, `multimodal/media_artifacts` or `processors/qwen_vl.py` since the pin.
- (b) Image host-prefix reuse: supplies the full artifact key (SHA-256 over content digest, modality, processor fingerprint, preprocess kwargs) that I1 needs, but at the pin the Qwen-VL artifact path is taken for `qwen4_exp` only when `--mm-preprocess-cache-size-mb > 0` and the request is image-only (`qwen4_exp` is not in `uses_media_artifacts_without_cache`); otherwise items are hashed from features. Only a 64-bit truncation reaches `MultimodalDataItem.hash` and the pad value is `1_000_000 + hash % 2**30`; the artifact key itself is not carried to the scheduler (pin and main).

### #22038 per-image ViT cache

- (a) Plugin: in the pin; no plugin patch target. Embedding cache granularity is per item (`embedding_cache.get_single(item.hash)`, now in `managers/mm_schedule.py`), ViT encoding is chunk-aware, device transfer is lazy.
- (b) Image host-prefix reuse: the basis for I3 inside-image boundaries (a suffix chunk that starts inside an image reuses or recomputes that image's embedding and slices by `extend_prefix_len`). The key is the 64-bit `item.hash`, not the full artifact key; LRU eviction can force ViT recompute on a warm prefix hit (performance only, if identity is correct).

### #38855 QSA FP8 prefix read

- (a) Plugin: in the pin. Adds a plain cast of gathered K/V to the Q dtype in `_sparse_gqa_chunk_prefill` (comment: the QSA backend writes the pool without per-tensor k/v scales). The fork's U6 descale work extends this path; the cast must not be re-fixed. Its test lives at `test/registered/kernel/qsa/test_qsa.py` at the pin, now `test/registered/kernels/ops/attention/qsa/test_qsa.py` on main.
- (b) Image host-prefix reuse: required for warm prefix reads only when the KV cache is FP8; no image-specific behavior.

### #35485 M-RoPE out-of-bounds continuation

- (a) Plugin: not in pin or main and does **not apply** to either (patch dry-run fails 1/1 hunk): #39144 (merged 2026-09-12, in the pin) already inserted a session-only tail branch at that site. The plugin cannot depend on it.
- (b) Image host-prefix reuse: a fresh request that hits a host prefix extends `[prefix_len, seq_len)` inside the prompt's precomputed table, so this bug does not trigger there. It matters only for extends past the prompt table outside sessions (for example a retracted request re-prefilled with output tokens), where the pin's `numel() == 0` fallback still returns a single column for a multi-token extend. Whether plugin flows reach that path is **unverified**; I4 should include a counterexample.

### #39862 HiCache carries PLE state

- (a) Plugin: merged after the pin, so it does not affect the plugin at the pin. It changes upstream HiCache (`MambaPoolHost`), which the fork does not use (the fork requires a chunk cache and rejects hierarchical cache). Relevant at the next pin cycle and as the upstream route for carrying PLE state.
- (b) Image host-prefix reuse: precedent for "complete recurrent/PLE state" on a host tier; the plugin's own snapshots must still capture PLE state at the pin.
- **What changes in PLE state handling.** Before: device `MambaPool` slot siblings registered via `register_slot_state` (the PLE `NGramPool`, `ShortConvPool`) were copied by device slot operations, but `MambaPoolHost` backed up/restored only conv and temporal state, so a host round trip left the previous occupant's side-state rows. After: `MambaPoolHost` collects `iter_transfer_state_entries()` of every registered sibling, allocates one host buffer per side-state tensor (own dtype, own slot axis), copies them on backup, and restores them on the first layer's load (`layer_id == 0`, blocking `index_copy_`, before that layer's completion event). Side-state row bytes are added once (not per layer) to `size_per_token`; L3 page serialization includes them and rejects pages written without them; zero-copy storage is rejected (`NotImplementedError`, `is_stride_page_aligned() == False`; open follow-up #42490). `HybridReqToTokenPool.get_ngram_context` now waits for the first Mamba layer's transfer when layer-wise loading is active. Test: `test/registered/unit/mem_cache/test_hicache_mamba_slot_side_states.py`.
- Related, not in the tracked list: #39893 (merged 2026-10-02, not in pin) preserves the QSA indexer state through HiCache (`mem_cache/pool_host/qsa.py`, `_wait_for_layer` in `get_qsa_compressed_k_buffer`). With #39862, upstream radix + HiCache now carries QSA indexer and PLE side state for text; neither adds multimodal identity.

## 3. Upstream changes since the pin per planned PR area

Counts are `git rev-list --count 76e06febab..upstream/main -- <paths>` (paths
under `python/sglang/srt/` unless noted).

| File / area | Commits | | File / area | Commits |
| --- | ---: | --- | --- | ---: |
| `managers/scheduler.py` | 60 | | `model_executor/model_runner.py` | 29 |
| `managers/schedule_policy.py` | 10 | | `runner/decode_cuda_graph_runner.py` | 17 |
| `managers/scheduler_components/*` | 48 | | `model_executor/forward_batch_info.py` | 35 |
| `.../batch_result_processor.py` | 7 | | `layers/attention/qwen_sparse_attn_backend.py` | 7 |
| `.../weight_updater.py` | 3 | | `layers/attention/qsa/*` | 9 |
| `mem_cache/allocation.py` | 5 | | `layers/quantization/auto_round.py` | 0 |
| `mem_cache/common.py` | 21 | | `layers/quantization/gptq/*` | 0 |
| `mem_cache/memory_pool.py` | 27 | | `hardware_backend/gpu/quantization/gptq_kernels.py` | 0 |
| `mem_cache/base_prefix_cache.py` | 18 | | `layers/moe/fused_moe_triton/*` | 11 |
| `mem_cache/chunk_cache.py` | 7 | | `kernels/jit/csrc/gemm/marlin_moe/*` (`python/sglang/`) | 1 |
| `mem_cache/allocator/paged.py` | 0 | | `kernels/ops/moe/moe_wna16_marlin.py` (`python/sglang/`) | 0 |
| `mem_cache/allocator/*` | 16 | | `layers/hyperconnection.py` | 3 |
| `model_executor/pool_configurator.py` | 18 | | `layers/hc_mix_triton.py` (moved) | 1 |
| `mem_cache/kv_cache_configurator.py` | 27 | | `models/qwen4_exp.py` | 16 |
| `mem_cache/qsa_kv_pool.py` | 3 | | `plugins/*` | **0** |
| `mem_cache/registry.py` | 6 | | `managers/hisparse_coordinator.py` | 3 |

Union per planned PR: U1 100, U2 37, U3 64, U4 40, U5 58, U6 11, U7 24, U8 21, U9 83.

### U1 scheduler gates on coordinator presence + coordinator protocol

- Not upstream. Main still gates on `self.enable_hisparse` / `get_memory().enable_hisparse` (scheduler.py and `batch_result_processor.py` at all fork-touched sites); the abort path still iterates only `collect_inflight_reqs()`.
- Refactors: #41950 moved internal-state readback into a collaborator (`scheduler_components/internal_state.py`); the parallel-context series (#40068, #40638, #41811, #42348) touches most scheduler files; #41520/#42270 renamed the release/checkpoint calls inside `batch_result_processor.py` (trial-merge conflict).
- Upstream rule: `model_runner.py` is a "frozen", orchestration-only file (`.agents/skills/large-class-style`); coordinator construction must be a `maybe_init_*` delegation, not inline logic.
- Open overlap: **#35488** "[HiSparse] HiCache as the plugin logical KV pool" (OPEN, review required, updated 2026-09-22) adds `managers/hisparse_protocol.py` with a `HiSparseCoordinator` Protocol (`backing`, `num_real_reqs`, `indexer_page_table`, `translate_page_table`, `swap_in_selected_pages`, `prepare_decode_batch`, `on_prefill_complete`, `on_prefill_finished_early`, `admit_pending`, `collect_ready_reqs`, `has_ongoing_staging`, `request_finished`, `retract_req`, `admit_budget`, `wait_for_pending_backup`, `get_token_stats`, `set_decode_producer_stream`, `set_tree_cache`, `destroy`). It still gates on `enable_hisparse`; its `batch_result_processor.py` call site replaces `admit_request_into_staging(req)` with `on_prefill_complete(req)` (the private-host backing keeps `admit_request_into_staging` internally). Also #32314 (HiSparse V2, same author), the HiSparse series #41780/#41781/#42168, #37771, #34345, and #42824 (scheduler `prefix_len`).

### U2 `BasePrefixCache` lifecycle protocol

- Large refactor of exactly this interface since the pin (KV-cache owners): #40988 (09-24) drops `is_insert`, `release_kv_cache` frees rows, adds `claim_kv_row` and `on_release`; #41281 (09-27) `cache_finished_req` -> `insert_req`; #41520 (10-01) `cache_unfinished_req` -> `checkpoint_req` (helper now `checkpoint_kv_cache`); #42202 (10-02) `insert_req` -> `checkpoint(req, *, up_to)`; #42194 (10-02) no release-time insert on optimistic requeue; #42362 (10-03) `is_chunk_cache`/`is_tree_cache` -> `supports_prefix_sharing()`; #42270 (10-05) `release_kv_cache(req, tree_cache, *, checkpoint)`; #42537 (10-05) `maybe_hand_to_session(req)` at the end of `alloc_for_extend`; #42686 (10-06) `TreeLock`, `lock()`/`unlock()`; #42467 (10-06) keeps `prefix_indices` current for chunked requests outside the tree.
- Already upstream (partial): `on_release(req, *, checkpointed)` runs after `free_kv_row`, `unlock` and over-allocation release, but **before** `req_to_token_pool.free(req)` and `mark_kv_released()`; `claim_kv_row` runs first; `maybe_hand_to_session` sits where the fork calls `restore_prefix_for_extend`. Not upstream: a pre-checkpoint hook (fork `before_release`), a hook after the row is freed (fork `after_release(lease)`), `prepare_prefix_for_extend` (before `prefix_indices` is read), `note_extend_allocation`, `rollback_prefix_for_extend`, `pending_prefix_tokens`/`prefill_checkpoint_limit` in `PrefillAdder`, `invalidate_model` on weight updates.
- **#42354 (10-03)**: hybrid-SSM models served with `--disable-radix-cache` now get `UnifiedRadixCache` (disabled mode) instead of `ChunkCache`, and `create_tree_cache` rejects any cache (including `--radix-cache-backend` registered ones) whose `supports_mamba()` is false for hybrid-SSM models. `Qwen4ExpTextConfig` subclasses `Qwen3NextConfig`, so `hybrid_gdn_config` makes Qwen4-Exp hybrid-SSM (code reading). The fork's `type(self.tree_cache) is not ChunkCache` guard, which precedes wrapping the tree cache in `QSAHostPrefixCache`, therefore rejects the default configuration at the next pin.
- Open, active (opened 2026-10-06/07 by the scheduler/KV-cache oncall): #42825 derives prefix KV indices from `last_node` at allocation and **drops `Req.prefix_indices`** (the field the fork's `alloc_for_extend` hooks read); #42923 makes `match_prefix` return only the matched length; #42823 folds match write-back into `match_kv_cache`; #42824 tracks prefill progress as `prefix_len`. Others: #41593, #38140, #35303 (extension points for SWA state capture/restore), #31057 (pluggable fuzzy radix backend), #40595 (out-of-tree unified-cache linkers).
- Existing interface not named in PLAN.md: `mem_cache/registry.py` `register_radix_cache_backend(name, factory)` + `--radix-cache-backend` (present at the pin). It could construct the host-prefix cache without patching `Scheduler.__init__`, but the factory runs from `kv_cache_builder` before `init_hisparse_coordinator` and needs a user flag; feasibility **unverified**.

### U3 allocator free-group/release callbacks + KV pool runtime attachment

- Not upstream. `allocator/paged.py` is unchanged (0 commits); there is no `free_group_end` callback. `allocator/base.py` gained `prealloc_fits*` and `set_hicache_transfer_done_event` only. Release order in `release_kv_cache` changed as described in U2 (trial-merge conflict in `common.py`). The unified-memory series (#38592, #40326-#40331, #39982) reworks pools and allocators for unified memory only.
- Open overlap: #28494 (allocators stop round-tripping through the KV pool; keeps `get_kvcache()`), #38064 (PoC: split hybrid SWA allocator and rename `*TokenToKVPoolAllocator` to `*KVAllocator`), #35594 (thread vendor KV pool classes through builders via `_resolve_kv_pool_class(kind=...)`; platform-scoped), #42520, #35401.

### U4 pool fixed reservation + `full_kv_pool` factory

- Not upstream for the default path. `DefaultPoolConfigurator` remains "coeff = cell_size, bias = 0"; fixed-byte bias exists only in `DSV4PoolConfigurator` (precedent). `HybridLinearKVPool` already accepts `full_kv_pool` at the pin (memory_pool.py:3805); only the `QSATokenToKVPool` pass-through is missing.
- Refactors: #39614 (09-29) adds `qsa_indexer_dtype` to `QSATokenToKVPool.__init__` next to the fork's new parameter and `compressed_dtype` to `qsa_bytes_per_token` (QSA cell size changes with FP8 indexer); #39893 adds a layer wait to `get_qsa_compressed_k_buffer`; DSV4 configurator refactors (#41048, #41049, #41090, #41091); #40743 MiniMax HiSparse full-pool accounting; #41325 SWA/spec padding extensibility; #40227 auxiliary cache accounting. Trial-merge conflicts: `pool_configurator.py`, `qsa_kv_pool.py`.
- Open: #38644 (declarative pricing table for hybrid SWA pools), #35594, #31967, #42957.

### U5 decode CUDA graph backend lifecycle hooks

- Not upstream as a coordinator/backend protocol. #40222 (09-18) added subclass seams `_next_token_logits_buffer_capacity_rows` and `_process_output_after_replay` (eager tail inside the replay timer). The pin already has the attention-backend `on_after_cuda_graph_warmup` hook the fork wraps. `num_real_reqs.fill_` is still unconditional at capture and replay.
- Partially upstream: #39695 (09-19) added `ForwardBatch.req_pool_indices_cpu`, **populated only for non-speculative extend** (None for decode); the fork adds the same field for all modes (trial-merge conflict in `forward_batch_info.py`). `kv_allocated_lens_cpu` is not upstream.
- Other changes: #40328 (unified-memory write-loc checks), #41817/#41813 (runner TP groups), #42239 (DiffusionGemma graphs). `model_runner.py` is frozen (see U1).
- Open: #28763 (rename `init_cuda_graph_state` -> `init_static_metadata_buffers` on attention backends), #41584, #42292, #41791 (draft, Foundry graph persistence plugin).

### U6 QSA backend extension points + FP8 descale / SM89 fallback

- Not upstream: main's `qwen_sparse_attn_backend.py` has no k/v descale and no device-capability (SM86/SM89) ragged FA2 fallback.
- Refactors (+479 lines in the backend, trial-merge conflict): #39721 context parallelism for QSA and the indexer; #40972 fused QSA KV preparation (new `qsa/fused_kv.py`); #41173 fused graph replay metadata; #41729 breakable prefill CUDA graphs for text-only Qwen3.8 (capture-safe metadata); #41175; #39614 FP8 compressed indexer cache; #39893; AMD paths #38875, #38876, #41513.
- Open overlap: #37798 (draft; quantized QSA KV pools fp8/int8/int4 with dequant-on-gather; also edits `allocator/paged.py`), #41933 (fuse QSA prefill KV gathering and dtype conversion: the #38855 code path), #42181 (scale fp8 KV writes once), #39575 (unaligned QSA extend prefixes from chunk-cache tails; relevant to host-prefix extends), #41461, #37275, #36787, #40143, #38180, #41932.

### U7 GPTQ MoE scale sizing/dtype, AutoRound g64 split, INT8-row PLE

- **GPTQ MoE scales already proposed upstream: #35955** (OPEN since 2026-08-22, review required, last update 2026-09-14) makes the same two `gptq_moe.py` changes as the fork (non-act-order `w2_scales_size = intermediate_size_per_partition`; scale dtype `params_dtype`), adds act-order full-w2 loading in `FusedMoE`, and a CPU test `test/registered/unit/layers/quantization/test_gptq_marlin_moe.py`. It does not touch `gptq_kernels.py` (the fork's `size_k` change for `marlin_moe_permute_scales`).
- AutoRound g128 -> g64 split: not upstream; open PRs touching `auto_round.py` are #41877 (grouped names) and #42119 (AMD GPTQ/AutoRound INT4); neither title describes the split (content of #42119 **unverified**).
- INT8-row PLE: not upstream (main PLE dtypes are bf16/fp8). The fork's "build the offloaded PLE table on the meta device" hunk is **already upstream** as #39928 (merged 2026-09-20, main only). Open PLE PRs: #40235, #40947, #40626, #42400, #36567, #41624, #38619.
- `qwen4_exp.py` refactors: see U8 (trial-merge conflict).

### U8 deterministic Marlin whole-K, QSA stable top-k, stable HC

- **QSA stable top-k already proposed upstream: #42087** "Fix batch-invariant kernels for Qwen3.8 Flash-Next" (OPEN, `mergeable=true`, blocked on review, updated 2026-10-03): under batch-invariant mode it uses FlashInfer deterministic ragged top-k with smaller-index tie breaking and canonical ascending index order (prefill and the decode shortcut), one QSA prefill launch configuration, a batch-invariant `aten::mm.out` for HC projections, and a shared-expert gate warp cap; tests `kernels/ops/attention/qsa/test_qsa_determinism.py` and `unit/batch_invariant_ops/test_qwen_batch_invariance.py`. Same tie and ordering contract as the fork's `_qsa_stable_topk`; it depends on FlashInfer `TopKTieBreak`/`top_k_ragged_transform` (main bumped FlashInfer to 0.7.0.post1 in #40709; availability at the pin and on SM89 **unverified**).
- Marlin: upstream touched `moe_wna16_marlin.cuh` and `fused_marlin_moe.py` once (#41251, SM90 configs and an SM90 swiglu fast path; trial-merge conflict in `fused_marlin_moe.py`). No whole-K deterministic MoE path. #39456 (OPEN) forces the dense Marlin atomicAdd reduction off under deterministic inference; it does not cover MoE.
- HC: `hc_mix_triton.py` moved to `kernels/ops/gemm/hc_mix.py` (#41243). `qwen4_exp.py` decoder layers were rebuilt on stage boundaries (#41549, #41550, #41552, #41553, #41555, #42301, #42478, #42481: `_attn_mix`, `attn_boundary.prepare`, `append_stages`), so the fork's `mix(..., stable=...)` call sites no longer exist in that form. Open: #40552 (HyperConnection performance; edits `hyperconnection.py` and adds hc kernels), #41880 (AMD hc_mix).
- Other open determinism work: #42145, #26374, #38144, #41893.

### U9 monotonic `req_generation`, HiSparse decode batch multimodal inputs

- Not upstream: `ReqToTokenPool.clear()` still calls `self.req_generation.zero_()`; `_build_hisparse_decode_batch` still does not set `batch.multimodal_inputs`.
- Open overlap: #36093 (OPEN, 2026-08-23) gates `req_generation` increments behind `enable_req_generation_tracking()`, which interacts with a monotonic-generation contract.

### Plugin loader (`sglang/srt/plugins`)

- **No change since the pin**: 0 commits; `hook_registry.py` and `__init__.py` are byte-identical on main. `load_plugins()` call sites are unchanged (`launch_server.py`, `cli/serve.py`, `Engine.__init__`, `Engine._launch_subprocesses`, `run_scheduler_process`). The loader still logs and continues on exceptions from `ep.load()` and from the plugin function. Diffusion got a separate hook (`multimodal_gen.runtime.platforms.plugins.apply_plugin_hooks`, #37547), not the srt registry.
- Open: #38464 (load plugins in `ServerArgs.__post_init__`, `run_detokenizer_process` and multi-tokenizer workers). At the pin, tokenizer-worker and detokenizer subprocesses do not load plugins; relevant if Track I patches code that runs there (whether it does is **unverified**). #41791 (draft) adds a Foundry-specific "required plugin missing" error, not a general fail-fast.

### Test path moves since the pin

`test/registered/kernel/qsa/*` -> `test/registered/kernels/ops/attention/qsa/*`
(#39966, #41243); `kernel/hyperconnection/test_hc_mix_triton.py` ->
`kernels/ops/gemm/test_hc_mix.py`; `quant/test_marlin_moe.py` ->
`kernels/ops/moe/test_marlin_moe.py`. PLAN.md's "registered `kernel/qsa/test_qsa.py`"
is the pin path only.

## 4. Contribution requirements (sgl-project/sglang)

Sources on main: `.github/pull_request_template.md`,
`docs/docs/developer_guide/contribution_guide.mdx`, `.github/MAINTAINER.md`,
`.github/CODEOWNERS`, `.github/labeler.yml`, `test/README.md`,
`test/registered/README.md`, `test/registered/unit/README.md`,
`python/sglang/test/ci/ci_register.py`, `.pre-commit-config.yaml`, `.claude/rules/*`,
`.agents/skills/large-class-style`.

- [ ] Branch from fresh `upstream/main` in a personal fork; never commit to `main`.
- [ ] `pre-commit run --all-files` passes (isort, ruff, ruff-format, codespell, clang-format, check-ast, local hooks including `check-chinese-characters`, `check-registered-tests`, `check-no-registered-tests-in-package`, `check-no-bare-pytest-main`). Lint failure stops all CI.
- [ ] PR body follows the template: Motivation, Modifications, Accuracy Tests (if outputs can change), Speed Tests and Profiling (if speed can change), Checklist, Review and Merge Process.
- [ ] Tests: CPU component tests in `test/registered/unit/<module>/` mirroring `python/sglang/srt/`; GPU kernel correctness in `test/registered/kernels/ops/<group>/`; E2E under topic directories. Prefer extending an existing file in the same subsystem.
- [ ] Each registered file: `register_cpu_ci(est_time=..., suite="base-a-test-cpu")` (or `register_cuda_ci(..., stage=..., runner_config=...)`) at module level with literal arguments; `CustomTestCase`; a standard `if __name__ == "__main__": unittest.main()` (or `sys.exit(pytest.main([__file__]))`) with no argv changes; under 500 s per file. Default PR registrations are capped at 1,200 weighted accelerator-seconds per backend; `disabled=` needs an issue and `until YYYY-MM-DD`.
- [ ] Unit-test admission: every case is a bug regression (fails before the fix), a derived property, or critical-path bookkeeping; no happy-path tautologies or mock-only assertions. Coverage guidance: at least 60% of changed lines (`diff-cover`).
- [ ] Code style (`.claude/rules`): no defensive `getattr`/`hasattr` (the fork's optional-hook probing style must become explicit base-class methods); new data containers use `msgspec.Struct`, not `@dataclass`; keyword arguments for 2+ args; functions under about 100 lines, files under about 2k lines; avoid mixins; minimal CPU-GPU sync; ASCII-only, short comments; `model_runner.py` changes stay orchestration-only (`maybe_init_*`). New `SGLANG_*` env vars follow `env-var-conventions`.
- [ ] Accuracy: `sgl-eval run gsm8k` (or a harder task) for output-affecting changes; benchmarks per the benchmark guide for speed claims.
- [ ] Review: a bot assigns a Merge Oncall; one CODEOWNER approval per modified protected file. Relevant owners: `managers` (merrymercy, Ying1123, hnyls2002, xiezhq-hermann), `managers/hisparse_coordinator.py` (+ hzh0425, ispobock, alphabetc1, huangtingwei9988), `mem_cache` and `mem_cache/allocator` (hnyls2002, ispobock, alphabetc1, ...), `model_executor` (merrymercy, Ying1123, hnyls2002, Fridge003, ispobock), `layers/attention` (merrymercy, Fridge003, ispobock, Qiaolin-Yu, ...), `layers/quantization` (ch-wan, BBuf, ...), `python/sglang/kernels` (DarkSharpness, BBuf, ...), `srt/multimodal` (mickqian, JustinTong0323, yhyang201, ...). Merge Oncalls: Scheduler (merrymercy, hnyls2002, cctry), KV Cache (ispobock, xiezhq-hermann), Kernel (BBuf), models/attention (Fridge003, ishandhanani, Qiaolin-Yu).
- [ ] CI: runs only with the `run-ci` label (`run-ci-extra` for the extra tier), added by users in `.github/CI_PERMISSIONS.json` via `/tag-run-ci-label`, `/tag-and-rerun-ci [extra]`, `/run-full-ci`, `/run-extra-ci`; PR authors may use `/rerun-failed-ci` on their own PR. Control labels `bypass-fail-fast`, `parallel-stages`, `max-concurrency`, `highest-priority` cost extra capacity. Selective `/rerun-test` does not build PR-local AOT kernels.
- [ ] Auto labels by path (`labeler.yml`): `memory-pool` (`mem_cache/allocator/**`, `*memory_pool*`), `unified-radix-cache` (`*radix*`, `chunk_cache.py`, `base_prefix_cache.py`), `hicache` (`*hicache*`, `pool_host/**`), `quant` (`*quant*`), `Multi-modal` (`*multimodal*`, `*vision*`), `jit-kernel` (`python/sglang/kernels/` except `aot`), `deterministic` (`batch_invariant_ops/**`).
- [ ] Kernels: lightweight kernels use the in-tree JIT path; AOT `sglang-kernel` changes must be released and pinned before callers that require them land.
- [ ] Project rule (GOAL.md): push and PR creation wait for explicit owner confirmation for each PR.

## 5. Findings that affect the plan

1. **U2 targets a moving interface.** Since the pin the prefix-cache lifecycle was renamed and reworked (`checkpoint`, `on_release`, `claim_kv_row`, `maybe_hand_to_session`, `TreeLock`, `supports_prefix_sharing`, `release_kv_cache(checkpoint=)`), and four open PRs from the scheduler/KV-cache oncall (#42823, #42824, #42825, #42923) are still changing it, including dropping `Req.prefix_indices`. Some hook points already exist; U2 should be written against main after those land or coordinated with that owner.
2. **#42354 removes `ChunkCache` for Qwen4-Exp on main.** Hybrid-SSM models with `--disable-radix-cache` now get `UnifiedRadixCache` (disabled mode), and caches without `supports_mamba()` are rejected. The plugin's chunk-cache wrapper and its guard will not work at the next pin; this does not affect the current pin.
3. **Parts of U7 and U8 are already proposed or merged upstream.** GPTQ MoE scale sizing and dtype: open #35955. QSA stable top-k (same tie and ordering contract): open #42087. Meta-device PLE table: merged #39928 (main only). These items should be coordinated with or rebased on those PRs, not duplicated.
4. **U1 overlaps open #35488**, which defines a `HiSparseCoordinator` Protocol with different method names (`on_prefill_complete`, etc.).
5. **U5 overlaps a partial upstream change.** `ForwardBatch.req_pool_indices_cpu` exists on main, but only for extend. U5 should widen it to decode instead of adding a second field.
6. **Track I identity.** Open draft #41792 adds a generic `MultimodalDataItem.identity` (full 32-byte digest) with wide 62-bit pad sentinels, but wires it only for Kimi. At the pin the Qwen-VL artifact key is used only when the preprocess cache is enabled, and it never reaches the scheduler. I1 must carry it and could align with the #41792 interface.
7. **#26161 is not a basis for the plugin.** It is DSA-only, has no QSA or multimodal support, and has conflicts with main.
8. **The plugin loader is unchanged.** No upstream fix for exception swallowing exists. Fail-closed activation stays plugin-side.
9. **Not in PLAN.md: the radix-cache backend registry.** `register_radix_cache_backend` + `--radix-cache-backend` already exist at the pin and could replace a scheduler patch for host-prefix cache construction. Not evaluated.

## Unverified

- Mergeability of open PRs for which GitHub returned `mergeable=null` (#35485, #35488, #35955, #38464); only #35485 and #26161 were dry-run against code.
- Semantic correctness of files the trial merge auto-merged.
- FlashInfer deterministic top-k availability at the pin's FlashInfer version and on SM89 (#42087).
- Whether #42119 includes an AutoRound group-size split.
- Whether plugin-patched or Track I code runs in tokenizer-worker/detokenizer subprocesses that do not load plugins at the pin.
- Feasibility of hosting the host-prefix cache through `register_radix_cache_backend`.
- Whether plugin request flows reach M-RoPE extends past the prompt table (retraction re-prefill).
