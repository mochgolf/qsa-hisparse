# Runtime package (W1)

Inventory rows N01–N04. The HiSparse runtime lives in
`src/sglang_qsa_hisparse/hisparse/` as verbatim copies of the reference with
the PLAN.md import rewrites. Since Phase 5 the reference is production
`897286b12a` (fork `ee8fe158d6` merged with upstream `35f3c96ff4`, the pin),
whose runtime differs from the fork only by its own pin adaptations (below),
so no pin edits remain. This file lists the proof, the ported tests, and
every pinned SGLang definition (`35f3c96ff4`) the runtime relies on.

## Moved modules

| Plugin module | Reference file | Row |
| --- | --- | --- |
| `__init__`, `config`, `coordinator`, `layout`, `runtime`, `single_request`, `slots` | `srt/mem_cache/qsa_hisparse/<same>.py` | N01 |
| `prefix`, `prefix_cache` (behavior: W6) | `srt/mem_cache/qsa_hisparse/<same>.py` | N02 |
| `graph` | `srt/layers/attention/qsa/hisparse_graph.py` | N03 |
| none | `srt/mem_cache/qsa_hisparse_{p2,slots,v3}.py` (re-export aliases) | N04, dropped |
| `depends` | none (plugin-owned, `RUNTIME_DEPENDS`) | |

`tests/runtime/test_moved_sources.py` reads the reference with
`git -C $QSA_FORK_ROOT show 897286b12a:<path>` (default root
`../qsa-hisparse`, the fork repository, which holds production's commit) and
asserts:

- every moved module equals its reference file byte for byte after replacing
  `sglang.srt.mem_cache.qsa_hisparse` → `sglang_qsa_hisparse.hisparse` and
  `sglang.srt.layers.attention.qsa.hisparse_graph` →
  `sglang_qsa_hisparse.hisparse.graph` and applying `PIN_EDITS` (empty), and
  nothing else (the Track I modules `prefix`, `prefix_cache`, `runtime`: plus
  exactly `track_i.diff`, whose content is unchanged since Phase 3);
- the package holds exactly those modules plus `depends.py` and the Track I
  identity modules, and the reference package holds exactly the N01/N02 files;
- no file under the reference's `python/`, `test/` or `scripts/` imports the
  N04 aliases (`qsa_hisparse_{p2,slots,v3}`, `QSAHiSparseP2`, `QSAHiSparseV3`);
- each ported test differs from its reference file only by the same rewrites
  (which also cover `mock.patch` target strings) plus an added
  `import pytest` and `@pytest.mark.integration(rows=...)` lines.

### Production's adaptations to the pin (Phase 5)

They replace the v0.5.21 pin edits of Phase 4.

| Module | Upstream change | Production's code |
| --- | --- | --- |
| `config.py`, `runtime.py`, `single_request.py` | `ModelRunner` has no `tp_rank` (35f3c96ff4 reads placement from the published parallel context) | `config.parallel_tp_rank()` returns `get_parallel().tp_rank`; both constructors use it (v0.5.21 edit: `runner.tp_rank`) |
| `prefix_cache.py` | `ChunkCache` has no `cache_unfinished_req`/`insert_req`; `release_kv_cache` calls `claim_kv_row`, `checkpoint` (when inserting), frees the row, then `on_release`; `checkpoint_kv_cache` calls `checkpoint` for unfinished requests | `claim_kv_row` refuses while a restore record is open (was `before_release`); `checkpoint` captures, then `ChunkCache.checkpoint` (was `cache_unfinished_req` and `before_release`'s capture); `on_release` drops the pending match, then the base (v0.5.21 edit: `on_release` without the base call). The release order is P5-A's M03 copy of production's `release_kv_cache` (docs/prefix-cache.md, site 7) |

## Ported tests

`tests/runtime/` holds production's `test/qsa_hisparse/{test_runtime,
test_slots,test_single_request,test_gather,test_parallel_rank_migration}.py`
(the last is production-added: the runtime takes its rank from the published
parallel bundle, never a stale runner attribute). Production changed
`test_runtime.py` and `test_gather.py` to build `QwenSparseAttnBackend()`
instead of `__new__` and added descale assertions to `test_gather.py`
(`bdb935d70f`), and `test_single_request.py` publishes the rank. Tests that
reach changed SGLang behavior are marked `integration` with the rows they need
(row IDs as at Phase 4; P5-C maps production's descale hunks):

| Test | Needs rows | Reference behavior used |
| --- | --- | --- |
| `test_runtime.py::test_extend_uses_hisparse_writer` | Q08 | `forward_extend` stores through `_store_kv` |
| `test_runtime.py::test_ragged_fa2_fast_path_is_p2_offload_graph_only` | Q09 | `_can_run_fa2_graph` |
| `test_runtime.py::test_ready_batch_preserves_position_metadata` | S03 | `_build_hisparse_decode_batch` carries `multimodal_inputs` |
| `test_runtime.py::test_batch_isolation_and_real_postflush` | K03, Q04, M02 | `QSATokenToKVPool(full_kv_pool=)`, `_store_kv`, `free_group_end` callback |
| `test_single_request.py::test_stale_release` | M02 | `free_group_end` calls `after_release(pending_release)` |
| `test_slots.py::test_two_requests_staging_reuse_and_release` | K03 | `QSATokenToKVPool(full_kv_pool=)` |
| `test_gather.py::test_gather_preserves_scales_and_padding` (4 cases) | A01, A07, A09 | `qwen_sparse_kv_extraction_compact_triton(k_scale=, v_scale=)` |
| `test_gather.py::test_paged_backend_passes_fp8_scales_to_gather` | Q04, Q10, A01, A07, A09 | `_forward_trtllm_sparse` passes FP8 descales; the paged kernel does not apply them again |

The other ported tests run on the pin. As a cross-check (2026-10-08), every
test of `tests/runtime` and `tests/prefix` except the two source/pin checks
(61, of them 26 `integration`) passes with the plugin's runtime modules and
production's SGLang on `PYTHONPATH` instead of the pin, so the integration
tests depend only on rows production has. Like the reference,
`test_gather.py` imports `qwen_sparse_kv_extraction_compact_triton` at module
import, so integration runs must activate the plugin before collection.

## SGLang definitions the runtime relies on

`hisparse/depends.py` lists these as `RUNTIME_DEPENDS`; their pinned records
are `fingerprints/runtime.json`. Row Q03's runtime-construction hook declares
them as `depends`, so any byte change in a module that defines one fails
`hisparse` activation. `tests/runtime/test_depends.py` checks that every entry
is pinned, matches the pin, and is a function or class. Line numbers are the
pinned definitions (`def`/`class` line). Module paths drop `sglang.srt.`.

### Runner, pools and allocator

| Definition | Relied on | Used in |
| --- | --- | --- |
| `model_executor.model_runner.ModelRunner.__init__` (324) | `server_args`, `model_config` | both constructors, prefix namespace, ledgers |
| `model_executor.model_runner.ModelRunner.alloc_memory_pool` (896) | `token_to_kv_pool`, `token_to_kv_pool_allocator`, `req_to_token_pool`; `_unified_memory_pool` must be None | both constructors |
| `runtime_context.get_parallel` (1261) | `tp_rank` of the published parallel bundle, read by `config.parallel_tp_rank` (v0.5.21: `ModelRunner.tp_rank`; before: `ps.tp_rank`) | ledger file names, events |
| `configs.model_config.ModelConfig.__init__` (457) | `hf_config.to_dict()` | `runtime.prefix_namespace` |
| `mem_cache.qsa_kv_pool.QSATokenToKVPool` (48) | class attribute `index_state_dtype` | prefix checkpoint index dtype |
| `mem_cache.qsa_kv_pool.QSATokenToKVPool.__init__` (75) | `qsa_compress_ratio`, `qsa_token_topk`, `qsa_compressed_flat`, `qsa_compressed_k_buffer_pool`, `qsa_key_state_buffer_pool`, `qsa_rope_position_buffer` (pending ring rows `[4 * req_pool_idx, 4 * req_pool_idx + 4)`) | `validate_configuration`, ledgers, prefix capture/restore |
| `mem_cache.memory_pool.HybridLinearKVPool.__init__` (3967) | `full_kv_pool`, `full_attention_layer_id_mapping`, `full_layer_nums`, `size`, `page_size`, `head_num`, `head_dim`, `dtype`, `device` | constructors, `validate_configuration` |
| `mem_cache.memory_pool.HybridLinearKVPool._transfer_full_attention_id` (4211) | layer id → full-attention index | `after_store`, `selected`, `capture_decode` |
| `mem_cache.memory_pool.KVCache.__init__` (1908) | raw pool `size`, `page_size`, `dtype`, `device`, `layer_num` | P2 raw geometry check |
| `mem_cache.memory_pool.MHATokenToKVPool` (2071) | exact type; `is_quantized_kv_cache` | constructors, `validate_configuration` |
| `mem_cache.memory_pool.MHATokenToKVPool.__init__` (2074) | `head_num`, `head_dim`, `kv_cache_layout`, `use_hnd`, `post_capture_active` | geometry checks |
| `mem_cache.memory_pool.MHATokenToKVPool._create_buffers` (2258) | `k_buffer`/`v_buffer`, shape `(size + page_size, 1, 256)`; called again to restore staging | P2 geometry, single-request `begin_batch` |
| `mem_cache.memory_pool.MHATokenToKVPool._init_data_ptrs_and_strides` (2369) | `k_data_ptrs`/`v_data_ptrs` | single-request handoff and restore |
| `mem_cache.allocator.base.BaseTokenToKVPoolAllocator.free_group_begin` (193), `.free_group_end` (197) | `free_group` is not None exactly inside a free group | `after_release`, `slots.logical_flushed` |
| `mem_cache.allocator.paged.PagedTokenToKVPoolAllocator` (116) | exact type | constructors |
| `mem_cache.allocator.paged.PagedTokenToKVPoolAllocator.available_size` (147) | logical capacity | handoff ownership check, ledgers |
| `mem_cache.allocator.paged.PagedTokenToKVPoolAllocator.free_group_begin` (324), `.free_group_end` (328) | the flush after which leases may be reused (row M02 adds the callback) | `after_logical_flush` |

### Request rows, generations, recurrent and PLE state

| Definition | Relied on | Used in |
| --- | --- | --- |
| `mem_cache.memory_pool.ReqToTokenPool.__init__` (286) | `req_to_token` (row table, width ≥ 262144), `req_generation` | `prefill_slots`, `_request`, `native_lease_snapshot` |
| `mem_cache.memory_pool.ReqToTokenPool.alloc_rows` (335) | each allocation increments `req_generation` | lease identity (`slots.acquire` rejects non-increasing generations) |
| `mem_cache.memory_pool.ReqToTokenPool.clear` (365) | zeroes `req_generation` at the pin; row M04 keeps it monotonic | lease identity after a flush |
| `mem_cache.memory_pool.HybridReqToTokenPool.__init__` (1245) | `enable_mamba_extra_buffer` | `native_lease_snapshot` |
| `mem_cache.memory_pool.HybridReqToTokenPool._init_mamba_pool` (1304) | `mamba_pool`, `mamba_allocator`, `mamba_ckpt_pool`, `req_index_to_mamba_index_mapping`, `req_index_to_mamba_ping_pong_track_buffer_mapping`; registers the PLE siblings | constructors, ledgers, prefix capture/restore |
| `mem_cache.memory_pool.HybridReqToTokenPool.alloc` (1450) | sets `req.kv.mamba_pool_idx` and `mamba_needs_clear = True` | `restore_prefix` clears the flag |
| `mem_cache.memory_pool.MambaPool` (401) | class default `_slot_siblings` | prefix capture/restore/validation |
| `mem_cache.memory_pool.MambaPool.State` (416) | `conv`, `temporal`, `replayssm_{d,k,g}` | prefix capture/restore, ReplaySSM rejection |
| `mem_cache.memory_pool.MambaPool.__init__` (532) | `mamba_cache` layout `[layers, slots, ...]` | prefix capture/restore |
| `mem_cache.memory_pool.MambaPool.register_slot_state` (410) | `_slot_siblings` list | prefix capture/restore |
| `mem_cache.memory_pool.MambaPool.get_contiguous_buf_infos` (1163) | buffer pointers and sizes | ledgers |
| `mem_cache.allocator.mamba.MambaSlotAllocator.available_size` (41) | free recurrent slots | ledgers |
| `mem_cache.ple_state_pool.ShortConvPool` (38) | `conv_state` `[layers, slots, ...]`, `iter_transfer_state_entries` | prefix checkpoint size, capture/restore |
| `mem_cache.ple_state_pool.NGramPool` (142) | `context` `[slots, ...]`, `iter_transfer_state_entries` | same |
| `managers.schedule_batch.ReqKvInfo` (928) | `req_pool_idx`, `mamba_pool_idx`, `mamba_needs_clear`, `mamba_cow_src_index`, `kv_allocated_len`, `kv_committed_len`, `cache_protected_len`, `mark_kv_released` | runtime, coordinator, `prefix_cache` |
| `managers.schedule_batch.Req.__init__` (1008) | `rid`, `kv`, `beam_group`, `hisparse_staging`, and the `prefix_cache` fields below | coordinator, runtime |
| `managers.schedule_batch.ScheduleBatch._collect_deferred_mamba_cow_and_clear` (3146) | consumes `mamba_needs_clear` / `mamba_cow_src_index`; `restore_prefix` resets both so restored state is not cleared or overwritten | `restore_prefix` |

### QSA state addressing copied by prefix capture/restore

| Definition | Relied on |
| --- | --- |
| `layers.attention.qsa.metadata.build_pending_ring_slots` (237) | pending ring row = `req_pool_idx * 4 + position % 4` |
| `layers.attention.qsa.metadata.build_group_ring_slots` (260) | decode compression reads the same ring rows |
| `layers.attention.qsa.graph_metadata._qsa_graph_row_metadata_kernel` (70) | graph decode ring rows, same layout |
| `layers.attention.qwen_sparse_attn_backend.QwenSparseAttnBackend._qsa_write_plan` (491) | compressed slot = first raw slot // 4, read as `req_to_token[idx, start:stop:4] // 4` |

### Batch metadata

| Definition | Relied on |
| --- | --- |
| `model_executor.forward_batch_info.ForwardMode` (210) | `is_idle`, `is_decode`, `is_extend` |
| `model_executor.forward_batch_info.ForwardBatch` (505) | `batch_size`, `req_pool_indices`, `seq_lens`, `seq_lens_cpu`, `rids`, `extend_seq_lens_cpu` (`req_pool_indices_cpu`: an upstream field since v0.5.21, copied from the schedule batch when it has one; row F01 sets it on every batch while the runtime is active) |
| `model_executor.forward_batch_info.ForwardBatch.init_new` (930) | fills those fields |

### Coordinator callers that no plugin patch replaces

| Definition | Relied on |
| --- | --- |
| `managers.schedule_batch.ScheduleBatch.prepare_for_decode` (3585) | calls `map_last_loc_to_buffer(seq_lens, out_cache_loc, req_pool_indices, seq_lens_cpu, req_pool_indices_cpu)` positionally |
| `managers.schedule_batch.release_req` (2264) | calls `retract_req(req)` for unfinished requests (QSA raises) |
| `managers.scheduler.Scheduler.release_host_resources` (1831) | calls `destroy()` |
| `model_executor.model_runner.ModelRunner._prepare_eager_forward_batch` (1629) | `num_real_reqs.fill_(batch_size)` on every eager forward; under p2-offload `num_real_reqs` is the runtime's `real` tensor |
| `model_executor.runner.decode_cuda_graph_runner.DecodeCudaGraphRunner.capture_prepare` (869) | `num_real_reqs.fill_(bs)` during capture |

The other callers (`Scheduler`, `SchedulerBatchResultProcessor`,
`DecodeCudaGraphRunner.load_batch`/`capture_one_shape`) are W2/W3 patch
targets and are pinned by those rows; `ModelRunner._forward_raw` and
`DecodeCudaGraphRunner.execute` are pinned as `depends` of the R02/G03 hooks
(D6, D7).

### CUDA graph configuration and native backend

| Definition | Relied on |
| --- | --- |
| `model_executor.cuda_graph_config.cuda_graph_fully_disabled` (226), `.check_cuda_graph_backend` (212) | startup graph-mode checks |
| `model_executor.cuda_graph_config.CudaGraphConfig` (154), `.PhaseConfig` (93) | `decode.bs == [1..max_requests]`, `decode.max_bs == max_requests` |
| `runtime_context.get_exec` (1294) | `get_exec().graph.cuda_graph_config` |
| `model_executor.runner.flashinfer_autotune.should_run_flashinfer_autotune` (56) | capture refuses autotune dummy forwards |
| `model_executor.runner_backend.full_cuda_graph_backend.FullCudaGraphBackend.__init__` (83), `.capture_session` (106) | `_graphs` keyed by `ShapeKey`, `_pool` |
| `model_executor.runner.shape_key.ShapeKey` (22) | `size`; `vars(key)` in ledgers |

### Resolver kernel and NVTX helpers

| Definition | Relied on |
| --- | --- |
| `sglang.kernels.ops.kvcache.hisparse.load_cache_to_device_buffer_mla` (356) | positional signature `(top_k_tokens, device_buffer_tokens, host_cache_locs, device_buffer_locs, host_cache, device_buffer, top_k_device_locs, req_pool_indices, seq_lens, lru_slots, item_size_bytes, num_top_k, hot_buffer_size, page_size, block_size, num_real_reqs, miss_src, miss_dst, miss_count)` and its hot-buffer/LRU semantics |
| `sglang.kernels.ops.kvcache.hisparse._load_cache_to_device_buffer_mla` (277), `._jit_sparse_module` (170) | JIT launch of `kvcacheio/hisparse.cuh` |
| `utils.nvtx_utils.profile_range` (84), `.profile_method` (100) | `operations_nvtx_range` (a `partial` of `profile_range`), `profile_method` decorators |

### Server arguments

`config.validate_configuration` and the runtime constructor read these
fields; a renamed field would make a `getattr(args, name, False)` rejection
pass silently, so their defining classes are pinned.

| Definition | Fields read |
| --- | --- |
| `server_args.ServerArgs` (223) | assembles the field classes below |
| `arg_groups.fields.schedule.Schedule` (19) | `max_running_requests`, `max_total_tokens`, `chunked_prefill_size`, `disable_overlap_schedule`, `enable_mixed_chunk` |
| `arg_groups.fields.parallel.Parallel` (17) | `tp_size`, `pp_size`, `enable_dp_attention` |
| `arg_groups.fields.memory.Memory` (21) | `disable_radix_cache`, `enable_hisparse` |
| `arg_groups.fields.exec_.ExecGraph` (484) | `cuda_graph_backend_{decode,prefill}`, `cuda_graph_bs_decode`, `cuda_graph_max_bs_decode`, `disable_cuda_graph_padding`, `enable_torch_compile`, `cuda_graph_config` |
| `arg_groups.fields.exec_.ExecDeterministic` (1015) | `enable_deterministic_inference` |
| `arg_groups.fields.exec_.ExecOverlap` (909) | `enable_two_batch_overlap`, `enable_single_batch_overlap` |
| `arg_groups.fields.exec_.ExecMamba` (344) | `enable_linear_replayssm`, `enable_mamba_extra_buffer` |
| `arg_groups.fields.model.Model` (31) | `context_length`, `model_path`, `revision` |
| `arg_groups.fields.serving.Serving` (22) | `skip_server_warmup`, `enable_streaming_session` |
| `arg_groups.fields.device.Device` (16) | `random_seed` |
| `arg_groups.fields.spec.Spec` (23) | `speculative_algorithm` |
| `arg_groups.fields.disagg.Disagg` (21) | `disaggregation_mode` |
| `arg_groups.fields.lora.Lora` (26) | `enable_lora` |

### `prefix_cache.py` (N02 code; behavior owned by W6)

| Definition | Relied on |
| --- | --- |
| `mem_cache.chunk_cache.ChunkCache` (35) | base class; instance `__dict__` copied; `match_prefix`, `checkpoint(req, *, up_to)` (sets `prefix_indices`); `page_size`, `token_to_kv_pool_allocator`, `req_to_token_pool` |
| `mem_cache.base_prefix_cache.BasePrefixCache.claim_kv_row` (519) | first call of `release_kv_cache`; the base keeps no row (returns False) |
| `mem_cache.common.checkpoint_kv_cache` (172) | calls `checkpoint(req, up_to=req.extend_range.end)` for unfinished requests (chunk boundaries), unless `skip_radix_cache_insert` |
| `mem_cache.base_prefix_cache.BasePrefixCache.on_release` (524) | `release_kv_cache` calls it after freeing the row and dropping the lock; the prefix cache drops a finished request's pending match there, then calls the base |
| `mem_cache.base_prefix_cache.MatchPrefixParams` (65), `.MatchResult` (260) | `params.req`, `params.key`; result fields |
| `mem_cache.radix_cache.RadixKey` (59) | iterates token ids, `len`, `is_bigram` |
| `environ.Envs` (257) | `SGLANG_RADIX_FORCE_MISS` |
| `utils.common.Range` (1282) | `req.extend_range.length` |
| `mem_cache.memory_pool.ReqToTokenPool.free` (360), `mem_cache.memory_pool.HybridReqToTokenPool.free_mamba_cache` (1693) | rollback of a failed restore |
| `mem_cache.allocator.paged.PagedTokenToKVPoolAllocator.alloc` (160), `.free` (270) | prefix pages, rollback |
| `Req.__init__` (above) | `cache_request_handle`, `full_untruncated_fill_ids`, `prefix_indices`, `extend_range`, `host_hit_length`, `host_loaded_length`, `skip_radix_cache_insert`, `multimodal_inputs`, `positional_embed_overrides`, `input_embeds`, `session`, `beam_group`, `return_hidden_states`, `extra_key`, `cache_salt`, `lora_id` |

## Findings (not changed; the modules stay verbatim)

1. The resolver's CUDA source (`sglang/kernels/jit/csrc/kvcacheio/hisparse.cuh`)
   is not a Python module, so fingerprints do not cover it; only its Python
   launcher is pinned (the file is identical at v0.5.21, 35f3c96ff4 and
   production). At v0.5.21 it gained block-id top-k (template
   parameters `SPARSE_BLOCK_SIZE`, `TopKIsBlocks`; the launcher passes 1 and
   false for the MLA path the runtime calls, which reproduces the old
   token-index path), and the miss count is now the number of top-k entries
   the miss pass actually finds (`s_total_misses`) instead of
   `NUM_TOP_K - hits - newest_hit`. The two are equal whenever the top-k
   indices are distinct and the hot buffer holds each token at most once:
   every hit marks exactly one top-k entry, the newest token marks at most
   one, and every other entry is a miss. The runtime's indices are distinct
   (`selected` passes one compressed slot per selected C4 group:
   `raw_indices[:, :2048:4] // 4`, from a top-k over distinct slots), so the
   miss plan (`miss_src`/`miss_dst`/`miss_count`) and the LRU writeback are
   unchanged. With repeated indices the old formula miscounted; the new one
   is exact.
2. `runtime.QSAHiSparseRuntime.__init__` rejects
   `server_args.enable_priority_preemption`, which exists neither at the pin
   nor in the fork (`Schedule` has `disable_priority_preemption`), so that
   check never fires. Priority preemption of a QSA request still fails
   closed: `PrefillAdder.preempt_to_schedule` reaches `release_req`, which
   calls `coordinator.retract_req`, which raises.
3. `prefix_cache._namespace` treats `position_ids` and `mrope_positions` on
   `Req` as cache-bypass conditions, but `Req` has neither attribute at the
   pin, so those two terms are inert (multimodal requests are still excluded
   through `multimodal_inputs`).
4. Activation cost: `fingerprint.verify` parses a module once per name, so
   the 89 entries take about 5 s per process on the development host.
5. At 35f3c96ff4 attention data parallelism is `--attn-dp-size`; the
   deprecated `enable_dp_attention` field is resolved into it and is always
   False after resolution (`parallel_hook._handle_deprecated_dp_attention`).
   `config.validate_configuration`'s DP rejection reads only
   `enable_dp_attention`, so it no longer fires for attention DP. Production
   has the same code; whether to add an `attn_dp_size` check is an
   orchestrator/owner decision (it would deviate from production).
6. At 35f3c96ff4 `--qsa-indexer-dtype fp8_e4m3` (CUDA SM90/SM100 only;
   `auto` resolves to bf16) stores the compressed index as FP8, while the
   prefix capture allocates checkpoint index tensors with
   `index_state_dtype` (bf16) and the single-request ledger counts 2 bytes
   per index element. Unreachable on the target SM89 host; production is
   identical; `validate_configuration` does not check the index dtype.
