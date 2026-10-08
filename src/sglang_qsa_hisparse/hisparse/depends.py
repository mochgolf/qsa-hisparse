"""Pinned SGLang definitions that the moved runtime package relies on.

The other modules of this package are verbatim copies of the reference
(production since Phase 5), so the list lives here. Each entry is the
function or class that provides something the runtime reads, calls or
mutates (for an instance attribute, the method creating it);
``docs/runtime.md`` says which. Row Q03's runtime-construction hook declares
``depends=RUNTIME_DEPENDS``, so with ``hisparse`` active any byte change in a
module defining one of these fails activation (``fingerprints/runtime.json``).
"""

RUNTIME_DEPENDS = (
    # Runner, pools and allocator handed to the runtime constructors.
    "sglang.srt.model_executor.model_runner.ModelRunner.__init__",
    "sglang.srt.model_executor.model_runner.ModelRunner.alloc_memory_pool",
    # config.parallel_tp_rank: the TP rank from the published parallel bundle.
    "sglang.srt.runtime_context.get_parallel",
    "sglang.srt.configs.model_config.ModelConfig.__init__",
    "sglang.srt.mem_cache.qsa_kv_pool.QSATokenToKVPool",
    "sglang.srt.mem_cache.qsa_kv_pool.QSATokenToKVPool.__init__",
    "sglang.srt.mem_cache.memory_pool.HybridLinearKVPool.__init__",
    "sglang.srt.mem_cache.memory_pool.HybridLinearKVPool._transfer_full_attention_id",
    "sglang.srt.mem_cache.memory_pool.KVCache.__init__",
    "sglang.srt.mem_cache.memory_pool.MHATokenToKVPool",
    "sglang.srt.mem_cache.memory_pool.MHATokenToKVPool.__init__",
    "sglang.srt.mem_cache.memory_pool.MHATokenToKVPool._create_buffers",
    "sglang.srt.mem_cache.memory_pool.MHATokenToKVPool._init_data_ptrs_and_strides",
    "sglang.srt.mem_cache.allocator.base.BaseTokenToKVPoolAllocator.free_group_begin",
    "sglang.srt.mem_cache.allocator.base.BaseTokenToKVPoolAllocator.free_group_end",
    "sglang.srt.mem_cache.allocator.paged.PagedTokenToKVPoolAllocator",
    "sglang.srt.mem_cache.allocator.paged.PagedTokenToKVPoolAllocator.available_size",
    "sglang.srt.mem_cache.allocator.paged.PagedTokenToKVPoolAllocator.free_group_begin",
    "sglang.srt.mem_cache.allocator.paged.PagedTokenToKVPoolAllocator.free_group_end",
    # Request rows, generations and recurrent/PLE state.
    "sglang.srt.mem_cache.memory_pool.ReqToTokenPool.__init__",
    "sglang.srt.mem_cache.memory_pool.ReqToTokenPool.alloc_rows",
    "sglang.srt.mem_cache.memory_pool.ReqToTokenPool.clear",
    "sglang.srt.mem_cache.memory_pool.HybridReqToTokenPool.__init__",
    "sglang.srt.mem_cache.memory_pool.HybridReqToTokenPool._init_mamba_pool",
    "sglang.srt.mem_cache.memory_pool.HybridReqToTokenPool.alloc",
    "sglang.srt.mem_cache.memory_pool.MambaPool",
    "sglang.srt.mem_cache.memory_pool.MambaPool.State",
    "sglang.srt.mem_cache.memory_pool.MambaPool.__init__",
    "sglang.srt.mem_cache.memory_pool.MambaPool.register_slot_state",
    "sglang.srt.mem_cache.memory_pool.MambaPool.get_contiguous_buf_infos",
    "sglang.srt.mem_cache.allocator.mamba.MambaSlotAllocator.available_size",
    "sglang.srt.mem_cache.ple_state_pool.ShortConvPool",
    "sglang.srt.mem_cache.ple_state_pool.NGramPool",
    "sglang.srt.managers.schedule_batch.ReqKvInfo",
    "sglang.srt.managers.schedule_batch.Req.__init__",
    "sglang.srt.managers.schedule_batch.ScheduleBatch._collect_deferred_mamba_cow_and_clear",
    # QSA state addressing copied by prefix capture/restore.
    "sglang.srt.layers.attention.qsa.metadata.build_pending_ring_slots",
    "sglang.srt.layers.attention.qsa.metadata.build_group_ring_slots",
    "sglang.srt.layers.attention.qsa.graph_metadata._qsa_graph_row_metadata_kernel",
    "sglang.srt.layers.attention.qwen_sparse_attn_backend.QwenSparseAttnBackend._qsa_write_plan",
    # Batch metadata read by begin_batch.
    "sglang.srt.model_executor.forward_batch_info.ForwardMode",
    "sglang.srt.model_executor.forward_batch_info.ForwardBatch",
    "sglang.srt.model_executor.forward_batch_info.ForwardBatch.init_new",
    # Coordinator callers that no plugin patch replaces.
    "sglang.srt.managers.schedule_batch.ScheduleBatch.prepare_for_decode",
    "sglang.srt.managers.schedule_batch.release_req",
    "sglang.srt.managers.scheduler.Scheduler.release_host_resources",
    "sglang.srt.model_executor.model_runner.ModelRunner._prepare_eager_forward_batch",
    "sglang.srt.model_executor.runner.decode_cuda_graph_runner.DecodeCudaGraphRunner.capture_prepare",
    # CUDA graph configuration and native graph backend.
    "sglang.srt.model_executor.cuda_graph_config.cuda_graph_fully_disabled",
    "sglang.srt.model_executor.cuda_graph_config.check_cuda_graph_backend",
    "sglang.srt.model_executor.cuda_graph_config.CudaGraphConfig",
    "sglang.srt.model_executor.cuda_graph_config.PhaseConfig",
    "sglang.srt.runtime_context.get_exec",
    "sglang.srt.model_executor.runner.flashinfer_autotune.should_run_flashinfer_autotune",
    "sglang.srt.model_executor.runner_backend.full_cuda_graph_backend.FullCudaGraphBackend.__init__",
    "sglang.srt.model_executor.runner_backend.full_cuda_graph_backend.FullCudaGraphBackend.capture_session",
    "sglang.srt.model_executor.runner.shape_key.ShapeKey",
    # Hot-buffer resolver kernel wrapper and NVTX helpers.
    "sglang.kernels.ops.kvcache.hisparse.load_cache_to_device_buffer_mla",
    "sglang.kernels.ops.kvcache.hisparse._load_cache_to_device_buffer_mla",
    "sglang.kernels.ops.kvcache.hisparse._jit_sparse_module",
    "sglang.srt.utils.nvtx_utils.profile_range",
    "sglang.srt.utils.nvtx_utils.profile_method",
    # Server arguments read by config.validate_configuration and the runtime.
    "sglang.srt.server_args.ServerArgs",
    "sglang.srt.arg_groups.fields.device.Device",
    "sglang.srt.arg_groups.fields.disagg.Disagg",
    "sglang.srt.arg_groups.fields.exec_.ExecDeterministic",
    "sglang.srt.arg_groups.fields.exec_.ExecGraph",
    "sglang.srt.arg_groups.fields.exec_.ExecMamba",
    "sglang.srt.arg_groups.fields.exec_.ExecOverlap",
    "sglang.srt.arg_groups.fields.lora.Lora",
    "sglang.srt.arg_groups.fields.memory.Memory",
    "sglang.srt.arg_groups.fields.model.Model",
    "sglang.srt.arg_groups.fields.parallel.Parallel",
    "sglang.srt.arg_groups.fields.schedule.Schedule",
    "sglang.srt.arg_groups.fields.serving.Serving",
    "sglang.srt.arg_groups.fields.spec.Spec",
    # prefix_cache.py (row N02 code; its behavior is W6's).
    "sglang.srt.mem_cache.chunk_cache.ChunkCache",
    "sglang.srt.mem_cache.base_prefix_cache.BasePrefixCache.claim_kv_row",
    "sglang.srt.mem_cache.base_prefix_cache.BasePrefixCache.on_release",
    "sglang.srt.mem_cache.common.checkpoint_kv_cache",
    "sglang.srt.mem_cache.base_prefix_cache.MatchPrefixParams",
    "sglang.srt.mem_cache.base_prefix_cache.MatchResult",
    "sglang.srt.mem_cache.radix_cache.RadixKey",
    "sglang.srt.environ.Envs",
    "sglang.srt.utils.common.Range",
    "sglang.srt.mem_cache.memory_pool.ReqToTokenPool.free",
    "sglang.srt.mem_cache.memory_pool.HybridReqToTokenPool.free_mamba_cache",
    "sglang.srt.mem_cache.allocator.paged.PagedTokenToKVPoolAllocator.alloc",
    "sglang.srt.mem_cache.allocator.paged.PagedTokenToKVPoolAllocator.free",
)
