"""Startup contract for the bounded QSA HiSparse execution modes."""

import os

import torch


def prefix_cache_options():
    try:
        mb = int(os.environ.get("SGLANG_QSA_HISPARSE_PREFIX_CACHE_MB", "0"))
        entries = int(
            os.environ.get("SGLANG_QSA_HISPARSE_PREFIX_CACHE_MAX_ENTRIES", "256")
        )
    except ValueError as error:
        raise ValueError(
            "QSA prefix host budget and entry limit must be integers"
        ) from error
    if mb < 0 or entries <= 0:
        raise ValueError(
            "QSA prefix host budget must be nonnegative and entry limit positive"
        )
    return mb * 1024 * 1024, entries


def validate_configuration(args, pool, *, p2=False, graph=False, strict=True):
    budget, _ = prefix_cache_options()
    if budget and not p2:
        raise ValueError("QSA host prefixes require the multi-request offload runtime")
    if graph and not p2:
        raise ValueError("only QSA P2 offload supports the bounded graph path")
    max_requests = getattr(args, "max_running_requests", None) if p2 else 1
    if p2 and max_requests not in (2, 4, 8):
        raise ValueError("QSA P2 requires max_running_requests in (2, 4, 8)")
    required = {
        "max_running_requests": max_requests,
        "tp_size": 2,
        "pp_size": 1,
        "disable_radix_cache": True,
        "disable_overlap_schedule": True,
        "cuda_graph_backend_decode": "full" if graph else "disabled",
        "cuda_graph_backend_prefill": "disabled",
        "context_length": 262144,
        "max_total_tokens": max_requests * 262144,
        "skip_server_warmup": True,
        "random_seed": 147342228,
        "speculative_algorithm": None,
        "disaggregation_mode": "null",
    }
    # The strict comparison oracle needs batch-invariant math. P2 light runtime
    # can use the user's native GEMM path; deterministic GEMM dominates B1 cost.
    if not p2 or strict:
        required["enable_deterministic_inference"] = True
    if graph:
        required.update(
            disable_cuda_graph_padding=True,
            cuda_graph_bs_decode=list(range(1, max_requests + 1)),
            cuda_graph_max_bs_decode=max_requests,
            enable_torch_compile=False,
        )
    for name, value in required.items():
        if getattr(args, name, None) != value:
            raise ValueError(f"QSA V3 requires {name}={value!r}")
    chunks = (2048, 4096) if p2 else (2048,)
    if getattr(args, "chunked_prefill_size", None) not in chunks:
        raise ValueError(f"QSA requires chunked_prefill_size in {chunks}")
    if getattr(args, "enable_hisparse", False):
        raise ValueError("QSA V3 cannot use the MLA HiSparse coordinator")
    if any(
        getattr(args, x, False)
        for x in (
            "enable_dp_attention",
            "enable_two_batch_overlap",
            "enable_single_batch_overlap",
            "enable_streaming_session",
        )
    ):
        raise ValueError("QSA V3 does not support DP/overlap")
    full = pool.full_kv_pool
    if (
        pool.size != max_requests * 262144
        or pool.page_size != 64
        or pool.qsa_compress_ratio != 4
        or pool.qsa_token_topk != 2048
        or pool.full_layer_nums != 12
        or pool.head_num != 1
        or pool.head_dim != 256
        or pool.dtype != torch.float8_e4m3fn
        or full.use_hnd
        or full.kv_cache_layout == "vectorized_5d"
        or full.post_capture_active
        or full.is_quantized_kv_cache
    ):
        raise ValueError("QSA V3 requires plain NHD FP8 TP2 C4/page64 pool")
