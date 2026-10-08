"""P2 offload pool sizing and the raw staging KV pool (rows K01-K03, C01, C02).

Fork files: ``model_executor/pool_configurator.py`` and
``mem_cache/{kv_cache_configurator,qsa_kv_pool}.py``. Row C03 (a docstring)
is dropped. Every hook here is hisparse: the behavior is gated on
``SGLANG_QSA_HISPARSE_V3=p2-offload`` or on a ``full_kv_pool`` that only the
p2-offload branch supplies, and is upstream behavior otherwise.
"""

from __future__ import annotations

import os
from contextvars import ContextVar

import torch

from sglang.srt.configs.model_config import (
    dsa_layer_skips_topk,
    get_dsa_index_head_dim,
    get_dsa_index_kpool,
    get_dsa_index_kpool_compress,
    is_deepseek_dsa,
)
from sglang.srt.mem_cache.kv_cache_configurator import calculate_mla_kv_cache_dim
from sglang.srt.mem_cache.memory_pool import (
    HybridLinearKVPool,
    KVCache,
    MHATokenToKVPool,
    MHATokenToKVPoolMXFP8,
    ReqToTokenPool,
)
from sglang.srt.model_executor import pool_configurator
from sglang.srt.model_executor.pool_configurator import logger
from sglang.srt.runtime_context import (
    get_exec,
    get_model,
    get_parallel,
    get_schedule,
    get_spec,
    max_speculative_num_draft_tokens,
)
from sglang_qsa_hisparse.features import HISPARSE
from sglang_qsa_hisparse.patching import patch

_KVC = "sglang.srt.mem_cache.kv_cache_configurator.KVCacheConfigurator"
_POOL_CONFIGURATOR = "sglang.srt.model_executor.pool_configurator.DefaultPoolConfigurator"

# K01 -> K02: ``sizes.max_running_requests`` of the enclosing
# ``_build_token_to_kv_pool`` call (the only caller of the K02 target).
_max_running_requests: ContextVar[int] = ContextVar("qsa_max_running_requests")
# K03: the ``full_kv_pool`` argument of the enclosing ``QSATokenToKVPool.__init__``.
_full_kv_pool: ContextVar = ContextVar("qsa_full_kv_pool", default=None)


# K01 ---------------------------------------------------------------------------


@patch(
    f"{_KVC}._build_token_to_kv_pool",
    "around",
    feature=HISPARSE,
    row="K01",
    depends=(f"{_KVC}._build_hybrid_linear_kv_pool",),
    reason=(
        "Fork passes max_running_requests=sizes.max_running_requests to "
        "_build_hybrid_linear_kv_pool. sizes is keyword-only, so the around binds "
        "it in a context variable for the K02 adapter; _build_hybrid_linear_kv_pool "
        "has exactly one caller (this method). hisparse: only K02's p2-offload "
        "branch reads the value."
    ),
)
def _bind_max_running_requests(original, self, *, sizes, **kwargs):
    token = _max_running_requests.set(sizes.max_running_requests)
    try:
        return original(self, sizes=sizes, **kwargs)
    finally:
        _max_running_requests.reset(token)


# K02 ---------------------------------------------------------------------------


@patch(
    f"{_KVC}._build_hybrid_linear_kv_pool",
    "replace",
    feature=HISPARSE,
    row="K02",
    depends=(
        "sglang.srt.configs.model_config.dsa_layer_skips_topk",
        "sglang.srt.configs.model_config.get_dsa_index_head_dim",
        "sglang.srt.configs.model_config.get_dsa_index_kpool",
        "sglang.srt.configs.model_config.get_dsa_index_kpool_compress",
        "sglang.srt.configs.model_config.is_deepseek_dsa",
        "sglang.srt.layers.attention.qsa.config.parse_qsa_profile",
        f"{_KVC}._build_mha_quant_method",
        "sglang.srt.mem_cache.kv_cache_configurator.calculate_mla_kv_cache_dim",
        "sglang.srt.mem_cache.memory_pool.HybridLinearKVPool",
        "sglang.srt.mem_cache.memory_pool.MHATokenToKVPool",
        "sglang.srt.mem_cache.memory_pool.MHATokenToKVPoolMXFP8",
        "sglang.srt.mem_cache.qsa_kv_pool.QSATokenToKVPool.__init__",
        "sglang.srt.mem_cache.qsa_kv_pool.resolve_qsa_indexer_dtype",
        "sglang.srt.runtime_context.get_exec",
        "sglang.srt.runtime_context.get_model",
        "sglang.srt.runtime_context.get_parallel",
        "sglang.srt.runtime_context.get_spec",
        "sglang.srt.runtime_context.max_speculative_num_draft_tokens",
    ),
    reason=(
        "Production's definition (_build_hybrid_linear_kv_pool below: the pinned "
        "body, which now resolves --qsa-indexer-dtype, with the fork change): "
        "with SGLANG_QSA_HISPARSE_V3=p2-offload it validates the "
        "bounded logical capacity (including the configurator's use_mla_backend "
        "and post_capture_kv_active) and builds the raw staging MHATokenToKVPool "
        "passed as full_kv_pool, which must enter extra_args mid-function, "
        "before pool_class(...). Mechanical edit: the QSAHiSparseSlots import is "
        "rewritten to the plugin package. "
        "The replacement keeps the pinned signature and supplies the fork's "
        "extra max_running_requests argument from K01. hisparse: env-gated."
    ),
)
def _build_hybrid_linear_kv_pool_adapter(self, **kwargs):
    return _build_hybrid_linear_kv_pool(
        self, max_running_requests=_max_running_requests.get(), **kwargs
    )


def _build_hybrid_linear_kv_pool(
    self,
    *,
    max_total_num_tokens: int,
    max_running_requests: int,
    req_to_token_pool: ReqToTokenPool,
    mha_pool_class: type,
) -> KVCache:
    full_attention_layer_ids = (
        [0]
        if self.is_draft_worker
        else [
            i
            for i in self.mambaish_config.full_attention_layer_ids
            if self.layer_info.start_layer <= i < self.layer_info.end_layer
        ]
    )
    extra_args = {}
    if self.use_mla_backend:
        extra_args = {
            "kv_lora_rank": self.model_config.kv_lora_rank,
            "qk_rope_head_dim": self.model_config.qk_rope_head_dim,
        }
        if is_deepseek_dsa(self.model_config.hf_config):
            dsa_index_kpool = get_dsa_index_kpool(self.model_config.hf_config)
            extra_args.update(
                use_dsa=True,
                index_head_dim=get_dsa_index_head_dim(self.model_config.hf_config),
                kv_cache_dim=calculate_mla_kv_cache_dim(
                    model_config=self.model_config,
                    kv_cache_dtype=self.kv_cache_dtype,
                ),
                index_kpool=dsa_index_kpool,
                index_kpool_compress=get_dsa_index_kpool_compress(
                    self.model_config.hf_config
                ),
                skip_topk_layers=(
                    None
                    if self.is_draft_worker
                    else [
                        dsa_layer_skips_topk(self.model_config.hf_config, layer_id)
                        for layer_id in full_attention_layer_ids
                    ]
                ),
            )
            if dsa_index_kpool > 1:
                extra_args.update(
                    tail_extra_slots=(max_speculative_num_draft_tokens() or 0),
                    max_running_requests=(req_to_token_pool.req_to_token.shape[0]),
                )
    quant_method = self._build_mha_quant_method(
        num_layers=len(full_attention_layer_ids)
    )
    # MXFP8 KV cache needs the block-scaled pool (data + UE8M0 scale
    # buffers) for the full-attention layers, same as the SWA branch.
    full_pool_class = (
        MHATokenToKVPoolMXFP8
        if self.kv_cache_dtype_str == "mxfp8" and not self.use_mla_backend
        else mha_pool_class
    )
    from sglang.srt.layers.attention.qsa.config import (
        parse_qsa_profile,
    )
    from sglang.srt.mem_cache.qsa_kv_pool import (
        QSATokenToKVPool,
        resolve_qsa_indexer_dtype,
    )

    qsa_profile = parse_qsa_profile(self.model_config.hf_config)
    qsa_indexer_dtype = get_model().qsa_indexer_dtype
    if qsa_indexer_dtype != "auto" and qsa_profile is None:
        raise ValueError(
            f"--qsa-indexer-dtype {qsa_indexer_dtype} needs a model with a "
            "compressed QSA indexer (Qwen4-Exp); this model has none"
        )
    if qsa_profile is None:
        pool_class = HybridLinearKVPool
        extra_args["use_mla"] = self.use_mla_backend
    else:
        pool_class = QSATokenToKVPool
        extra_args.update(
            qsa_index_kv_heads=qsa_profile.kv_heads,
            qsa_index_head_dim=qsa_profile.head_dim,
            qsa_compress_ratio=qsa_profile.compress_ratio,
            qsa_token_topk=qsa_profile.budget,
            num_request_slots=req_to_token_pool.req_to_token.shape[0],
            qsa_indexer_dtype=resolve_qsa_indexer_dtype(qsa_indexer_dtype),
        )
        if os.environ.get("SGLANG_QSA_HISPARSE_V3") == "p2-offload":
            from sglang_qsa_hisparse.hisparse.slots import QSAHiSparseSlots

            if (max_running_requests not in (2, 4, 8)
                    or max_total_num_tokens != max_running_requests * 262144
                    or full_pool_class is not MHATokenToKVPool
                    or self.pool_page_size != 64 or quant_method is not None
                    or self.post_capture_kv_active or self.use_mla_backend):
                raise ValueError("QSA P2 requires bounded logical capacity and plain static MHA staging")
            slots = QSAHiSparseSlots(262144, 64, max_running_requests)
            extra_args["full_kv_pool"] = MHATokenToKVPool(
                size=slots.raw_pool_size, page_size=64, dtype=self.kv_cache_dtype,
                head_num=self.model_config.get_num_kv_heads(
                    get_parallel().attn_tp_size, get_parallel().attn_dcp_size),
                head_dim=self.model_config.head_dim,
                layer_num=len(full_attention_layer_ids), device=self.device,
                enable_memory_saver=get_exec().features.enable_memory_saver,
            )
    token_to_kv_pool = pool_class(
        page_size=self.pool_page_size,
        size=max_total_num_tokens,
        dtype=self.kv_cache_dtype,
        head_num=self.model_config.get_num_kv_heads(
            get_parallel().attn_tp_size, get_parallel().attn_dcp_size
        ),
        head_dim=self.model_config.head_dim,
        # if draft worker, we only need 1 attention layer's kv pool
        full_attention_layer_ids=full_attention_layer_ids,
        device=self.device,
        mamba_pool=req_to_token_pool.mamba_pool,
        enable_memory_saver=get_exec().features.enable_memory_saver,
        enable_kv_cache_copy=(get_spec().speculative_algorithm is not None),
        start_layer=self.layer_info.start_layer,
        full_kv_pool_class=full_pool_class,
        quant_method=quant_method,
        post_capture_active=self.post_capture_kv_active and quant_method is None,
        **extra_args,
    )
    return token_to_kv_pool


# K03 ---------------------------------------------------------------------------


@patch(
    "sglang.srt.mem_cache.qsa_kv_pool.QSATokenToKVPool.__init__",
    "around",
    feature=HISPARSE,
    row="K03",
    depends=("sglang.srt.mem_cache.memory_pool.HybridLinearKVPool.__init__",),
    reason=(
        "Fork adds full_kv_pool=None to QSATokenToKVPool.__init__ (keyword-only "
        "parameters, so its place before the pin's qsa_indexer_dtype is immaterial) "
        "and forwards it to HybridLinearKVPool.__init__, which already accepts it "
        "at the pin. The "
        "around pops it into a context variable for the duration of the call; "
        "QSA's super().__init__ is the only pool construction nested in it that "
        "reaches HybridLinearKVPool.__init__. hisparse: None (the default) is "
        "the upstream call."
    ),
)
def _qsa_pool_accepts_full_kv_pool(original, self, *args, full_kv_pool=None, **kwargs):
    token = _full_kv_pool.set(full_kv_pool)
    try:
        return original(self, *args, **kwargs)
    finally:
        _full_kv_pool.reset(token)


@patch(
    "sglang.srt.mem_cache.memory_pool.HybridLinearKVPool.__init__",
    "before",
    feature=HISPARSE,
    row="K03",
    reason=(
        "Second half of K03: inject QSATokenToKVPool's full_kv_pool into its "
        "super().__init__ call while the context variable is set; otherwise "
        "the call is unchanged."
    ),
)
def _forward_full_kv_pool(self, *args, **kwargs):
    full_kv_pool = _full_kv_pool.get()
    if full_kv_pool is None:
        return None
    return (self, *args), {**kwargs, "full_kv_pool": full_kv_pool}


# C01, C02 ----------------------------------------------------------------------


@patch(
    f"{_POOL_CONFIGURATOR}.__init__",
    "after",
    feature=HISPARSE,
    row="C01",
    depends=(
        f"{_POOL_CONFIGURATOR}._compute_qsa_cell_size",
        "sglang.srt.configs.hybrid_arch.mambaish_config",
        "sglang.srt.layers.attention.qsa.config.parse_qsa_profile",
        "sglang.srt.mem_cache.qsa_kv_pool.QSATokenToKVPool",
        "sglang.srt.runtime_context.get_schedule",
    ),
    reason=(
        "Fork sets _bias = 0 and, with SGLANG_QSA_HISPARSE_V3=p2-offload, prices "
        "the fixed raw staging + ring bytes as bias and makes the cell size "
        "compressed-only (_compute_qsa_cell_size: compressed keys at the pin's "
        "--qsa-indexer-dtype; the ring stays index_state_dtype). The block is "
        "appended at the end of __init__, which has no early return and does not "
        "read _bias, and the class has no subclasses. The block is copied "
        "verbatim from production; mechanical edits: num_layers (a local of "
        "__init__) is re-derived with the pinned expression, and the QSAHiSparseSlots import "
        "is rewritten to the plugin package. hisparse: env-gated; _bias is 0 "
        "otherwise."
    ),
)
def _price_p2_offload_staging(result, self, kvc):
    self._bias = 0
    # Pinned DefaultPoolConfigurator.__init__ derivation of its local num_layers.
    if mambaish := pool_configurator.mambaish_config(kvc.model_config):
        effective_layer_ids = [
            i
            for i in mambaish.full_attention_layer_ids
            if kvc.layer_info.start_layer <= i < kvc.layer_info.end_layer
        ]
        num_layers = len(effective_layer_ids)
    else:
        num_layers = kvc.layer_info.num_effective_layers

    if os.environ.get("SGLANG_QSA_HISPARSE_V3") == "p2-offload":
        from sglang.srt.layers.attention.qsa.config import (
            parse_qsa_profile,
        )
        from sglang_qsa_hisparse.hisparse.slots import QSAHiSparseSlots
        from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool

        schedule = get_schedule()
        max_requests = schedule.max_running_requests
        profile = parse_qsa_profile(kvc.model_config.hf_text_config)
        if (
            profile is None
            or max_requests not in (2, 4, 8)
            or schedule.max_total_tokens != max_requests * 262144
            or schedule.page_size != 64
            or not kvc.spec_algorithm.is_none()
            or kvc.is_draft_worker
        ):
            raise ValueError("QSA P2 offload requires its bounded logical capacity")
        qsa_cell_size = self._compute_qsa_cell_size(
            hf_config=kvc.model_config.hf_text_config, num_layers=num_layers
        )
        raw_cell_size = self._cell_size - qsa_cell_size
        if raw_cell_size <= 0 or qsa_cell_size <= 0:
            raise ValueError(
                "QSA P2 offload requires raw and compressed KV storage"
            )
        slots = QSAHiSparseSlots(262144, 64, max_requests)
        ring_slots = max_requests * profile.compress_ratio
        ring_bytes = ring_slots * (
            profile.kv_heads
            * profile.head_dim
            * QSATokenToKVPool.index_state_dtype.itemsize
            * num_layers
            + 3 * torch.int64.itemsize
        )
        self._bias = (
            raw_cell_size * (slots.raw_pool_size + 64)
            + qsa_cell_size * 64
            + ring_bytes
        )
        self._cell_size = qsa_cell_size
        logger.info(
            "QSA P2 offload pool budget: fixed_bytes=%d, "
            "logical_bytes_per_token=%d",
            self._bias,
            self._cell_size,
        )


@patch(
    f"{_POOL_CONFIGURATOR}.calculate_pool_sizes",
    "before",
    feature=HISPARSE,
    row="C02",
    reason=(
        "Fork subtracts self._bias (C01) in the first statement, "
        "max(available_bytes - self._bias, 0), which is the argument's only use; "
        "passing available_bytes - self._bias to the pinned body is identical. "
        "hisparse: _bias is 0 unless p2-offload."
    ),
)
def _subtract_fixed_bias(self, available_bytes, page_size):
    return (self, available_bytes - self._bias, page_size), {}
