"""QSA attention shared-path patches (inventory W4 rows, feature ``model_compat``).

FP8 K/V descales on store, extraction, prefill and reference reads, the
SM86/SM89 flash-attention fallback, the eager decode score width,
deterministic stable top-k, the FP8-aware sparse GQA kernels and the
fast_topk overflow fix. Rule 2: this feature owns every QSA backend REPLACE;
each copy equals the reference's definition (production ``897286b12a``,
whose upstream base is the pin; PLAN.md Phase 5 rule Q1), including its
``qsa_hisparse`` branches, which stay inert unless
``patches/hisparse/qsa_backend.py`` attaches a runtime.

Copies run with this module's globals (inventory section 6, G3). Mechanical
edits, all at the import level: the kernel launchers, the packed decode
(A11), the torch reference (A12), ``is_fp8_kv_dtype`` and the stable top-k
resolve to the plugin copies in ``kernels/qsa_*.py``;
``_resolve_flash_attn_varlen_func`` resolves to this module's Q01 copy;
``qsa_fast_topk`` is the pinned name, which activation rebinds to the T02 hook;
methods are module-level functions attached to or installed on the class.

Target-model scope (PLAN.md rule 9): the backend ``__init__`` hook (Q02)
evaluates ``scope.target_model_active()`` once per backend and stores it as
``_qsa_target_model``; the backend method hooks (Q05, Q08, Q10-Q13) read that
attribute. Module-level targets (Q01, T02, T03, T05, A03, A05, A09) have no
backend reference and call the process-cached predicate. Out of scope each of
them runs the pinned definition. The remaining rows change no behavior by
themselves: Q02's other attributes, the Q04/Q09 members (called only from the
scoped copies and the hisparse Q07 hook) and the T04 field, which stays None
unless Q05 sets it.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Dict, Optional, Tuple

import msgspec
import torch
import torch.nn.functional as F

from sglang.srt.layers.attention import qwen_sparse_attn_backend as _backend
from sglang.srt.layers.attention.qsa import metadata as _metadata
from sglang.srt.layers.attention.qsa import qsa_indexer as _indexer
from sglang.srt.layers.attention.qsa import sparse_attn as _sparse_attn
from sglang.srt.layers.attention.qsa.fused_kv import fused_kv_prepare
from sglang.srt.layers.attention.qsa.kernel import (
    expand_qsa_block_indices,
    qsa_fast_topk,
)
from sglang.srt.layers.attention.qsa.metadata import compressed_decode_view
from sglang.srt.layers.attention.qsa.mqa import qsa_mqa_decode
from sglang.srt.layers.attention.qsa.sparse_attn import (
    qwen_sparse_fa2_cu_seqlens_triton,
    qwen_sparse_valid_counts_triton,
)
from sglang.srt.layers.attention.qwen_sparse_attn_backend import (
    _TRTLLM_SPARSE_PAGE_SIZE,
    _resolve_trtllm_sparse_decode,
    logger,
)
from sglang.srt.layers.cp.utils import is_cp_active
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    is_in_breakable_cuda_graph,
)
from sglang.srt.runtime_context import get_context, get_exec
from sglang.srt.utils import is_hip
from sglang.srt.utils.nvtx_utils import operations_nvtx_range
from sglang_qsa_hisparse import scope
from sglang_qsa_hisparse.features import MODEL_COMPAT
from sglang_qsa_hisparse.kernels import fast_topk as _plugin_fast_topk
from sglang_qsa_hisparse.kernels import qsa_sparse_attn as _plugin_sparse_attn
from sglang_qsa_hisparse.kernels.qsa_sparse_attn import (
    is_fp8_kv_dtype,
    qsa_sparse_attention,
    qwen_sparse_kv_extraction_compact_triton,
    sparse_gqa_fwd_interface_triton,
    sparse_gqa_fwd_interface_triton_ck,
    sparse_gqa_packed_decode_triton,
)
from sglang_qsa_hisparse.kernels.qsa_topk import _qsa_stable_topk
from sglang_qsa_hisparse.patching import attach_value, patch

_BACKEND_MODULE = "sglang.srt.layers.attention.qwen_sparse_attn_backend"
_BACKEND = f"{_BACKEND_MODULE}.QwenSparseAttnBackend"
_QSA = "sglang.srt.layers.attention.qsa"
_SPARSE_ATTN = f"{_QSA}.sparse_attn"
_RUNTIME_CONTEXT = "sglang.srt.runtime_context"
_FAST_TOPK = "sglang.kernels.ops.attention.fast_topk"

_BACKEND_METHOD_SCOPE = (
    "Scope: decided once per backend by the Q02 __init__ hook "
    "(self._qsa_target_model); out of scope the pinned method runs."
)
_FUNCTION_SCOPE = (
    "Scope: module-level target without a backend reference, so it calls the "
    "process-cached scope.target_model_active() per call; out of scope the "
    "pinned definition runs."
)
_DIRECT_BINDING = (
    " The in-scope backend copies bind the plugin copy directly, so only "
    "pinned callers pass through this check."
)


def _scoped_function(upstream, plugin_impl):
    """REPLACE body for a module-level target: the fork copy only in scope."""

    def replacement(*args, **kwargs):
        if scope.target_model_active():
            return plugin_impl(*args, **kwargs)
        return upstream(*args, **kwargs)

    replacement.__qualname__ = replacement.__name__ = plugin_impl.__name__
    return replacement


def _in_scope(backend) -> bool:
    """Scope decided once per backend: by Q02 at construction, or on first use
    for backends built without ``__init__`` (the fork's tests use ``__new__``)."""
    decided = backend.__dict__.get("_qsa_target_model")
    if decided is None:
        decided = backend._qsa_target_model = scope.target_model_active()
    return decided


def _scoped_method(name, plugin_impl):
    """REPLACE body for a backend method: the fork copy only in scope."""
    upstream = _backend.QwenSparseAttnBackend.__dict__[name]

    def replacement(self, *args, **kwargs):
        if _in_scope(self):
            return plugin_impl(self, *args, **kwargs)
        return upstream(self, *args, **kwargs)

    replacement.__qualname__ = replacement.__name__ = name
    return replacement


# Q01: SM86/SM89 flash-attention fallback ------------------------------------


@lru_cache(maxsize=1)
def _resolve_flash_attn_varlen_func():
    from sglang.srt.utils import is_hip, is_sm121

    if is_sm121():
        from sglang.kernels.ops.attention import (
            qwen38_qsa_sm121_varlen,
        )

        return qwen38_qsa_sm121_varlen
    if is_hip():
        # ROCm ships neither flash_attn (FA2) nor flash-attn-4; aiter provides an
        # FA2-compatible varlen kernel (CK) with the same call signature. Fall
        # through to the flash_attn probes below if aiter is unavailable.
        try:
            from aiter import flash_attn_varlen_func as aiter_varlen_func
        except ImportError:
            aiter_varlen_func = None
        if aiter_varlen_func is not None:

            def flash_attn_varlen_func(*args, **kwargs):
                output = aiter_varlen_func(*args, **kwargs)
                # aiter returns a tuple when return_lse/return_attn_probs is set.
                return output[0] if isinstance(output, tuple) else output

            return flash_attn_varlen_func
    try:
        from flash_attn import flash_attn_varlen_func

        return flash_attn_varlen_func
    except ImportError:
        pass
    if torch.cuda.get_device_capability() in ((8, 6), (8, 9)):
        # FA4's SM80 head-dim-256 tile needs 128 KiB shared memory, more than
        # consumer Ampere/Ada allow. SGLang's vendored kernel uses a fitting
        # tile and is already the native flash-attention backend on these GPUs.
        from sglang.kernels.ops.attention.flash_attention import (
            flash_attn_varlen_func,
        )

        return flash_attn_varlen_func
    try:
        from flash_attn.cute.interface import flash_attn_varlen_func as cute_varlen_func

        def flash_attn_varlen_func(*args, **kwargs):
            output = cute_varlen_func(*args, **kwargs)
            # The cute interface returns (out, lse); lse is None here.
            return output[0] if isinstance(output, tuple) else output

        return flash_attn_varlen_func
    except ImportError as exc:
        raise ImportError(
            "QSA decode requires flash_attn (FA2), flash-attn-4 (FA4 cute), "
            "or aiter (ROCm) for its packed varlen fallback."
        ) from exc


def _flash_attn_resolver_cache_clear(upstream):
    def cache_clear():
        _resolve_flash_attn_varlen_func.cache_clear()
        upstream.cache_clear()

    return cache_clear


patch(
    f"{_BACKEND_MODULE}._resolve_flash_attn_varlen_func",
    "replace",
    feature=MODEL_COMPAT,
    row="Q01",
    depends=(
        "sglang.kernels.ops.attention.flash_attention.flash_attn_varlen_func",
        "sglang.srt.utils.common.is_hip",
        "sglang.srt.utils.common.is_sm121",
    ),
    reason=(
        "model_compat: selects SGLang's vendored varlen flash-attention instead "
        "of FA4 cute on SM86/SM89 without flash_attn. replace: the branch sits "
        "between two try-blocks (after the pin's ROCm aiter probe). The copy "
        "equals the reference's definition and keeps @lru_cache(maxsize=1). "
        + _FUNCTION_SCOPE
        + _DIRECT_BINDING
    ),
)(_scoped_function(_backend._resolve_flash_attn_varlen_func, _resolve_flash_attn_varlen_func))

attach_value(
    f"{_BACKEND_MODULE}._resolve_flash_attn_varlen_func",
    "cache_clear",
    _flash_attn_resolver_cache_clear(_backend._resolve_flash_attn_varlen_func),
    feature=MODEL_COMPAT,
    row="Q01",
    reason=(
        "Inventory section 6, G5: HookRegistry's REPLACE wrapper drops "
        "lru_cache's cache_clear, which the fork and pinned tests call on the "
        "module attribute; it clears the plugin copy's and the pinned cache."
    ),
)


# Q02: backend construction state ---------------------------------------------


@patch(
    f"{_BACKEND}.__init__",
    "after",
    feature=MODEL_COMPAT,
    row="Q02",
    reason=(
        "model_compat (rule 2): the Q09/Q12 copies read the FA2 graph-wrapper "
        "state; inert without the runtime. after: the fork's block is followed "
        "only by plain None assignments. qsa_hisparse is set to None only when "
        "absent, so the order relative to the hisparse Q03 hook is irrelevant "
        "(section 2, C3). Scope: decided here once per backend and stored as "
        "_qsa_target_model, which every backend method hook reads. The other "
        "attributes are set for every backend: no pinned code reads them."
    ),
)
def _init_compat_state(result, self, *args, **kwargs):
    self._qsa_target_model = scope.target_model_active()
    self._fa2_graph_wrappers = {}
    self._fa2_graph_workspace = None
    self._fa2_graph_shape = None
    self._fa2_graph_unavailable = set()
    self._fa2_graph_active_logged = set()
    if not hasattr(self, "qsa_hisparse"):
        self.qsa_hisparse = None


# Q04: FP8 K/V descales on store ----------------------------------------------


@staticmethod
def _kv_descales(layer, kv_dtype: torch.dtype) -> Tuple[float, float]:
    if not is_fp8_kv_dtype(kv_dtype):
        return 1.0, 1.0
    k_scale = getattr(layer, "k_scale_float", None)
    v_scale = getattr(layer, "v_scale_float", None)
    k_scale = 1.0 if k_scale is None else float(k_scale)
    v_scale = 1.0 if v_scale is None else float(v_scale)
    return (
        k_scale if k_scale > 0.0 else 1.0,
        v_scale if v_scale > 0.0 else 1.0,
    )


def _kv_descale_kwargs(self, layer, buffer: torch.Tensor) -> Dict[str, float]:
    """Descales every reader of a quantized pool must apply in-kernel."""

    if not is_fp8_kv_dtype(buffer.dtype):
        return {}
    k_scale, v_scale = self._kv_descales(layer, buffer.dtype)
    return {"k_scale": k_scale, "v_scale": v_scale}


def _store_kv(self, layer, loc, k: torch.Tensor, v: torch.Tensor) -> None:
    if self.qsa_hisparse is not None:
        loc = self.qsa_hisparse.write_locations(loc)
    cache_dtype = getattr(self.token_to_kv_pool, "dtype", k.dtype)
    if not is_fp8_kv_dtype(cache_dtype):
        self.token_to_kv_pool.set_kv_buffer(layer, loc, k, v)
        return
    k_scale, v_scale = self._kv_descales(layer, cache_dtype)
    if k_scale == 1.0 and v_scale == 1.0:
        self.token_to_kv_pool.set_kv_buffer(layer, loc, k, v)
        return
    # The pool divides by non-unit scales in-place before casting.
    # Preserve live prefill K/V, which are consumed after this write.
    self.token_to_kv_pool.set_kv_buffer(
        layer, loc, k.clone(), v.clone(), k_scale, v_scale
    )


_Q04_REASON = (
    "model_compat: FP8 KV store path and the descales of the prefill and "
    "reference reads (production bdb935d70f); the qsa_hisparse branch is inert "
    "(rule 2). attach: members added by the reference; is_fp8_kv_dtype is the "
    "plugin copy (A01). Scope: called only from the scoped Q08/Q10-Q13 code, "
    "so they need no check of their own."
)
_Q04_DEPENDS = (
    "sglang.srt.mem_cache.memory_pool.HybridLinearKVPool.set_kv_buffer",
    "sglang.srt.mem_cache.memory_pool.MHATokenToKVPool.set_kv_buffer",
)
for _name, _member in (
    ("_kv_descales", _kv_descales),
    ("_kv_descale_kwargs", _kv_descale_kwargs),
    ("_store_kv", _store_kv),
):
    attach_value(
        _BACKEND, _name, _member,
        feature=MODEL_COMPAT, row="Q04", depends=_Q04_DEPENDS, reason=_Q04_REASON,
    )
del _name, _member


# Q05: eager decode score width ------------------------------------------------


@patch(
    f"{_BACKEND}._metadata_from_forward_batch",
    "after",
    feature=MODEL_COMPAT,
    row="Q05",
    depends=(
        f"{_BACKEND}.should_reuse_mtp_sparse_indices",
        f"{_BACKEND}._empty_metadata",
    ),
    reason=(
        "model_compat: eager plain decode uses the CUDA graph score width "
        "(section 4 item 6). after: the fork sets decode_score_width exactly when "
        "the batch is neither idle nor empty (else the _empty_metadata return), "
        "should_reuse_mtp_sparse_indices (pure) is false, the mode is decode and "
        "spec_info is None; the width depends only on max_context_len (final "
        "after the body), compress_ratio and the pool page size, and nothing in "
        "the body reads it, so it is set on the returned indexer metadata (T04 "
        "field). Fallback: replace (193). " + _BACKEND_METHOD_SCOPE
    ),
)
def _match_graph_decode_score_width(result, self, forward_batch):
    if not _in_scope(self):
        return None
    forward_mode = forward_batch.forward_mode
    if (
        forward_mode.is_idle()
        or forward_batch.seq_lens.numel() == 0
        or self.should_reuse_mtp_sparse_indices(forward_batch)
        or not forward_mode.is_decode()
        or getattr(forward_batch, "spec_info", None) is not None
    ):
        return None
    # Reference qwen_sparse_attn_backend.py:793-802.
    max_blocks = math.ceil(self.max_context_len / self.compress_ratio)
    page_size = self.token_to_kv_pool.qsa_compressed_page_size
    decode_score_width = max(1, math.ceil(max_blocks / page_size)) * page_size
    return msgspec.structs.replace(
        result,
        indexer_metadata=msgspec.structs.replace(
            result.indexer_metadata, decode_score_width=decode_score_width
        ),
    )


# Q08: forward_extend ----------------------------------------------------------


def forward_extend(
    self,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layer,
    forward_batch,
    save_kv_cache: bool = True,
    topk_indices: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    if topk_indices is None:
        raise ValueError("QSA sparse attention requires topk_indices")
    if getattr(
        forward_batch, "attn_cp_metadata", None
    ) is not None and is_cp_active(forward_batch):
        return self._forward_extend_cp(
            q, k, v, layer, forward_batch, topk_indices, save_kv_cache
        )
    if save_kv_cache:
        fused_output = self._try_fused_kv_attention(
            q, k, v, layer, forward_batch, topk_indices
        )
        if fused_output is not None:
            return fused_output
        self._store_kv(layer, forward_batch.out_cache_loc, k, v)
    q = q.reshape(-1, layer.tp_q_head_num, layer.head_dim)
    num_output_rows = q.shape[0]
    num_valid_rows = topk_indices.shape[0]
    if num_valid_rows > num_output_rows:
        raise ValueError(
            "QSA top-k rows exceed query rows: "
            f"topk={num_valid_rows}, query={num_output_rows}"
        )
    # DP attention may pad q beyond the indexer's query rows.
    # The kernels see only valid rows; the output is zero-padded to q's row count.
    q = q[:num_valid_rows]
    if self._is_speculative_paged_mode(forward_batch.forward_mode):
        output = self._forward_paged_attention(
            q, layer, forward_batch, topk_indices
        )
        return self._pad_extend_output(output, num_output_rows)
    if not q.is_cuda:
        metadata = self._resolve_metadata(forward_batch)
        slots = self._logical_to_physical(topk_indices, metadata)
        pool = self.token_to_kv_pool
        k_buffer = pool.get_key_buffer(layer.layer_id)
        v_buffer = pool.get_value_buffer(layer.layer_id)
        output = qsa_sparse_attention(
            q,
            k_buffer,
            v_buffer,
            slots,
            layer.scaling,
            **self._kv_descale_kwargs(layer, k_buffer),
        )
        return self._pad_extend_output(output, num_output_rows)

    topk_indices = topk_indices.to(torch.int32).contiguous()
    extend_lens = [int(x) for x in forward_batch.extend_seq_lens_cpu]
    sequence_lens = [int(x) for x in forward_batch.seq_lens_cpu]
    prefix_lens = [
        sequence_lens[i] - extend_lens[i] for i in range(len(extend_lens))
    ]
    cu_seqlens_q = F.pad(
        forward_batch.extend_seq_lens.to(q.device, dtype=torch.int32).cumsum(0),
        (1, 0),
    ).contiguous()
    if not any(prefix_lens):
        output = sparse_gqa_fwd_interface_triton(
            q.contiguous(),
            k[:num_valid_rows].contiguous(),
            v[:num_valid_rows].contiguous(),
            max(sequence_lens, default=1),
            topk_indices,
            cu_seqlens_q,
            layer.scaling,
        )
        return self._pad_extend_output(output, num_output_rows)

    # The validated chunk-prefill kernel consumes tightly packed full-context
    # K/V. Current-chunk K/V has already been committed to the cache above.
    pool = self.token_to_kv_pool
    k_buffer = pool.get_key_buffer(layer.layer_id)
    v_buffer = pool.get_value_buffer(layer.layer_id)
    req_to_token = self.req_to_token_pool.req_to_token
    if is_in_breakable_cuda_graph():
        # Reuse the prepared slot table to avoid a D2H sync per layer on replay.
        req_to_token = self._resolve_metadata(forward_batch).token_slot_table
        req_indices = range(len(sequence_lens))
    else:
        req_indices = forward_batch.req_pool_indices.tolist()
    raw_slots = [
        self.qsa_hisparse.prefill_slots(req_indices[i], sequence_lens[i])
        if self.qsa_hisparse is not None
        else req_to_token[req_indices[i], : sequence_lens[i]].long()
        for i in range(len(sequence_lens))
    ]
    k_parts = [
        k_buffer.index_select(0, raw_slots[i]) for i in range(len(sequence_lens))
    ]
    v_parts = [
        v_buffer.index_select(0, raw_slots[i]) for i in range(len(sequence_lens))
    ]
    sequence_lens_tensor = torch.tensor(
        sequence_lens, dtype=torch.int32, device=q.device
    )
    cu_seqlens_k = F.pad(sequence_lens_tensor.cumsum(0), (1, 0)).contiguous()
    output = sparse_gqa_fwd_interface_triton_ck(
        q.contiguous(),
        torch.cat(k_parts),
        torch.cat(v_parts),
        topk_indices,
        cu_seqlens_q,
        cu_seqlens_k,
        sequence_lens_tensor,
        layer.scaling,
        **self._kv_descale_kwargs(layer, k_buffer),
    )
    return self._pad_extend_output(output, num_output_rows)


patch(
    f"{_BACKEND}.forward_extend",
    "replace",
    feature=MODEL_COMPAT,
    row="Q08",
    depends=(
        f"{_BACKEND}._forward_extend_cp",
        f"{_BACKEND}._is_speculative_paged_mode",
        f"{_BACKEND}._logical_to_physical",
        f"{_BACKEND}._pad_extend_output",
        f"{_BACKEND}._resolve_metadata",
        f"{_BACKEND}._try_fused_kv_attention",
        "sglang.srt.layers.cp.utils.is_cp_active",
        "sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph."
        "context.is_in_breakable_cuda_graph",
    ),
    reason=(
        "model_compat (both features; rule 2 copies the inert qsa_hisparse "
        "branch): stores through _store_kv, gathers chunk-prefill raw slots via "
        "qsa_hisparse.prefill_slots, and passes the FP8 descales to the "
        "reference and chunk-prefill reads (production bdb935d70f). replace "
        "(120): mid-function call changes. The sparse GQA launchers (A03/A05) "
        "and the torch reference (A12) are the plugin copies. "
        + _BACKEND_METHOD_SCOPE
    ),
)(_scoped_method("forward_extend", forward_extend))


# Q09: SM89 FlashInfer ragged FA2 graph wrapper -------------------------------


@lru_cache(maxsize=1)
def _resolve_flashinfer_qsa_ragged():
    if torch.cuda.get_device_capability() != (8, 9):
        return None
    try:
        from flashinfer.prefill import BatchPrefillWithRaggedKVCacheWrapper
    except ImportError:
        return None
    return BatchPrefillWithRaggedKVCacheWrapper


def _qsa_local_head_shape(self):
    if self.runner is None:
        return None
    from sglang.srt.runtime_context import get_parallel

    config = self.runner.model_config
    parallel = get_parallel()
    head_dim = getattr(config, "head_dim", None)
    if head_dim is None:
        head_dim = config.hidden_size // config.num_attention_heads
    return (
        config.get_num_attention_heads(parallel.attn_tp_size),
        config.get_num_kv_heads(parallel.attn_tp_size, parallel.attn_dcp_size),
        head_dim,
        getattr(self.runner, "dtype", torch.bfloat16),
    )


def _ensure_fa2_graph_wrapper(
    self, batch: int, num_kv_heads: int, head_dim: int, dtype: torch.dtype
) -> None:
    wrapper_cls = _resolve_flashinfer_qsa_ragged()
    if wrapper_cls is None or batch in self._fa2_graph_unavailable:
        return
    shape = self._qsa_local_head_shape()
    if shape is None or (num_kv_heads, head_dim, dtype) != shape[1:]:
        return
    if batch in self._fa2_graph_wrappers:
        return
    if self._fa2_graph_workspace is None:
        self._fa2_graph_workspace = torch.zeros(
            128 * 1024 * 1024, dtype=torch.uint8, device=self.device
        )
    topk = int(self.token_to_kv_pool.qsa_token_topk)
    if self.token_to_kv_pool.qsa_compress_ratio > 1:
        topk += self.token_to_kv_pool.qsa_compress_ratio - 1
    try:
        wrapper = wrapper_cls(
            self._fa2_graph_workspace,
            "NHD",
            use_cuda_graph=True,
            qo_indptr_buf=self._graph_cu_seqlens_q[: batch + 1],
            kv_indptr_buf=self._graph_fa2_cu_seqlens_k[: batch + 1],
            backend="fa2",
        )
        qo_indptr = torch.arange(batch + 1, dtype=torch.int32, device=self.device)
        wrapper.plan(
            qo_indptr,
            qo_indptr * topk,
            shape[0],
            num_kv_heads,
            head_dim,
            head_dim_vo=head_dim,
            causal=False,
            q_data_type=dtype,
            kv_data_type=dtype,
            o_data_type=dtype,
            sm_scale=1.0 / math.sqrt(head_dim),
        )
    except (RuntimeError, ValueError) as exc:
        self._fa2_graph_unavailable.add(batch)
        logger.warning_once("QSA FlashInfer ragged B%d unavailable: %s", batch, exc)
        return
    self._fa2_graph_wrappers[batch] = wrapper
    self._fa2_graph_shape = (dtype, num_kv_heads, head_dim)
    logger.info("QSA HiSparse SM89 ragged FA2 B%d enabled", batch)


def _can_run_fa2_graph(
    self,
    q: torch.Tensor,
    k_buffer: torch.Tensor,
    layer,
    forward_batch,
    metadata,
    topk: int,
) -> bool:
    hisparse = self.qsa_hisparse
    expected_topk = int(self.token_to_kv_pool.qsa_token_topk)
    if self.token_to_kv_pool.qsa_compress_ratio > 1:
        expected_topk += self.token_to_kv_pool.qsa_compress_ratio - 1
    return (
        q.shape[0] in self._fa2_graph_wrappers
        and hisparse is not None
        and getattr(hisparse, "uses_qsa_hisparse_leases", False)
        and getattr(hisparse, "mode", None) == "p2-offload"
        and getattr(hisparse, "graph_enabled", False)
        and hisparse.offloaded
        and metadata.is_cuda_graph
        and forward_batch.forward_mode.is_decode()
        and getattr(forward_batch, "spec_info", None) is None
        and topk == expected_topk
        and self._fa2_graph_shape == (q.dtype, k_buffer.shape[1], k_buffer.shape[2])
        and math.isclose(
            float(layer.scaling),
            1.0 / math.sqrt(k_buffer.shape[2]),
            rel_tol=0.0,
            abs_tol=1e-7,
        )
    )


_Q09_REASON = (
    "model_compat (rule 2): _can_run_fa2_graph is called by the Q12 copy; "
    "the hisparse Q07 hook calls the other two. All are inert without the "
    "runtime. attach: members added by the fork; the module-level "
    "_resolve_flashinfer_qsa_ragged lives in this module. Scope: called only "
    "from the scoped Q12 copy and the hisparse Q07 hook."
)
_Q09_DEPENDS = (
    f"{_BACKEND}.init_cuda_graph_state",
    f"{_RUNTIME_CONTEXT}.get_parallel",
    "sglang.srt.configs.model_config.ModelConfig.get_num_attention_heads",
    "sglang.srt.configs.model_config.ModelConfig.get_num_kv_heads",
)
for _member in (_qsa_local_head_shape, _ensure_fa2_graph_wrapper, _can_run_fa2_graph):
    attach_value(
        _BACKEND, _member.__name__, _member,
        feature=MODEL_COMPAT, row="Q09", depends=_Q09_DEPENDS, reason=_Q09_REASON,
    )
del _member


# Q10: trtllm sparse decode with FP8 descales ----------------------------------


def _forward_trtllm_sparse(
    self,
    q: torch.Tensor,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    layer,
    forward_batch,
    metadata,
    topk_indices: torch.Tensor,
    trtllm_decode,
    new_kv=None,
) -> torch.Tensor:
    """Pack selected KV at page-aligned row strides for FlashInfer's paged decode,
    driven by a static arange block table and the per-row valid counts."""
    batch, topk = topk_indices.shape
    compress_ratio = (
        self.compress_ratio if self._uses_block_indices(topk_indices) else 1
    )
    topk = topk * compress_ratio + compress_ratio - 1
    page = _TRTLLM_SPARSE_PAGE_SIZE
    pages_per_row = (topk + page - 1) // page
    stride = pages_per_row * page
    device = q.device
    sequence_lens = metadata.sequence_lengths
    if metadata.is_cuda_graph:
        valid_counts = metadata.fa2_valid_counts
        if valid_counts is None:
            raise RuntimeError("QSA CUDA graph metadata is incomplete")
    else:
        valid_counts = torch.empty(batch, dtype=torch.int32, device=device)
    cu_strided, block_tables = self._get_trtllm_sparse_tables(
        batch, pages_per_row, page, device
    )
    capacity_rows = self._cuda_graph_max_tokens if metadata.is_cuda_graph else batch
    # Gather into the query dtype: an FP8 pool is dequantized on the way in, so the
    # paged kernel always runs the bf16 q + bf16 KV path.
    packed_k, packed_v = self._get_fa2_scratch(
        max(capacity_rows, batch) * stride,
        k_buffer.shape[1],
        k_buffer.shape[2],
        q.dtype,
        k_buffer.device,
    )
    k_scale, v_scale = self._kv_descales(layer, k_buffer.dtype)
    if new_kv is not None:
        k, v, loc, width = new_kv
        row_requests = metadata.row_req_pool_indices
        if row_requests is None:
            row_requests = forward_batch.req_pool_indices
        fused_kv_prepare(
            k_buffer,
            v_buffer,
            k,
            v,
            loc,
            self.req_to_token_pool.req_to_token,
            row_requests,
            topk_indices,
            sequence_lens,
            valid_counts,
            packed_k[: batch * stride],
            packed_v[: batch * stride],
            width,
            compress_ratio=compress_ratio,
            chain_positions=True,
            query_positions=(
                metadata.indexer_metadata.decode_logical_positions
                if compress_ratio > 1
                else None
            ),
        )
    else:
        qwen_sparse_valid_counts_triton(
            sequence_lens, topk_indices, valid_counts, batch, topk
        )
        qwen_sparse_kv_extraction_compact_triton(
            k_buffer,
            v_buffer,
            self.req_to_token_pool.req_to_token,
            (
                metadata.row_req_pool_indices
                if metadata.row_req_pool_indices is not None
                else forward_batch.req_pool_indices
            ),
            topk_indices,
            sequence_lens,
            cu_strided,
            packed_k,
            packed_v,
            batch,
            topk,
            k_scale=k_scale,
            v_scale=v_scale,
            zero_fill_cols=stride,
        )
    num_kv_heads = k_buffer.shape[1]
    head_dim = k_buffer.shape[2]
    kc = (
        packed_k[: batch * stride]
        .view(-1, page, num_kv_heads, head_dim)
        .permute(0, 2, 1, 3)
    )
    vc = (
        packed_v[: batch * stride]
        .view(-1, page, num_kv_heads, head_dim)
        .permute(0, 2, 1, 3)
    )
    if self._trtllm_workspace is None:
        self._trtllm_workspace = torch.zeros(
            128 * 1024 * 1024, dtype=torch.uint8, device=device
        )
    from flashinfer.utils import get_trtllm_gen_multi_ctas_kv_counter_bytes

    key = (batch, q.shape[1], device)
    if key not in self._trtllm_counters:
        size = get_trtllm_gen_multi_ctas_kv_counter_bytes(
            batch,
            q.shape[1],
            torch.cuda.get_device_properties(device).multi_processor_count,
        )
        self._trtllm_counters[key] = torch.zeros(
            size, dtype=torch.uint8, device=device
        )
    output = trtllm_decode(
        query=q.contiguous(),
        kv_cache=(kc, vc),
        workspace_buffer=self._trtllm_workspace,
        block_tables=block_tables,
        seq_lens=valid_counts,
        max_seq_len=stride,
        bmm1_scale=layer.scaling,
        bmm2_scale=1.0,
        multi_ctas_kv_counter_buffer=self._trtllm_counters[key],
    )
    return output.reshape(q.shape[0], -1)


patch(
    f"{_BACKEND}._forward_trtllm_sparse",
    "replace",
    feature=MODEL_COMPAT,
    row="Q10",
    depends=(
        f"{_BACKEND}._get_trtllm_sparse_tables",
        f"{_BACKEND}._get_fa2_scratch",
        f"{_BACKEND}._uses_block_indices",
        f"{_QSA}.fused_kv.fused_kv_prepare",
        f"{_SPARSE_ATTN}.qwen_sparse_valid_counts_triton",
    ),
    reason=(
        "model_compat: FP8 KV descales for compact extraction (the gather "
        "branch; the pin's fused #40972 branch is reached only with unit "
        "descales, Q13). replace (135): extra arguments in a mid-function call; "
        "the launcher is the plugin copy (A09). " + _BACKEND_METHOD_SCOPE
    ),
)(_scoped_method("_forward_trtllm_sparse", _forward_trtllm_sparse))


# Q11: forward_decode ----------------------------------------------------------


def forward_decode(
    self,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layer,
    forward_batch,
    save_kv_cache: bool = True,
    topk_indices: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    if topk_indices is None:
        raise ValueError("QSA sparse attention requires topk_indices")
    if save_kv_cache:
        fused_output = self._try_fused_kv_attention(
            q, k, v, layer, forward_batch, topk_indices
        )
        if fused_output is not None:
            return fused_output
        self._store_kv(layer, forward_batch.out_cache_loc, k, v)
        if self.qsa_hisparse is not None:
            if (
                getattr(self.qsa_hisparse, "graph_enabled", False)
                and self._resolve_metadata(forward_batch).is_cuda_graph
            ):
                self.qsa_hisparse.after_store(layer, graph=True)
            else:
                self.qsa_hisparse.after_store(layer)
    q = q.reshape(-1, layer.tp_q_head_num, layer.head_dim)
    return self._forward_paged_attention(q, layer, forward_batch, topk_indices)


patch(
    f"{_BACKEND}.forward_decode",
    "replace",
    feature=MODEL_COMPAT,
    row="Q11",
    depends=(f"{_BACKEND}._resolve_metadata", f"{_BACKEND}._try_fused_kv_attention"),
    reason=(
        "model_compat (both; rule 2): stores through _store_kv, then "
        "qsa_hisparse.after_store. replace (30): after_store must follow the "
        "store (after the pin's fused #40972 attempt, which the HiSparse "
        "runtime disables, Q03) and precede paged attention. "
        + _BACKEND_METHOD_SCOPE
    ),
)(_scoped_method("forward_decode", forward_decode))


# Q12: _forward_paged_attention ------------------------------------------------


def _forward_paged_attention(
    self,
    q: torch.Tensor,
    layer,
    forward_batch,
    topk_indices: torch.Tensor,
) -> torch.Tensor:
    pool = self.token_to_kv_pool
    k_buffer = pool.get_key_buffer(layer.layer_id)
    v_buffer = pool.get_value_buffer(layer.layer_id)
    metadata = self._resolve_metadata(forward_batch)
    topk_indices = self._expand_block_indices(topk_indices, metadata)
    if not q.is_cuda:
        slots = self._logical_to_physical(topk_indices, metadata)
        output = qsa_sparse_attention(
            q,
            k_buffer,
            v_buffer,
            slots,
            layer.scaling,
            **self._kv_descale_kwargs(layer, k_buffer),
        )
        return output.reshape(q.shape[0], -1)

    topk_indices = topk_indices.to(torch.int32).contiguous()
    req_table = self.req_to_token_pool.req_to_token
    row_req_indices = (
        metadata.row_req_pool_indices
        if metadata.row_req_pool_indices is not None
        else forward_batch.req_pool_indices
    )
    if self.qsa_hisparse is not None and self.qsa_hisparse.offloaded:
        graph_args = (
            {"graph": True}
            if metadata.is_cuda_graph
            and getattr(self.qsa_hisparse, "graph_enabled", False)
            else {}
        )
        k_buffer, v_buffer, req_table, row_req_indices = self.qsa_hisparse.selected(
            layer, topk_indices, **graph_args
        )
    # Both V3 arms use the same FA2 decode implementation.
    trtllm_decode = (
        None if self.qsa_hisparse is not None else _resolve_trtllm_sparse_decode()
    )
    if trtllm_decode is not None:
        return self._forward_trtllm_sparse(
            q,
            k_buffer,
            v_buffer,
            layer,
            forward_batch,
            metadata,
            topk_indices,
            trtllm_decode,
        )

    with operations_nvtx_range("qsa.fa2_metadata_scratch"):
        batch, topk = topk_indices.shape
        sequence_lens = metadata.sequence_lengths
        if metadata.is_cuda_graph:
            valid_counts = metadata.fa2_valid_counts
            cu_seqlens_k = metadata.fa2_cu_seqlens_k
            cu_seqlens_q = metadata.fa2_cu_seqlens_q
            if valid_counts is None or cu_seqlens_k is None or cu_seqlens_q is None:
                raise RuntimeError("QSA CUDA graph FA2 metadata is incomplete")
        else:
            valid_counts = torch.empty(batch, dtype=torch.int32, device=q.device)
            cu_seqlens_k = torch.empty(
                batch + 1, dtype=torch.int32, device=q.device
            )
            cu_seqlens_q = torch.arange(
                batch + 1, dtype=torch.int32, device=q.device
            )
        qwen_sparse_fa2_cu_seqlens_triton(
            sequence_lens,
            topk_indices,
            valid_counts,
            cu_seqlens_k,
            batch,
            topk,
        )
        scratch_capacity = (
            self._cuda_graph_max_tokens * topk
            if metadata.is_cuda_graph
            else batch * topk
        )
        scratch_dtype = (
            q.dtype if is_fp8_kv_dtype(k_buffer.dtype) else k_buffer.dtype
        )
        packed_k, packed_v = self._get_fa2_scratch(
            scratch_capacity,
            k_buffer.shape[1],
            k_buffer.shape[2],
            scratch_dtype,
            k_buffer.device,
        )
        k_scale, v_scale = self._kv_descales(layer, k_buffer.dtype)
    with operations_nvtx_range("qsa.fa2_extract"):
        qwen_sparse_kv_extraction_compact_triton(
            k_buffer,
            v_buffer,
            req_table,
            row_req_indices,
            topk_indices,
            sequence_lens,
            cu_seqlens_k,
            packed_k,
            packed_v,
            batch,
            topk,
            k_scale=k_scale,
            v_scale=v_scale,
        )
    with operations_nvtx_range("qsa.fa2_attention"):
        if is_hip():
            relative_indices = torch.arange(
                topk, dtype=torch.int32, device=q.device
            ).expand(batch, -1)
            relative_indices = relative_indices.masked_fill(
                relative_indices >= valid_counts[:, None], -1
            ).contiguous()
            output = sparse_gqa_packed_decode_triton(
                q.contiguous(),
                packed_k,
                packed_v,
                relative_indices,
                cu_seqlens_q,
                cu_seqlens_k,
                valid_counts,
                layer.scaling,
            )
        elif self._can_run_fa2_graph(
            q, k_buffer, layer, forward_batch, metadata, topk
        ):
            batch = q.shape[0]
            if batch not in self._fa2_graph_active_logged:
                logger.info("QSA HiSparse SM89 ragged FA2 B%d active", batch)
                self._fa2_graph_active_logged.add(batch)
            output = self._fa2_graph_wrappers[batch].run(
                q.contiguous(), packed_k[: batch * topk], packed_v[: batch * topk]
            )
        else:
            flash_attn_varlen_func = _resolve_flash_attn_varlen_func()
            output = flash_attn_varlen_func(
                q=q,
                k=packed_k,
                v=packed_v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=1,
                max_seqlen_k=topk,
                softmax_scale=layer.scaling,
                causal=True,
            )
    if self.qsa_hisparse is not None:
        self.qsa_hisparse.capture_decode(
            layer,
            q,
            packed_k,
            packed_v,
            topk_indices,
            output,
            k_scale,
            v_scale,
            valid_counts,
            cu_seqlens_q,
            cu_seqlens_k,
        )

    return output.reshape(q.shape[0], -1)


patch(
    f"{_BACKEND}._forward_paged_attention",
    "replace",
    feature=MODEL_COMPAT,
    row="Q12",
    depends=(
        f"{_BACKEND_MODULE}._resolve_trtllm_sparse_decode",
        f"{_SPARSE_ATTN}.qwen_sparse_fa2_cu_seqlens_triton",
        f"{_BACKEND}._expand_block_indices",
        f"{_BACKEND}._get_fa2_scratch",
        f"{_BACKEND}._logical_to_physical",
        f"{_BACKEND}._resolve_metadata",
        "sglang.srt.utils.common.is_hip",
        "sglang.srt.utils.nvtx_utils.profile_range",
    ),
    reason=(
        "model_compat (both; rule 2): qsa_hisparse.selected buffers, trtllm "
        "disabled under HiSparse, NVTX ranges, FP8 scratch dtype and descales "
        "(also on the reference read, production bdb935d70f), FA2 graph "
        "wrapper, capture_decode. replace (171). Uses the plugin Q01 resolver, "
        "A09 launcher, A11 packed decode, A12 reference and A01 "
        "is_fp8_kv_dtype. Copy: the reference's definition (production merged "
        "the pin's ROCm branch into its qsa.fa2_attention range). "
        + _BACKEND_METHOD_SCOPE
    ),
)(_scoped_method("_forward_paged_attention", _forward_paged_attention))


# Q13: no fused #40972 KV path with non-unit FP8 descales ----------------------


@patch(
    f"{_BACKEND}._try_fused_kv_attention",
    "around",
    feature=MODEL_COMPAT,
    row="Q13",
    depends=(
        f"{_BACKEND}._supports_fused_kv_pool",
        "sglang.srt.mem_cache.memory_pool.MHATokenToKVPool.get_key_buffer",
    ),
    reason=(
        "model_compat: the pin's fused KV prepare stores cast_fp8(x) without the "
        "write divide and packs cached FP8 without the read scale, so the "
        "reference (production merge 80dc48ddfc) returns None, i.e. keeps the "
        "descaling store and gather path, when the K buffer is FP8 and the "
        "layer's descales are not unit. around: the reference evaluates that "
        "guard after side-effect-free checks that only return None, so "
        "returning None first gives the same result; it is evaluated under the "
        "reference's first preconditions (eligible pool, CUDA q, K/V present, "
        "decode or target-verify), where every caller (Q08, Q11) reads the "
        "same layer's K buffer in _forward_paged_attention when this returns "
        "None, so the only side effect of get_key_buffer (HiCache's per-layer "
        "load wait) happens in the same layer step, only earlier. "
        + _BACKEND_METHOD_SCOPE
    ),
)
def _no_fused_kv_with_fp8_descales(original, self, q, k, v, layer, forward_batch, indices):
    if (
        self._fused_kv_pool_eligible
        and q.is_cuda
        and k is not None
        and v is not None
        and _in_scope(self)
    ):
        mode = forward_batch.forward_mode
        if mode.is_decode() or mode.is_target_verify():
            # Reference qwen_sparse_attn_backend.py:1526-1531.
            kv_dtype = self.token_to_kv_pool.get_key_buffer(layer.layer_id).dtype
            if self._kv_descales(layer, kv_dtype) != (1.0, 1.0):
                return None
    return original(self, q, k, v, layer, forward_batch, indices)


# T02: deterministic stable top-k ---------------------------------------------


def _deterministic_inference() -> bool:
    # The fork's expression at every qsa_fast_topk call site (T02, T03).
    return (
        get_context().is_config_namespace_published("exec")
        and get_exec().deterministic.enable_deterministic_inference
    )


@patch(
    f"{_QSA}.kernel.qsa_fast_topk",
    "around",
    feature=MODEL_COMPAT,
    row="T02",
    depends=(
        f"{_QSA}.metadata.QSAIndexerMetadata.topk_transform",
        f"{_QSA}.qsa_indexer.QSAIndexer.select_prefill_tokens",
        f"{_RUNTIME_CONTEXT}.get_context",
        f"{_RUNTIME_CONTEXT}.get_exec",
        f"{_RUNTIME_CONTEXT}.RuntimeContext.is_config_namespace_published",
    ),
    reason=(
        "model_compat: deterministic inference selects with the stable top-k "
        "(T01). around: accepts the fork's deterministic argument; when omitted "
        "it is computed with the fork's expression, which is exactly what each "
        "fork caller passes (topk_transform, select_prefill_tokens, T03). False "
        "delegates to the original. Fallback: replace topk_transform (24) and "
        "select_prefill_tokens (61). " + _FUNCTION_SCOPE
        + " The predicate is consulted only when deterministic is true."
    ),
)
def _qsa_fast_topk(original, logits, row_starts, row_ends, topk, deterministic=None):
    if deterministic is None:
        deterministic = _deterministic_inference()
    if not deterministic or not scope.target_model_active():
        return original(logits, row_starts, row_ends, topk)
    # Reference qsa/kernel.py:87-90.
    lengths = (row_ends - row_starts).to(device=logits.device, dtype=torch.int32)
    starts = row_starts.to(device=logits.device, dtype=torch.int32)
    return _qsa_stable_topk(logits, starts, lengths, topk)


# T03: select_decode_tokens ----------------------------------------------------


def select_decode_tokens(
    self,
    q: torch.Tensor,
    compressed_cache: torch.Tensor,
    compressed_page_table: torch.Tensor,
    compressed_lengths: torch.Tensor,
    max_model_len: int,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    defer_expansion: bool = False,
) -> torch.Tensor:
    logits = qsa_mqa_decode(
        q,
        compressed_cache,
        compressed_page_table,
        compressed_lengths,
        max_model_len,
    )
    deterministic = (
        get_context().is_config_namespace_published("exec")
        and get_exec().deterministic.enable_deterministic_inference
    )
    if logits.is_cuda and self.block_topk == 512 and not deterministic:
        # Decode rows start at zero, so compressed lengths double as row lengths;
        # skip the generic zero-fill + subtract.
        from sglang.kernels.ops.attention.fast_topk import fast_topk

        block_indices = fast_topk(
            logits,
            compressed_lengths.to(torch.int32),
            topk=self.block_topk,
            row_starts=None,
        )
    else:
        row_starts = torch.zeros_like(compressed_lengths, dtype=torch.int32)
        block_indices = qsa_fast_topk(
            logits,
            row_starts,
            compressed_lengths,
            topk=self.block_topk,
            deterministic=deterministic,
        )
    if defer_expansion:
        return block_indices
    return expand_qsa_block_indices(
        block_indices,
        query_positions,
        sequence_lengths,
        compress_ratio=self.compress_ratio,
        token_topk=self.token_topk,
    )


patch(
    f"{_QSA}.qsa_indexer.QSAIndexer.select_decode_tokens",
    "replace",
    feature=MODEL_COMPAT,
    row="T03",
    depends=(
        f"{_QSA}.mqa.qsa_mqa_decode",
        f"{_QSA}.kernel.expand_qsa_block_indices",
        "sglang.kernels.ops.attention.fast_topk.fast_topk",
        f"{_RUNTIME_CONTEXT}.get_context",
        f"{_RUNTIME_CONTEXT}.get_exec",
    ),
    reason=(
        "model_compat: deterministic decode skips the JIT fast_topk and passes "
        "the flag to qsa_fast_topk (rebound to the T02 hook at activation). "
        "replace (51): mid-function branch predicate. Copy: the reference's "
        "definition (the pin's defer_expansion plus the fork's change); its "
        "inline fast_topk import reads the module attribute, the T05 hook. "
        "Per-layer indexer code without a backend reference: "
        + _FUNCTION_SCOPE
    ),
)(_scoped_function(_indexer.QSAIndexer.__dict__["select_decode_tokens"], select_decode_tokens))


# T05: fast_topk keeps every candidate on radix overflow ----------------------


@patch(
    f"{_FAST_TOPK}.fast_topk",
    "around",
    feature=MODEL_COMPAT,
    row="T05",
    depends=(
        f"{_FAST_TOPK}._jit_fast_topk_module",
        "sglang.kernels.jit.utils.arch.is_arch_support_pdl",
        "sglang.kernels.jit.utils.compile.loader.load_jit",
        "sglang.kernels.jit.utils.compile.spec.resolve_sources",
        "sglang.kernels.jit.utils.compile.cpp_args.make_cpp_args",
        "sglang.kernels.jit.utils.common.cache_once",
    ),
    reason=(
        "model_compat: the reference's fast_topk.cuh (production 773f3c2d84, "
        "fork PR #6) refines from a full-row rescan when a threshold bin "
        "overflows the 4K candidate stage instead of dropping candidates. The "
        "kernel source is moved/owned (kernels/csrc/fast_topk, byte-identical) "
        "and compiled by the plugin copy of the unchanged op wrapper "
        "(kernels/fast_topk.py: own JIT module name and export). around: in "
        "scope every call goes to the plugin wrapper, so the in-tree module is "
        "never built or loaded there; out of scope the original runs. Callers "
        "(the T03 copy and the pinned qsa_fast_topk) import the module "
        "attribute at call time. " + _FUNCTION_SCOPE
    ),
)
def _fast_topk_keeps_overflow_candidates(original, *args, **kwargs):
    if scope.target_model_active():
        return _plugin_fast_topk.fast_topk(*args, **kwargs)
    return original(*args, **kwargs)


# T04: QSAIndexerMetadata.decode_score_width ------------------------------------


class QSAIndexerMetadata(_metadata.QSAIndexerMetadata, frozen=True):
    """Pinned metadata plus the fork's ``decode_score_width`` field.

    The base is read from the module at class creation (section 6, G6). The
    field is appended after the pinned fields (the fork placed it after
    ``decode_lengths``); every pinned construction passes it by keyword or not
    at all.
    """

    # Match graph's native score stride. Deterministic selection separately
    # canonicalizes the selected indices, independent of collector order.
    decode_score_width: Optional[int] = None

    def get_decode_mqa_inputs(
        self, layer_id: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Paged compressed-K cache inputs for decode MQA, one row per query row."""

        pool = self.token_to_kv_pool
        num_rows = self.sequence_lengths.numel()
        if self.token_slot_table.shape[0] != num_rows:
            raise ValueError(
                "QSA decode page-table rows must match the per-query sequence "
                f"lengths: table_rows={self.token_slot_table.shape[0]}, "
                f"rows={num_rows}"
            )
        compressed_cache = pool.get_qsa_compressed_k_buffer(layer_id).reshape(
            -1,
            pool.qsa_compressed_page_size,
            pool.qsa_index_kv_heads,
            pool.qsa_index_head_dim,
        )
        if self.is_cuda_graph:
            if (
                self.graph_compressed_page_table is None
                or self.graph_compressed_lengths is None
            ):
                raise RuntimeError("QSA CUDA graph decode metadata is incomplete")
            return (
                compressed_cache,
                self.graph_compressed_page_table,
                self.graph_compressed_lengths,
                self.graph_compressed_page_table.shape[1]
                * pool.qsa_compressed_page_size,
            )
        if self.decode_page_table is not None and self.decode_lengths is not None:
            return (
                compressed_cache,
                self.decode_page_table,
                self.decode_lengths,
                self.decode_score_width
                or self.decode_page_table.shape[1] * pool.qsa_compressed_page_size,
            )
        compressed_page_table, compressed_lengths = compressed_decode_view(
            compressed_page_size=pool.qsa_compressed_page_size,
            compress_ratio=self.compress_ratio,
            sequence_lengths=self.sequence_lengths,
            token_slot_table=self.token_slot_table,
        )
        return (
            compressed_cache,
            compressed_page_table,
            compressed_lengths,
            self.decode_score_width
            or compressed_page_table.shape[1] * pool.qsa_compressed_page_size,
        )


patch(
    f"{_QSA}.metadata.QSAIndexerMetadata",
    "replace",
    feature=MODEL_COMPAT,
    row="T04",
    depends=(f"{_QSA}.metadata.compressed_decode_view",),
    reason=(
        "model_compat (with Q05): decode_score_width field and its use as the "
        "decode score stride. replace (class): a frozen msgspec subclass adds "
        "the field (default None) and overrides get_decode_mqa_inputs verbatim "
        "(53). Scope: none needed; with the field None the override equals the "
        "pinned method, and only the scoped Q05 hook sets it."
    ),
)(QSAIndexerMetadata)


# A03/A05/A09: FP8-aware sparse GQA launchers ----------------------------------
# A11 (packed decode on the A04 kernel) and A12 (torch reference with FP8
# descales) need no hook: their only pinned callers are the Q12 and Q08/Q12
# targets, whose in-scope copies bind the plugin copies above (imports).

_KERNEL_REASON = (
    "model_compat: K/V dtype validation, FP8 scales and the plugin kernel "
    "(kernels/qsa_sparse_attn.py, moved rows A01/A02/A04/A07). replace: the "
    "kernel signature changed, so the launcher must launch the plugin copy. "
    + _FUNCTION_SCOPE
    + _DIRECT_BINDING
)
patch(
    f"{_SPARSE_ATTN}.sparse_gqa_fwd_interface_triton",
    "replace",
    feature=MODEL_COMPAT,
    row="A03",
    depends=(f"{_SPARSE_ATTN}._get_best_config",),
    reason="A03 (54 lines). " + _KERNEL_REASON,
)(
    _scoped_function(
        _sparse_attn.sparse_gqa_fwd_interface_triton,
        _plugin_sparse_attn.sparse_gqa_fwd_interface_triton,
    )
)
patch(
    f"{_SPARSE_ATTN}.sparse_gqa_fwd_interface_triton_ck",
    "replace",
    feature=MODEL_COMPAT,
    row="A05",
    depends=(f"{_SPARSE_ATTN}._get_best_config",),
    reason="A05 (59 lines). " + _KERNEL_REASON,
)(
    _scoped_function(
        _sparse_attn.sparse_gqa_fwd_interface_triton_ck,
        _plugin_sparse_attn.sparse_gqa_fwd_interface_triton_ck,
    )
)
patch(
    f"{_SPARSE_ATTN}.qwen_sparse_kv_extraction_compact_triton",
    "replace",
    feature=MODEL_COMPAT,
    row="A09",
    reason="A09 (63 lines). " + _KERNEL_REASON,
)(
    _scoped_function(
        _sparse_attn.qwen_sparse_kv_extraction_compact_triton,
        _plugin_sparse_attn.qwen_sparse_kv_extraction_compact_triton,
    )
)
