"""Plugin-specific contracts of the W4 hooks (not covered by the fork tests).

- Target-model scope (PLAN.md rule 9): out of scope every model_compat hook
  runs the pinned definition; each case uses an input on which the pinned and
  fork definitions behave observably differently.
- FP8 descales (Q04, A09) and the hook-order independence of the two backend
  ``__init__`` hooks (inventory section 2, C3).
- HiSparse runtime wiring (Q03, Q06, Q07) with stand-in runtimes.
- The ROCm packed decode of Q12 (the pin's branch, which the reference merged
  into its NVTX ranges) and the routing of the fast_topk overflow fix (T05).
"""

import sys
from types import SimpleNamespace
from unittest.mock import PropertyMock, patch

import pytest
import torch

from sglang.srt.layers.attention import qwen_sparse_attn_backend as backend_module
from sglang.srt.layers.attention.qsa import kernel as qsa_kernel
from sglang.srt.layers.attention.qsa import qsa_indexer
from sglang.srt.layers.attention.qsa import sparse_attn
from sglang.srt.model_executor.forward_batch_info import ForwardMode

COMPAT = "sglang_qsa_hisparse.patches.model_compat.qsa_attention"
FP8 = torch.float8_e4m3fn


class _Stop(Exception):
    """Ends a call at the observed point."""


class _Pool:
    qsa_compress_ratio = 4
    qsa_token_topk = 8
    qsa_compressed_page_size = 64

    def __init__(self, dtype=torch.bfloat16):
        self.dtype = dtype
        self.stored = None

    def set_kv_buffer(self, *args):
        self.stored = args
        raise _Stop

    def get_key_buffer(self, layer_id):
        return torch.zeros(16, 1, 8, dtype=self.dtype)

    get_value_buffer = get_key_buffer


def _runner(pool):
    return SimpleNamespace(
        device="cpu",
        token_to_kv_pool=pool,
        req_to_token_pool=SimpleNamespace(
            req_to_token=torch.arange(16, dtype=torch.int32).reshape(1, 16)
        ),
        model_config=SimpleNamespace(
            context_len=16, hf_config=SimpleNamespace(indexer_compress_ratio=4)
        ),
    )


def _backend(pool, target_model, active):
    target_model(active)
    return backend_module.QwenSparseAttnBackend(_runner(pool))


def _layer():
    return SimpleNamespace(
        layer_id=0, tp_q_head_num=1, head_dim=8, scaling=1.0,
        k_scale_float=2.0, v_scale_float=0.5,
    )


# Target-model scope -----------------------------------------------------------


@pytest.mark.parametrize("method", ["forward_extend", "forward_decode"])
@pytest.mark.parametrize("active", [True, False])
def test_kv_store_descales_fp8_only_for_the_target_model(target_model, method, active):
    pool = _Pool(dtype=FP8)
    backend = _backend(pool, target_model, active)
    k, v = torch.ones(1, 8), torch.ones(1, 8)
    batch = SimpleNamespace(out_cache_loc=torch.tensor([3]))
    with pytest.raises(_Stop):
        getattr(backend, method)(
            k, k, v, _layer(), batch, topk_indices=torch.zeros(1, 1, dtype=torch.int32)
        )
    if active:
        # Q04: the pool divides by non-unit scales in place, so live K/V are cloned.
        layer, loc, k_stored, v_stored, k_scale, v_scale = pool.stored
        assert (k_scale, v_scale) == (2.0, 0.5)
        assert k_stored is not k and torch.equal(k_stored, k)
        assert v_stored is not v and torch.equal(v_stored, v)
    else:
        assert pool.stored[2] is k and pool.stored[3] is v
        assert len(pool.stored) == 4


@pytest.mark.parametrize("active", [True, False])
def test_trtllm_extraction_gets_descales_only_for_the_target_model(
    target_model, monkeypatch, active
):
    backend = _backend(_Pool(), target_model, active)
    calls = []

    def extraction(*args, **kwargs):
        calls.append(kwargs)
        raise _Stop

    monkeypatch.setattr(f"{COMPAT}.qwen_sparse_kv_extraction_compact_triton", extraction)
    monkeypatch.setattr(
        backend_module, "qwen_sparse_kv_extraction_compact_triton", extraction
    )
    k_buffer = torch.zeros(16, 1, 8, dtype=FP8)
    metadata = SimpleNamespace(
        sequence_lengths=torch.tensor([8], dtype=torch.int32),
        is_cuda_graph=False,
        row_req_pool_indices=torch.tensor([0], dtype=torch.int32),
    )
    with pytest.raises(_Stop):
        backend._forward_trtllm_sparse(
            torch.zeros(1, 1, 8, dtype=torch.bfloat16), k_buffer, k_buffer, _layer(),
            None, metadata, torch.tensor([[0, 1, -1, -1]], dtype=torch.int32), None,
        )
    assert ("k_scale" in calls[0]) is active
    if active:
        assert (calls[0]["k_scale"], calls[0]["v_scale"]) == (2.0, 0.5)


@pytest.mark.parametrize("active", [True, False])
def test_paged_attention_routes_through_the_runtime_only_for_the_target_model(
    target_model, monkeypatch, active
):
    backend = _backend(_Pool(), target_model, active)
    backend.forward_metadata = SimpleNamespace(
        row_req_pool_indices=torch.tensor([0]), is_cuda_graph=False
    )

    def selected(*args, **kwargs):
        raise _Stop("fork path")

    def pinned_resolver():
        raise _Stop("pinned path")

    backend.qsa_hisparse = SimpleNamespace(offloaded=True, selected=selected)
    monkeypatch.setattr(backend_module, "_resolve_trtllm_sparse_decode", pinned_resolver)
    with (
        patch.object(torch.Tensor, "is_cuda", new_callable=PropertyMock, return_value=True),
        pytest.raises(_Stop, match="fork path" if active else "pinned path"),
    ):
        backend._forward_paged_attention(
            torch.zeros(1, 1, 8), _layer(), None, torch.zeros(1, 4, dtype=torch.int32)
        )


def test_rocm_paged_decode_runs_the_packed_kernel(target_model, monkeypatch):
    """Q12: the pin's ROCm branch runs after extraction and flash-attention is
    never resolved on ROCm; the reference's merge keeps both."""
    backend = _backend(_Pool(), target_model, True)
    backend.forward_metadata = SimpleNamespace(
        row_req_pool_indices=torch.tensor([0]),
        is_cuda_graph=False,
        sequence_lengths=torch.tensor([8], dtype=torch.int32),
    )
    backend._get_fa2_scratch = lambda capacity, heads, dim, dtype, device: (
        torch.zeros(capacity, heads, dim, dtype=dtype),
        torch.zeros(capacity, heads, dim, dtype=dtype),
    )
    calls = []

    def packed_decode(q, k, v, indices, cu_q, cu_k, kv_lens, scale):
        calls.append("packed")
        return torch.ones_like(q)

    def resolver():
        raise AssertionError("flash-attention resolved on ROCm")

    monkeypatch.setattr(f"{COMPAT}.is_hip", lambda: True)
    monkeypatch.setattr(f"{COMPAT}._resolve_trtllm_sparse_decode", lambda: None)
    monkeypatch.setattr(f"{COMPAT}.qwen_sparse_fa2_cu_seqlens_triton", lambda *a: None)
    monkeypatch.setattr(
        f"{COMPAT}.qwen_sparse_kv_extraction_compact_triton",
        lambda *a, **k: calls.append("extract"),
    )
    monkeypatch.setattr(f"{COMPAT}.sparse_gqa_packed_decode_triton", packed_decode)
    monkeypatch.setattr(f"{COMPAT}._resolve_flash_attn_varlen_func", resolver)
    with patch.object(torch.Tensor, "is_cuda", new_callable=PropertyMock, return_value=True):
        output = backend._forward_paged_attention(
            torch.zeros(1, 1, 8, dtype=torch.bfloat16),
            _layer(),
            None,
            torch.tensor([[0, 1, -1, -1]], dtype=torch.int32),
        )
    assert calls == ["extract", "packed"]
    assert output.shape == (1, 8) and bool((output == 1).all())


def test_backend_copies_bind_the_packed_decode_and_reference_copies():
    """A11/A12 have no hook: the Q08/Q12 copies, the only in-scope callers,
    bind the reference's FP8-aware definitions, never the pinned ones."""
    import importlib

    from sglang_qsa_hisparse.kernels import qsa_sparse_attn as plugin

    copies = importlib.import_module(COMPAT)
    assert copies.sparse_gqa_packed_decode_triton is plugin.sparse_gqa_packed_decode_triton
    assert copies.qsa_sparse_attention is plugin.qsa_sparse_attention
    assert sparse_attn.sparse_gqa_packed_decode_triton is not plugin.sparse_gqa_packed_decode_triton
    assert qsa_kernel.qsa_sparse_attention is not plugin.qsa_sparse_attention


class _TopKModule:
    def __init__(self, calls, name):
        self.calls, self.name = calls, name

    def __getattr__(self, export):
        return lambda score, starts, indices, lengths: self.calls.append((self.name, export))


@pytest.mark.parametrize("active", [True, False])
def test_fast_topk_builds_the_overflow_fixed_kernel_only_for_the_target_model(
    target_model, monkeypatch, active
):
    """T05: in scope the module attribute every caller imports at call time
    loads the plugin JIT module (production's fast_topk.cuh); out of scope the
    in-tree one."""
    from sglang.kernels.ops.attention import fast_topk as pinned
    from sglang_qsa_hisparse.kernels import fast_topk as plugin

    calls = []
    monkeypatch.setattr(pinned, "_jit_fast_topk_module", lambda k: _TopKModule(calls, "pin"))
    monkeypatch.setattr(plugin, "_jit_fast_topk_module", lambda k: _TopKModule(calls, "plugin"))
    target_model(active)
    from sglang.kernels.ops.attention.fast_topk import fast_topk

    indices = fast_topk(torch.zeros(2, 8), torch.full((2,), 8, dtype=torch.int32), 512)
    assert indices.shape == (2, 512)
    assert calls == (
        [("plugin", "qsa_hisparse_fast_topk")] if active else [("pin", "fast_topk")]
    )


class _Fp8MhaPool:
    is_quantized_kv_cache = False
    dtype = FP8

    def __init__(self):
        self.k = torch.zeros((4, 1, 256)).to(FP8)

    def get_key_buffer(self, layer_id):
        return self.k

    get_value_buffer = get_key_buffer


@pytest.mark.parametrize("active", [True, False])
def test_fused_kv_path_refuses_fp8_descales_only_for_the_target_model(
    target_model, monkeypatch, active
):
    """Q13: out of scope the pinned fused attempt proceeds past the guard."""
    monkeypatch.setattr(backend_module, "MHATokenToKVPool", _Fp8MhaPool)
    monkeypatch.setattr(backend_module, "_resolve_trtllm_sparse_decode", lambda: object())
    target_model(active)
    backend = backend_module.QwenSparseAttnBackend()
    backend.token_to_kv_pool = _Fp8MhaPool()
    backend._fused_kv_pool_eligible = True
    backend._resolve_metadata = lambda forward_batch: (_ for _ in ()).throw(_Stop())
    layer = SimpleNamespace(
        layer_id=0, tp_q_head_num=2, head_dim=256, k_scale_float=2.0, v_scale_float=0.5
    )
    batch = SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        out_cache_loc=torch.arange(1, dtype=torch.int32),
        req_pool_indices=torch.tensor([0]),
    )
    kv = torch.zeros((1, 1, 256), dtype=torch.bfloat16)
    with patch.object(torch.Tensor, "is_cuda", new_callable=PropertyMock, return_value=True):
        call = lambda: backend._try_fused_kv_attention(  # noqa: E731
            torch.zeros((1, 2, 256), dtype=torch.bfloat16), kv, kv, layer, batch,
            torch.zeros((1, 1), dtype=torch.int32),
        )
        if active:
            assert call() is None
        else:
            with pytest.raises(_Stop):
                call()


@pytest.mark.parametrize("active", [True, False])
def test_flash_attention_fallback_only_for_the_target_model(target_model, active):
    resolver = backend_module._resolve_flash_attn_varlen_func
    target_model(active)
    resolver.cache_clear()
    try:
        with (
            patch.dict(sys.modules, {"flash_attn": None}),
            patch("torch.cuda.get_device_capability", return_value=(8, 9)),
            patch("sglang.srt.utils.is_sm121", return_value=False),
        ):
            if active:
                assert resolver().__module__ == "sglang.kernels.ops.attention.flash_attention"
            else:
                with pytest.raises(ImportError, match="FA4 cute"):
                    resolver()
    finally:
        resolver.cache_clear()


@pytest.mark.parametrize("active", [True, False])
def test_stable_topk_only_for_the_target_model(target_model, active):
    target_model(active)
    scores = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    starts, ends = torch.tensor([0]), torch.tensor([4])
    actual = qsa_kernel.qsa_fast_topk(scores, starts, ends, 2, deterministic=True)
    # Stable selection emits ascending indices; the pinned CPU path keeps top-k order.
    assert actual.tolist() == ([[2, 3]] if active else [[3, 2]])


@pytest.mark.parametrize("active", [True, False])
def test_deterministic_decode_skips_native_topk_only_for_the_target_model(
    target_model, monkeypatch, active
):
    target_model(active)
    published = SimpleNamespace(is_config_namespace_published=lambda name: True)
    config = SimpleNamespace(
        deterministic=SimpleNamespace(enable_deterministic_inference=True)
    )
    native = torch.full((1, 512), 7, dtype=torch.int32)
    for module in (COMPAT, qsa_indexer.__name__):
        monkeypatch.setattr(f"{module}.qsa_mqa_decode", lambda *a: torch.zeros(1, 513))
        monkeypatch.setattr(f"{module}.expand_qsa_block_indices", lambda b, *a, **k: b)
    monkeypatch.setattr(f"{COMPAT}.get_context", lambda: published)
    monkeypatch.setattr(f"{COMPAT}.get_exec", lambda: config)
    monkeypatch.setattr(
        "sglang.kernels.ops.attention.fast_topk.fast_topk", lambda *a, **k: native
    )
    indexer = SimpleNamespace(block_topk=512, compress_ratio=4, token_topk=2048)
    with patch.object(torch.Tensor, "is_cuda", new_callable=PropertyMock, return_value=True):
        actual = qsa_indexer.QSAIndexer.select_decode_tokens(
            indexer, None, None, None, torch.tensor([513]), 2052, None, None
        )
    assert (actual is native) is not active


@pytest.mark.parametrize("active", [True, False])
@pytest.mark.parametrize(
    "launcher, args",
    [
        ("sparse_gqa_fwd_interface_triton", lambda q: (q, q, q, 1, None, None, 1.0)),
        (
            "sparse_gqa_fwd_interface_triton_ck",
            lambda q: (q, q, q, None, torch.tensor([0, 1]), None, None, 1.0),
        ),
    ],
)
def test_sparse_gqa_validation_only_for_the_target_model(
    target_model, monkeypatch, active, launcher, args
):
    target_model(active)

    def pinned_config(total_q):
        raise _Stop

    monkeypatch.setattr(sparse_attn, "_get_best_config", pinned_config)
    q = torch.zeros(1, 1, 8, dtype=torch.float32)
    with pytest.raises(ValueError if active else _Stop):
        getattr(sparse_attn, launcher)(*args(q))


@pytest.mark.parametrize("active", [True, False])
def test_compact_extraction_descales_only_for_the_target_model(target_model, active):
    target_model(active)
    torch.manual_seed(0)
    k = torch.randn(16, 1, 8).to(FP8)
    out_k, out_v = torch.zeros(4, 1, 8, dtype=torch.bfloat16), torch.zeros(4, 1, 8, dtype=torch.bfloat16)
    args = (
        k, k, torch.arange(16, dtype=torch.int32).reshape(1, 16),
        torch.tensor([0], dtype=torch.int32), torch.tensor([[3, 1, 7, -1]], dtype=torch.int32),
        torch.tensor([8], dtype=torch.int32), torch.tensor([0, 3], dtype=torch.int32),
        out_k, out_v, 1, 4,
    )
    if not active:
        # The pinned launcher has no scale arguments.
        with pytest.raises(TypeError):
            sparse_attn.qwen_sparse_kv_extraction_compact_triton(*args, k_scale=2.0)
        return
    sparse_attn.qwen_sparse_kv_extraction_compact_triton(*args, k_scale=2.0, v_scale=0.5)
    gathered = k[[3, 1, 7]].float()
    assert torch.equal(out_k[:3], (gathered * 2.0).to(torch.bfloat16))
    assert torch.equal(out_v[:3], (gathered * 0.5).to(torch.bfloat16))


def test_eager_score_width_hook_matches_the_fork_conditions(target_model):
    from sglang_qsa_hisparse.patches.model_compat.qsa_attention import (
        _match_graph_decode_score_width as hook,
    )

    def batch(mode=ForwardMode.DECODE, rows=1, spec_info=None):
        return SimpleNamespace(
            forward_mode=mode, seq_lens=torch.ones(rows), spec_info=spec_info
        )

    def backend(in_scope=True, mtp_reuse=False):
        return SimpleNamespace(
            _qsa_target_model=in_scope,
            should_reuse_mtp_sparse_indices=lambda forward_batch: mtp_reuse,
        )

    # Every case where the fork leaves decode_score_width None.
    for self, forward_batch in [
        (backend(in_scope=False), batch()),
        (backend(), batch(mode=ForwardMode.IDLE)),
        (backend(), batch(rows=0)),
        (backend(mtp_reuse=True), batch()),
        (backend(), batch(mode=ForwardMode.EXTEND)),
        (backend(), batch(mode=ForwardMode.TARGET_VERIFY)),
        (backend(), batch(spec_info=object())),
    ]:
        assert hook(object(), self, forward_batch) is None


# Backend construction -----------------------------------------------------------


def test_init_hooks_are_order_independent(monkeypatch):
    from sglang_qsa_hisparse.hisparse import runtime
    from sglang_qsa_hisparse.patches.hisparse.qsa_backend import _attach_hisparse_runtime
    from sglang_qsa_hisparse.patches.model_compat.qsa_attention import _init_compat_state

    monkeypatch.setenv("SGLANG_QSA_HISPARSE_V3", "p2-offload")
    monkeypatch.setattr(runtime, "QSAHiSparseRuntime", lambda runner, mode: (runner, mode))
    runner = object()
    states = []
    for order in ([_init_compat_state, _attach_hisparse_runtime],
                  [_attach_hisparse_runtime, _init_compat_state]):
        backend = SimpleNamespace(token_to_kv_pool=SimpleNamespace())
        # The reference's _fused_kv_eligible (Q03) reads the attached runtime.
        backend._fused_kv_eligible = lambda: backend.qsa_hisparse is None
        for hook in order:
            hook(None, backend, runner)
        states.append(
            (
                backend.qsa_hisparse,
                backend.token_to_kv_pool.qsa_hisparse,
                backend._fused_kv_pool_eligible,
            )
        )
    assert states[0] == states[1] == (((runner, "p2-offload"),) * 2 + (False,))


@pytest.mark.parametrize(
    "mode, runtime_class",
    [
        ("p2-offload", "runtime.QSAHiSparseRuntime"),
        ("p2-resident", "runtime.QSAHiSparseRuntime"),
        ("offload", "single_request.QSAHiSparseSingleRequest"),
        ("resident", "single_request.QSAHiSparseSingleRequest"),
        (None, None),
    ],
)
def test_backend_attaches_the_runtime_for_the_mode(target_model, monkeypatch, mode, runtime_class):
    built = []

    def fake(runner, mode):
        built.append((runner, mode))
        return SimpleNamespace(mode=mode)

    for name in ("runtime.QSAHiSparseRuntime", "single_request.QSAHiSparseSingleRequest"):
        monkeypatch.setattr(f"sglang_qsa_hisparse.hisparse.{name}", fake)
    if mode is None:
        monkeypatch.delenv("SGLANG_QSA_HISPARSE_V3", raising=False)
    else:
        monkeypatch.setenv("SGLANG_QSA_HISPARSE_V3", mode)
    pool = _Pool()
    backend = _backend(pool, target_model, True)
    if mode is None:
        assert backend.qsa_hisparse is None and not built
        return
    assert built == [(backend.runner, mode)]
    assert backend.qsa_hisparse.mode == mode
    assert pool.qsa_hisparse is backend.qsa_hisparse


@pytest.mark.parametrize("mode", ["p2-offload", None])
def test_attached_runtime_turns_the_fused_kv_path_off(target_model, monkeypatch, mode):
    """Q03: the reference computes _fused_kv_pool_eligible after attaching the
    runtime, so a fused-eligible pool keeps the fused #40972 path only
    without one."""

    class _EligiblePool(_Pool):
        is_quantized_kv_cache = False

    monkeypatch.setattr(backend_module, "MHATokenToKVPool", _EligiblePool)
    monkeypatch.setattr(
        "sglang_qsa_hisparse.hisparse.runtime.QSAHiSparseRuntime",
        lambda runner, mode: SimpleNamespace(mode=mode),
    )
    if mode is None:
        monkeypatch.delenv("SGLANG_QSA_HISPARSE_V3", raising=False)
    else:
        monkeypatch.setenv("SGLANG_QSA_HISPARSE_V3", mode)
    backend = _backend(_EligiblePool(), target_model, True)
    assert backend._fused_kv_pool_eligible is (mode is None)
    assert backend._fused_kv_eligible() is (mode is None)


def test_runtime_begins_non_idle_batches_before_metadata(target_model, monkeypatch):
    backend = _backend(_Pool(), target_model, True)
    events = []
    backend.qsa_hisparse = SimpleNamespace(begin_batch=lambda batch: events.append("begin"))
    monkeypatch.setattr(
        backend, "_metadata_from_forward_batch", lambda batch: events.append("metadata")
    )
    backend.init_forward_metadata(SimpleNamespace(forward_mode=ForwardMode.IDLE))
    assert events == [] and backend.forward_metadata is None
    backend.init_forward_metadata(SimpleNamespace(forward_mode=ForwardMode.DECODE))
    assert events == ["begin", "metadata"]


def test_graph_capture_plans_the_fa2_wrapper_for_plain_decode_only():
    from sglang_qsa_hisparse.patches.hisparse.qsa_backend import _plan_fa2_graph_wrapper

    planned = []
    backend = SimpleNamespace(
        _is_speculative_paged_mode=backend_module.QwenSparseAttnBackend._is_speculative_paged_mode,
        qsa_profile=object(),
        qsa_hisparse=SimpleNamespace(uses_qsa_hisparse_leases=True, graph_enabled=True),
        _qsa_local_head_shape=lambda: (8, 2, 256, torch.bfloat16),
        _ensure_fa2_graph_wrapper=lambda *args: planned.append(args),
    )
    for mode, spec_info in [
        (ForwardMode.DECODE, None),
        (ForwardMode.DECODE, object()),
        (ForwardMode.TARGET_VERIFY, None),
    ]:
        _plan_fa2_graph_wrapper(
            None, backend, bs=3, num_tokens=12, req_pool_indices=None,
            seq_lens=None, forward_mode=mode, spec_info=spec_info,
        )
    assert planned == [(3, 2, 256, torch.bfloat16)]
