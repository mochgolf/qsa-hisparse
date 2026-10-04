"""QSA runtime construction must take the rank from the published parallel bundle.

After the parallel-context migration, ``ModelRunner`` has no ``tp_rank``
attribute; reading it raised ``AttributeError`` during production startup
(``QSAHiSparseRuntime.__init__``). These regressions build a runner that does
not expose ``tp_rank`` (and, in the stale case, still carries the old ``0``)
while the published parallel context reports rank ``1``, so a silent default or
a leftover runner read cannot pass.
"""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.mem_cache.qsa_hisparse import config as qsa_config
from sglang.srt.mem_cache.qsa_hisparse import runtime as qsa_runtime
from sglang.srt.mem_cache.qsa_hisparse import single_request as qsa_single
from sglang.srt.mem_cache.qsa_hisparse.slots import QSAHiSparseSlots

CAPACITY = 262144
MAX_REQUESTS = 2
LAYER_IDS = [3, 7]
LAYER_COUNT = 12


def _real_class_instance(cls, **attributes):
    """Type checks in the runtime use ``type(x) is Cls``; build a real instance."""

    instance = cls.__new__(cls)
    for name, value in attributes.items():
        setattr(instance, name, value)
    return instance


def _fake_runner(pool, *, stale_rank=None):
    from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator

    runner = SimpleNamespace(
        token_to_kv_pool=pool,
        token_to_kv_pool_allocator=_real_class_instance(
            PagedTokenToKVPoolAllocator,
            page_size=64,
            free_pages=list(range(CAPACITY // 64)),
            num_staged_pages=0,
        ),
        req_to_token_pool=SimpleNamespace(
            req_to_token=torch.zeros((MAX_REQUESTS, CAPACITY), dtype=torch.int32),
            mamba_pool=SimpleNamespace(
                mamba_cache=SimpleNamespace(
                    replayssm_d=None, replayssm_k=None, replayssm_g=None
                ),
                get_contiguous_buf_infos=lambda: ([], [], None),
            ),
            mamba_allocator=SimpleNamespace(available_size=lambda: MAX_REQUESTS),
        ),
        server_args=SimpleNamespace(
            max_running_requests=MAX_REQUESTS,
            cuda_graph_backend_decode="disabled",
            enable_mixed_chunk=False,
            enable_priority_preemption=False,
            enable_lora=False,
            enable_linear_replayssm=False,
            enable_mamba_extra_buffer=False,
            tp_size=2,
            model_path="fixture-model",
            revision=None,
        ),
        model_config=SimpleNamespace(hf_config=SimpleNamespace(to_dict=lambda: {})),
        dtype=torch.bfloat16,
    )
    if stale_rank is not None:
        # The pre-migration attribute: must never win over the published rank.
        runner.tp_rank = stale_rank
    return runner


def _fake_pool():
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

    slots = QSAHiSparseSlots(CAPACITY, 64, MAX_REQUESTS)
    raw = slots.raw_pool_size
    device = torch.device("cuda:0")
    dtype = torch.float8_e4m3fn
    full = _real_class_instance(
        MHATokenToKVPool,
        size=raw,
        page_size=64,
        dtype=dtype,
        device=device,
        head_num=1,
        head_dim=256,
        layer_num=LAYER_COUNT,
        kv_cache_layout="NHD",
        k_buffer=[
            torch.zeros((raw + 64, 1, 256), dtype=dtype) for _ in range(LAYER_COUNT)
        ],
        v_buffer=[
            torch.zeros((raw + 64, 1, 256), dtype=dtype) for _ in range(LAYER_COUNT)
        ],
    )
    return SimpleNamespace(
        size=CAPACITY,
        page_size=64,
        dtype=dtype,
        device=device,
        full_kv_pool=full,
        full_attention_layer_id_mapping=list(LAYER_IDS),
        qsa_compressed_flat=torch.zeros(16, dtype=torch.uint8),
        qsa_hisparse=None,
    )


@pytest.fixture
def _published_rank(monkeypatch):
    def publish(rank):
        monkeypatch.setattr(
            "sglang.srt.runtime_context.get_parallel",
            lambda: SimpleNamespace(tp_rank=rank, tp_size=2),
        )

    return publish


@pytest.fixture
def _init_env(monkeypatch, tmp_path):
    """Neutralize the CPU-hostile parts of runtime construction."""

    monkeypatch.delenv("SGLANG_QSA_HISPARSE_V3_CAPTURE", raising=False)
    monkeypatch.setenv("SGLANG_QSA_HISPARSE_V3_EVENTS", str(tmp_path))
    monkeypatch.setenv("SGLANG_QSA_HISPARSE_PREFIX_CACHE_MB", "0")
    monkeypatch.setattr(
        "sglang.srt.model_executor.cuda_graph_config.cuda_graph_fully_disabled",
        lambda: True,
    )
    monkeypatch.setattr(qsa_runtime, "validate_configuration", lambda *a, **k: None)
    monkeypatch.setattr(qsa_single, "validate_configuration", lambda *a, **k: None)
    monkeypatch.setattr(
        qsa_runtime.QSAHiSparseRuntime,
        "_allocate_decode_workspace",
        lambda self: None,
    )
    monkeypatch.setattr(
        torch.cuda, "Stream", lambda *a, **k: SimpleNamespace(synchronize=lambda: None)
    )
    real_empty = torch.empty

    def capped_empty(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        kwargs.pop("device", None)
        return real_empty((1,), dtype=kwargs.get("dtype", torch.uint8))

    # The pinned host slabs are multi-gigabyte on the real geometry; the rank
    # wiring under test does not depend on their contents.
    monkeypatch.setattr(qsa_runtime.torch, "empty", capped_empty)
    monkeypatch.setattr(qsa_single.torch, "empty", capped_empty)
    return tmp_path


def test_helper_reads_the_published_rank(_published_rank):
    for rank in (0, 1):
        _published_rank(rank)
        assert qsa_config.parallel_tp_rank() == rank


def test_runtime_init_takes_published_rank(_init_env, _published_rank):
    _published_rank(1)
    runner = _fake_runner(_fake_pool())
    runtime = qsa_runtime.QSAHiSparseRuntime.__new__(qsa_runtime.QSAHiSparseRuntime)
    runtime.__init__(runner, "p2-offload")

    assert runtime.rank == 1
    assert runtime.path is not None and runtime.path.name == "rank-1.jsonl"


def test_runtime_init_ignores_a_stale_runner_tp_rank(_init_env, _published_rank):
    _published_rank(1)
    runner = _fake_runner(_fake_pool(), stale_rank=0)
    runtime = qsa_runtime.QSAHiSparseRuntime.__new__(qsa_runtime.QSAHiSparseRuntime)
    runtime.__init__(runner, "p2-offload")

    assert runner.tp_rank == 0  # the old attribute is present but not consulted
    assert runtime.rank == 1


@pytest.mark.parametrize("stale_rank", [None, 0])
def test_single_request_init_takes_published_rank(
    _init_env, _published_rank, stale_rank
):
    _published_rank(1)
    runner = _fake_runner(_fake_pool(), stale_rank=stale_rank)
    if stale_rank is None:
        assert not hasattr(runner, "tp_rank")  # the production condition
    else:
        assert runner.tp_rank == 0  # the pre-migration attribute

    runtime = qsa_single.QSAHiSparseSingleRequest.__new__(
        qsa_single.QSAHiSparseSingleRequest
    )
    runtime.__init__(runner, "offload")

    assert runtime.rank == 1
    assert runtime.path is not None and runtime.path.name == "rank-1.jsonl"
