"""CPU byte checks for main's state-storage and typed-KV routing refactors.

The runtime remains the production implementation: these checks exercise its
capture/restore through the new request-table properties and its leased ring
views through a real upstream router. CUDA DMA and kernels are not exercised.
Expectations are fixed integer bytes and explicit slot positions; no numerical
tolerance or candidate-generated fixture is used.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.srt.mem_cache.kv.state_storage import StateStorage
from sglang.srt.mem_cache.memory_pool import (
    HybridLinearKVPool,
    HybridReqToTokenPool,
    MHATokenToKVPool,
    MambaPool,
)
from sglang.srt.mem_cache.ple_state_pool import NGramPool, ShortConvPool
from sglang_qsa_hisparse.hisparse.runtime import QSAHiSparseRuntime, _RequestCache
from sglang_qsa_hisparse.hisparse.slots import QSAHiSparseSlots


def test_prefix_state_bytes_use_the_request_tables_state_storage(monkeypatch):
    mamba = MambaPool.__new__(MambaPool)
    mamba.mamba_cache = MambaPool.State(
        conv=[torch.arange(6, dtype=torch.int16).reshape(1, 3, 2)],
        temporal=torch.arange(10, 16, dtype=torch.float32).reshape(1, 3, 2),
    )
    short = ShortConvPool.__new__(ShortConvPool)
    short.conv_state = torch.arange(20, 26, dtype=torch.bfloat16).reshape(1, 3, 2)
    ngram = NGramPool.__new__(NGramPool)
    ngram.context = torch.arange(30, 36, dtype=torch.int64).reshape(3, 2)
    mamba._slot_siblings = (short, ngram)
    table = HybridReqToTokenPool.__new__(HybridReqToTokenPool)
    table.state = StateStorage(
        sub_pool="mamba",
        pool=mamba,
        slots=None,
        layer_map={3: 0},
        short_conv_pool=short,
        ngram_pool=ngram,
    )
    runtime = QSAHiSparseRuntime.__new__(QSAHiSparseRuntime)
    runtime.req_pool, runtime.device = table, "cpu"
    assert table.mamba_pool is table.state.pool

    snapshot = runtime._capture_mamba(1)
    assert snapshot[0][0].tolist() == [[[2, 3]]]
    assert snapshot[1].tolist() == [[[12, 13]]]
    assert snapshot[2][0].tolist() == [[[22, 23]]]
    assert snapshot[2][1].tolist() == [[32, 33]]

    tensors = [mamba.mamba_cache.conv[0], mamba.mamba_cache.temporal, short.conv_state]
    for tensor in [*tensors, ngram.context]:
        tensor.fill_(-1)
    stream = Mock()
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: stream)
    runtime._restore_mamba(snapshot, 2)
    stream.synchronize.assert_called_once_with()
    assert [tensor[:, 2].tolist() for tensor in tensors] == [
        [[2, 3]], [[12, 13]], [[22, 23]]
    ]
    assert ngram.context[2].tolist() == [32, 33]
    assert all(torch.all(tensor[:, :2] == -1) for tensor in tensors)
    assert torch.all(ngram.context[:2] == -1)
    assert runtime._check_restored_mamba(snapshot, 2)
    ngram.context[2, 0] = 99
    assert not runtime._check_restored_mamba(snapshot, 2)


def test_typed_router_reads_the_runtime_request_ring_backing():
    slots = QSAHiSparseSlots(128, 64, 2)
    raw = MHATokenToKVPool(
        size=slots.raw_pool_size,
        page_size=64,
        dtype=torch.float8_e4m3fn,
        head_num=1,
        head_dim=256,
        layer_num=1,
        device="cpu",
        enable_memory_saver=False,
        enable_alt_stream=False,
    )
    pool = HybridLinearKVPool.__new__(HybridLinearKVPool)
    pool.full_kv_pool = raw
    pool.full_attention_layer_id_mapping = {3: 0}
    pool.layer_transfer_counter, pool.start_layer = None, 0
    routed_k, routed_v = pool.layer_router.read(3).kv()
    # The MHA reader views its uint8 FP8 storage as float8; it aliases the bytes.
    assert routed_k.data_ptr() == raw.k_buffer[0].data_ptr()
    assert routed_v.data_ptr() == raw.v_buffer[0].data_ptr()
    adapter = SimpleNamespace(
        pool=pool, device="cpu", strict=True, capacity=128,
        copy_stream=None, slots=slots, full=raw, host_slabs=None,
    )
    lease = slots.acquire(1, 1, "route-test")
    state = _RequestCache(adapter, lease)
    state.full.k_buffer[0].view(torch.uint8)[1:5].fill_(7)
    state.full.v_buffer[0].view(torch.uint8)[1:5].fill_(19)
    # Staging is [64, 192), the first lease's ring is [192, 197).
    assert torch.all(routed_k.view(torch.uint8)[193:197] == 7)
    assert torch.all(routed_v.view(torch.uint8)[193:197] == 19)
