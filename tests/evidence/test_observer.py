"""W8 checkpoint observer on the real runtime methods, on CPU.

Expected bytes are built from position/layer arithmetic, independently of the
runtime's capture and restore code and of the observer's read-back.
"""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import sglang_qsa_hisparse.hisparse.runtime as runtime_module
from sglang.srt.mem_cache.ple_state_pool import NGramPool, ShortConvPool
from sglang_qsa_hisparse.hisparse.prefix import HostPrefixCache, token_bytes
from sglang_qsa_hisparse.hisparse.slots import QSAHiSparseSlots

Runtime = runtime_module.QSAHiSparseRuntime
REPO = Path(__file__).resolve().parents[2]
SITE = REPO / "tools" / "evidence" / "site"
LAYERS = 2
NAMESPACE = ("fixture-model", "text", "TP2")


def make_runtime():
    a = Runtime.__new__(Runtime)
    a.mode, a.device, a.strict, a.path, a.rank = "p2-offload", "cpu", True, None, 0
    a.capacity, a.max_requests, a.layer_ids = 256, 2, [3, 7]
    a.slots = QSAHiSparseSlots(a.capacity, 64, a.max_requests)
    rows = a.slots.raw_pool_size + 64
    a.full = SimpleNamespace(
        k_buffer=[torch.zeros((rows, 1, 256), dtype=torch.uint8) for _ in range(LAYERS)],
        v_buffer=[torch.zeros((rows, 1, 256), dtype=torch.uint8) for _ in range(LAYERS)],
    )
    a.pool = SimpleNamespace(
        index_state_dtype=torch.bfloat16,
        qsa_key_state_buffer_pool=[
            torch.zeros((12, 1, 16), dtype=torch.bfloat16) for _ in range(LAYERS)
        ],
        qsa_rope_position_buffer=torch.zeros((12, 3), dtype=torch.int64),
        qsa_compressed_k_buffer_pool=[
            torch.zeros((144, 1, 16), dtype=torch.bfloat16) for _ in range(LAYERS)
        ],
    )
    short = ShortConvPool.__new__(ShortConvPool)
    short.conv_state = torch.zeros((2, 3, 2, 3), dtype=torch.bfloat16)
    short.layer_map = {3: 0, 7: 1}
    ngram = NGramPool.__new__(NGramPool)
    ngram.context = torch.zeros((3, 3), dtype=torch.int64)
    mamba = SimpleNamespace(
        mamba_cache=SimpleNamespace(
            conv=[torch.zeros((3, 3, 2, 3), dtype=torch.bfloat16)],
            temporal=torch.zeros((3, 3, 2, 2), dtype=torch.float32),
        ),
        _slot_siblings=(short, ngram),
    )
    a.req_pool = SimpleNamespace(
        req_generation=torch.ones(3, dtype=torch.int64), mamba_pool=mamba
    )
    a.req_table = torch.zeros((3, a.capacity), dtype=torch.int32)
    a.copy_stream, a.producer_stream = Mock(), Mock()
    a.host_slabs = None
    a.requests, a.pending_releases = {}, []
    a.prefix_cache = HostPrefixCache(4_000_000)
    return a


def request(rid, row, first_slot):
    """Request in ``row`` whose logical pages start at ``first_slot``, page order reversed."""
    slots = (torch.arange(128) + first_slot).reshape(2, 64).flip(0).flatten()
    return SimpleNamespace(rid=rid, kv=SimpleNamespace(req_pool_idx=row, mamba_pool_idx=row)), slots


def reference():
    """Independent expected state of the 128-token source request."""
    t = torch.arange(128 * 256).reshape(128, 1, 256)
    k = [((t + li * 31) % 256).to(torch.uint8) for li in range(LAYERS)]
    v = [((t * 3 + li * 19) % 256).to(torch.uint8) for li in range(LAYERS)]
    index = [
        (torch.arange(32 * 16).reshape(32, 1, 16) + li * 61).to(torch.bfloat16)
        for li in range(LAYERS)
    ]
    return SimpleNamespace(
        k=k,
        v=v,
        index=index,
        pending=[torch.full((4, 1, 16), li + 200, dtype=torch.bfloat16) for li in range(LAYERS)],
        rope=torch.arange(12).reshape(4, 3) + 1000,
        conv=torch.arange(18).reshape(3, 2, 3).to(torch.bfloat16),
        temporal=torch.arange(12).reshape(3, 2, 2) / 8,
        short=(torch.arange(12).reshape(2, 2, 3) + 50).to(torch.bfloat16),
        ngram=torch.tensor([91, 92, 93]),
    )


def fill_source(a, req, slots, ref):
    row, mid = req.kv.req_pool_idx, req.kv.mamba_pool_idx
    a.req_table[row, :128] = slots.int()
    for li in range(LAYERS):
        a.full.k_buffer[li][64:192] = ref.k[li]
        a.full.v_buffer[li][64:192] = ref.v[li]
        a.pool.qsa_compressed_k_buffer_pool[li][slots[::4] // 4] = ref.index[li]
        a.pool.qsa_key_state_buffer_pool[li][row * 4 : row * 4 + 4] = ref.pending[li]
    a.pool.qsa_rope_position_buffer[row * 4 : row * 4 + 4] = ref.rope
    cache = a.req_pool.mamba_pool.mamba_cache
    cache.conv[0][:, mid] = ref.conv
    cache.temporal[:, mid] = ref.temporal
    short, ngram = a.req_pool.mamba_pool._slot_siblings
    short.conv_state[:, mid] = ref.short
    ngram.context[mid] = ref.ngram


def poison(a):
    for t in a.full.k_buffer + a.full.v_buffer:
        t.fill_(255)
    for t in a.pool.qsa_compressed_k_buffer_pool + a.pool.qsa_key_state_buffer_pool:
        t.fill_(-7)
    a.pool.qsa_rope_position_buffer.fill_(-9)
    cache = a.req_pool.mamba_pool.mamba_cache
    cache.conv[0].fill_(-1)
    cache.temporal.fill_(-2)
    short, ngram = a.req_pool.mamba_pool._slot_siblings
    short.conv_state.fill_(-3)
    ngram.context.fill_(-4)


def expected_state(obs, ref, length):
    segments = []
    for start in range(0, length, 64):
        stop = start + 64
        raw = torch.stack(
            [torch.stack([ref.k[li][start:stop], ref.v[li][start:stop]]) for li in range(LAYERS)]
        )
        index = torch.stack([ref.index[li][start // 4 : stop // 4] for li in range(LAYERS)])
        segments.append([start, stop, obs.digest(raw), obs.digest(index)])
    return {
        "tokens": length,
        "token_sha256": hashlib.sha256(token_bytes(range(length))).hexdigest(),
        "segments": segments,
        "pending": [obs.digest(t) for t in ref.pending],
        "rope": obs.digest(ref.rope),
        "mamba": [
            [obs.digest(ref.conv.unsqueeze(1))],
            obs.digest(ref.temporal.unsqueeze(1)),
            [obs.digest(ref.short.unsqueeze(1)), obs.digest(ref.ngram.unsqueeze(0))],
        ],
    }


def state(record):
    return {k: record[k] for k in ("tokens", "token_sha256", "segments", "pending", "rope", "mamba")}


@pytest.fixture
def observed(evidence, monkeypatch, tmp_path):
    """Observer module with the runtime's original methods restored afterwards."""
    monkeypatch.setattr(torch.cuda, "current_stream", Mock())
    monkeypatch.setattr(Runtime, "capture_prefix", Runtime.capture_prefix)
    monkeypatch.setattr(Runtime, "restore_prefix", Runtime.restore_prefix)
    monkeypatch.setenv("QSA_EVIDENCE_OBSERVER_DIR", str(tmp_path / "observer"))
    return evidence("observer")


def capture_source(a):
    """Publish checkpoints at 64 and 128 tokens; the second shares the first segment."""
    ref = reference()
    req, slots = request("source", 1, 64)
    fill_source(a, req, slots, ref)
    state = a._acquire_request(1, "source")
    snapshots = []
    for length in (64, 128):
        state.seq_len = length
        reservation, snapshot = a.capture_prefix(req, NAMESPACE, token_bytes(range(length)))
        entry = a.prefix_cache.publish(reservation, snapshot, completed=True)
        a.note_prefix_basis(req, snapshot, entry)
        snapshots.append(snapshot)
    return ref, snapshots


def restore_warm(b, snapshot):
    req, slots = request("warm", 2, 320)
    b.req_table[2, :128] = slots.int()
    poison(b)
    b.restore_prefix(req, snapshot)


def records(tmp_path):
    lines = (tmp_path / "observer" / "rank-0.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines]


def test_records_reference_bytes_and_exact_restore(observed, monkeypatch, tmp_path):
    raw_digests = []
    digest = observed.digest

    def counting(tensor):
        if tensor.dim() == 5:
            raw_digests.append(tensor.shape)
        return digest(tensor)

    monkeypatch.setattr(observed, "digest", counting)
    observed.wrap(runtime_module)
    a = make_runtime()
    ref, snapshots = capture_source(a)
    # The 128-token capture reuses the shared 64-token segment's digests.
    assert len(raw_digests) == 2
    assert snapshots[1].segments[0] is snapshots[0].segments[0]

    restore_warm(make_runtime(), snapshots[1])
    first, second, restored = records(tmp_path)
    assert [r["event"] for r in (first, second, restored)] == ["capture", "capture", "restore"]
    assert [r["checkpoint"] for r in (first, second, restored)] == [0, 1, 1]
    assert [r["rid"] for r in (first, second, restored)] == ["source", "source", "warm"]
    assert state(first) == expected_state(observed, ref, 64)
    assert state(second) == expected_state(observed, ref, 128)
    assert state(restored) == state(second)


def test_restore_read_back_detects_one_corrupted_byte(observed, evidence, monkeypatch, tmp_path):
    original = Runtime.restore_prefix

    def corrupting(self, req, snapshot):
        original(self, req, snapshot)
        self.full.v_buffer[1][64 + 70, 0, 9] ^= 1

    monkeypatch.setattr(Runtime, "restore_prefix", corrupting)
    observed.wrap(runtime_module)
    _, snapshots = capture_source(make_runtime())
    b = make_runtime()
    b.strict = False  # The runtime's own exact check would raise first.
    restore_warm(b, snapshots[1])
    _, second, restored = records(tmp_path)
    assert restored["segments"][0] == second["segments"][0]
    assert restored["segments"][1][2] != second["segments"][1][2]
    assert {k: v for k, v in state(restored).items() if k != "segments"} == {
        k: v for k, v in state(second).items() if k != "segments"
    }
    message = evidence("compare").observer_consistency(tmp_path / "observer" / "rank-0.jsonl")
    assert "record 2" in message and "checkpoint 1" in message


def test_observer_failure_propagates_and_releases_the_reservation(observed, tmp_path):
    observed.wrap(runtime_module)
    (tmp_path / "observer").mkdir()
    (tmp_path / "observer" / "rank-0.jsonl").write_text("stale\n")
    a = make_runtime()
    req, slots = request("source", 1, 64)
    fill_source(a, req, slots, reference())
    a._acquire_request(1, "source").seq_len = 64
    with pytest.raises(RuntimeError, match="already exists"):
        a.capture_prefix(req, NAMESPACE, token_bytes(range(64)))
    assert a.prefix_cache.pending_bytes == a.prefix_cache.pending_entries == 0


def test_digest_covers_dtype_shape_and_bytes(evidence):
    obs = evidence("observer")
    x = torch.tensor([[1, 2]], dtype=torch.int16)
    expected = hashlib.sha256(b"torch.int16:(1, 2):" + x.numpy().tobytes()).hexdigest()
    assert obs.digest(x) == expected
    assert obs.digest(x.view(torch.uint8)) != expected
    assert obs.digest(x.reshape(2, 1)) != expected


def run_python(code, pythonpath, observer_dir=None):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(map(str, pythonpath)))
    env.pop("QSA_EVIDENCE_OBSERVER_DIR", None)
    if observer_dir is not None:
        env["QSA_EVIDENCE_OBSERVER_DIR"] = str(observer_dir)
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.split()


def wrapped_flags(module):
    return (
        f"import {module} as m; c = m.QSAHiSparseRuntime; "
        "print(*(getattr(f, '__qsa_evidence__', False) "
        "for f in (c.capture_prefix, c.restore_prefix)))"
    )


def test_site_hook_wraps_the_runtime_in_fork_and_plugin_trees(tmp_path):
    fork = tmp_path / "fork"
    package = fork / "sglang" / "srt" / "mem_cache" / "qsa_hisparse"
    package.mkdir(parents=True)
    for directory in (package, *package.parents):
        if directory == fork:
            break
        (directory / "__init__.py").write_text("")
    (package / "runtime.py").write_text(
        "class QSAHiSparseRuntime:\n"
        "    def capture_prefix(self, req, namespace, tokens): pass\n"
        "    def restore_prefix(self, req, snapshot): pass\n"
    )
    chained = tmp_path / "chained"
    chained.mkdir()
    (chained / "sitecustomize.py").write_text("print('chained')\n")
    out = tmp_path / "observer"
    fork_module = "sglang.srt.mem_cache.qsa_hisparse.runtime"

    assert run_python(wrapped_flags(fork_module), [SITE, chained, fork], out) == [
        "chained",
        "True",
        "True",
    ]
    assert run_python(wrapped_flags(fork_module), [SITE, chained, fork]) == [
        "chained",
        "False",
        "False",
    ]
    plugin_path = [SITE, *os.environ["PYTHONPATH"].split(os.pathsep)]
    assert run_python(
        wrapped_flags("sglang_qsa_hisparse.hisparse.runtime"), plugin_path, out
    ) == ["True", "True"]
