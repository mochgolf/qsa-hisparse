"""Neutral checkpoint-byte observer for fork (F) and plugin (P) GPU runs.

Enable it identically in both arms: put ``tools/evidence/site`` first on the
server's ``PYTHONPATH`` and set ``QSA_EVIDENCE_OBSERVER_DIR`` to a new absolute
directory, ``<arm dir>/observer`` for ``compare.py``. Spawned TP schedulers
inherit both, and the site hook wraps ``QSAHiSparseRuntime.capture_prefix``
and ``restore_prefix`` in whichever runtime module the process imports:

    sglang.srt.mem_cache.qsa_hisparse.runtime   fork
    sglang_qsa_hisparse.hisparse.runtime        plugin (moved verbatim)

Each TP rank appends JSON lines to ``<dir>/rank-<rank>.jsonl``:

- ``capture``: after ``capture_prefix`` returns a checkpoint, the digests of
  its host tensors: per segment raw FP8 K/V ``[layers, K/V, tokens, 1, 256]``
  and compressed index; the pending C4 ring per layer; the RoPE positions;
  every recurrent and PLE slot tensor (``mamba``, nested as captured).
- ``restore``: after ``restore_prefix`` returns, the same digests of the
  device state read back from the request's staging rows, index pages, ring
  rows and Mamba slot, at the locations ``capture_prefix`` reads.
  ``checkpoint`` is the per-rank ordinal of the capture that was restored.

A digest is SHA-256 over ``"<dtype>:<shape>:"`` followed by the raw bytes.
Records hold no times, pointers or module names, so equal behavior gives
byte-identical files in F and P (``compare.py``).

Synchronization added (the observer never writes a tensor and never calls
synchronize):
- capture: none. ``capture_prefix`` already synchronized the producer and
  current streams and copied into pageable host tensors; hashing reads host
  memory only. Segments shared with an ancestor checkpoint are hashed once.
- restore: after ``restore_prefix`` returns, blocking device-to-host copies
  of exactly the restored regions on the current stream (each waits for that
  stream), plus small index-gather kernels and their transient device
  buffers. ``restore_prefix`` has already synchronized that stream at its end
  (``_restore_mamba``), so the copies add host latency but no new ordering
  between device work; other streams are untouched.
The cost is host time on the scheduler thread: hashing newly captured bytes,
and reading back and hashing every restored checkpoint before its forward.
"""

import functools
import hashlib
import json
import os
import sys
import weakref
from pathlib import Path

ENV = "QSA_EVIDENCE_OBSERVER_DIR"
RUNTIME_MODULES = (
    "sglang.srt.mem_cache.qsa_hisparse.runtime",
    "sglang_qsa_hisparse.hisparse.runtime",
)
_MARK = "__qsa_evidence__"
_OBSERVERS = {}  # TP rank -> _Observer; one runtime per rank per process


def digest(tensor):
    import torch

    if tensor.device.type != "cpu":
        raise ValueError("observer digests host tensors only")
    tensor = tensor.detach().contiguous()
    h = hashlib.sha256(f"{tensor.dtype}:{tuple(tensor.shape)}:".encode())
    h.update(tensor.reshape(-1).view(torch.uint8).numpy())
    return h.hexdigest()


def digests(value):
    if isinstance(value, (list, tuple)):
        return [digests(item) for item in value]
    return digest(value)


def _remembered(table, obj):
    entry = table.get(id(obj))
    return entry[1] if entry is not None and entry[0]() is obj else None


def _remember(table, obj, value):
    table[id(obj)] = (weakref.ref(obj), value)


class _Observer:
    def __init__(self, directory, rank):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"rank-{rank}.jsonl"
        if self.path.exists():
            raise RuntimeError(f"observer output already exists: {self.path}")
        self.rank = rank
        self.captures = 0
        self.checkpoints = {}  # id(snapshot) -> (weakref, capture ordinal)
        self.segments = {}  # id(segment) -> (weakref, [raw, index] digests)

    def captured(self, req, snapshot):
        ordinal = self.captures
        self.captures += 1
        _remember(self.checkpoints, snapshot, ordinal)
        segments = []
        for segment in snapshot.segments:
            pair = _remembered(self.segments, segment)
            if pair is None:
                pair = [digest(segment.raw), digest(segment.index)]
                _remember(self.segments, segment, pair)
            segments.append([segment.start, segment.stop, *pair])
        self._write(
            "capture",
            ordinal,
            req,
            snapshot,
            segments,
            snapshot.pending,
            snapshot.rope,
            snapshot.mamba,
        )

    def restored(self, runtime, req, snapshot):
        segments, pending, rope, mamba = _read_back(runtime, req, snapshot)
        ordinal = _remembered(self.checkpoints, snapshot)
        self._write("restore", ordinal, req, snapshot, segments, pending, rope, mamba)

    def _write(self, event, ordinal, req, snapshot, segments, pending, rope, mamba):
        row = {
            "event": event,
            "rank": self.rank,
            "checkpoint": ordinal,
            "rid": req.rid,
            "tokens": snapshot.length,
            "token_sha256": hashlib.sha256(snapshot.tokens).hexdigest(),
            "segments": segments,
            "pending": digests(pending),
            "rope": digest(rope),
            "mamba": digests(mamba),
        }
        with self.path.open("a") as stream:
            stream.write(json.dumps(row) + "\n")


def _read_back(runtime, req, snapshot):
    """Copy the restored state to the host the way capture_prefix reads it."""
    import torch

    idx = req.kv.req_pool_idx
    lease = runtime.requests[idx].lease
    pool = runtime.pool
    layers = len(runtime.layer_ids)
    shape = pool.qsa_compressed_k_buffer_pool[0].shape[1:]
    segments = []
    for segment in snapshot.segments:
        start, stop = segment.start, segment.stop
        raw = torch.empty((layers, 2, stop - start, 1, 256), dtype=torch.uint8)
        index = torch.empty(
            (layers, (stop - start) // 4, *shape), dtype=pool.index_state_dtype
        )
        physical = runtime.slots.staging_slice(lease, start, stop)
        logical = runtime.req_table[idx, start:stop:4].long() // 4
        for li in range(layers):
            raw[li, 0].copy_(runtime.full.k_buffer[li][physical].view(torch.uint8))
            raw[li, 1].copy_(runtime.full.v_buffer[li][physical].view(torch.uint8))
            index[li].copy_(pool.qsa_compressed_k_buffer_pool[li][logical])
        segments.append([start, stop, digest(raw), digest(index)])
    ring = slice(idx * 4, (idx + 1) * 4)
    pending = [t[ring].to("cpu", copy=True) for t in pool.qsa_key_state_buffer_pool]
    rope = pool.qsa_rope_position_buffer[ring].to("cpu", copy=True)
    mamba = runtime._capture_mamba(int(req.kv.mamba_pool_idx))
    return segments, pending, rope, mamba


def _observer(runtime):
    observer = _OBSERVERS.get(runtime.rank)
    if observer is None:
        observer = _OBSERVERS[runtime.rank] = _Observer(os.environ[ENV], runtime.rank)
    return observer


def wrap(module):
    """Wrap the runtime class of ``module`` in place; idempotent."""
    cls = module.QSAHiSparseRuntime
    capture, restore = cls.capture_prefix, cls.restore_prefix
    if getattr(capture, _MARK, False):
        return

    @functools.wraps(capture)
    def capture_prefix(self, req, namespace, tokens):
        result = capture(self, req, namespace, tokens)
        if result is not None:
            try:
                _observer(self).captured(req, result[1])
            except BaseException:
                result[0].close()  # Release the reservation the caller never sees.
                raise
        return result

    @functools.wraps(restore)
    def restore_prefix(self, req, snapshot):
        result = restore(self, req, snapshot)
        _observer(self).restored(self, req, snapshot)
        return result

    setattr(capture_prefix, _MARK, True)
    setattr(restore_prefix, _MARK, True)
    cls.capture_prefix = capture_prefix
    cls.restore_prefix = restore_prefix


def install():
    """Wrap each runtime module when it is imported; called by the site hook."""
    import importlib.abc
    import importlib.machinery

    if not Path(os.environ[ENV]).is_absolute():
        raise ValueError(f"{ENV} must be an absolute directory")

    class Loader(importlib.abc.Loader):
        def __init__(self, original):
            self.original = original

        def __getattr__(self, name):
            return getattr(self.original, name)

        def create_module(self, spec):
            return self.original.create_module(spec)

        def exec_module(self, module):
            self.original.exec_module(module)
            wrap(module)

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname not in RUNTIME_MODULES:
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is not None and spec.loader is not None:
                spec.loader = Loader(spec.loader)
            return spec

    sys.meta_path.insert(0, Finder())
    for name in RUNTIME_MODULES:
        if name in sys.modules:
            wrap(sys.modules[name])
