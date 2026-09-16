"""Immutable, bounded host checkpoints for QSA's private prefill lane.

No cached object owns logical KV pages, request rows, or CUDA tensors. Readers
pin checkpoints (including after reset) and writers reserve their entire new
host footprint before allocating. Segments shared by checkpoints count once.
"""

import hashlib
from array import array
from collections import OrderedDict
from dataclasses import dataclass, fields, is_dataclass
from threading import RLock

import torch


def token_bytes(tokens):
    return array("q", tokens).tobytes()


def tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif is_dataclass(value):
        for field in fields(value):
            yield from tensors(getattr(value, field.name))
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from tensors(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from tensors(item)


@dataclass(frozen=True)
class PrefixSegment:
    start: int
    stop: int
    raw: torch.Tensor  # [layers, K/V, tokens, 1, 256], exact FP8 bytes
    index: torch.Tensor  # [layers, tokens / 4, index heads, index dim]


@dataclass(frozen=True)
class PrefixSnapshot:
    namespace: tuple
    tokens: bytes
    segments: tuple
    pending: tuple
    rope: torch.Tensor
    mamba: tuple

    @property
    def length(self):
        return len(self.tokens) // 8

    @property
    def signature(self):
        return self.namespace, self.length, hashlib.sha256(self.tokens).hexdigest()


class PrefixReader:
    def __init__(self, cache, entry_id, snapshot):
        self.cache, self.entry_id, self.snapshot = cache, entry_id, snapshot

    def close(self):
        if self.snapshot is not None:
            self.cache._release(self.entry_id)
            self.snapshot = None


class PrefixReservation:
    def __init__(self, cache, size, epoch):
        self.cache, self.size, self.epoch = cache, size, epoch
        self.active = True

    def close(self):
        with self.cache.lock:
            if self.active:
                self.cache.pending_bytes -= self.size
                self.cache.pending_entries -= 1
                self.active = False
                reader = getattr(self, "basis_reader", None)
                if reader is not None:
                    reader.close()


class HostPrefixCache:
    def __init__(self, budget_bytes, max_entries=256, page_size=64):
        if budget_bytes <= 0 or max_entries <= 0 or page_size != 64:
            raise ValueError("QSA prefixes need a positive host budget and page64")
        self.budget_bytes, self.max_entries = budget_bytes, max_entries
        self.lock = RLock()
        self.entries = OrderedDict()
        self.retired = {}
        self.pending_bytes = self.pending_entries = self.epoch = self.next_id = 0
        self.hits = self.misses = self.reused_tokens = self.evictions = 0

    def _footprint(self, snapshots):
        storages, keys = {}, {}
        for snapshot in snapshots:
            keys[id(snapshot.tokens)] = len(snapshot.tokens)
            for tensor in tensors(snapshot):
                if tensor.device.type != "cpu":
                    raise ValueError("QSA prefix snapshots must be host resident")
                storage = tensor.untyped_storage()
                storages[(storage.data_ptr(), storage.nbytes())] = storage.nbytes()
        return sum(storages.values()) + sum(keys.values())

    @property
    def used_bytes(self):
        return self._footprint(
            [entry[0] for entry in self.entries.values()]
            + [entry[0] for entry in self.retired.values()]
        )

    def _evict_one(self):
        for entry_id, (_, readers) in self.entries.items():
            if readers == 0:
                del self.entries[entry_id]
                self.evictions += 1
                return True
        return False

    def acquire(self, namespace, tokens, limit=None, *, count=True):
        with self.lock:
            cap = len(tokens) // 8 if limit is None else limit
            best = None
            for entry_id, (snapshot, _) in self.entries.items():
                if (
                    snapshot.namespace == namespace
                    and snapshot.length <= cap
                    and tokens.startswith(snapshot.tokens)
                    and (best is None or snapshot.length > best[1].length)
                ):
                    best = entry_id, snapshot
            if best is None:
                if count:
                    self.misses += 1
                return None
            entry_id, snapshot = best
            self.entries[entry_id][1] += 1
            self.entries.move_to_end(entry_id)
            if count:
                self.hits += 1
            return PrefixReader(self, entry_id, snapshot)

    def _release(self, entry_id):
        with self.lock:
            table = self.entries if entry_id in self.entries else self.retired
            entry = table[entry_id]
            if entry[1] <= 0:
                raise RuntimeError("QSA prefix reader underflow")
            entry[1] -= 1
            if table is self.retired and entry[1] == 0:
                del table[entry_id]

    def reserve(self, size):
        with self.lock:
            if size <= 0 or size > self.budget_bytes:
                return None
            while (
                self.used_bytes + self.pending_bytes + size > self.budget_bytes
                or len(self.entries) + len(self.retired) + self.pending_entries
                >= self.max_entries
            ):
                if not self._evict_one():
                    return None
            self.pending_bytes += size
            self.pending_entries += 1
            return PrefixReservation(self, size, self.epoch)

    def publish(self, reservation, snapshot, *, completed):
        with self.lock:
            if reservation.cache is not self or not reservation.active:
                raise RuntimeError("QSA prefix publication has no reservation")
            try:
                if not completed:
                    raise RuntimeError("QSA prefix publication has incomplete writes")
                if reservation.epoch != self.epoch:
                    raise RuntimeError(
                        "QSA prefix publication belongs to a flushed epoch"
                    )
                if not snapshot.length or snapshot.length % 64:
                    raise ValueError("QSA prefix checkpoint must end on page64")
                stop = 0
                for segment in snapshot.segments:
                    if (
                        segment.start != stop
                        or segment.stop <= stop
                        or segment.stop % 64
                    ):
                        raise ValueError(
                            "QSA prefix segments have a gap or partial page"
                        )
                    if segment.raw.shape[2] != segment.stop - stop:
                        raise ValueError("QSA prefix raw segment length differs")
                    if segment.index.shape[1] != (segment.stop - stop) // 4:
                        raise ValueError("QSA prefix index segment length differs")
                    stop = segment.stop
                if stop != snapshot.length:
                    raise ValueError("QSA checkpoint is missing prefix pages")
                old = self.used_bytes
                all_snapshots = [entry[0] for entry in self.entries.values()] + [
                    entry[0] for entry in self.retired.values()
                ]
                new = self._footprint(all_snapshots + [snapshot]) - old
                if new > reservation.size:
                    raise RuntimeError(
                        "QSA prefix writer exceeded its host reservation"
                    )
                self.next_id += 1
                self.entries[self.next_id] = [snapshot, 0]
                return self.next_id
            finally:
                reservation.close()

    def reset(self):
        with self.lock:
            self.epoch += 1
            self.retired.update(
                (entry_id, entry)
                for entry_id, entry in self.entries.items()
                if entry[1]
            )
            self.entries.clear()

    def stats(self):
        with self.lock:
            return dict(
                prefix_cache_budget_bytes=self.budget_bytes,
                prefix_cache_host_bytes=self.used_bytes,
                prefix_cache_pending_bytes=self.pending_bytes,
                prefix_cache_pending_entries=self.pending_entries,
                prefix_cache_entries=len(self.entries),
                prefix_cache_readers=sum(e[1] for e in self.entries.values())
                + sum(e[1] for e in self.retired.values()),
                prefix_cache_hits=self.hits,
                prefix_cache_misses=self.misses,
                prefix_cache_reused_tokens=self.reused_tokens,
                prefix_cache_evictions=self.evictions,
                prefix_cache_epoch=self.epoch,
            )
