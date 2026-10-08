"""Image identity for exact host prefix reuse (Track I contract).

Fixed by the orchestrator before Track I; I1/I2 implement against it.
A host prefix of length ``L`` (a multiple of 64) may be reused for a request
only if, besides equal token bytes, ``identity.key_at(L)`` equals the key the
snapshot recorded at capture. The key covers exactly what determines the
prefix's KV, recurrent and PLE state:

- every image whose token span starts before ``L`` (including one that
  straddles ``L``, with its full span and grid), by full artifact key
  (content and preprocessing), ordinal, span and grid;
- the M-RoPE positions of tokens ``[0, L)``, through a page-cumulative digest.

Images starting at or after ``L`` are not part of the key, so requests that
differ only after ``L`` share the prefix. There is no whole-prompt digest.
"""

import hashlib
from dataclasses import dataclass

PAGE = 64


@dataclass(frozen=True)
class ImageRecord:
    artifact_key: str  # Full artifact identity (content + processor + kwargs).
    order: int  # Ordinal among the request's image items.
    start: int  # First token offset of the image span in input_ids.
    stop: int  # One past the last token offset.
    grid: tuple[int, int, int]  # image_grid_thw (t, h, w).


@dataclass(frozen=True)
class ImagePrefixIdentity:
    records: tuple[ImageRecord, ...]  # Sorted by start.
    page_digests: tuple[bytes, ...]  # page_digests[k-1] covers tokens [0, 64k).

    def records_at(self, length: int) -> tuple[ImageRecord, ...]:
        return tuple(r for r in self.records if r.start < length)

    def digest_at(self, length: int) -> bytes:
        if length % PAGE or length <= 0 or length // PAGE > len(self.page_digests):
            raise ValueError(f"No M-RoPE digest for prefix length {length}")
        return self.page_digests[length // PAGE - 1]

    def key_at(self, length: int) -> tuple:
        return self.records_at(length), self.digest_at(length)


def page_digests(mrope_positions, length: int) -> tuple[bytes, ...]:
    """Cumulative SHA-256 over int64 M-RoPE positions ``[3, length]``, per page.

    ``d[k] = H(d[k-1] || positions[:, 64(k-1):64k])``; only complete pages.
    """
    positions = mrope_positions[:, : length - length % PAGE]
    digests, previous = [], b""
    for page in range(positions.shape[1] // PAGE):
        block = positions[:, page * PAGE : (page + 1) * PAGE]
        data = block.contiguous().to("cpu").to(dtype=_int64()).numpy().tobytes()
        previous = hashlib.sha256(previous + data).digest()
        digests.append(previous)
    return tuple(digests)


def _int64():
    import torch

    return torch.int64
