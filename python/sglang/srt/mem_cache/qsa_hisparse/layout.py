"""C4 byte layout shared by the single- and multi-request runtimes.

One record stores four K rows followed by four V rows: 8 x 256 FP8 bytes.
Logical token/index-K addresses remain in SGLang's ordinary paged allocator.
"""

import torch


def pack_c4(k, v):
    """Pack complete raw byte rows; also used by the CPU layout check."""
    if k.shape != v.shape or k.shape[-1] != 256 or k.numel() % 1024:
        raise ValueError("complete C4 K/V rows required")
    return torch.cat((k.reshape(-1, 1024), v.reshape(-1, 1024)), dim=1)


def unpack_index(device):
    rows = torch.arange(2048, device=device, dtype=torch.int64)
    k = (rows // 4) * 8 + rows % 4
    return torch.cat((k, k + 4))


def stage_short_prefix(hot, tokens, records):
    """HiSparse's <=HOT fast path assumes a fully resident ordered prefix."""
    count = records.shape[0]
    if count <= 2048:
        hot[:count].copy_(records, non_blocking=True)
        tokens[0, :count] = torch.arange(count, dtype=torch.int32, device=hot.device)
