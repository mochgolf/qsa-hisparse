"""Fixed-capacity byte movement for QSA HiSparse decode graphs.

Slots own persistent storage; rows only route a forward. No host transfers or
lease lifecycle operations belong in these kernels.
"""

import triton as tr
import triton.language as tl


@tr.jit
def close_c4(K, V, Hot, Tokens, Slots, Lengths, Real,
             RING_START: tl.constexpr):
    row = tl.program_id(0)
    if row < tl.load(Real):
        seq = tl.load(Lengths + row)
        if seq % 4 == 0:
            slot = tl.load(Slots + row)
            i = tl.arange(0, 1024)
            ring = (RING_START + slot * 5 + 1) * 256 + i
            k, v = tl.load(K + ring), tl.load(V + ring)
            newest = (slot * 2112 + 2048) * 2048
            tl.store(Hot + newest + i, k)
            tl.store(Hot + newest + 1024 + i, v)
            block = seq // 4 - 1
            tl.store(Tokens + slot * 2112 + 2048, block)
            if block < 2048:
                dst = (slot * 2112 + block) * 2048
                tl.store(Hot + dst + i, k)
                tl.store(Hot + dst + 1024 + i, v)
                tl.store(Tokens + slot * 2112 + block, block)


@tr.jit
def finish_compact(Unpacked, K, V, Compact, Table, Raw, Slots, Lengths, Real,
                   CAPACITY: tl.constexpr, RING_START: tl.constexpr,
                   PLANE_STRIDE: tl.constexpr):
    row, chunk = tl.program_id(0), tl.program_id(1)
    if row < tl.load(Real):
        slot, seq = tl.load(Slots + row), tl.load(Lengths + row)
        tail = seq % 4
        byte = chunk * 1024 + tl.arange(0, 1024)
        member, dim = byte // 256, byte % 256
        selected = member < 2048
        pending = (member >= 2048) & (member < 2048 + tail)
        ring = (RING_START + slot * 5 + 1 + member - 2048) * 256 + dim
        for plane in tl.static_range(2):
            u = tl.load(Unpacked + (row * 2 + plane) * 2048 * 256 + byte,
                        selected, other=0)
            if plane == 0:
                t = tl.load(K + ring, pending, other=0)
            else:
                t = tl.load(V + ring, pending, other=0)
            dst = ((plane * PLANE_STRIDE + slot * 2052 + 1 + member) * 256 + dim)
            tl.store(Compact + dst, tl.where(selected, u, t), selected | pending)
        member_id = chunk * 4 + tl.arange(0, 4)
        valid = member_id < 2048 + tail
        logical = tl.load(Raw + row * 2051 + member_id, valid, other=0)
        tl.store(Table + slot * CAPACITY + logical,
                 slot * 2052 + 1 + member_id, valid)
