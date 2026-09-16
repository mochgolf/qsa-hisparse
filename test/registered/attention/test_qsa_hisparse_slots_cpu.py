"""Scaled CPU proof of physical leases and real QSA/paged-pool separation.

This does not execute a CUDA writer, DMA, TP convergence, or a service batch.
"""

from types import SimpleNamespace
import unittest

import torch

from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.mem_cache.qsa_hisparse_slots import QSAHiSparseSlots
from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool


class Event:
    def __init__(self, ready=True, on_wait=lambda: None):
        self.ready, self.on_wait = ready, on_wait

    def query(self):
        return self.ready

    def synchronize(self):
        self.on_wait()
        self.ready = True


class TestQSAHiSparseSlots(unittest.TestCase):
    def test_two_requests_staging_reuse_and_release(self):
        slots = QSAHiSparseSlots(128, 64, 2)
        raw = MHATokenToKVPool(
            size=slots.raw_pool_size, page_size=64, dtype=torch.float8_e4m3fn,
            head_num=1, head_dim=256, layer_num=1, device="cpu",
            enable_memory_saver=False, enable_alt_stream=False,
        )
        pool = QSATokenToKVPool(
            size=256, page_size=64, dtype=torch.float8_e4m3fn, head_num=1,
            head_dim=256, full_attention_layer_ids=[3], device="cpu",
            mamba_pool=SimpleNamespace(), qsa_index_kv_heads=1,
            qsa_index_head_dim=128, qsa_compress_ratio=4, qsa_token_topk=2048,
            num_request_slots=3, full_kv_pool=raw,
        )
        logical = PagedTokenToKVPoolAllocator(256, 64, pool.dtype, "cpu", pool, False)
        reqs = ReqToTokenPool(2, 128, "cpu", False)
        self.assertEqual(pool.size, 256)
        self.assertEqual(pool.qsa_compressed_capacity, 80)
        self.assertIs(pool.full_kv_pool, raw)
        self.assertEqual(raw.k_buffer[0].shape[0], 128 + 64 + 10)
        raw_ptrs = (raw.k_buffer[0].data_ptr(), raw.v_buffer[0].data_ptr())

        def claim(rid):
            idx = reqs.alloc_rows(1)[0]
            lease = slots.acquire(idx, int(reqs.req_generation[idx]), rid)
            rows = logical.alloc(128)
            reqs.req_to_token[idx] = rows.to(torch.int32)
            return lease, rows

        a, ar = claim("A")
        with self.assertRaisesRegex(RuntimeError, "only one"):
            slots.acquire(1, 1, "too-early")
        for plane, byte in ((raw.k_buffer[0], 7), (raw.v_buffer[0], 19)):
            plane.view(torch.uint8)[slots.staging_slice(a, 0, 128)].fill_(byte)
        akeys = torch.full((32, 1, 128), 11, dtype=torch.bfloat16)
        pool.set_qsa_compressed_k_buffer(3, ar[3::4] // 4, akeys)
        # CPU copy models ownership of completed host bytes, not DMA correctness.
        host_a = [p.view(torch.uint8)[slots.staging_slice(a, 0, 128)].clone()
                  for p in (raw.k_buffer[0], raw.v_buffer[0])]
        slots.begin_handoff(a)
        with self.assertRaisesRegex(RuntimeError, "before all layer copies"):
            slots.finish_handoff(a, Event(False))
        self.assertEqual(slots.snapshot()["staging_available_tokens"], 0)
        slots.finish_handoff(a, Event())
        self.assertEqual(logical.available_size(), 128)
        with self.assertRaises(RuntimeError):
            slots.ring_write_location(a, 65)  # local ready is not TP admission
        slots.admit_decode(a)
        b, br = claim("B")
        self.assertFalse(torch.isin(ar, br).any())
        for plane, byte in ((raw.k_buffer[0], 31), (raw.v_buffer[0], 43)):
            plane.view(torch.uint8)[slots.staging_slice(b, 0, 128)].fill_(byte)
        bkeys = torch.full((32, 1, 128), 29, dtype=torch.bfloat16)
        pool.set_qsa_compressed_k_buffer(3, br[3::4] // 4, bkeys)
        self.assertTrue(torch.equal(pool.get_qsa_compressed_k_buffer(3)[ar[3::4] // 4], akeys))
        self.assertTrue(torch.all(host_a[0] == 7) and torch.all(host_a[1] == 19))
        # Both prefix reads and writes use the same fixed-position helper.
        self.assertTrue(torch.all(raw.v_buffer[0].view(torch.uint8)[slots.staging_slice(b, 0, 65)] == 43))
        with self.assertRaises(IndexError):
            raw.k_buffer[0].view(torch.uint8).index_select(0, br)
        with self.assertRaises(RuntimeError):
            slots.staging_slice(a, 0, 64)
        with self.assertRaises(ValueError):
            slots.staging_slice(b, 0, 129)
        slots.begin_handoff(b)
        slots.finish_handoff(b, Event())
        slots.admit_decode(b)
        with self.assertRaisesRegex(RuntimeError, "capacity exhausted"):
            slots.acquire(3, 1, "overflow")

        # Reordered batch rows and independent tail phases retain physical identity.
        for seq_a, seq_b in ((65, 66), (67, 68), (69, 70), (71, 72)):
            wa, wb = slots.ring_write_location(a, seq_a), slots.ring_write_location(b, seq_b)
            raw.k_buffer[0].view(torch.uint8)[wa].fill_(seq_a)
            raw.k_buffer[0].view(torch.uint8)[wb].fill_(seq_b + 64)
            self.assertNotEqual(wa, wb)
            for lease, seq, value in ((b, seq_b, seq_b + 64), (a, seq_a, seq_a)):
                self.assertTrue(torch.all(raw.k_buffer[0].view(torch.uint8)[slots.ring_write_location(lease, seq)] == value))
            self.assertEqual(logical.available_size(), 0)
        rb = slots.ring_slice(b)
        b_ring = raw.k_buffer[0].view(torch.uint8)[rb].clone()
        with self.assertRaises(RuntimeError):
            slots.commit_release(a)
        waits = []

        def waited(name):
            self.assertIn(a.req_pool_idx, slots.active)
            self.assertEqual(slots.snapshot()["lease_free"], 0)
            waits.append(name)

        slots.drain(a, Event(on_wait=lambda: waited("consumer")),
                    [Event(on_wait=lambda: waited("copy"))])
        self.assertEqual(waits, ["consumer", "copy"])
        with self.assertRaises(RuntimeError):
            slots.commit_release(a)
        logical.free_group_begin()
        logical.free(ar)
        self.assertEqual(logical.available_size(), 0)
        self.assertEqual(slots.snapshot()["lease_free"], 0)
        with self.assertRaisesRegex(RuntimeError, "has not flushed"):
            slots.logical_flushed(a, logical)
        logical.free_group_end()
        slots.logical_flushed(a, logical)
        slots.commit_release(a)
        reqs.free_rows([a.req_pool_idx])
        self.assertEqual(logical.available_size(), 128)
        self.assertTrue(torch.equal(raw.k_buffer[0].view(torch.uint8)[rb], b_ring))
        self.assertTrue(torch.equal(pool.get_qsa_compressed_k_buffer(3)[br[3::4] // 4], bkeys))

        c, cr = claim("C")
        self.assertEqual(c.req_pool_idx, a.req_pool_idx)
        self.assertEqual(c.generation, a.generation + 1)
        for callback in (lambda: slots.finish_handoff(a, Event()),
                         lambda: slots.drain(a, Event(), []),
                         lambda: slots.logical_flushed(a, logical),
                         lambda: slots.commit_release(a)):
            with self.assertRaisesRegex(RuntimeError, "stale"):
                callback()
        with self.assertRaises(ValueError):
            slots.drain(c, None, [])
        # Failed drains retain the staging lease; a successful retry may free it.
        def fail():
            raise RuntimeError("copy failed")

        with self.assertRaisesRegex(RuntimeError, "copy failed"):
            slots.drain(c, Event(), [Event(on_wait=fail)])
        self.assertEqual(slots.prefill_owner, c)
        self.assertEqual(slots.phases[c.req_pool_idx], "failed")
        for lease, rows in ((c, cr), (b, br)):
            slots.drain(lease, Event(), [Event()])
            logical.free(rows)
            slots.logical_flushed(lease, logical)
            slots.commit_release(lease)
            reqs.free_rows([lease.req_pool_idx])
        with self.assertRaisesRegex(RuntimeError, "stale"):
            slots.acquire(c.req_pool_idx, c.generation, "old-generation")
        self.assertEqual(logical.available_size(), 256)
        self.assertEqual(reqs.available_size(), 2)
        self.assertEqual(slots.snapshot()["lease_free"], 2)
        self.assertEqual(slots.snapshot()["staging_available_tokens"], 128)
        self.assertEqual(len(torch.unique(logical.get_all_free_pages())), 4)
        self.assertEqual((raw.k_buffer[0].data_ptr(), raw.v_buffer[0].data_ptr()), raw_ptrs)


if __name__ == "__main__":
    unittest.main()
