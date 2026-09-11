"""Scaled CPU runtime checks. CUDA events/writer/resolver are substitutes.

The real adapter, coordinator, backend addressing, pools and postflush hooks run;
this is neither a DMA/kernel check nor live TP2/service acceptance.
"""

from contextlib import nullcontext
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import Mock, patch
import unittest

import torch

from sglang.srt.layers.attention.qwen_sparse_attn_backend import QwenSparseAttnBackend
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.mem_cache.qsa_hisparse_p2 import QSAHiSparseP2, QSAHiSparseCoordinator, _RequestCache
from sglang.srt.mem_cache.qsa_hisparse_slots import QSAHiSparseSlots
from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool


class Event:
    def __init__(self, **kwargs):
        self.recorded = False

    def record(self, stream=None):
        self.recorded = True

    def query(self):
        return self.recorded

    def synchronize(self):
        assert self.recorded, "cannot drain an unrecorded event"

    def elapsed_time(self, other):
        return 0.0


class TestQSAHiSparseP2(unittest.TestCase):
    def test_extend_uses_hisparse_writer(self):
        backend = QwenSparseAttnBackend.__new__(QwenSparseAttnBackend)
        backend._store_kv = Mock()
        backend._is_speculative_paged_mode = Mock(return_value=True)
        backend._forward_paged_attention = Mock(return_value=torch.ones((1, 1)))
        backend._pad_extend_output = Mock(side_effect=lambda output, _: output)
        layer = SimpleNamespace(tp_q_head_num=1, head_dim=1)
        loc = torch.tensor([262143])
        batch = SimpleNamespace(out_cache_loc=loc, forward_mode=object())
        q = k = v = torch.ones((1, 1, 1))

        backend.forward_extend(q, k, v, layer, batch, topk_indices=torch.zeros((1, 1)))

        backend._store_kv.assert_called_once_with(layer, loc, k, v)

    def test_capture_environment_is_rejected(self):
        with patch.dict(os.environ, {"SGLANG_QSA_HISPARSE_V3_CAPTURE": "/tmp/old-trace"}), \
                self.assertRaisesRegex(ValueError, "does not support V3 capture"):
            QSAHiSparseP2(None, "p2-offload")

    def test_graph_routes_events_and_failed_submission(self):
        from sglang.srt.model_executor.runner.shape_key import ShapeKey
        a = QSAHiSparseP2.__new__(QSAHiSparseP2)
        a.mode, a.device, a.strict, a.path = "p2-offload", "cpu", False, None
        a.graph_enabled, a.graph_capture_size, a.graph_batch = True, None, None
        a.graph_copy_pending, a.forward_id, a.offloaded = False, 0, False
        a.graph_failed_leases = set()
        a.graph_copy_epoch = 0
        a.requests, a.pending_releases, a.batch_requests = {}, [], []
        a.capacity, a.layer_ids = 2112, [3]
        a.max_requests, a.graph_batch_sizes = 2, (1, 2)
        a.slots = QSAHiSparseSlots(a.capacity, 64, 2)
        a.full = SimpleNamespace(**{key: [torch.zeros((a.slots.raw_pool_size + 64, 1, 256),
                                                      dtype=torch.uint8)] for key in ("k_buffer", "v_buffer")})
        a.pool = SimpleNamespace(dtype=torch.uint8, _transfer_full_attention_id=lambda lid: 0)
        a.req_table = torch.zeros((3, a.capacity), dtype=torch.int32)
        a.req_pool = SimpleNamespace(req_generation=[0, 1, 1], enable_mamba_extra_buffer=False,
                                    req_index_to_mamba_index_mapping=torch.tensor([0, 7, 9]))
        a.host_slabs = torch.zeros((2, 1, a.capacity // 4, 2048), dtype=torch.uint8)
        a.copy_stream, a.producer_stream = Mock(), Mock()
        a._allocate_decode_workspace()
        a.record = Mock()
        with patch.object(torch.cuda, "Event", Event), \
                patch.object(torch.cuda, "stream", side_effect=lambda _: nullcontext()), \
                patch.object(torch.cuda, "current_stream", return_value=a.producer_stream):
            a._allocate_graph_workspace()
            pointers = [t.data_ptr() for t in a.workspace]
            coord = QSAHiSparseCoordinator(a, None)
            self.assertIs(coord.num_real_reqs, a.real)
            a.runner = SimpleNamespace()
            persistent = [a.compact, a.compact_table, a.host_slabs,
                          *(t for s in a.layer_states for n, t in s.items() if n in ("hot", "tokens", "lru"))]
            for tensor in persistent:
                tensor.fill_(17)
            preserved = [t.clone() for t in persistent]
            with patch("sglang.srt.model_executor.runner.flashinfer_autotune.should_run_flashinfer_autotune",
                       return_value=False):
                for count in (1, 2):
                    before = a.slots.snapshot()
                    with self.assertRaisesRegex(RuntimeError, "capture fault"):
                        with a.graph_capture(count):
                            self.assertEqual(int(coord.num_real_reqs[0]), 0)
                            self.assertEqual(a.raw_write_locs.tolist(), [0] * count)
                            raise RuntimeError("capture fault")
                    self.assertEqual(a.slots.snapshot(), before)
                    self.assertIsNone(a.graph_capture_size)
                    self.assertFalse(a.offloaded)
                with patch.object(a.batch_write_locs, "zero_", side_effect=RuntimeError("reset fault")), \
                        self.assertRaisesRegex(RuntimeError, "reset fault"):
                    with a.graph_capture(1):
                        self.fail("failed capture setup must not yield")
                self.assertIsNone(a.graph_capture_size)
                self.assertFalse(a.offloaded)
            for actual, expected in zip(persistent, preserved):
                self.assertTrue(torch.equal(actual, expected))
            for idx, rid in ((1, "A"), (2, "B")):
                lease = a.slots.acquire(idx, 1, rid)
                state = _RequestCache(a, lease)
                a.requests[idx] = state
                a.slots.begin_handoff(lease)
                state.states.append(state.make_state())
                ready = Event(); ready.record()
                a.slots.finish_handoff(lease, ready)
                a.slots.admit_decode(lease)
                state.graph_identity = a.native_lease_snapshot(state)

            def batch(indices, seqs):
                return SimpleNamespace(batch_size=len(indices), req_pool_indices_cpu=torch.tensor(indices),
                    req_pool_indices=torch.tensor(indices), seq_lens_cpu=torch.tensor(seqs),
                    seq_lens=torch.tensor(seqs), rids=[a.requests[i].lease.rid for i in indices],
                    forward_mode=SimpleNamespace(is_idle=lambda: False, is_decode=lambda: True,
                                                  is_extend=lambda: False))

            routes = a.batch_slots.clone()
            original = a.requests[2].lease
            alias = replace(original, slot=a.requests[1].lease.slot)
            a.requests[2].lease = a.slots.active[2] = alias
            with self.assertRaisesRegex(RuntimeError, "aliases physical slots"), a.graph_replay_scope():
                a.prepare_graph_replay(batch([1, 2], [2051, 2052]), 2)
            self.assertEqual(a.forward_id, 0)
            self.assertEqual([s.decode_steps for s in a.requests.values()], [0, 0])
            self.assertTrue(torch.equal(a.batch_slots, routes))
            a.requests[2].lease = a.slots.active[2] = original
            a.req_pool.req_generation[2] = 2
            with a.graph_replay_scope(), self.assertRaises(RuntimeError):
                a.prepare_graph_replay(batch([1, 2], [2051, 2052]), 2)
            self.assertEqual(a.forward_id, 0)
            self.assertTrue(torch.equal(a.batch_slots, routes))
            self.assertIsNone(a.graph_batch)
            a.req_pool.req_generation[2] = 1
            state = a.requests[1]
            state.full.k_buffer[0][1:5].fill_(7)
            state.full.v_buffer[0][1:5].fill_(13)
            expected = torch.cat((state.full.k_buffer[0][1:5].flatten(),
                                  state.full.v_buffer[0][1:5].flatten()))
            state.states[0]["hot"][2048].copy_(expected)
            state.host[0, 512].copy_(expected)
            a._check_graph_close([(state, 512)], host=False)
            a._check_graph_close([(state, 512)], host=True)
            # A corrupted close cannot make its own expected host bytes pass.
            state.states[0]["hot"][2048].fill_(99)
            state.host[0, 512].fill_(99)
            for host in (False, True):
                with self.assertRaisesRegex(AssertionError, "raw K/V ring"):
                    a._check_graph_close([(state, 512)], host=host)
            state.states[0]["hot"][2048].copy_(expected)
            state.host[0, 512].copy_(expected)
            a.requests[2].states[0]["hot"][2048].fill_(73)
            with a.graph_replay_scope():
                a.prepare_graph_replay(batch([2, 1], [2052, 2051]), 2)
                self.assertEqual(a.batch_slots.tolist(), [1, 0])
                self.assertEqual(a.graph_seq_lens.tolist(), [2052, 2051])
                key = ShapeKey(size=2)
                backend = SimpleNamespace(_graphs={key: object()})
                a.finish_graph_replay(2, native_key=key, native_backend=backend)
                self.assertEqual(a.record.call_args.kwargs["graph_key"], vars(key))
                self.assertTrue(a.record.call_args.kwargs["native_key_present"])
            self.assertTrue(torch.all(a.requests[2].host[0, 512] == 73))
            self.assertEqual(a.requests[1].states[0]["writeback_bytes"], 0)
            self.assertEqual(a.requests[2].states[0]["writeback_bytes"], 2048)
            self.assertIs(a.requests[2].states[0]["done"], a.graph_copy_done)
            with a.graph_replay_scope():
                a.prepare_graph_replay(batch([1], [2052]), 1)
                a.producer_stream.wait_event.assert_called_with(a.graph_copy_done)
                a.finish_graph_replay(1)
            # All tail/close destinations are host metadata; device correctness
            # is covered separately by the interpreter and the later CUDA gate.
            for ta in range(4):
                for tb in range(4):
                    with a.graph_replay_scope():
                        a.prepare_graph_replay(batch([1, 2], [2056 + ta, 2060 + tb]), 2)
                        a.finish_graph_replay(2)
            self.assertEqual([t.data_ptr() for t in a.workspace], pointers)
            good_b = a.requests[2].host.clone()
            with self.assertRaisesRegex(RuntimeError, "injected") as caught, a.graph_replay_scope():
                a.prepare_graph_replay(batch([1], [2072]), 1)
                with patch.object(a.graph_copy_done, "record", side_effect=RuntimeError("injected record failure")), \
                        patch.object(a.producer_stream, "synchronize", side_effect=RuntimeError("producer drain")), \
                        patch.object(a.copy_stream, "synchronize", side_effect=RuntimeError("copy drain")) as copy_drain:
                    a.finish_graph_replay(1)
            self.assertEqual(a.slots.phases, {1: "failed", 2: "decode"})
            self.assertIsNone(a.graph_batch)
            self.assertEqual(set(a.requests), {1, 2})
            self.assertTrue(torch.equal(good_b, a.requests[2].host))
            copy_drain.assert_called_once()
            self.assertTrue(caught.exception.__notes__)
            lease = a.requests[1].lease
            self.assertEqual(a.graph_failed_leases, {lease})
            with patch.object(a.copy_stream, "synchronize", side_effect=RuntimeError("still pending")), \
                    self.assertRaisesRegex(RuntimeError, "still pending"):
                a.release(1, "A")
            self.assertEqual(a.slots.phases[1], "failed")
            self.assertIn(lease, a.graph_failed_leases)
            # Release retry must drain the registered producer, regardless of
            # the caller's current stream after the failed forward unwinds.
            other_stream = Mock()
            with patch.object(torch.cuda, "current_stream", return_value=other_stream):
                self.assertEqual(a.release(1, "A"), lease)
            other_stream.synchronize.assert_not_called()
            self.assertNotIn(lease, a.graph_failed_leases)
            self.assertEqual(a.slots.phases[1], "drained")

    @unittest.skipUnless(os.environ.get("TRITON_INTERPRET") == "1", "explicit CPU interpreter run only")
    def test_graph_byte_kernels_cpu_interpreter(self):
        from sglang.srt.layers.attention.qsa.hisparse_graph import close_c4, finish_compact

        batch, capacity, start = 8, 2112, 2176
        plane_stride = batch * 2052
        positions = torch.arange(start + 5 * batch, dtype=torch.int32)[:, None]
        dims = torch.arange(256)[None]
        k = ((positions >> ((dims % 2) * 8)) + dims * 3).to(torch.uint8)
        v = (k.int() ^ 137).to(torch.uint8)
        compact = torch.full((2, plane_stride, 1, 256), 213, dtype=torch.uint8)
        table = torch.full((batch, capacity), -1, dtype=torch.int32)
        hot = torch.full((batch * 2112, 2048), 91, dtype=torch.uint8)
        tokens = torch.full((batch, 2112), -1, dtype=torch.int32)
        slots = torch.tensor([7, 0, 6, 1, 5, 2, 4, 3], dtype=torch.int32)
        lengths = torch.tensor([2052 + row % 4 for row in range(batch)], dtype=torch.int32)
        real = torch.tensor([batch], dtype=torch.int32)
        members = k[:2048].view(2048, 1, 256)
        unpacked = torch.stack([
            torch.stack((members ^ row, members ^ (137 + row))) for row in range(batch)])
        raw = torch.full((batch, 2051), -1, dtype=torch.int32)
        raw[:, :2048] = torch.arange(2048)
        for row, length in enumerate(lengths.tolist()):
            raw[row, 2048:2048 + length % 4] = torch.arange(2048, 2048 + length % 4)
        for real_count in (0, batch):
            real.fill_(real_count)
            before = [x.clone() for x in (hot, tokens, compact, table)]
            close_c4[(batch,)](k, v, hot, tokens, slots, lengths, real, start)
            finish_compact[(batch, 513)](
                unpacked, k, v, compact, table, raw, slots, lengths, real,
                capacity, start, plane_stride)
            if real_count == 0:
                for actual, expected in zip((hot, tokens, compact, table), before):
                    self.assertTrue(torch.equal(actual, expected))
            else:
                for row, slot in enumerate(slots.tolist()):
                    base, tail = slot * 2052, int(lengths[row]) % 4
                    self.assertTrue(torch.equal(compact[:, base + 1:base + 2049], unpacked[row]))
                    for plane, source in enumerate((k, v)):
                        ring = start + slot * 5 + 1
                        self.assertTrue(torch.equal(compact[plane, base + 2049:base + 2049 + tail],
                                                    source[ring:ring + tail].view(tail, 1, 256)))
                    self.assertTrue(torch.equal(table[slot, raw[row, :2048 + tail].long()],
                                                torch.arange(base + 1, base + 2049 + tail, dtype=torch.int32)))
                    if tail == 0:
                        ring = start + slot * 5 + 1
                        expected = torch.cat((k[ring:ring + 4].flatten(), v[ring:ring + 4].flatten()))
                        self.assertTrue(torch.equal(hot[slot * 2112 + 2048], expected))
                        self.assertEqual(int(tokens[slot, 2048]), int(lengths[row]) // 4 - 1)

    def test_light_decode_ledger_uses_only_host_schedule(self):
        adapter = QSAHiSparseP2.__new__(QSAHiSparseP2)
        adapter.observe, adapter.mode, adapter.rank = "light", "p2-offload", 0
        adapter.forward_id = 17
        # No pools exist in this fixture: entering the inventory path must fail.
        rows = [{"req_pool_idx": 2, "generation": 3, "rid": "B", "lease_slot": 1,
                 "seq_len": 2051, "tail": 3, "compressed_len": 512,
                 "closes_c4": False, "ring_location": 8}]
        with tempfile.TemporaryDirectory() as directory:
            adapter.path = Path(directory) / "events.jsonl"
            with patch.object(torch.cuda, "memory_allocated", side_effect=AssertionError("GPU query")), \
                    patch.object(adapter, "native_lease_snapshot", side_effect=AssertionError("page query")):
                for event in ("decode_batch", "prefill_chunk"):
                    adapter.record(event, batch_size=1, rows=rows)
            records = [json.loads(line) for line in adapter.path.read_text().splitlines()]
        self.assertEqual([r["event"] for r in records], ["decode_batch", "prefill_chunk"])
        for record in records:
            self.assertEqual(record["rows"], rows)
            self.assertEqual(record["forward_id"], 17)
            self.assertEqual(set(record), {"event", "time_ns", "rank", "mode", "observe",
                                          "forward_id", "batch_size", "rows"})

    def test_b8_workspace_row_routing_and_slot_reuse(self):
        a = QSAHiSparseP2.__new__(QSAHiSparseP2)
        a.mode, a.device, a.strict, a.path = "p2-offload", "cpu", False, None
        a.graph_enabled, a.graph_capture_size, a.graph_batch = True, None, None
        a.graph_copy_pending, a.forward_id, a.offloaded = False, 0, False
        a.graph_failed_leases, a.requests, a.pending_releases, a.batch_requests = set(), {}, [], []
        a.capacity, a.layer_ids, a.max_requests = 2112, [3], 8
        a.graph_batch_sizes = tuple(range(1, a.max_requests + 1))
        a.slots = QSAHiSparseSlots(a.capacity, 64, a.max_requests)
        a.full = SimpleNamespace(**{key: [torch.zeros(
            (a.slots.raw_pool_size + 64, 1, 256), dtype=torch.uint8)]
            for key in ("k_buffer", "v_buffer")})
        a.pool = SimpleNamespace(dtype=torch.uint8, _transfer_full_attention_id=lambda lid: 0)
        a.req_table = torch.zeros((9, a.capacity), dtype=torch.int32)
        a.req_pool = SimpleNamespace(
            req_generation=torch.tensor([0] + [1] * 8), enable_mamba_extra_buffer=False,
            req_index_to_mamba_index_mapping=torch.arange(9))
        a.host_slabs = torch.zeros((a.max_requests, 1, a.capacity // 4, 2048), dtype=torch.uint8)
        a.copy_stream, a.producer_stream, a.runner = Mock(), Mock(), SimpleNamespace()
        a.record = Mock()
        a._allocate_decode_workspace()
        with patch.dict(os.environ, {"SGLANG_QSA_P2_VALIDATE_INITIAL_PAIR": "4"}):
            coordinator = QSAHiSparseCoordinator(a, None)
        self.assertTrue(coordinator.wait_initial_pair)
        self.assertEqual(coordinator.initial_batch_target, 4)
        with patch.dict(os.environ, {"SGLANG_QSA_P2_VALIDATE_INITIAL_PAIR": "3"}), \
                self.assertRaisesRegex(ValueError, "must be"):
            QSAHiSparseCoordinator(a, None)
        with patch.object(torch.cuda, "Event", Event), patch(
                "sglang.srt.model_executor.runner.flashinfer_autotune.should_run_flashinfer_autotune",
                return_value=False):
            a._allocate_graph_workspace()
            for count in a.graph_batch_sizes:
                with a.graph_capture(count):
                    pass
            with self.assertRaisesRegex(RuntimeError, "exact configured batch"):
                with a.graph_capture(9):
                    pass
        with patch.object(torch.distributed, "all_reduce"), patch.object(
                torch.distributed, "get_world_size", return_value=1), patch.object(
                torch.distributed, "all_gather_object",
                side_effect=lambda out, vote, **kwargs: out.__setitem__(0, vote)):
            for idx in range(1, 9):
                lease = a.slots.acquire(idx, 1, str(idx))
                state = _RequestCache(a, lease)
                a.requests[idx] = state
                a.slots.begin_handoff(lease)
                state.states.append(state.make_state())
                ready = Event(); ready.record()
                a.slots.finish_handoff(lease, ready)
                self.assertIsNone(a.slots.prefill_owner)
                if idx <= 4:
                    req = SimpleNamespace(rid=str(idx), kv=SimpleNamespace(req_pool_idx=idx))
                    coordinator.ack_staging_queue.append(
                        SimpleNamespace(req=req, lease=lease, event=ready))
                    released = coordinator.collect_ready_reqs()
                    if idx < 4:
                        self.assertEqual(released, [])
                    else:
                        self.assertEqual([req.rid for req in released], ["1", "2", "3", "4"])
                        self.assertFalse(coordinator.wait_initial_pair)
                        self.assertEqual(coordinator.initial_batch_target, 0)
                else:
                    a.slots.admit_decode(lease)

        indices = list(range(8, 0, -1))
        seqs = [2052 + row % 4 for row in range(8)]
        batch = SimpleNamespace(
            batch_size=8, req_pool_indices_cpu=torch.tensor(indices),
            req_pool_indices=torch.tensor(indices), seq_lens_cpu=torch.tensor(seqs),
            seq_lens=torch.tensor(seqs), rids=[str(i) for i in indices],
            forward_mode=SimpleNamespace(is_idle=lambda: False, is_decode=lambda: True,
                                         is_extend=lambda: False))
        a.begin_batch(batch)
        self.assertEqual(a.batch_slots.tolist(), list(range(7, -1, -1)))
        self.assertEqual(a.batch_lens.tolist(), [seq // 4 for seq in seqs])
        self.assertEqual(a.batch_write_locs.tolist(), [
            a.slots.ring_write_location(a.requests[idx].lease, seq)
            for idx, seq in zip(indices, seqs)])
        self.assertEqual(a.unpacked.shape[:2], (8, 2))
        self.assertEqual(a.compact.shape[:2], (2, 8 * 2052))
        self.assertEqual(a.layer_states[0]["tokens"].shape, (8, 2112))

        released = a.requests[4].lease
        survivors = {idx: state.lease for idx, state in a.requests.items() if idx != 4}
        terminal = Event(); terminal.record()
        a.slots.drain(released, terminal, [])
        a.slots.logical_flushed(released, SimpleNamespace(free_group=None))
        a.slots.commit_release(released)
        replacement = a.slots.acquire(4, 2, "replacement")
        self.assertEqual(replacement.slot, released.slot)
        self.assertEqual({idx: a.slots.active[idx] for idx in survivors}, survivors)
        with self.assertRaisesRegex(RuntimeError, "stale"):
            a.slots.require(released)

    def test_ready_batch_preserves_position_metadata(self):
        from array import array

        from sglang.srt.managers.schedule_batch import Req
        from sglang.srt.managers.scheduler import Scheduler
        from sglang.srt.model_executor.forward_batch_info import ForwardBatch
        from sglang.srt.sampling.sampling_params import SamplingParams
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        reqs = [Req(str(i), "", array("q", range(n)), SamplingParams(max_new_tokens=4))
                for i, n in enumerate((7, 10, 13), 1)]
        for i, req in enumerate(reqs, 1):
            req.kv.req_pool_idx = i
            req.output_ids = [42]
        scheduler = SimpleNamespace(
            device="cpu", req_to_token_pool=SimpleNamespace(device="cpu"),
            token_to_kv_pool_allocator=None, tree_cache=None,
            model_config=SimpleNamespace(vocab_size=128, is_encoder_decoder=False),
            enable_overlap=False, spec_algorithm=SpeculativeAlgorithm.NONE,
            future_map=Mock(),
        )

        def build(rows):
            # Sampling and token relay are unrelated to metadata propagation.
            with patch("sglang.srt.managers.scheduler.SamplingBatchInfo.from_schedule_batch",
                       return_value=Mock()), patch(
                    "sglang.srt.managers.schedule_batch.get_spec",
                    return_value=SimpleNamespace(speculative_algorithm=None)):
                batch = Scheduler._build_hisparse_decode_batch(scheduler, rows)
            self.assertEqual(len(batch.multimodal_inputs), len(rows))
            for actual, req in zip(batch.multimodal_inputs, rows):
                self.assertIs(actual, req.multimodal_inputs)
            return batch

        def positions(batch, expected):
            forward = ForwardBatch.__new__(ForwardBatch)
            # First decode consumes the prefill output at the next sequence length.
            forward.seq_lens = batch.seq_lens + 1
            forward.seq_lens_cpu = batch.seq_lens_cpu + 1
            with patch("sglang.srt.model_executor.forward_batch_info.get_exec",
                       return_value=SimpleNamespace(
                           deterministic=SimpleNamespace(rl_on_policy_target=None))):
                forward._compute_mrope_positions_decode(scheduler, batch)
            self.assertTrue(torch.equal(
                forward.mrope_positions, torch.tensor([expected] * 3)))

        positions(build(reqs[:1]), [7])
        positions(build([reqs[1], reqs[0]]), [10, 7])
        # Distinct metadata proves row/object preservation, not just None padding.
        reqs[1].multimodal_inputs = SimpleNamespace(
            mrope_positions=None, mrope_position_delta=torch.tensor(5))
        batch = build([reqs[1], reqs[0]])
        positions(batch, [15, 7])
        batch.filter_batch(keep_indices=[0])
        self.assertIs(batch.multimodal_inputs[0], reqs[1].multimodal_inputs)
        batch.merge_batch(build([reqs[2]]))
        self.assertEqual(batch.reqs, [reqs[1], reqs[2]])
        self.assertIs(batch.multimodal_inputs[0], reqs[1].multimodal_inputs)
        self.assertIsNone(batch.multimodal_inputs[1])
        positions(batch, [15, 13])

    def test_batch_isolation_and_real_postflush(self):
        a = QSAHiSparseP2.__new__(QSAHiSparseP2)
        a.mode, a.device, a.rank, a.strict, a.observe = "p2-offload", "cpu", 0, True, "strict"
        a.graph_failed_leases = set()
        a.capacity = 2112
        a.max_requests = 2
        a.slots = QSAHiSparseSlots(a.capacity, 64, 2)
        a.full = MHATokenToKVPool(
            a.slots.raw_pool_size, 64, torch.float8_e4m3fn, 1, 256, 1,
            "cpu", False, enable_alt_stream=False,
        )
        a.req_pool = ReqToTokenPool(2, a.capacity, "cpu", False)
        # Native mapping-shaped CPU tensors; this scenario checks observation,
        # not Mamba allocation or state numerics (those require the live model).
        a.req_pool.enable_mamba_extra_buffer = False
        a.req_pool.req_index_to_mamba_index_mapping = torch.tensor([0, 7, 9])
        a.req_table = a.req_pool.req_to_token
        a.pool = QSATokenToKVPool(
            size=2 * a.capacity, dtype=torch.float8_e4m3fn, page_size=64,
            head_num=1, head_dim=256, full_attention_layer_ids=[3], device="cpu",
            mamba_pool=SimpleNamespace(), qsa_index_kv_heads=1, qsa_index_head_dim=128,
            qsa_compress_ratio=4, qsa_token_topk=2048, num_request_slots=3,
            full_kv_pool=a.full,
        )
        a.pool.qsa_hisparse_v3 = a
        logical = PagedTokenToKVPoolAllocator(2 * a.capacity, 64, a.pool.dtype, "cpu", a.pool, False)
        a.runner = SimpleNamespace(token_to_kv_pool_allocator=logical)
        a.copy_stream, a.producer_stream = Mock(), Mock()
        a.requests, a.pending_releases, a.batch_requests = {}, [], []
        a.offloaded, a.forward_id, a.raw_write_locs = False, 0, None
        a.path, a.layer_ids = None, [3]
        a.host_slabs = torch.empty((2, 1, a.capacity // 4, 2048), dtype=torch.uint8)
        a._allocate_decode_workspace()
        backing = a.workspace + [s["hot"] for s in a.layer_states]
        backing_ptrs = [t.data_ptr() for t in backing]
        backend = QwenSparseAttnBackend.__new__(QwenSparseAttnBackend)
        backend.hisparse_v3, backend.token_to_kv_pool = a, a.pool
        layer = SimpleNamespace(layer_id=3)

        def cpu_writer(layer, loc, k, v):
            a.full.k_buffer[0].view(torch.uint8)[loc] = k
            a.full.v_buffer[0].view(torch.uint8)[loc] = v

        def row(req, seq):
            return SimpleNamespace(req=req, seq=seq)

        def batch(rows, decode, extend=None):
            indices = torch.tensor([r.req.kv.req_pool_idx for r in rows])
            return SimpleNamespace(
                batch_size=len(rows), req_pool_indices_cpu=indices,
                req_pool_indices=indices, rids=[r.req.rid for r in rows],
                seq_lens_cpu=torch.tensor([r.seq for r in rows]),
                seq_lens=torch.tensor([r.seq for r in rows]),
                extend_seq_lens_cpu=[extend] if extend is not None else None,
                forward_mode=SimpleNamespace(is_idle=lambda: False,
                    is_decode=lambda: decode, is_extend=lambda: not decode),
            )

        def claim(rid):
            req_idx = a.req_pool.alloc_rows(1)[0]
            rows = logical.alloc(a.capacity)
            a.req_table[req_idx] = rows.to(torch.int32)
            return SimpleNamespace(rid=rid, kv=SimpleNamespace(req_pool_idx=req_idx)), rows

        def votes(out, vote, **kwargs):
            out[:] = [vote, vote]

        resolver_calls = []
        def resolver(*args):
            # Only the short fully resident prefix path is modeled on CPU.
            blocks, tokens, host_locs, device_locs, host, hot, out = args[:7]
            slots, lengths = args[7:9]
            self.assertEqual(int(args[15][0]), len(blocks))
            self.assertEqual(lengths.tolist(), [s.seq_len // 4 for s in a.batch_requests])
            for row, slot in enumerate(slots.tolist()):
                ids = blocks[row].long()
                self.assertTrue(torch.equal(tokens[slot, ids], blocks[row]))
                self.assertTrue(torch.equal(host[host_locs[slot, ids]],
                                           a.host_slabs[slot, 0, ids]))
                out[row].copy_(device_locs[slot, ids])
            args[18].zero_()
            resolver_calls.append(len(blocks))

        with patch.object(torch.cuda, "Event", Event), \
                patch.object(torch.cuda, "stream", side_effect=lambda _: nullcontext()), \
                patch.object(torch.cuda, "current_stream", return_value=Mock()), \
                patch.object(a.pool, "set_kv_buffer", side_effect=cpu_writer), \
                patch.object(torch.distributed, "get_world_size", return_value=2), \
                patch.object(torch.distributed, "all_reduce"), \
                patch.object(torch.distributed, "all_gather_object", side_effect=votes), \
                patch("sglang.kernels.ops.kvcache.hisparse.load_cache_to_device_buffer_mla", side_effect=resolver):
            coord = QSAHiSparseCoordinator(a, None)
            self.assertFalse(coord.wait_initial_pair)
            with patch.dict(os.environ, {"SGLANG_QSA_P2_VALIDATE_INITIAL_PAIR": "1"}):
                coord = QSAHiSparseCoordinator(a, None)
            coord.set_decode_producer_stream(a.producer_stream)
            def peer_nonempty(out, vote, **kwargs):
                out[:] = [vote, ([(1, 1, "peer-only", 0)], 1, vote[2])]
            with patch.object(torch.distributed, "all_reduce", side_effect=lambda total, **kw: total.fill_(1)), \
                    patch.object(torch.distributed, "all_gather_object", side_effect=peer_nonempty), \
                    self.assertRaisesRegex(RuntimeError, "identities"):
                coord.collect_ready_reqs()
            req_a, ar = claim("A")
            req_b = None
            saved_a = None
            for req, rows, length, kbyte, vbyte in ((req_a, ar, 2049, 7, 19),):
                for end, extend in ((2048, 2048), (length, length - 2048)):
                    a.begin_batch(batch([row(req, end)], False, extend))
                    backend._store_kv(layer, rows[end - extend:end],
                        torch.full((extend, 1, 256), kbyte, dtype=torch.uint8),
                        torch.full((extend, 1, 256), vbyte, dtype=torch.uint8))
                coord.admit_request_into_staging(req)
                # A remote rank that is not ready cannot admit local-ready A.
                def slow(out, vote, **kwargs):
                    out[:] = [vote, (vote[0], 0, vote[2])]
                with patch.object(torch.distributed, "all_gather_object", side_effect=slow):
                    self.assertEqual(coord.collect_ready_reqs(), [])
                self.assertEqual(coord.collect_ready_reqs(), [])
                self.assertEqual(a.slots.phases[req.kv.req_pool_idx], "host_ready")
                self.assertIsNone(a.slots.prefill_owner)
                saved_a = a.requests[req.kv.req_pool_idx].host.clone()

            req_b, br = claim("B")
            a.begin_batch(batch([row(req_b, 2050)], False, 2050))
            backend._store_kv(layer, br[:2050], torch.full((2050, 1, 256), 31, dtype=torch.uint8),
                              torch.full((2050, 1, 256), 43, dtype=torch.uint8))
            physical = a.prefill_slots(req_b.kv.req_pool_idx, 2050)
            self.assertTrue(torch.all(a.full.k_buffer[0].view(torch.uint8)[physical] == 31))
            # B's first logical slots numerically fit, but alias ring storage.
            wrong = a.full.k_buffer[0].view(torch.uint8)[br[:4]]
            self.assertFalse(torch.all(wrong == 31))
            self.assertTrue(torch.equal(a.requests[req_a.kv.req_pool_idx].host, saved_a))
            coord.admit_request_into_staging(req_b)
            def mismatch(out, vote, **kwargs):
                out[:] = [vote, ([(999, 1, "wrong", 0)], 1, vote[2])]
            with patch.object(torch.distributed, "all_gather_object", side_effect=mismatch), \
                    self.assertRaisesRegex(RuntimeError, "identities"):
                coord.collect_ready_reqs()
            def barrier_mismatch(out, vote, **kwargs):
                out[:] = [vote, (vote[0], vote[1], not vote[2])]
            with patch.object(torch.distributed, "all_gather_object", side_effect=barrier_mismatch), \
                    self.assertRaisesRegex(RuntimeError, "identities"):
                coord.collect_ready_reqs()
            def slot_mismatch(out, vote, **kwargs):
                ids = [(*identity[:3], 1 - identity[3]) for identity in vote[0]]
                out[:] = [vote, (ids, vote[1], vote[2])]
            with patch.object(torch.distributed, "all_gather_object", side_effect=slot_mismatch), \
                    self.assertRaisesRegex(RuntimeError, "identities"):
                coord.collect_ready_reqs()
            self.assertEqual(coord.collect_ready_reqs(), [req_a, req_b])
            self.assertFalse(coord.wait_initial_pair)

            state_b = a.requests[req_b.kv.req_pool_idx]
            a.req_pool.req_index_to_mamba_index_mapping[req_b.kv.req_pool_idx] = 9
            native = a.native_lease_snapshot(state_b, include_pages=True)
            self.assertEqual(native["logical_row_ptr"], a.req_table[req_b.kv.req_pool_idx].data_ptr())
            self.assertEqual(native["logical_page_ids"], (br[:2050:64] // 64).tolist())
            self.assertEqual(native["mamba_pool_idx"], 9)
            self.assertFalse(native["mamba_extra_buffer_enabled"])
            self.assertEqual(native["mamba_track_slots"], [])
            # Same backing pointer does not imply an unchanged native mapping.
            a.req_table[req_b.kv.req_pool_idx, 0] = ar[0]
            a.req_pool.req_index_to_mamba_index_mapping[req_b.kv.req_pool_idx] = 7
            changed = a.native_lease_snapshot(state_b, include_pages=True)
            self.assertEqual(changed["logical_row_ptr"], native["logical_row_ptr"])
            self.assertNotEqual(changed["logical_page_ids"], native["logical_page_ids"])
            self.assertNotEqual(changed["mamba_pool_idx"], native["mamba_pool_idx"])
            a.req_table[req_b.kv.req_pool_idx, 0] = br[0] + 1
            with self.assertRaisesRegex(RuntimeError, "not aligned"):
                a.native_lease_snapshot(state_b, include_pages=True)
            a.req_table[req_b.kv.req_pool_idx, 0] = br[0]
            a.req_pool.req_index_to_mamba_index_mapping[req_b.kv.req_pool_idx] = 9
            a.req_pool.enable_mamba_extra_buffer = True
            a.req_pool.req_index_to_mamba_ping_pong_track_buffer_mapping = torch.tensor([[0], [11], [13]])
            a.req_pool.req_index_to_mamba_ping_pong_track_buffer_mapping[req_b.kv.req_pool_idx] = 13
            self.assertEqual(a.native_lease_snapshot(state_b)["mamba_track_slots"], [13])
            a.req_pool.enable_mamba_extra_buffer = False

            a_steps = a.requests[req_a.kv.req_pool_idx].decode_steps
            a.req_pool.req_generation[req_b.kv.req_pool_idx] += 1
            with self.assertRaisesRegex(RuntimeError, "stale native"):
                a.native_lease_snapshot(state_b)
            with self.assertRaisesRegex(RuntimeError, "generation changed"):
                a.begin_batch(batch([row(req_a, 2050), row(req_b, 2051)], True))
            self.assertEqual(a.requests[req_a.kv.req_pool_idx].decode_steps, a_steps)
            a.req_pool.req_generation[req_b.kv.req_pool_idx] -= 1

            for seq_a, seq_b in ((2050, 2051), (2051, 2052), (2052, 2053), (2053, 2054)):
                rows = [row(req_b, seq_b), row(req_a, seq_a)]
                a.begin_batch(batch(rows, True))
                loc = torch.tensor([br[seq_b - 1], ar[seq_a - 1]])
                keep_logical = loc.clone()
                coord.map_last_loc_to_buffer(None, loc, None, None,
                                             torch.tensor([req_b.kv.req_pool_idx, req_a.kv.req_pool_idx]))
                self.assertTrue(torch.equal(loc, keep_logical))
                backend._store_kv(layer, loc, torch.tensor([31, 7], dtype=torch.uint8)[:, None, None].expand(2, 1, 256),
                                 torch.tensor([43, 19], dtype=torch.uint8)[:, None, None].expand(2, 1, 256))
                a.after_store(layer)
                topk = []
                for seq in (seq_b, seq_a):
                    ids = torch.cat((torch.arange(511), torch.tensor([seq // 4 - 1])))
                    tail = seq % 4
                    topk.append(torch.cat(((ids[:, None] * 4 + torch.arange(4)).flatten(),
                                          torch.arange(seq - tail, seq), torch.full((3 - tail,), -1))))
                topk = torch.stack(topk).to(torch.int32)
                k, v, table, physical_rows = a.selected(layer, topk)
                for r, seq, kb, vb in ((0, seq_b, 31, 43), (1, seq_a, 7, 19)):
                    locs = table[physical_rows[r], topk[r, :2048 + seq % 4].long()].long()
                    self.assertTrue(torch.all(k.view(torch.uint8)[locs] == kb))
                    self.assertTrue(torch.all(v.view(torch.uint8)[locs] == vb))
                self.assertNotEqual(int(physical_rows[0]), int(physical_rows[1]))
            self.assertEqual(resolver_calls, [2] * 4)

            state_b = a.requests[req_b.kv.req_pool_idx]
            host_b = state_b.host.clone()
            state_a = a.requests[req_a.kv.req_pool_idx]
            lease_a = a.release(req_a.kv.req_pool_idx, "A")
            logical.free_group_begin()
            logical.free(ar)
            a.after_release(lease_a)
            self.assertEqual(a.slots.phases[lease_a.req_pool_idx], "drained")
            self.assertEqual(len(a.slots.free_slots), 0)
            logical.free_group_end()  # Executes the real P2 allocator callback.
            a.req_pool.free_rows([req_a.kv.req_pool_idx])
            self.assertEqual(len(a.slots.free_slots), 1)
            self.assertIs(a.requests[req_b.kv.req_pool_idx], state_b)
            self.assertTrue(torch.equal(state_b.host, host_b))
            # B1 in slot 1 must neither read nor write the released slot 0.
            inactive = a.layer_states[0]["hot"].view(2, 2112, 2048)[lease_a.slot]
            inactive.fill_(173)
            a.begin_batch(batch([row(req_b, 2054)], True))
            a.selected(layer, topk[:1])
            self.assertEqual(resolver_calls[-1], 1)
            self.assertTrue(torch.all(inactive == 173))
            b_hot = state_b.states[0]["hot"].clone()
            req_c, cr = claim("C")
            a.begin_batch(batch([row(req_c, 2048)], False, 2048))
            self.assertEqual(req_c.kv.req_pool_idx, req_a.kv.req_pool_idx)
            self.assertEqual(a.requests[req_c.kv.req_pool_idx].generation, lease_a.generation + 1)
            with self.assertRaisesRegex(RuntimeError, "stale"):
                a.after_release(lease_a)
            with self.assertRaisesRegex(RuntimeError, "stale"):
                state_a.make_state()
            self.assertTrue(torch.equal(state_b.states[0]["hot"], b_hot))
            backend._store_kv(layer, cr[:2048], torch.full((2048, 1, 256), 53, dtype=torch.uint8),
                              torch.full((2048, 1, 256), 67, dtype=torch.uint8))
            coord.admit_request_into_staging(req_c)
            self.assertEqual(coord.collect_ready_reqs(), [req_c])
            self.assertEqual(a.requests[req_c.kv.req_pool_idx].states[0]["hot"].data_ptr(), inactive.data_ptr())
            self.assertTrue(torch.equal(state_b.states[0]["hot"], b_hot))
            self.assertEqual([t.data_ptr() for t in backing], backing_ptrs)
            logical.free_group_begin()
            for req, rows in ((req_c, cr), (req_b, br)):
                lease = a.release(req.kv.req_pool_idx, req.rid)
                logical.free(rows)
                a.after_release(lease)
            self.assertEqual(len(a.pending_releases), 2)
            first, second = a.pending_releases
            # A pre-commit failure retains both leases, including the current
            # logical_flushed lease; native pages have already flushed once.
            with patch.object(a.slots, "commit_release", side_effect=RuntimeError("commit fault")), \
                    self.assertRaisesRegex(RuntimeError, "commit fault"):
                logical.free_group_end()
            self.assertEqual(a.pending_releases, [first, second])
            self.assertEqual(a.slots.phases[first.req_pool_idx], "logical_flushed")
            self.assertEqual(a.slots.phases[second.req_pool_idx], "drained")
            self.assertEqual(a.slots.free_slots, [])
            self.assertEqual(logical.available_size(), 2 * a.capacity)

            def record_fault(event, *args, **kwargs):
                if event == "logical_release_complete":
                    raise RuntimeError("ledger fault")

            # A post-commit ledger failure removes only the completed head.
            with patch.object(a, "record", side_effect=record_fault), \
                    self.assertRaisesRegex(RuntimeError, "ledger fault"):
                a.after_logical_flush()
            self.assertEqual(a.pending_releases, [second])
            self.assertNotIn(first.req_pool_idx, a.requests)
            self.assertEqual(a.slots.active[second.req_pool_idx], second)
            self.assertEqual(a.slots.phases[second.req_pool_idx], "drained")
            self.assertEqual(len(a.slots.free_slots), 1)
            a.after_logical_flush()
            self.assertEqual(a.pending_releases, [])
            self.assertEqual(a.requests, {})
            self.assertEqual(logical.available_size(), 2 * a.capacity)
            self.assertEqual(len(a.slots.free_slots), 2)


if __name__ == "__main__":
    unittest.main()
