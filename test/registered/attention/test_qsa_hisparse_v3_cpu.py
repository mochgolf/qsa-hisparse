"""CPU checks for the actual V3 byte-layout helpers and startup guards."""

import json
import unittest
import tempfile
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.mem_cache.qsa_hisparse_v3 import (
    QSAHiSparseV3,
    pack_c4,
    stage_short_prefix,
    unpack_index,
    validate_configuration,
)


class TestQSAHiSparseV3(unittest.TestCase):
    def test_startup_slab_reuse_and_appended_writeback(self):
        # Exercise the real constructor/handoff/release with scaled CPU backing.
        class Full(SimpleNamespace):
            pass

        class Allocator(SimpleNamespace):
            pass

        slab = torch.empty((1, 514, 2048), dtype=torch.uint8)
        full = Full(k_buffer=[torch.zeros((2056, 1, 256), dtype=torch.uint8)],
                    v_buffer=[torch.zeros((2056, 1, 256), dtype=torch.uint8)],
                    _init_data_ptrs_and_strides=Mock())
        mamba = SimpleNamespace(get_contiguous_buf_infos=lambda: ([], [], None))
        pool = SimpleNamespace(full_kv_pool=full, device="cpu", size=262144,
                               full_attention_layer_id_mapping=[3],
                               qsa_compressed_flat=torch.zeros(1),
                               _transfer_full_attention_id=lambda _: 0)
        allocator = Allocator(available_size=lambda: 0, free_group=None)
        runner = SimpleNamespace(token_to_kv_pool=pool, server_args=None,
                                 token_to_kv_pool_allocator=allocator,
                                 ps=SimpleNamespace(tp_rank=0),
                                 req_to_token_pool=SimpleNamespace(
                                     req_to_token=torch.arange(262144)[None], mamba_pool=mamba))
        empty = torch.empty
        allocations = []

        def pinned_empty(shape, **kwargs):
            if kwargs.get("pin_memory"):
                allocations.append((shape, kwargs))
                return slab
            return empty(shape, **kwargs)

        with patch.dict("os.environ", {}, clear=True), \
                patch("sglang.srt.mem_cache.qsa_hisparse_v3.validate_configuration"), \
                patch("sglang.srt.model_executor.cuda_graph_config.cuda_graph_fully_disabled", return_value=True), \
                patch("sglang.srt.mem_cache.memory_pool.MHATokenToKVPool", Full), \
                patch("sglang.srt.mem_cache.allocator.paged.PagedTokenToKVPoolAllocator", Allocator), \
                patch.object(torch, "empty", side_effect=pinned_empty), \
                patch.object(torch.cuda, "Stream"), \
                patch.object(QSAHiSparseV3, "record") as record:
            resident = QSAHiSparseV3(runner, "resident")
            self.assertIsNone(resident.host_slab)
            self.assertEqual(allocations, [])
            adapter = QSAHiSparseV3(runner, "offload")
            self.assertEqual(allocations, [((12, 65536, 2048),
                                           {"dtype": torch.uint8, "pin_memory": True})])
            self.assertEqual(record.call_args.kwargs["allocation_phase"], "startup")
            with patch.object(torch, "empty", side_effect=RuntimeError("pin failure")), \
                    self.assertRaisesRegex(RuntimeError, "pin failure"):
                QSAHiSparseV3(runner, "offload")

        adapter.capacity = 2056
        def check_ledger(event, **extra):
            if event != "handoff_begin":
                return
            with tempfile.TemporaryDirectory() as directory, \
                    patch.object(torch.cuda, "memory_reserved", return_value=0), \
                    patch.object(torch.cuda, "max_memory_allocated", return_value=0), \
                    patch.object(torch.cuda, "max_memory_reserved", return_value=0):
                adapter.path = Path(directory) / "events.jsonl"
                QSAHiSparseV3.record(adapter, event, **extra)
                row = json.loads(adapter.path.read_text())
                self.assertEqual(row["host_reserved_bytes"], slab.nbytes)
                self.assertEqual(row["host_bytes"], slab.nbytes)
                self.assertEqual(row["host_free_bytes"], 0)
                self.assertEqual(row["host_slab_ptr"], slab.data_ptr())

        runner.req_to_token_pool.mamba_allocator = SimpleNamespace(available_size=lambda: 40)
        adapter.record = Mock(side_effect=check_ledger)
        events = []

        def make_event(**kwargs):
            event = Mock()
            event.elapsed_time.return_value = 0.0
            events.append(event)
            return event

        def drain():
            # Active backing cannot be surrendered before consumer/copy drains.
            self.assertIs(adapter.host, slab)
            events[-1].synchronize.assert_called_once_with()

        adapter.copy_stream.synchronize.side_effect = drain
        with patch.object(torch.cuda, "Event", side_effect=make_event), \
                patch.object(torch.cuda, "memory_allocated", return_value=0), \
                patch.object(torch.cuda, "current_stream"), \
                patch.object(torch.cuda, "stream", side_effect=lambda _: nullcontext()), \
                patch.object(torch, "empty", side_effect=pinned_empty):
            for generation in (1, 2):
                adapter.owner, adapter.owner_rid, adapter.generation = 0, str(generation), generation
                full.k_buffer = [torch.full((2056, 1, 256), generation, dtype=torch.uint8)]
                full.v_buffer = [torch.full((2056, 1, 256), generation + 17, dtype=torch.uint8)]
                adapter.handoff(2048)
                self.assertIs(adapter.host, slab)
                self.assertTrue(torch.all(slab[0, :512, :1024] == generation))
                self.assertTrue(torch.all(slab[0, :512, 1024:] == generation + 17))
                with self.assertRaisesRegex(RuntimeError, "unclaimed host slab"):
                    adapter.handoff(2048)
                adapter.seq_len = 2052
                full.k_buffer[0][1:5].fill_(generation + 31)
                full.v_buffer[0][1:5].fill_(generation + 53)
                adapter.after_store(SimpleNamespace(layer_id=3))
                # This fails if handoff leases only host_slab[:, :prompt_blocks].
                self.assertTrue(torch.all(slab[0, 512, :1024] == generation + 31))
                self.assertTrue(torch.all(slab[0, 512, 1024:] == generation + 53))
                lease = adapter.release(0, str(generation))
                self.assertIsNone(adapter.host)
                self.assertIs(adapter.host_slab, slab)
                adapter.after_release(lease)
                self.assertIsNone(adapter.owner)
            self.assertEqual(len(allocations), 1)
            adapter.owner, adapter.failed = 0, True
            with self.assertRaisesRegex(RuntimeError, "unclaimed host slab"):
                adapter.handoff(2048)

    def test_light_observation_keeps_writeback_dependencies(self):
        for strict in (False, True):
            adapter = QSAHiSparseV3.__new__(QSAHiSparseV3)
            adapter.strict, adapter.device = strict, "cpu"
            adapter.offloaded, adapter.seq_len, adapter.generation = True, 2052, 1
            adapter.pool = SimpleNamespace(_transfer_full_attention_id=lambda _: 0)
            k = torch.arange(5 * 256).reshape(5, 1, 256).to(torch.uint8)
            v = (k + 17).clone()
            adapter.full = SimpleNamespace(k_buffer=[k], v_buffer=[v])
            adapter.capacity = 2056
            state = adapter.make_state()
            previous = Mock(name="previous_writeback")
            state["done"] = previous
            adapter.states = [state]
            adapter.host = torch.zeros((1, 514, 2048), dtype=torch.uint8)
            adapter.copy_stream = Mock(name="copy_stream")
            current = Mock(name="current_stream")
            producer, copy_begin, done = Mock(), Mock(), Mock()
            events = [producer, copy_begin, done] if strict else [producer, done]
            with patch.object(torch.cuda, "Event", side_effect=events), \
                    patch.object(torch.cuda, "current_stream", return_value=current), \
                    patch.object(torch.cuda, "stream", return_value=nullcontext()):
                adapter.after_store(SimpleNamespace(layer_id=3))
            current.wait_event.assert_called_once_with(previous)
            producer.record.assert_called_once_with()
            adapter.copy_stream.wait_event.assert_called_once_with(producer)
            done.record.assert_called_once_with(adapter.copy_stream)
            done.synchronize.assert_not_called()
            self.assertIs(state["done"], done)
            self.assertEqual(state["writeback_bytes"], 2048)
            self.assertTrue(torch.equal(adapter.host[0, 512, :1024], k[1:5].flatten()))
            self.assertTrue(torch.equal(adapter.host[0, 512, 1024:], v[1:5].flatten()))
            # A light per-step record must not inspect pool/memory or open files.
            adapter.path = Path("must-not-write.jsonl")
            if not strict:
                for event in ("decode_step", "prefill_step", "selected_check"):
                    adapter.record(event)
            # Boundary evidence must remain active in either observation mode.
            with self.assertRaises(AttributeError):
                adapter.record("handoff_complete")

    def test_observation_rejects_invalid_or_contaminated_mode(self):
        # Stop at pool validation, before constructing CUDA resources.
        for env, expected in (({}, "strict"),
                              ({"SGLANG_QSA_HISPARSE_V3_OBSERVE": "light"}, "light")):
            adapter = QSAHiSparseV3.__new__(QSAHiSparseV3)
            with patch.dict("os.environ", env, clear=True), patch(
                "sglang.srt.mem_cache.qsa_hisparse_v3.validate_configuration",
                side_effect=RuntimeError("CPU configuration boundary"),
            ), self.assertRaisesRegex(RuntimeError, "CPU configuration boundary"):
                adapter.__init__(SimpleNamespace(token_to_kv_pool=None, server_args=None), "offload")
            self.assertEqual(adapter.observe, expected)
            self.assertEqual(adapter.strict, expected == "strict")
        for env in (
            {"SGLANG_QSA_HISPARSE_V3_OBSERVE": "typo"},
            {"SGLANG_QSA_HISPARSE_V3_OBSERVE": "light",
             "SGLANG_QSA_HISPARSE_V3_CAPTURE": "/tmp/capture"},
        ):
            with patch.dict("os.environ", env, clear=True), self.assertRaises(ValueError):
                QSAHiSparseV3(None, "offload")

    def test_decode_ring_preserves_writer_padding(self):
        adapter = QSAHiSparseV3.__new__(QSAHiSparseV3)
        adapter.failed = adapter.releasing = False
        adapter.owner, adapter.owner_rid = 0, "ring-check"
        adapter.mode, adapter.offloaded = "offload", True
        adapter.decode_steps = 0
        adapter.ring_loc = torch.zeros(1, dtype=torch.int64)
        adapter.compressed_len = torch.zeros(1, dtype=torch.int32)
        adapter.record = lambda *args, **kwargs: None
        mode = SimpleNamespace(is_idle=lambda: False, is_decode=lambda: True)
        ring = torch.zeros(5, dtype=torch.int64)
        for position in range(261120, 261128):
            batch = SimpleNamespace(forward_mode=mode, batch_size=1,
                                    req_pool_indices=torch.tensor([0]),
                                    rids=["ring-check"], seq_lens=torch.tensor([position + 1]))
            adapter.begin_batch(batch)
            slot = int(adapter.ring_loc[0])
            # The real CUDA writer's reserved_skip_index defaults to zero.
            self.assertIn(slot, (1, 2, 3, 4))
            ring[slot] = position
            tail = (position + 1) % 4
            count = tail or 4
            self.assertEqual(ring[1:1 + count].tolist(),
                             list(range(position + 1 - count, position + 1)))
            self.assertEqual(int(ring[0]), 0)
        self.assertEqual(adapter.decode_steps, 8)

    def test_capture_uses_actual_consumer_length(self):
        adapter = QSAHiSparseV3.__new__(QSAHiSparseV3)
        adapter.decode_steps, adapter.seq_len = 1, 2049
        adapter.layer_ids, adapter.capture_layers = [3], {}
        adapter.owner_rid, adapter.generation, adapter.rank = "capture-check", 1, 0
        q = torch.ones((1, 12, 256), dtype=torch.bfloat16)
        packed = torch.zeros((2051, 1, 256), dtype=torch.bfloat16)
        with tempfile.TemporaryDirectory() as directory:
            adapter.capture_dir = Path(directory)
            adapter.capture_decode(
                SimpleNamespace(layer_id=3), q, packed, packed + 1,
                torch.arange(2051)[None], q + 2, 0.5, 2.0,
                torch.tensor([3]), torch.tensor([0, 1]), torch.tensor([0, 3]),
            )
            paths = list(adapter.capture_dir.glob("*.pt"))
            saved = torch.load(paths[0], weights_only=True)["layers"][3]
            # Observe cu_k, even if a bug makes it disagree with sequence-derived counts.
            self.assertEqual(saved["k"].shape[0], 3)
            self.assertTrue(torch.equal(saved["output"], q + 2))
            self.assertEqual(saved["k_stride"], packed.stride())

    def test_layout_and_tail(self):
        # Independent raw source rows; neither expectation uses candidate indices.
        raw = torch.arange(2052 * 256, dtype=torch.int64).reshape(2052, 1, 256)
        k = (raw % 251).to(torch.uint8)
        v = ((raw * 7 + 13) % 251).to(torch.uint8)
        packed = pack_c4(k[:2048], v[:2048])
        for block in (0, 15, 16, 511):
            self.assertTrue(torch.equal(packed[block, :1024], k[block * 4:block * 4 + 4].flatten()))
            self.assertTrue(torch.equal(packed[block, 1024:], v[block * 4:block * 4 + 4].flatten()))
        got = packed.view(-1, 256).index_select(0, unpack_index("cpu")).view(2, 2048, 1, 256)
        self.assertTrue(torch.equal(got[0], k[:2048]))
        self.assertTrue(torch.equal(got[1], v[:2048]))
        for tail in range(4):
            compact = torch.zeros((2, 2052, 1, 256), dtype=torch.uint8)
            compact[:, 1:2049].copy_(got)
            compact[0, 2049:2049 + tail].copy_(k[2048:2048 + tail])
            compact[1, 2049:2049 + tail].copy_(v[2048:2048 + tail])
            self.assertTrue(torch.equal(compact[0, 1:2049 + tail], k[:2048 + tail]))
            self.assertTrue(torch.equal(compact[1, 1:2049 + tail], v[:2048 + tail]))
        with self.assertRaises(ValueError):
            pack_c4(k[:3], v[:3])
        # The real resolver bypasses H2D at <=2048 C4: its ordered rows must exist.
        records = torch.arange(2049 * 8).reshape(2049, 8)
        for count in (512, 2048, 2049):
            hot = torch.full((2112, 8), -1)
            tokens = torch.full((1, 2112), -1, dtype=torch.int32)
            stage_short_prefix(hot, tokens, records[:count])
            if count <= 2048:
                self.assertTrue(torch.equal(hot[:count], records[:count]))
                self.assertTrue(torch.equal(tokens[0, :count], torch.arange(count)))
            else:
                self.assertTrue(torch.all(hot == -1))

    def test_configuration(self):
        args = SimpleNamespace(max_running_requests=1, tp_size=2, pp_size=1,
                               disable_radix_cache=True, disable_overlap_schedule=True,
                               cuda_graph_backend_decode="disabled", cuda_graph_backend_prefill="disabled",
                               context_length=262144, max_total_tokens=262144,
                               chunked_prefill_size=2048, skip_server_warmup=True,
                               enable_deterministic_inference=True,
                               random_seed=147342228,
                               speculative_algorithm=None,
                               disaggregation_mode="null")
        full = SimpleNamespace(use_hnd=False, kv_cache_layout="NHD",
                               post_capture_active=False, is_quantized_kv_cache=False)
        pool = SimpleNamespace(full_kv_pool=full, size=262144, page_size=64, qsa_compress_ratio=4,
                               qsa_token_topk=2048, full_layer_nums=12, head_num=1,
                               head_dim=256, dtype=torch.float8_e4m3fn)
        validate_configuration(args, pool)
        for name, bad in (("max_running_requests", 2), ("disable_radix_cache", False),
                          ("disable_overlap_schedule", False), ("cuda_graph_backend_decode", "full"),
                          ("cuda_graph_backend_prefill", "full"), ("context_length", 8192),
                          ("max_total_tokens", 8192), ("chunked_prefill_size", 4096),
                          ("skip_server_warmup", False), ("enable_streaming_session", True),
                          ("enable_deterministic_inference", False),
                          ("random_seed", 42),
                          ("speculative_algorithm", "NEXTN"), ("enable_hisparse", True)):
            changed = SimpleNamespace(**vars(args))
            setattr(changed, name, bad)
            with self.assertRaises(ValueError):
                validate_configuration(changed, pool)
        graph_args = SimpleNamespace(**vars(args))
        graph_args.max_running_requests, graph_args.max_total_tokens = 2, 524288
        graph_args.cuda_graph_backend_decode = "full"
        graph_args.disable_cuda_graph_padding = True
        graph_args.cuda_graph_bs_decode, graph_args.cuda_graph_max_bs_decode = [1, 2], 2
        graph_args.enable_torch_compile = False
        graph_pool = SimpleNamespace(**vars(pool))
        graph_pool.size = 524288
        validate_configuration(graph_args, graph_pool, p2=True, graph=True)
        validate_configuration(graph_args, graph_pool, p2=True, graph=True, strict=False)
        for max_requests in (4, 8):
            scaled_args = SimpleNamespace(**vars(graph_args))
            scaled_args.max_running_requests = max_requests
            scaled_args.max_total_tokens = max_requests * 262144
            scaled_args.cuda_graph_bs_decode = list(range(1, max_requests + 1))
            scaled_args.cuda_graph_max_bs_decode = max_requests
            scaled_pool = SimpleNamespace(**vars(graph_pool))
            scaled_pool.size = scaled_args.max_total_tokens
            validate_configuration(scaled_args, scaled_pool, p2=True, graph=True)
            scaled_args.max_total_tokens -= 64
            with self.assertRaisesRegex(ValueError, "max_total_tokens"):
                validate_configuration(scaled_args, scaled_pool, p2=True, graph=True)
        unsupported = SimpleNamespace(**vars(graph_args))
        unsupported.max_running_requests = 3
        with self.assertRaisesRegex(ValueError, "max_running_requests"):
            validate_configuration(unsupported, graph_pool, p2=True, graph=True)
        native_args = SimpleNamespace(**vars(graph_args))
        native_args.enable_deterministic_inference = False
        validate_configuration(native_args, graph_pool, p2=True, graph=True, strict=False)
        with self.assertRaisesRegex(ValueError, "enable_deterministic_inference"):
            validate_configuration(native_args, graph_pool, p2=True, graph=True)
        legacy_args = SimpleNamespace(**vars(args))
        legacy_args.enable_deterministic_inference = False
        with self.assertRaisesRegex(ValueError, "enable_deterministic_inference"):
            validate_configuration(legacy_args, pool, strict=False)
        native_args.disable_overlap_schedule = False
        with self.assertRaisesRegex(ValueError, "disable_overlap_schedule"):
            validate_configuration(native_args, graph_pool, p2=True, graph=True, strict=False)
        graph_args.chunked_prefill_size = 4096
        validate_configuration(graph_args, graph_pool, p2=True, graph=True)
        eager_args = SimpleNamespace(**vars(graph_args))
        eager_args.cuda_graph_backend_decode = "disabled"
        validate_configuration(eager_args, graph_pool, p2=True)
        for chunk in (None, 0, 4095, 8192):
            changed = SimpleNamespace(**vars(graph_args))
            changed.chunked_prefill_size = chunk
            with self.assertRaises(ValueError):
                validate_configuration(changed, graph_pool, p2=True, graph=True)
        for name, bad in (("disable_cuda_graph_padding", False), ("cuda_graph_bs_decode", [2]),
                          ("cuda_graph_max_bs_decode", 4), ("enable_torch_compile", True),
                          ("cuda_graph_backend_decode", "breakable")):
            changed = SimpleNamespace(**vars(graph_args))
            setattr(changed, name, bad)
            with self.assertRaises(ValueError):
                validate_configuration(changed, graph_pool, p2=True, graph=True)
        with self.assertRaises(ValueError):
            validate_configuration(graph_args, graph_pool, graph=True)

    def test_stale_release(self):
        adapter = QSAHiSparseV3.__new__(QSAHiSparseV3)
        adapter.owner, adapter.owner_rid, adapter.generation = 1, "B", 2
        adapter.releasing = False
        adapter.pending_release = None
        allocator = SimpleNamespace(free_group=None)
        adapter.runner = SimpleNamespace(token_to_kv_pool_allocator=allocator)
        with self.assertRaises(RuntimeError):
            adapter.release(1, "A")
        adapter.releasing = True
        with self.assertRaises(RuntimeError):
            adapter.after_release((1, "B", 1))
        self.assertEqual((adapter.owner, adapter.owner_rid, adapter.generation), (1, "B", 2))
        adapter.record = lambda *args, **kwargs: None
        adapter.after_release((1, "B", 2))
        self.assertIsNone(adapter.owner)
        self.assertFalse(adapter.releasing)
        # Exercise the real allocator flush: the lease stays held until free runs.
        from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator

        adapter.owner, adapter.owner_rid, adapter.generation = 1, "C", 3
        adapter.releasing = True
        allocator = PagedTokenToKVPoolAllocator(
            size=192, page_size=64, dtype=torch.uint8, device="cpu",
            kvcache=SimpleNamespace(qsa_hisparse_v3=adapter), need_sort=True,
        )
        adapter.runner.token_to_kv_pool_allocator = allocator
        events = []
        adapter.record = lambda event: events.append((event, allocator.available_size()))
        rows = allocator.alloc(128)
        allocator.free_group_begin()
        allocator.free(rows[:64])
        allocator.free_segment(rows[64:], start_pos=64)
        adapter.after_release((1, "C", 3))
        self.assertEqual(adapter.owner_rid, "C")
        self.assertEqual(allocator.available_size(), 64)
        allocator.free_group_end()
        self.assertEqual(events[-1], ("logical_release_complete", 192))
        self.assertIsNone(adapter.owner)


if __name__ == "__main__":
    unittest.main()
