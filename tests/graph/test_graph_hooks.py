"""Decode graph, model runner and ForwardBatch hooks (rows G01-G03, R01, R02, F01).

The patched methods run on instances built without ``__init__`` and on fakes
that record the order of runtime calls (``kvcache.qsa_hisparse`` and the
coordinator's ``adapter``), so the ordering contract is checked without CUDA.
"""

import contextlib
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from pools.activation import activated
from sglang.srt.batch_overlap.two_batch_overlap import TboForwardBatchPreparer
from sglang.srt.environ import envs
from sglang.srt.layers.attention.base_attn_backend import SharedReadEnds
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.cuda_graph_buffer_registry import build_eager_registry
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.model_executor.model_runner import ModelRunner, ModelRunnerOutput
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import DecodeCudaGraphRunner
from sglang.srt.model_executor.runner.eager_runner import EagerRunner
from sglang.srt.model_executor.runner.shape_key import ShapeKey
from sglang.srt.model_executor.runner_backend.full_cuda_graph_backend import (
    FullCudaGraphBackend,
)
from sglang.srt.runtime_context import get_context, get_parallel
from sglang_qsa_hisparse.hisparse.coordinator import QSAHiSparseCoordinator
from sglang_qsa_hisparse.patches.hisparse import graph


class Events(list):
    def call(self, name, result=None):
        def record(*args, **kwargs):
            self.append((name, args, kwargs))
            return result

        return record

    def context(self, name):
        @contextlib.contextmanager
        def scope(*args, **kwargs):
            self.append((f"{name}:enter", args, kwargs))
            yield
            self.append((f"{name}:exit", (), {}))

        return scope

    @property
    def names(self):
        return [name for name, _, _ in self]


@pytest.fixture
def events():
    return Events()


@pytest.fixture
def published():
    with get_context().override_server_args():
        yield


def _qsa(events, graph_enabled=True):
    return SimpleNamespace(
        graph_enabled=graph_enabled,
        graph_capture=events.context("qsa.graph_capture"),
        after_graph_warmup=events.call("qsa.after_graph_warmup"),
        prepare_graph_replay=events.call("qsa.prepare_graph_replay"),
        graph_replay_scope=events.context("qsa.graph_replay_scope"),
        finish_graph_replay=events.call("qsa.finish_graph_replay"),
    )


def _coordinator(events, adapter):
    coordinator = SimpleNamespace(
        wait_for_pending_backup=events.call("wait_for_pending_backup"),
        num_real_reqs=SimpleNamespace(fill_=events.call("num_real_reqs.fill_")),
    )
    if adapter is not None:
        coordinator.adapter = adapter
    return coordinator


class _FullBackend(FullCudaGraphBackend):
    """The native full backend's call contract, recorded instead of captured."""

    def __init__(self, events):  # No CUDA state.
        self.events = events

    def capture_one(self, shape_key, forward_fn, capture_inputs=None, post_warmup_hook=None):
        self.events.append(("backend.capture_one", (shape_key,), {}))
        for _ in range(2):  # Warmups, each followed by the hook, then the capture.
            forward_fn()
            if post_warmup_hook is not None:
                post_warmup_hook()
        forward_fn()

    @contextlib.contextmanager
    def replay_session(self):
        self.events.append(("backend.replay_session:enter", (), {}))
        yield
        self.events.append(("backend.replay_session:exit", (), {}))

    def replay(self, shape_key, static_forward_batch, **kwargs):
        self.events.append(("backend.replay", (shape_key,), {}))
        return LogitsProcessorOutput(next_token_logits=torch.arange(8.0).reshape(4, 2))


def _graph_runner(events, coordinator, backend):
    runner = DecodeCudaGraphRunner.__new__(DecodeCudaGraphRunner)
    runner.model_runner = SimpleNamespace(
        hisparse_coordinator=coordinator,
        lora_manager=None,
        canary_manager=None,
        capture_tail_hooks=[],
        is_draft_worker=False,
        device_timer=None,
        spec_algorithm=SimpleNamespace(
            is_speculative=lambda: False, is_dflash_family=lambda: False
        ),
    )
    runner.backend = backend
    runner.captured_req_width = 1
    runner.ragged_verify_mode = False
    runner.pp_size = 1
    runner.capture_bs = [1, 2, 4, 8]
    runner.require_mlp_tp_gather = False
    runner.require_attn_tp_gather = False
    runner.enable_two_batch_overlap = False
    runner.enable_pdmux = False
    runner.record_nolora_graph = False
    runner.attention_graph_variants = None
    runner._metadata_glue = None
    runner.is_dllm = False
    runner.dllm_uses_input_embeds = False
    runner.capture_hidden_mode = CaptureHiddenMode.NULL
    runner.capture_forward_mode = ForwardMode.DECODE
    runner.seq_len_fill_value = 1
    runner.is_encoder_decoder = False
    runner.in_graph_metadata_prep_done = None
    runner.device_module = SimpleNamespace(is_current_stream_capturing=lambda: False)
    runner.tbo_plugin = SimpleNamespace(
        capture_one_batch_size=events.call("tbo.capture_one_batch_size")
    )
    runner.deepep_adapter = SimpleNamespace(
        capture=events.call("deepep.capture"), replay=events.call("deepep.replay")
    )
    runner.buffers = SimpleNamespace()
    runner.buffer_registry = SimpleNamespace(fill_from=events.call("buffer_registry.fill_from"))
    runner.attn_backend = SimpleNamespace(
        init_forward_metadata_out_graph=events.call("attn.init_forward_metadata_out_graph"),
        init_forward_metadata_in_graph=events.call("attn.init_forward_metadata_in_graph"),
        on_after_cuda_graph_warmup=events.call("attn.on_after_cuda_graph_warmup"),
        shared_read_ends=lambda forward_mode: SharedReadEnds.IN_REPLAY,
    )
    return runner


# G01 capture ----------------------------------------------------------------------


def _capture(events, runner, bs=2):
    forward_batch = SimpleNamespace(
        lora_ids=None,
        global_dp_buffer_len=None,
        dp_padding_mode=SimpleNamespace(is_max_len=lambda: False),
        global_num_tokens_cpu=None,
        input_ids="input_ids",
        positions="positions",
    )
    runner.capture_prepare = lambda bs, stream_idx=None, num_tokens=None: (
        forward_batch,
        runner.attn_backend,
        None,
    )
    with (
        mock.patch.object(graph, "set_dp_buffer_len", events.call("set_dp_buffer_len")),
        mock.patch.object(graph, "set_is_extend_in_batch", events.call("set_is_extend")),
    ):
        runner.capture_one_shape(bs, events.call("model.forward", "out"))


def test_capture_enters_graph_capture_and_chains_warmup_hooks(events, published):
    qsa = _qsa(events)
    backend = _FullBackend(events)
    runner = _graph_runner(events, _coordinator(events, qsa), backend)
    with activated("G01"):
        _capture(events, runner)

    key = ShapeKey(size=2, stream_idx=None, variant_label=None, attention_variant=None)
    assert events[0] == ("qsa.graph_capture:enter", (2,), {"native_key": key, "native_backend": backend})
    assert events[-1][0] == "qsa.graph_capture:exit"
    assert ("backend.capture_one", (key,), {}) in events
    # Two warmups (each followed by the chained hook: attention backend first,
    # then the runtime) and the capture, all inside graph_capture.
    warmup = ["model.forward", "attn.on_after_cuda_graph_warmup", "qsa.after_graph_warmup"]
    order = [n for n in events.names if n in warmup]
    assert order == warmup * 2 + ["model.forward"]


def test_capture_without_attention_warmup_hook_still_calls_runtime(events, published):
    qsa = _qsa(events)
    runner = _graph_runner(events, _coordinator(events, qsa), _FullBackend(events))
    del runner.attn_backend.on_after_cuda_graph_warmup
    with activated("G01"):
        _capture(events, runner)
    assert [n for n in events.names if n.endswith("warmup")] == ["qsa.after_graph_warmup"] * 2


def test_capture_requires_the_native_full_backend(events, published):
    qsa = _qsa(events)
    runner = _graph_runner(events, _coordinator(events, qsa), SimpleNamespace())
    with activated("G01"), pytest.raises(RuntimeError, match="native full graph backend"):
        _capture(events, runner)
    assert events == []


@pytest.mark.parametrize("adapter", ["absent", "disabled"])
def test_capture_without_qsa_graph_is_upstream(events, published, adapter):
    qsa = None if adapter == "absent" else _qsa(events, graph_enabled=False)
    runner = _graph_runner(events, _coordinator(events, qsa), _FullBackend(events))
    with activated("G01"):
        _capture(events, runner)
    assert not [n for n in events.names if n.startswith("qsa.")]
    assert events.names.count("attn.on_after_cuda_graph_warmup") == 2


# G02 load_batch -------------------------------------------------------------------


def _decode_batch(preplanned=False, batch_size=3):
    return SimpleNamespace(
        batch_size=batch_size,
        forward_mode=ForwardMode.DECODE,
        capture_hidden_mode=CaptureHiddenMode.NULL,
        spec_info=None,
        lora_ids=None,
        input_ids=torch.zeros(batch_size, dtype=torch.int64),
        needs_forward_metadata_init=lambda: not preplanned,
    )


def _load(events, runner, forward_batch):
    with mock.patch.object(graph, "build_replay_fb_view", events.call("build_replay_fb_view", "view")):
        runner.load_batch(forward_batch)


def test_replay_prepares_runtime_before_buffer_fill(events, published):
    qsa = _qsa(events)
    runner = _graph_runner(events, _coordinator(events, qsa), _FullBackend(events))
    forward_batch = _decode_batch()
    with activated("G02"):
        _load(events, runner, forward_batch)

    assert events.names == [
        "deepep.replay",
        "qsa.prepare_graph_replay",
        "buffer_registry.fill_from",
        "build_replay_fb_view",
        "attn.init_forward_metadata_out_graph",
    ]
    assert events[1][1] == (forward_batch, 4)
    assert events[2][2]["raw_bs"] == 3 and events[2][2]["padded_bs"] == 4
    assert events[4][1] == ("view",)
    assert (runner.raw_bs, runner.bs, runner.raw_num_token) == (3, 4, 3)
    assert runner._replay_graph_key == ShapeKey(
        size=4, stream_idx=None, variant_label=None, attention_variant=None
    )


def test_replay_rejects_external_preplanning(events, published):
    runner = _graph_runner(events, _coordinator(events, _qsa(events)), _FullBackend(events))
    with activated("G02"), pytest.raises(RuntimeError, match="external preplanning"):
        _load(events, runner, _decode_batch(preplanned=True))
    assert events.names == ["deepep.replay"]


@pytest.mark.parametrize("adapter", ["absent", "disabled"])
def test_replay_without_qsa_graph_fills_real_requests(events, published, adapter):
    qsa = None if adapter == "absent" else _qsa(events, graph_enabled=False)
    runner = _graph_runner(events, _coordinator(events, qsa), _FullBackend(events))
    with activated("G02"):
        _load(events, runner, _decode_batch())
    assert not [n for n in events.names if n.startswith("qsa.")]
    assert ("num_real_reqs.fill_", (3,), {}) in events


# G03 execute ----------------------------------------------------------------------


def _execute(events, runner):
    def load_batch(forward_batch, pp_proxy_tensors=None):
        events.append(("load_batch", (forward_batch,), {}))
        runner.bs, runner.raw_num_token = 4, 3
        runner._replay_graph_key = ShapeKey(size=4)

    runner.load_batch = load_batch
    runner._publish_read_done = events.call("publish_read_done")
    return runner.execute(_decode_batch())


def test_replay_scope_wraps_execute_and_finish_follows_publish(events, published):
    qsa = _qsa(events)
    backend = _FullBackend(events)
    runner = _graph_runner(events, _coordinator(events, qsa), backend)
    runner.in_graph_metadata_prep_done = "in-graph event"  # IN_REPLAY stays IN_REPLAY.
    with activated("G03"):
        output = _execute(events, runner)

    # Deviation D6: the scope encloses the whole execute call (the fork enters
    # it inside replay_session), and finish runs in _process_output_after_replay,
    # after the IN_REPLAY publish (the fork calls it right after replay).
    assert events.names == [
        "qsa.graph_replay_scope:enter",
        "backend.replay_session:enter",
        "load_batch",
        "backend.replay",
        "publish_read_done",
        "qsa.finish_graph_replay",
        "backend.replay_session:exit",
        "qsa.graph_replay_scope:exit",
    ]
    assert events[5][1:] == ((4,), {"native_key": ShapeKey(size=4), "native_backend": backend})
    assert events[4][2] == {"in_graph": True}
    assert output.next_token_logits.shape == (3, 2)


def test_replay_without_qsa_graph_is_upstream(events, published):
    runner = _graph_runner(events, _coordinator(events, None), _FullBackend(events))
    runner.in_graph_metadata_prep_done = "in-graph event"
    with activated("G03"):
        _execute(events, runner)
    assert events.names == [
        "backend.replay_session:enter",
        "load_batch",
        "backend.replay",
        "publish_read_done",
        "backend.replay_session:exit",
    ]


# R02 _forward_raw, _prepare_eager_forward_batch -------------------------------------


def _model_runner(events, coordinator, can_run_graph):
    runner = ModelRunner.__new__(ModelRunner)
    runner.device = "cuda"
    runner.attn_backend = SimpleNamespace()
    runner.hisparse_coordinator = coordinator
    runner.decode_cuda_graph_runner = SimpleNamespace(
        can_run_graph=lambda forward_batch: can_run_graph,
        execute=events.call("graph.execute", "graph output"),
    )
    return runner


def test_qsa_graph_decode_cannot_fall_back_to_eager(events):
    runner = _model_runner(events, _coordinator(events, _qsa(events)), can_run_graph=False)
    forward_batch = SimpleNamespace(forward_mode=ForwardMode.DECODE, batch_size=2)
    with activated("R02"), pytest.raises(RuntimeError, match="cannot fall back to eager"):
        runner._forward_raw(forward_batch, None)
    # Deviation D7: the raise comes from the eager branch's first call, after
    # the pinned coordinator block (the fork raises before it).
    assert events.names == ["wait_for_pending_backup", "num_real_reqs.fill_"]


@pytest.mark.parametrize("graph_enabled", [True, False])
def test_qsa_graph_decode_replays_after_the_pinned_fill(events, graph_enabled):
    coordinator = _coordinator(events, _qsa(events, graph_enabled=graph_enabled))
    runner = _model_runner(events, coordinator, can_run_graph=True)
    forward_batch = SimpleNamespace(forward_mode=ForwardMode.DECODE, batch_size=2)
    with activated("R02"):
        output = runner._forward_raw(forward_batch, "pp")

    # Deviation D7: the pinned fill also runs under the QSA graph (the fork skips
    # it); it writes the batch size that prepare_graph_replay writes again.
    assert events.names == ["wait_for_pending_backup", "num_real_reqs.fill_", "graph.execute"]
    assert events[1][1] == (2,)
    assert events[-1][2] == {"pp_proxy_tensors": "pp"}
    assert forward_batch.hisparse_coordinator is coordinator
    assert output == ModelRunnerOutput(logits_output="graph output", can_run_graph=True)


@pytest.fixture
def stub_prepare_eager(events):
    original = ModelRunner.__dict__["_prepare_eager_forward_batch"]
    ModelRunner._prepare_eager_forward_batch = events.call("prepare_eager")
    yield
    ModelRunner._prepare_eager_forward_batch = original


@pytest.mark.parametrize(
    "mode, graph_enabled",
    [(ForwardMode.DECODE, True), (ForwardMode.DECODE, False), (ForwardMode.EXTEND, True)],
    ids=["graph-decode", "eager-decode", "graph-extend"],
)
def test_eager_preparation_rejects_only_qsa_graph_decode(
    events, stub_prepare_eager, mode, graph_enabled
):
    runner = _model_runner(
        events, _coordinator(events, _qsa(events, graph_enabled)), can_run_graph=False
    )
    forward_batch = SimpleNamespace(forward_mode=mode, batch_size=2)
    with activated("R02"):
        if mode.is_decode() and graph_enabled:
            with pytest.raises(RuntimeError, match="cannot fall back to eager"):
                runner._prepare_eager_forward_batch(forward_batch)
            assert events == []
        else:
            runner._prepare_eager_forward_batch(forward_batch)
            assert events.names == ["prepare_eager"]


# R01 init_attention_backends -------------------------------------------------------


@pytest.fixture
def stub_init_attention_backends(events):
    original = ModelRunner.__dict__["init_attention_backends"]
    ModelRunner.init_attention_backends = events.call("init_attention_backends")
    yield
    ModelRunner.init_attention_backends = original


def _runner_with_pool(runtime):
    runner = ModelRunner.__new__(ModelRunner)
    runner.token_to_kv_pool = SimpleNamespace()
    if runtime is not None:
        runner.token_to_kv_pool.qsa_hisparse = runtime
    runner.hisparse_coordinator = None
    return runner


def test_coordinator_follows_attention_backends(events, stub_init_attention_backends):
    adapter = SimpleNamespace(
        uses_qsa_hisparse_leases=True, max_requests=2, mode="p2-offload", real="real rows"
    )
    runner = _runner_with_pool(adapter)
    # Production passes the live TP CPU group (the pin's ModelRunner has no tp_group).
    tp_group = SimpleNamespace(cpu_group="tp cpu group")
    with activated("R01"), get_parallel().override(tp_group=tp_group):
        runner.init_attention_backends()
    assert events.names == ["init_attention_backends"]
    coordinator = runner.hisparse_coordinator
    assert isinstance(coordinator, QSAHiSparseCoordinator)
    assert coordinator.adapter is adapter
    assert coordinator.tp_group == "tp cpu group"
    assert coordinator.num_real_reqs == "real rows"


@pytest.mark.parametrize("runtime", [None, SimpleNamespace(mode="offload")])
def test_no_coordinator_without_lease_runtime(events, stub_init_attention_backends, runtime):
    runner = _runner_with_pool(runtime)
    with activated("R01"):
        runner.init_attention_backends()
    assert events.names == ["init_attention_backends"]
    assert runner.hisparse_coordinator is None


# F01 ForwardBatch.init_new ---------------------------------------------------------


def _forward_batch(req_pool_indices_cpu=None):
    return ForwardBatch(
        forward_mode=ForwardMode.DECODE,
        batch_size=2,
        input_ids=torch.tensor([5, 6]),
        req_pool_indices=torch.tensor([1, 2]),
        seq_lens=torch.tensor([9, 10]),
        out_cache_loc=torch.tensor([100, 200]),
        seq_lens_sum=19,
        seq_lens_cpu=torch.tensor([9, 10]),
        positions=torch.tensor([8, 9]),
        req_pool_indices_cpu=req_pool_indices_cpu,
    )


@pytest.fixture
def stub_init_new():
    """Stand-in for the pinned init_new; the test sets ``upstream["rows"]``, the
    value upstream leaves in ``req_pool_indices_cpu`` (None except for extend
    without speculative decoding)."""
    upstream = {"rows": None}
    original = ForwardBatch.__dict__["init_new"]
    ForwardBatch.init_new = classmethod(
        lambda cls, batch, model_runner, **kwargs: _forward_batch(upstream["rows"])
    )
    yield upstream
    ForwardBatch.init_new = original


def _init_new(runtime):
    pool = SimpleNamespace() if runtime is None else SimpleNamespace(qsa_hisparse=runtime)
    batch = SimpleNamespace(req_pool_indices_cpu=torch.tensor([1, 2]))
    return batch, ForwardBatch.init_new(
        batch,
        SimpleNamespace(token_to_kv_pool=pool),
        capture_hidden_mode=None,
        return_hidden_states_before_norm=False,
    )


def test_forward_batch_carries_cpu_request_rows_with_runtime(stub_init_new):
    with activated("F01"):  # Decode: upstream leaves the field None.
        batch, forward_batch = _init_new(runtime=object())
    assert forward_batch.req_pool_indices_cpu is batch.req_pool_indices_cpu


@pytest.mark.parametrize("upstream_rows", [None, torch.tensor([7, 8])], ids=["unset", "set"])
def test_two_batch_overlap_unaffected_without_runtime(stub_init_new, published, upstream_rows):
    stub_init_new["rows"] = upstream_rows
    fields = set(ForwardBatch.__dataclass_fields__)
    with activated("F01"):
        batch, forward_batch = _init_new(runtime=None)
        # No class replace: the plugin adds no ForwardBatch field (the fork's
        # kv_allocated_lens_cpu is dropped).
        assert set(ForwardBatch.__dataclass_fields__) == fields
    assert "kv_allocated_lens_cpu" not in fields
    # Without the runtime the field is exactly what upstream set, never the
    # ScheduleBatch rows the hook would copy.
    assert forward_batch.req_pool_indices_cpu is upstream_rows
    assert upstream_rows is None or upstream_rows is not batch.req_pool_indices_cpu
    child = TboForwardBatchPreparer.filter_batch(
        forward_batch,
        start_token_index=0,
        end_token_index=1,
        start_seq_index=0,
        end_seq_index=1,
        out_num_token_non_padded=torch.tensor(1),
    )
    assert child.batch_size == 1
    assert child.req_pool_indices.tolist() == [1]
    if upstream_rows is None:
        assert child.req_pool_indices_cpu is None
    else:
        assert child.req_pool_indices_cpu.tolist() == [7]


def _eager_runner():
    runner = EagerRunner.__new__(EagerRunner)
    # The pinned eager registry (the slot set EagerRunner.__init__ builds).
    runner._eager_registry = build_eager_registry(
        device="cpu", max_bs=8, max_num_token=64, cache_loc_dtype=torch.int64
    )
    return runner


@pytest.mark.parametrize("no_copy", [False, True])
def test_eager_batch_copy_keeps_cpu_request_rows(published, no_copy):
    runner = _eager_runner()
    rows = torch.tensor([1, 2])
    with_rows = _forward_batch(rows)
    without_rows = _forward_batch()
    with envs.SGLANG_EAGER_INPUT_NO_COPY.override(no_copy):
        pinned = runner.load_batch(with_rows)
        with activated("F01"):
            active = runner.load_batch(with_rows)
            inert = runner.load_batch(without_rows)
    # The eager runner hands a dataclasses.replace copy to init_forward_metadata,
    # where the runtime's begin_batch reads the rows. Upstream's field survives
    # that copy (the eager registry has no slot for it), so F01 needs no hook on
    # EagerRunner.load_batch.
    for copy in (pinned, active):
        assert copy is not with_rows
        assert copy.req_pool_indices_cpu is rows
    assert inert is not without_rows and inert.req_pool_indices_cpu is None
