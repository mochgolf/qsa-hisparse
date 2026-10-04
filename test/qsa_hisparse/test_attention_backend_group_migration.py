"""``ModelRunner.init_attention_backends`` must pass the live TP CPU group.

Upstream removed ``ModelRunner.tp_group``; our QSA coordinator construction read
it and failed production startup on both TP ranks. This executes the real
``init_attention_backends`` body with the backend/coordinator setup mocked, using
a runner without the old field (the production condition) and a runner carrying
a stale group, while the live parallel context publishes a distinct group. Only
the live group may reach ``QSAHiSparseCoordinator``.
"""

from types import SimpleNamespace

import pytest

from sglang.srt.model_executor import model_runner as model_runner_module
from sglang.srt.model_executor.model_runner import ModelRunner

LIVE_GROUP = "live-tp-cpu-group"
STALE_GROUP = "stale-tp-cpu-group"


def _runner(**extra):
    runner = ModelRunner.__new__(ModelRunner)
    runner.model = SimpleNamespace()
    runner.spec_aux_config = SimpleNamespace(
        eagle_use_aux_hidden_state=False,
        eagle_aux_hidden_state_layer_ids=None,
        dflash_use_aux_hidden_state=False,
        dflash_target_layer_ids=None,
    )
    runner.spec_algorithm = SimpleNamespace(is_dspark=lambda: False)
    runner.kv_index_translator = SimpleNamespace(
        bind_and_verify_backends=lambda *a: None
    )
    runner.token_to_kv_pool = SimpleNamespace(
        qsa_hisparse=SimpleNamespace(uses_qsa_hisparse_leases=True)
    )
    for name, value in extra.items():
        setattr(runner, name, value)
    return runner


@pytest.fixture
def _live_context(monkeypatch):
    """Publish the live parallel bundle; the runner must not need its own copy."""

    monkeypatch.setattr(
        model_runner_module,
        "get_parallel",
        lambda: SimpleNamespace(
            tp_group=SimpleNamespace(cpu_group=LIVE_GROUP),
            dcp_enabled=False,
        ),
    )
    monkeypatch.setattr(
        model_runner_module, "configure_aux_hidden_state_capture", lambda **kwargs: None
    )
    monkeypatch.setattr(
        model_runner_module,
        "resolve_attention_backend_strs",
        lambda **kwargs: SimpleNamespace(prefill="qsa", decode="qsa"),
    )
    monkeypatch.setattr(
        model_runner_module,
        "build_attention_backends",
        lambda **kwargs: SimpleNamespace(
            attn_backend="prefill-backend",
            decode_attn_backend="decode-backend",
            decode_attn_backend_group="decode-group",
        ),
    )


@pytest.fixture
def _recorded_coordinator(monkeypatch):
    recorded = []

    class _Coordinator:
        def __init__(self, adapter, tp_group):
            recorded.append((adapter, tp_group))

    monkeypatch.setattr(
        "sglang.srt.mem_cache.qsa_hisparse.coordinator.QSAHiSparseCoordinator",
        _Coordinator,
    )
    return recorded


def test_init_attention_backends_passes_the_live_tp_cpu_group(
    _live_context, _recorded_coordinator
):
    runner = _runner()
    assert not hasattr(runner, "tp_group")  # the production condition

    runner.init_attention_backends()

    assert len(_recorded_coordinator) == 1
    adapter, group = _recorded_coordinator[0]
    assert group == LIVE_GROUP
    assert adapter is runner.token_to_kv_pool.qsa_hisparse
    assert runner.attn_backend == "prefill-backend"
    assert runner.decode_attn_backend == "decode-backend"


def test_init_attention_backends_ignores_a_stale_runner_group(
    _live_context, _recorded_coordinator
):
    runner = _runner(tp_group=SimpleNamespace(cpu_group=STALE_GROUP))
    assert runner.tp_group.cpu_group == STALE_GROUP  # pre-migration attribute

    runner.init_attention_backends()

    _, group = _recorded_coordinator[0]
    assert group == LIVE_GROUP
    assert group != STALE_GROUP


def test_coordinator_is_only_built_for_lease_runtimes(
    _live_context, _recorded_coordinator
):
    runner = _runner()
    runner.token_to_kv_pool = SimpleNamespace(qsa_hisparse=None)

    runner.init_attention_backends()

    assert _recorded_coordinator == []
