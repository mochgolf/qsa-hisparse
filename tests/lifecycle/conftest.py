"""Fixtures of W2's tests: in-process activation of W2's rows only
(``activation.py``) and a published SGLang config."""

from types import SimpleNamespace

import pytest

from lifecycle.activation import W2_ROWS, import_w2_modules, w2_active, w2_targets


@pytest.fixture(scope="module")
def w2_patches():
    with w2_active():
        yield


@pytest.fixture
def w2():
    import_w2_modules()
    return SimpleNamespace(rows=W2_ROWS, targets=w2_targets(), active=w2_active)


@pytest.fixture
def published_context():
    """A published SGLang config context, as upstream unit tests install it."""
    from sglang.srt.runtime_context import get_context

    # A disabled UnifiedRadixCache's mamba component reads the mamba cache chunk
    # size, which loads the model's HF config unless seeded (as upstream's
    # unit tests seed it): FLA's chunk size for the dummy model.
    with get_context().override_server_args(
        attention_backend="torch_native", dcp_size=1, _mamba_cache_chunk_size=64
    ):
        yield


@pytest.fixture
def target_model(monkeypatch):
    """Answer the rule-9 scope question for scoped model_compat hooks."""
    from sglang_qsa_hisparse import scope

    def set_active(active: bool) -> None:
        monkeypatch.setattr(scope, "target_model_active", lambda: active)

    set_active(True)
    return set_active
