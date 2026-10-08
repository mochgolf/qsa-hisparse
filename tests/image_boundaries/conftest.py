import pytest

from sglang.srt.runtime_context import get_context
from sglang_qsa_hisparse.hisparse import prefix_cache


@pytest.fixture
def identities(monkeypatch):
    """Stand-in for I-A's ``identity_for``: rid -> what the test assigns."""
    table = {}
    monkeypatch.setattr(prefix_cache, "identity_for", lambda req: table.get(req.rid))
    return table


@pytest.fixture
def published_context():
    """A published SGLang config context, as upstream unit tests install it."""
    with get_context().override_server_args(attention_backend="torch_native", dcp_size=1):
        yield
