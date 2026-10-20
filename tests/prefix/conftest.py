import pytest

from lifecycle.activation import w2_active
from sglang.srt.runtime_context import get_context


@pytest.fixture(autouse=True)
def published_context():
    """A published default SGLang config, as upstream unit tests install it.

    At v0.5.21 prefill admission reads the published config
    (``PrefillAdder._commit_prefill_admission`` calls ``get_exec()``); the
    fork's tests ran at a pin whose admission path did not. At v0.5.22 the
    host prefix cache's mamba component reads the mamba cache chunk size,
    which loads the model's HF config unless seeded (as upstream's unit
    tests seed it): FLA's chunk size for the dummy model.
    """
    override = get_context().override_server_args(_mamba_cache_chunk_size=64)
    override.install()
    yield
    override.restore()


@pytest.fixture(scope="module")
def w2_patches():
    """The host prefix cache's hooks (M01, M03, B06; W2's rows), in-process."""
    with w2_active():
        yield
