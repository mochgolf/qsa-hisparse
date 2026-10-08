import pytest

from sglang.srt.runtime_context import get_context


@pytest.fixture(autouse=True)
def published_context():
    """A published default SGLang config, as upstream unit tests install it.

    At v0.5.21 prefill admission reads the published config
    (``PrefillAdder._commit_prefill_admission`` calls ``get_exec()``); the
    fork's tests ran at a pin whose admission path did not.
    """
    override = get_context().override_server_args()
    override.install()
    yield
    override.restore()
