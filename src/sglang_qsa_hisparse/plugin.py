"""SGLang general-plugin entry point (``sglang.srt.plugins`` group)."""

from sglang_qsa_hisparse.errors import PluginActivationError
from sglang_qsa_hisparse.features import read_features


def load() -> None:
    """Activate requested features; with no switch set, change nothing.

    SGLang's loader logs and ignores ``Exception`` from plugins, so every
    failure on the enabled path (imports, fingerprints, registration) is
    converted to ``PluginActivationError``, which it does not catch.
    """
    features = read_features()
    if not features.active:
        return
    try:
        from sglang_qsa_hisparse.patching import activate

        activate(features)
    except PluginActivationError:
        raise
    except Exception as error:
        raise PluginActivationError(
            f"QSA HiSparse activation of {features.active} failed: {error!r}"
        ) from error
