"""SGLang general-plugin entry point (``sglang.srt.plugins`` group)."""

from sglang_qsa_hisparse.features import read_features


def load() -> None:
    """Activate requested features; with no switch set, change nothing."""
    features = read_features()
    if not features.active:
        return
    from sglang_qsa_hisparse.patching import activate

    activate(features)
