"""Activation failures that SGLang's plugin loader cannot swallow.

``sglang.srt.plugins.load_plugins`` and ``HookRegistry.apply_hooks`` catch
``Exception`` and only log it. A requested feature that fails to activate
must stop the process instead of serving unpatched upstream behavior, so
activation errors derive from ``BaseException``.
"""


class PluginActivationError(BaseException):
    """A requested feature could not be activated exactly as pinned."""


class PluginConfigError(PluginActivationError):
    """Feature switches are malformed or form an unsupported combination."""


class FingerprintMismatch(PluginActivationError):
    """A patch target's source differs from the pinned SGLang revision."""
