"""Target-model scope for generic ``model_compat`` hooks (PLAN.md rule 9).

Contract (fixed; implementation owned by W7):

- ``TARGET_ARCHITECTURES`` is the fork's validated model set.
- ``target_model_active()`` returns True iff the model this process serves
  has an architecture in that set, as resolved from SGLang's published model
  configuration (``get_model().model_path`` and any architecture override in
  the published model arguments, read the way SGLang's ``ModelConfig`` does).
  The answer is computed once per process and cached.
- If the published configuration is unavailable or ambiguous, it raises
  ``ScopeUndecidable``; it never guesses.

Generic hooks call the original SGLang code when this returns False.
Unit tests of hooks monkeypatch ``target_model_active``.
"""

from sglang_qsa_hisparse.errors import PluginActivationError

TARGET_ARCHITECTURES = frozenset({"Qwen4ExpForConditionalGeneration"})


class ScopeUndecidable(PluginActivationError):
    """The served model's architecture cannot be determined."""


def target_model_active() -> bool:
    raise ScopeUndecidable("target_model_active is not implemented yet (W7)")
