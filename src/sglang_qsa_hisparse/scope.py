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

Implementation: the published ``model`` namespace supplies exactly the
arguments ``ModelConfig.from_server_args`` hands to SGLang's ``get_config``
(model path, revision, ``trust_remote_code``, ``--json-model-override-args``,
``--model-config-parser`` and the decrypted config file), so the
architectures are those of ``ModelConfig.hf_config`` before any draft-model
rewrite. Unreadable or unpublished configuration, a missing or empty
``architectures`` list, or a list mixing target and non-target entries is
undecidable.
"""

import functools
import json

from sglang_qsa_hisparse.errors import PluginActivationError

TARGET_ARCHITECTURES = frozenset({"Qwen4ExpForConditionalGeneration"})


class ScopeUndecidable(PluginActivationError):
    """The served model's architecture cannot be determined."""


def _served_architectures() -> object:
    """``hf_config.architectures`` of the published model, as ModelConfig reads it."""
    from sglang.srt.runtime_context import get_model
    from sglang.srt.utils.hf_transformers_utils import get_config

    model = get_model()
    kwargs = {}
    # ModelConfig.from_server_args passes decrypted_config_file as
    # override_config_file, which ModelConfig forwards as _configuration_file.
    if model.decrypted_config_file and model.decrypted_config_file.strip():
        kwargs["_configuration_file"] = model.decrypted_config_file.strip()
    config = get_config(
        model.model_path,
        trust_remote_code=model.trust_remote_code,
        revision=model.revision,
        model_override_args=json.loads(model.json_model_override_args),
        model_config_parser=model.model_config_parser,
        **kwargs,
    )
    return config.architectures


@functools.cache
def target_model_active() -> bool:
    try:
        architectures = _served_architectures()
    except Exception as error:
        raise ScopeUndecidable(
            f"Cannot read the served model's architectures: {error!r}"
        ) from error
    if (
        not isinstance(architectures, (list, tuple))
        or not architectures
        or not all(isinstance(name, str) for name in architectures)
    ):
        raise ScopeUndecidable(
            f"The served model declares no usable architectures: {architectures!r}"
        )
    decisions = {name in TARGET_ARCHITECTURES for name in architectures}
    if len(decisions) != 1:
        raise ScopeUndecidable(
            f"The served model's architectures {list(architectures)} mix target "
            f"and non-target entries"
        )
    return decisions.pop()
