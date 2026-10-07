"""Feature switches. Reading them imports nothing from SGLang.

``model_compat`` carries the reference fork's shared-path behavior: quantized
MoE loading, INT8-row PLE, QSA FP8 descales, and deterministic kernels.
``hisparse`` is the request-scoped CPU KV offload runtime and host prefix
cache; it keeps the fork's ``SGLANG_QSA_HISPARSE_V3`` spelling and requires
``model_compat`` because the fork never ran it without those changes.
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Optional

from sglang_qsa_hisparse.errors import PluginConfigError

MODEL_COMPAT_ENV = "SGLANG_QSA_MODEL_COMPAT"
HISPARSE_ENV = "SGLANG_QSA_HISPARSE_V3"
HISPARSE_MODES = ("p2-offload", "p2-resident", "offload", "resident")

MODEL_COMPAT = "model_compat"
HISPARSE = "hisparse"
FEATURES = (MODEL_COMPAT, HISPARSE)


@dataclass(frozen=True)
class Features:
    model_compat: bool = False
    hisparse_mode: Optional[str] = None

    @property
    def active(self) -> tuple[str, ...]:
        names = []
        if self.model_compat:
            names.append(MODEL_COMPAT)
        if self.hisparse_mode is not None:
            names.append(HISPARSE)
        return tuple(names)


def read_features(environ: Optional[Mapping[str, str]] = None) -> Features:
    environ = os.environ if environ is None else environ
    compat = environ.get(MODEL_COMPAT_ENV, "")
    if compat not in ("", "0", "1"):
        raise PluginConfigError(f"{MODEL_COMPAT_ENV} must be 0 or 1, got {compat!r}")
    mode = environ.get(HISPARSE_ENV) or None
    if mode is not None and mode not in HISPARSE_MODES:
        raise PluginConfigError(
            f"{HISPARSE_ENV} must be one of {HISPARSE_MODES}, got {mode!r}"
        )
    if mode is not None and compat != "1":
        raise PluginConfigError(f"{HISPARSE_ENV} requires {MODEL_COMPAT_ENV}=1")
    return Features(model_compat=compat == "1", hisparse_mode=mode)
