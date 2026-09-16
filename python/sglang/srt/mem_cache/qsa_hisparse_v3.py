"""Compatibility imports for the former single-request V3 experiment module."""

from sglang.srt.mem_cache.qsa_hisparse.config import validate_configuration
from sglang.srt.mem_cache.qsa_hisparse.layout import (
    pack_c4,
    stage_short_prefix,
    unpack_index,
)
from sglang.srt.mem_cache.qsa_hisparse.single_request import QSAHiSparseSingleRequest

QSAHiSparseV3 = QSAHiSparseSingleRequest

__all__ = [
    "QSAHiSparseV3",
    "validate_configuration",
    "pack_c4",
    "stage_short_prefix",
    "unpack_index",
]
