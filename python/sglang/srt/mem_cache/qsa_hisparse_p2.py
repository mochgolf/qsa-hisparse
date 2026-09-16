"""Compatibility imports for the former P2 experiment module."""

from sglang.srt.mem_cache.qsa_hisparse.coordinator import QSAHiSparseCoordinator
from sglang.srt.mem_cache.qsa_hisparse.runtime import QSAHiSparseRuntime, _RequestCache

QSAHiSparseP2 = QSAHiSparseRuntime

__all__ = ["QSAHiSparseP2", "QSAHiSparseCoordinator", "_RequestCache"]
