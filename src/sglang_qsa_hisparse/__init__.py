"""QSA HiSparse as an out-of-tree SGLang general plugin.

Importing this package has no side effects. Patches are applied only by
``sglang_qsa_hisparse.plugin.load`` when a feature switch is set.
"""

PINNED_SGLANG_COMMIT = "e00930c5489053f26d86b179cee0d087f846acbb"  # v0.5.21
REFERENCE_FORK_COMMIT = "ee8fe158d64186b47236b007a299696030c372e8"
# The upstream commit the reference fork branched from (its changes are
# defined relative to it).
FORK_BASE_COMMIT = "76e06febab732d28a61b75a61b7835284568cdfb"

__all__ = ["FORK_BASE_COMMIT", "PINNED_SGLANG_COMMIT", "REFERENCE_FORK_COMMIT"]
