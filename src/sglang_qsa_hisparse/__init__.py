"""QSA HiSparse as an out-of-tree SGLang general plugin.

Importing this package has no side effects. Patches are applied only by
``sglang_qsa_hisparse.plugin.load`` when a feature switch is set.
"""

PINNED_SGLANG_COMMIT = "76e06febab732d28a61b75a61b7835284568cdfb"
REFERENCE_FORK_COMMIT = "ee8fe158d64186b47236b007a299696030c372e8"

__all__ = ["PINNED_SGLANG_COMMIT", "REFERENCE_FORK_COMMIT"]
