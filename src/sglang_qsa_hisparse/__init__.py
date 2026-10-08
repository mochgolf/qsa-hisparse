"""QSA HiSparse as an out-of-tree SGLang general plugin.

Importing this package has no side effects. Patches are applied only by
``sglang_qsa_hisparse.plugin.load`` when a feature switch is set.
"""

PINNED_SGLANG_COMMIT = "35f3c96ff4794a4de15daf12caad371084a037ee"  # upstream main, 2026-10-04
# The reference whose behavior the plugin reproduces: the production build
# (fork ee8fe158d6 merged with upstream 35f3c96ff4, plus production fixes).
REFERENCE_FORK_COMMIT = "897286b12a128d4bcb8229a5dadedc1ea49fcc16"
# The upstream commit the reference branched from (its changes are defined
# relative to it).
FORK_BASE_COMMIT = "35f3c96ff4794a4de15daf12caad371084a037ee"

__all__ = ["FORK_BASE_COMMIT", "PINNED_SGLANG_COMMIT", "REFERENCE_FORK_COMMIT"]
