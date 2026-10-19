# G6-CPU response

Review: `reviews/G6CPU.md` (one P2 finding, not cleared).

**P2 — host-cache fingerprints omit the tree core.** Accepted (it is within
the activation threat model: byte drift in a module the plugin relies on).
M05's `depends` now also names
`mem_cache.unified_cache.tree_core_registry.{create_tree_core,select_tree_core_backend}`,
`mem_cache.unified_cache.unified_tree_core.UnifiedTreeCore` and
`mem_cache.rust_tree_core.adapter.RustUnifiedTreeCore` (they supply the
disabled cache's empty root match), fingerprinted against v0.5.22: 406
fingerprints match. No dedicated drift test: every `depends` entry already
fails activation on any module byte change through the generic mechanism
covered by `tests/test_framework.py`. The Rust sources themselves are not
Python modules and stay outside fingerprints, like the in-tree `.cuh`
headers (W5 precedent): native behavior is established by tests and the
GPU gate.

Full CPU suite after the change: 723 + 4 + 11 passed, 0 failed.

## Re-check (`reviews/G6CPUr.md`)

P2 resolved, one new P2: `RustUnifiedTreeCore` in `depends` makes
activation import the adapter, which loads the optional native extension and
would fail activation where upstream falls back to the Python core.
Accepted: `RustUnifiedTreeCore` is removed from M05's `depends` (and its
record from `fingerprints/hisparse_lifecycle.json`); `create_tree_core`,
`select_tree_core_backend` and `UnifiedTreeCore` stay (the cache module
imports them anyway; importing it does not import the adapter, checked).
The adapter is the Python side of the native tree core and is treated like
the Rust sources and `.cuh` headers: covered by the CPU tests that build the
cache and by the GPU gate, not by activation fingerprints. No "byte check
without import" machinery is added (PLAN rule 10). 405 fingerprints; CPU
suite 723 + 4 + 11 passed.
