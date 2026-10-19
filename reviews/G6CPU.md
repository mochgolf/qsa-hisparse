- **P2 — Host-cache fingerprints omit the tree core.** [lifecycle.py:268](/home/zyk/projects/interests/ai-video/qwen/qsa-hisparse-plugin/src/sglang_qsa_hisparse/patches/hisparse/lifecycle.py:268) fingerprints `UnifiedRadixCache`, but omits the tree-core registry, Rust adapter and Python core that supply its empty match, page size and capacity values.

  Concrete scenario: drift in the Rust adapter reports a nonzero device prefix for an empty match. Activation passes, but host-hit admission gains that bogus prefix and M01 rejects the restore. I demonstrated this read-only with an in-memory adapter mutation: the empty match became length 64 while all 402 fingerprint checks still passed.

  **Smallest fix:** add M05 dependencies covering `create_tree_core`, `RustUnifiedTreeCore` and `UnifiedTreeCore`; regenerate fingerprints and add a focused drift-rejection test.

No other in-scope blocker found in the restore/release paths, narrowed hooks, hand merges or test changes. The NUMA removals match upstream; removing the renamed-copy tests is justified.

Fingerprints, manifest, and source-only inventory/moved-source/REPLACE guards passed. Normal CPU pytest runs were blocked by unwritable temporary storage. No files changed.

G6-CPU: not cleared