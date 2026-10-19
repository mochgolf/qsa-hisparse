The original P2 fingerprint gap is resolved: all 406 fingerprints match, and in-memory byte changes to each newly covered module are rejected.

- **P2 — Activation now unconditionally loads the optional Rust backend.** [lifecycle.py:284](/home/zyk/projects/interests/ai-video/qwen/qsa-hisparse-plugin/src/sglang_qsa_hisparse/patches/hisparse/lifecycle.py:284) adds `RustUnifiedTreeCore` to `depends`. Activation freezes dependencies through `_raw_attribute`, importing the adapter and immediately loading its native extension. When Rust tooling is unavailable, upstream selects the Python fallback; this new import instead fails activation, even with `SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python`. **Smallest fix:** retain the adapter’s source-byte verification without eagerly importing the optional backend to freeze its identity.

Manifest verification passed. The runtime probe was blocked by unwritable temporary storage; the finding follows the import and fallback paths. No files changed.

G6-CPU: not cleared