# I-A image identity and matching (Track I: I1 + I2)

Contract: `src/sglang_qsa_hisparse/hisparse/image_identity.py` (fixed; do
not change it without the orchestrator). Design: `docs/prefix-cache.md` §6.
Acceptance: PLAN.md "Track I acceptance" criteria 1, 2, 4, 7.

1. **Transport (I1).** Make the full artifact key of each Qwen-VL image item
   reach the scheduler: where the fast path builds items from
   `QwenVLImagePreprocessArtifact` (pinned `multimodal/processors/qwen_vl.py`),
   record `artifact.artifact_key` on the item (e.g. in `model_specific_data`)
   with the narrowest hook. Confirm it survives the tokenizer→scheduler
   transport. Items without an artifact key are not reusable.
2. **`identity_for(req)`** in a new plugin module: return an
   `ImagePrefixIdentity` for a supported image request, `None` for a text
   request, and a bypass marker for unsupported ones: video/audio,
   precomputed embeddings, `SGLANG_MM_SKIP_COMPUTE_HASH`, items with more
   than one offset span, items without an artifact key, missing
   `mrope_positions`. Compute once per request attempt (cache on the
   request object).
3. **Matching (I2).** `PrefixSnapshot` gains the identity key at its length;
   `HostPrefixCache.acquire` accepts a candidate only when tokens are a prefix
   and the key at the candidate length equals the request's; `capture_prefix`
   ancestor lookup uses the same rule; snapshot `signature` and `_capture`'s
   TP signature include a digest of the key. Text requests behave exactly as
   before (no identity, match only snapshots without images).
4. **Tests (CPU), counterexamples first:** forced `pad_value` collision with
   different artifact keys (miss), same content with different preprocessing
   (miss), swapped image order (miss), different grid (miss), different image
   after `L` (hit), straddling image different (miss), M-RoPE digest differs
   (miss), text request unchanged (all existing prefix tests pass).

Owned paths: new modules under `src/sglang_qsa_hisparse/hisparse/` for
identity, edits to `hisparse/prefix.py` and `hisparse/prefix_cache.py` limited
to the matching rule and signatures, processor hook in
`patches/hisparse/image_identity.py` (manifest row `I1`, report it), tests in
`tests/image/`. These intentionally diverge from the fork: update
`tests/runtime/test_moved_sources.py` to list `prefix.py` and
`prefix_cache.py` as Track I divergences and compare them against the fork
plus your recorded diff.
