# I-B image requests at every page64 boundary (Track I: I3)

Contract: `src/sglang_qsa_hisparse/hisparse/image_identity.py` (fixed). I-A
implements `identity_for(req)` and matching in parallel; code against the
contract and stub `identity_for` in your tests. Design: `docs/prefix-cache.md`
§6. Acceptance: PLAN.md "Track I acceptance" criteria 3, 5, 6 (CPU part), 7.

1. Let supported image requests through `QSAHostPrefixCache._namespace` and
   `prefill_checkpoint_limit` (the bypass set stays bypassed, via I-A's
   marker), so checkpoints are captured and restored at every page64
   boundary, including inside an image.
2. After a hit of length `L` inside or after an image, the suffix must be
   computed exactly as an uncached chunked prefill would at the same
   boundary: verify on the pin that `general_mm_embed_routine` slices the
   straddling image's embedding by `extend_prefix_len`, and that the per-image
   ViT cache is reused for images that end before `L` (skipped work) while a
   straddling image's embedding is computed whole and sliced. Patch only what
   the pin gets wrong, with the narrowest hook (manifest rows `I3*`, report
   them).
3. PLE n-gram history over image pad tokens: confirm the captured recurrent
   and PLE state covers it (it is part of the restored state); add a test.
4. A hit always leaves the logits tail to compute, and input-logprob limits
   are unchanged, for image requests too.
5. M-RoPE suffix positions come from the request's own `mrope_positions`
   sliced at `L`; test a hit whose suffix starts inside an image.
6. CPU tests with fake models/embedding paths where needed; every claim in
   items 2–5 has a test. Text-only behavior unchanged.

Owned paths: `patches/hisparse/image_boundaries.py`, edits to
`hisparse/prefix_cache.py` limited to `_namespace` and
`prefill_checkpoint_limit` (coordinate through the orchestrator if I-A's edits
overlap), `tests/image_boundaries/`.
