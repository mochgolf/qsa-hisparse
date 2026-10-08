# I-C image prefix GPU evidence harness (Track I: I5 preparation)

Build the tooling for the Track I GPU evidence run; the orchestrator runs it
in an owner-approved GPU window. CPU-only development with a fake server.
Acceptance: PLAN.md "Track I acceptance" criteria 1–7 (GPU part).

1. `tools/evidence/image_fixtures.py freeze --output <file>`: deterministic
   synthetic images (fixed seeds, PNG bytes inside the fixture file, sizes
   giving image token spans well over 64 tokens) and chat prompts built
   from them with the model's tokenizer-independent OpenAI chat format.
   Cases, frozen before any run: base prompt (text, image A, text, image B,
   question); divergent suffix (same through B, different question: hit);
   different image after the hit boundary (hit up to it); different image A
   (miss); A and B swapped (miss); same image A content with different
   preprocessing (e.g. `max_pixels`; miss); a boundary inside image B (a
   prefix length chosen so a page64 boundary falls inside B's span).
2. `tools/evidence/image_prefix_harness.py --url ... --fixtures ... --output
   ...`: for each case a salted cold control and a seeded warm request
   (greedy, deterministic profile, fixed `max_tokens`), recording output
   token IDs, `cached_tokens` and logprobs. Pass criteria per case: warm IDs
   equal cold IDs; `cached_tokens` > 0 for hit cases and 0 for miss cases
   and cold controls. Text control: rerun G2-2 qualification cases up to
   8192 tokens and compare their token IDs with Phase 2's fork arm
   (`qwen:results/plugin-g2-20261007/run2/F/g22/qualification.json`).
3. `tools/evidence/run_image.py` (reuse `run_g2.py` helpers): one plugin
   server on the deterministic base profile plus
   `--mm-preprocess-cache-size-mb 512`, observer on, `compare.py`-style
   checks that every observer restore equals its capture; run the harness
   twice: with the per-image ViT cache on, and with `SGLANG_VLM_CACHE_SIZE_MB=0`
   (ViT recomputed on hits; tests ViT batch invariance, I-B finding).
4. CPU tests with a fake OpenAI-compatible server for the harness's pass
   logic (each criterion has a failing counterexample) and the fixture
   freezer's determinism.

Owned paths: `tools/evidence/{image_fixtures,image_prefix_harness,run_image}.py`,
`tests/evidence/test_image_*.py`. Keep it simple (PLAN rule 10).
