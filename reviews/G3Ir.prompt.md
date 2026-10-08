You are an independent reviewer re-checking gate G3-I (CPU part) before a GPU
window. Work read-only. Be proportionate (owner's instruction; PLAN rule 10).

Read `reviews/G3I.md`, `reviews/G3I-response.md`, `docs/PLAN.md` ("Track I
acceptance"), `docs/tasks/I-C.md`. Code: `src/sglang_qsa_hisparse/hisparse/image_request.py`;
harness: `tools/evidence/{image_fixtures,image_prefix_harness,run_image,vit_observer}.py`,
`tools/evidence/site/sitecustomize.py`; tests: `tests/image/`,
`tests/evidence/test_image_*.py`. The I-C agent's deviations from its card
(accepted by the orchestrator): `/generate` prompts with `<image>`
placeholders instead of OpenAI chat; `detail: "high"` as the
different-preprocessing miss; ViT batch-invariance measured from the
cache-off session's own traffic.

1. For each G3-I finding: resolved / not, with evidence.
2. Would the planned GPU run (`run_image.py`) establish the GPU part of the
   seven acceptance criteria, or can it pass while a criterion is violated?
   Check the frozen expected hit lengths, the inside-image span assertion,
   ViT skip and invariance checks, the text control, and that the server
   profile really enables image reuse (`--mm-preprocess-cache-size-mb`).
3. Risks that would waste the GPU window (e.g. a harness assumption that
   only holds for the fake server).

Output: findings (at most eight) with file:line, scenario and smallest fix;
final line "G3-I (CPU): cleared" or "G3-I (CPU): not cleared".
