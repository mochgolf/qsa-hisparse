Reviewed main `630b69d`, read-only.

1. **Page64 coverage gap — resolved.** [Fixtures](/tools/evidence/image_fixtures.py:139) require **4352 inside B** and **3904 inside the second C**. Exact-length and observed-span checks enforce both. The [counterexamples](/tests/evidence/test_image_prefix_harness.py:329) reject chunk-only checkpoints and every-other-page64 checkpoints.

2. **Per-request restore evidence — not resolved (Medium).** Request IDs and frozen lengths are checked, but [required ranks are inferred from records already present](/tools/evidence/image_prefix_harness.py:177). The [text control runs afterward](/tools/evidence/image_prefix_harness.py:395), without rechecking image restores.

   **Concrete reproduction:** use the existing fake model with consistent capture/restore states; omit rank 1’s captures and restores for image requests, while retaining rank 0, ViT records, and subsequent text-control records. Reproduced in both cache sessions:

   - Harness exit **0**, with no failures.
   - `observer_problems(output, 2)` returns **[]**.
   - All **7 image warm hits lack rank 1 restores**; rank 1’s **12 text restores** satisfy the global observer requirements.
   - The runner’s final acceptance expression returns **True**.

   Thus the GPU run could pass without every image hit’s own restore evidence on every TP rank. **Smallest fix:** check each request against the configured TP rank set.

Verification: 31 existing fixture/harness CPU cases passed using process-local file and JSON transport substitutes. Subprocess runner tests and GPU execution were not run. No files changed.

G3-I (CPU): not cleared