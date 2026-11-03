**[P3] Correct the selector ownership description** — [docs/patch-inventory.md:816](/home/zyk/projects/interests/ai-video/qwen/qsa-hisparse-plugin/docs/patch-inventory.md:816) still says plugin kernels use main’s batch-invariant fixed tile. The implementation now uses the local legacy `_get_best_config`. Smallest fix: describe the legacy selector and fingerprint-protected native table data.

No code defects found in the reviewed remediation. The D01 inventory-placement failure was corrected during review; its focused regression passes.

Recorded final CPU results: **772 passed**, 54 skipped, 29 deselected, 1 xfailed; integration stages: **4 passed / 1 skipped** and **11 passed**.

Full GPU rerun remains pending. This review establishes no scientific validity or full-model qualification.

G7-CPU-remediation: not cleared.