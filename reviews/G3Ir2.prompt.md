You are an independent reviewer doing the final, scoped re-check of gate
G3-I (CPU part) before a GPU window. Work read-only. Be proportionate
(owner's instruction); this is the last round before the GPU run.

Scope: only the two findings of `reviews/G3Ir.md` (page64 coverage gap;
per-request image restore evidence). Read `tools/evidence/{image_fixtures,image_prefix_harness,run_image}.py`
and `tests/evidence/test_image_{fixtures,prefix_harness,run}.py` at main
`630b69d`. Expected reuse lengths now include 4352 (inside B) and 3904
(= 61 × 64, inside the second C); restores must carry the request's own
`meta_info.id` on every TP rank at the frozen length.

For each finding: resolved / not, with evidence. Report a new finding only
if the GPU run could pass while an acceptance criterion is violated, with a
concrete reproduction. Final line "G3-I (CPU): cleared" or
"G3-I (CPU): not cleared".
