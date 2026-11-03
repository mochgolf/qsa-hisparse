Review the finalized Phase 7 GPU gate read-only under GOAL.md/PLAN.md. No
delegation, GPU jobs, service actions, environment changes, publication,
credential/config reads or duplicate full CPU tests. Production must already
be restored before this review. Check evidence and claims; do not repeat the
cleared source-port/D01/D02 reviews (reviews/G7CPU*.md).

Scope: unmodified main582168dd7c, maintenance interpreter, 2xRTX4090 48GB
SM89/driver595.84, declared deterministic TP2 and native profiles. Numerical
authority is frozen legacy897286b12a; this is compatibility qualification,
not scientific validation or all-hardware/all-feature certification.

Read docs/phase7-results.md and the bounded aggregate qualification-summary.json
under ../results/qsa-maintenance-20261102/. G22 and memory-g24.json are under
gpu-final; G21, compat, image/native and final restoration are under
gpu-remainder. Prefer their per-stage exit codes, compare-g22-schema.log,
compare-g21.log, compare-g21-tests.log, compare-compat.log, i5.log and its
final summary, compare-g23.log, restored-production status/smoke and source
hash audit. The original gpu-final/status.json retains its truthful failure
from a metadata representation mismatch; the separate complete offline
compare-g22-schema.log qualifies those unchanged raw numerical/byte reports.
Do not print giant observer/HTTP arrays or recursively read server logs.
Use Python to summarize selected JSON fields when necessary. F G21/G22
reference is ../results/qsa-maintenance-20261102/gpu/F; corrected G22 is
gpu-final/P/g22 and subsequent arms are under gpu-remainder. Preserve failed runs.

Verify:
1. G22: both arms' complete 16 frozen cases through262016, concurrency B2/B8,
   flush/abort/lifecycle/ledger all pass. Full offline comparator compares
   tokenIDs, cachedcounts, logprobs and every captured/restored state field
   on both ranks. Missing observers cannot pass. The progress-only reader
   does not replace this complete comparison.
2. G21: five exact reports match; all91 common GPU outcomes match,14
   plugin-only outcomes pass, existing SM121 skip preserved. Two parameter
   ID aliases come from exact upstream source renames; new FP8-to-FP8 case
   remains required. No added known-failure allowance. Prior topk29/arm
   results remain applicable because that code/dependencies did not change.
3. Compat-only complete equality, I5 two-session image-prefix byte/token
   checks and text reference controls, nativeG23 smoke/lifecycle/concurrency
   and cachecounts. Native outputs have no exact-equality obligation and
   latency medians have no threshold; do not inflate either claim.
4. G24 all12 fixed pool fields are present/non-null and exactly match both
   ranks. Measured startup/peak allocations are informational and distinct.
5. The package computational source stays frozen throughout the final GPU
   matrix, including new files. Any later environment.lock.json freeform
   qualification-note edit must be explicitly metadata-only with unchanged
   versions/runtime code. Source-hash audit must document that distinction.
6. Exact original production profile restored, health/ready/model verified,
   synthetic text and image cold/warm smoke passes with warm prefix hits.
   No production promotion or publication is claimed.
7. Main commit a076243c (#43898) deleted ShapeKey.stream_idx, whose previous
   default None means a single stream. Both profiles disable PDMux/overlap;
   every other graph-capture field matches across all16 records. The checker
   adapter ONLY equates missing with null at this exact graph_key location;
   non-null0/1 and other metadata/buffer differences still fail. See
   graph-key-schema-audit.{md,json}, tools/evidence/compare.py and its16 new
   regression cases. Existing raw files/failed log were not rewritten.
8. Compat emits TileLang static race-check warnings. Checks stayed enabled.
   Read tilelang-warning-audit/proof.md and ownership.json: inspected source
   and seven actual cached CUDA specializations have unique Logits store
   ownership/disjoint CTA intervals; warning examples violate the defining
   indices. CPU enumeration passes. This closes only those reported write
   sites for valid declared metadata; no whole-compiler/race-sanitizer claim
   is made. Judge whether the final report preserves that limited scope.

Earlier failures led to reference-local QSA config plus D01 (skip only new
mm.out registration for target NVIDIA CUDA) and D02 (exact legacy shared
gate launcher under target/global-ON). Both have meaningful old-fails/new-
passes CPU evidence, source guards and cleared CPU reviews. Same-input HC
operator control isolates mm.out; shared-gate layer0 is a preserved negative
control and layer3 reproduces a local delta restored by only the warp cap.
That sample's TP sum masks the delta, so it never proved sole cause or model
sufficiency. Only the complete frozen GPU matrix supports the final profile
claim. No golden, tolerance or numeric/byte comparison rule changed. Only
the obsolete optional single-stream field representation was normalized
from source-backed schema authority; entire evidence CPU suite141 passes.

Report only concrete unresolved correctness/evidence/reporting issues with
verified file:line and smallest fix, or no actionable findings. State any
material missing/failed obligation explicitly. End with G7-GPU: cleared or
G7-GPU: not cleared.
