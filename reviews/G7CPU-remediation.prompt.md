Review only the Phase 7 numerical-remediation CPU gate, read-only. Follow
GOAL.md/PLAN.md's prescribed review protocol. No delegation, GPU work,
service actions, environment changes, publication, or credential/config reads.
Do not repeat the cleared G7CPU source-port review or scan the whole checkout.

The unchanged contract is exact legacy compatibility, reference897286b12a,
on the documented SM89 profiles. Read docs/phase7-results.md's diagnostic
section and these bounded evidence summaries (not giant observer files):
../results/qsa-maintenance-20261102/tensor-diag/run/result.json (inspect first
differences and controls only), hc-mm-out-source-evidence.json and
hc-op-diag/run/result.json. Both fresh F/P instrumented runs reproduce their
original checkpoint bytes exactly. Earliest difference is HC mixed output;
normalized input and both complete saved weight tensors match. Executing
the original F compiled function in fresh P processes: native mm.out
registration matches P exactly, omitting only it matches F exactly, both ranks.
This is a compatibility diagnosis, not scientific validation or full-model
qualification. Source pin remains main582168dd7c; no golden/tolerance change.

Review:
- src/sglang_qsa_hisparse/patches/model_compat/batch_invariant.py (new D01),
  its native namespace/Library/bmm enable/disable ownership, idempotency,
  supported CUDA scope and delegation for other models/platforms, call-site
  alias propagation and activation before ModelRunner enables the mode.
- The local _get_best_config in kernels/qsa_sparse_attn.py, now copied from
  the pinned legacy authority. Main #42087 added global-invariant selection;
  plugin reference kernels preserve historical launch geometry. Imported
  native table data stays under full-module fingerprint protection. Check
  tests/qsa/test_reference_launch_config.py and its old-fails evidence.
- tests/model_compat/test_batch_invariant_initialization.py, the added legacy
  copy digest in test_model_compat_copies.py, dedicated fingerprint JSON,
  manifest D01 and docs/patch-inventory.md's new main-only ownership entry.

Use the targeted logs and final CPU-remediation runner results under
../results/qsa-maintenance-20261102; do not rerun the whole suite. Small
CPU-only checks are allowed if needed. Preserve the earlier G7CPU findings
as already resolved unless this fix concretely invalidates them. Report
actionable defects with verified file:line and smallest fix, or no findings.
End with G7-CPU-remediation: cleared or G7-CPU-remediation: not cleared.
Full GPU rerun is a subsequent gate and remains pending during this review.
