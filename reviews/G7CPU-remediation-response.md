# Phase 7 CPU numerical-remediation response

The review's inventory finding is fixed: D01 is a main-only compatibility
adaptation and now lives in patch-inventory section 9 beside S10. It is not
claimed as a legacy fork hunk. The focused inventory suite passes all five
checks (`../results/qsa-maintenance-20261102/remediation-inventory.log`), and
the generated manifest remains current. This correction is documentation
only; the reviewed product source is unchanged.

The final P3 finding is also fixed: the maintenance subsection now describes
the local historical selector and full-module-fingerprinted native table
data, replacing the stale claim that plugin kernels use main's invariant
fixed tile. No product source or acceptance criteria changed.

Final complete CPU runner: 772 passed / 54 skipped / 29 deselected /
1 xfailed; subprocess integration 4 passed / 1 skipped; activated-runtime
integration 11 passed; runner exit zero (`cpu-remediation.log`). All 455
fingerprints match the pristine main snapshot, and the manifest has 64 rows
with 25 REPLACE hooks. The independent agent review's NVIDIA/HIP scope
finding was fixed before the prescribed review and full CPU run.

The same-input/weight HC operator diagnostic establishes the first numerical
divergence and its registration cause. Full-model GPU qualification remains
pending; neither these CPU results nor the operator diagnostic clear it.
