# G7-CPU corrections

The candidate was updated while the prescribed read-only review was running.
All three code concerns have concrete regressions; source and numerical
expectations were not relaxed.

1. **Coordinator protocol.** Main's `Scheduler.request_lists()` calls each
   collaborator's method unconditionally. Added the native ordered
   `hisparse_staging` projection to `QSAHiSparseCoordinator`, with an exact
   `PIN_EDITS` record. Before: real scheduler raises `AttributeError`.
   After: scheduler/row-observer and duplicate-owner checks pass (17 tests
   with moved-source checks). Added caller/invariant-checker dependencies.
2. **Running ownership.** Main's launch predicate checks `enable_hisparse`,
   which QSA deliberately leaves off. S10's pure AFTER hook returns false
   only for the QSA lease coordinator; S04 joins running after host-ready.
   Before: actual `_join_launched_reqs` prematurely owns the staged request.
   After: real scheduler/coordinator transition and invariants pass, including
   a native/unrelated-coordinator guard matrix. Lifecycle/prefix: 103 passed.
3. **Compat-only router source protection.** Added Q08/Q12 dependencies for
   typed KV route factories and the actual read/key/value methods. A temp-only
   mutation swaps key for value in router source; real compat-only activation
   rejects it before registry mutation. Router/scope checks: 46 passed.
   All 444 pinned fingerprints match the immutable main source.

The complete post-code-fix run is
`../results/qsa-maintenance-20261102/cpu-final.log`: its unit stage found only
an inventory classification error (742 passed, one failure). S10 was listed
among original fork hunks, but it adapts a main-only interface introduced
after that fork. It now lives in inventory section 9 (like Track I's
plugin-only rows), preserving the exact fork-hunk mapping assertions and
the unchanged 63-row manifest. All five inventory checks pass afterwards
(`inventory-final.log`). This is a documentation classification correction;
the S10 source and manifest did not change.

A clean complete unit rerun with the historical reference enabled is in
`../results/qsa-maintenance-20261102/cpu-unit-final.log`: 743 passed, 54 skipped,
29 deselected, 1 xfailed; exit 0 (completed after review clearance). The same
source's integration stages in `cpu-final.log` are reused; their code and
inputs did not change when the inventory row moved. GPU qualification is
still explicitly pending; production and shared interpreters are untouched.
