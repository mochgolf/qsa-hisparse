# G0 re-review response

Review: `reviews/G0r.md`. All findings accepted.

| Finding | Resolution |
| --- | --- |
| New 1 (blocker) verifier removable | Framework hook moved to module-level `configure_scheduler_process` (BEFORE), called by global name after `load_plugins()` in `run_scheduler_process`; registry internals and both functions are pinned `depends`. Test: a later foreign class REPLACE leaves the verifier running; a later hook on the verifier's target fails it. |
| New 2 (blocker) `functools.wraps` | No unwrapping. Records carry the live binding chain per mode (cpu/cuda): every level's type and code location through `__wrapped__`/descriptors, captured by importing the pinned checkout. Activation requires the exact chain. Tests: raw and `wraps` replacements fail; pinned `lru_cache` passes; missing mode chain fails. |
| New 3 (major) premature global apply | Activation applies only this plugin's targets via `_apply_target` (class REPLACE first); the expected hook set is frozen; `verify_final` fails on any later entry on a frozen target or any later QSA-sourced hook. Test reproduces the reviewer's 2-vs-12 case (now 13 = 2+1+10). |
| New 4 (major) empty features | `manifest.json` generated from the inventory (`tools/manifest.py`); a feature's declarations must equal its rows exactly; features without rows fail. Real loader with compat=1 now fails ("differ from manifest") until W-tasks land; a success test injects a one-row manifest. |
| New 5 (major) property hooks | Properties are rejected as hook targets (still allowed as dependencies). Test added. |
| New 6 (major) F01 depends | F01 depends on the `ScheduleBatch` class. Inventory §2 C4, §5 counts (37 REPLACE: 36 functions + T04 class), Appendix B, service ownership (W7) and U23 references reconciled. |
| New 7 (major) GOAL coverage | Integration merges W1–W8 and checks the manifest; Track I has I1–I4 agent owners and I5 orchestrator-owned GPU evidence, with seven acceptance criteria mapped to GOAL image obligations (identity, order/offset/grid, M-RoPE without whole-prompt digest, every page64 boundary incl. inside images, exact state, logits tail, deterministic cold/warm/divergent outputs). |
| New 8 (minor) binding counting | Bindings counted per Python binding form in scope: def/class, imports, assignment/aug/ann-with-value, del, for/with targets, except aliases, match captures, walrus (incl. inside comprehensions); attribute/subscript targets, annotation-only declarations, comprehension variables and nested scopes excluded. 15 tests. |
| New 9 (minor) runner exclusions | `tests/conftest.py` deselects `ServiceLifecycleTests`/`service_lifecycle` and skips `gpu` tests unless explicit opt-in variables are set; the runner clears them and other inherited switches. |
| Old 5 partial | Framework part done (records from every scheduler process); `launch.py` with spawned-process discovery tests is W7's Phase 1 deliverable. |
| Old 7 partial | `scope.py` is a fixed contract implemented by W7 in Phase 1; W4's card no longer allows unscoped QSA changes. |
| Old 8 partial | Baseline opt-out removed; W8 freezes the reduced compat-only profile in Phase 1. |

Framework tests: 56 passed.
