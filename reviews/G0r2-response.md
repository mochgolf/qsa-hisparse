# G0 round 3 response

Review: `reviews/G0r2.md`. All findings accepted.

| Finding | Resolution |
| --- | --- |
| 1 (blocker) dependencies unprotected | Declared dependencies join targets and attachments in the foreign-overlap check at activation and in `verify_final`; their live objects are frozen after activation and re-checked by identity. Test: foreign REPLACE on a dependency before activation fails; after activation (applied by the loader's final pass) `verify_final` fails on overlap and, with the registry entry removed, on the changed binding. Framework dependencies (registry internals, `load_plugins`, `run_scheduler_process`) are covered by the same mechanism. |
| 2 (blocker) chains do not identify behavior | Each level must be an identifiable kind: function (code location, qualified name, and for code outside the pinned tree its closure, defaults and kwdefaults described by value), bound method (owner type and instance state), staticmethod/classmethod/property, `lru_cache` (cache parameters), `functools.partial` (function, args, keywords), operator packet (qualified op name), class (qualified name, file and file hash). Internal code levels also record their file hash. Anything else raises `Undescribable` at record time; patches may opt out per name via `unchecked_bindings` (module bytes still pinned). Tests: no_grad vs enable_grad, aten.add vs aten.mul, lru_cache configurations and partial arguments differ; opaque callables and opaque closure values are rejected; a swapped decorator configuration fails activation. |
| 3 (major) duplicate declarations | Duplicate (feature, row, target, hook type) declarations are rejected before registration; manifest comparison uses multisets. Test added. |
| 4 (major) wheel omits manifest | `manifest.json` added to package data. Test builds a wheel (system `python3` by default, `QSA_WHEEL_PYTHON` to override; the validation env's `setuptools_scm` plugin is broken independently) and asserts manifest, fingerprints and the entry point are present. |
| 5 (non-blocking) C01 field dependency | Inventory now names the `QSATokenToKVPool` class. |
| 6 (minor) header expressions | Decorators, defaults, keyword defaults, argument/return annotations, class bases/keywords, and annotation-only annotations are scanned for walrus bindings in the enclosing scope. Three tests added. |
| Stale prose (G6) | F01 removed from class-REPLACE guidance. |
| CUDA chains | Framework records now carry `cpu` and `cuda` chains (identical); `check` passes in both modes. |

Framework tests: 68 passed.
