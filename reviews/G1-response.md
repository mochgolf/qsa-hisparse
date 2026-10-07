# G1 response

Review: `reviews/G1.md`. All findings accepted and fixed.

| # | Resolution |
| --- | --- |
| 1 (major) plugin GPU probes | `tools/evidence/plugin_probe.py` runs the unmodified fork probe scripts (or `-m pytest` on pinned test files) in the plugin arm: model_compat activated in-process with the target-model scope forced on, HiSparse off, the in-tree Marlin GEMM name pointed at the plugin op, the scripts' relative `stable_align` helper replaced by the plugin copy in a temporary tree, and stable top-k reached through the T02 hook; it prints what it redirected. The whole-K GPU test uses it and fails (not skips) when the script is missing. CPU tests cover activation and redirection without CUDA; P-arm commands are in `docs/baseline.md` (W5, `46669a3`). |
| 2 (major) ignored evidence | The comparator compares the fork ledger summary fields, every per-case per-stage Marlin SHA-256 plus `differing_cases`/repeatability flags, and top-k exactness results; a report with nothing compared in either arm is a difference. Mutation tests for each (W8, `af67564`). |
| 3 (major) missing observer evidence | `compare.py --require-observer RANKS` requires non-empty observer files with capture and restore records for every rank in both arms; the HiSparse G2 run uses `--require-observer 2` (W8, `af67564`). |
| 4 (minor) compat readiness | The plugin arm starts the harness only after the launcher's `ready.json` (written after every TP rank's activation record) and `/health` (W8, `1f4299d`). |
| 5 (minor) wheel lock | `environment.lock.json` moved into package data and read beside `launch.py`; the wheel test asserts it is packaged (`fb4f627`). |

CPU runner: 408 + 3 + 26 passed (12 GPU skipped, 1 known fork xfail);
358 fingerprints match; manifest current.
