# G0 response

Review: `reviews/G0.md` (gpt-6.1-sol, xhigh, read-only). All findings accepted.

| # | Severity | Resolution |
| --- | --- | --- |
| 1 | blocker | `plugin.load` converts every exception on the enabled path to `PluginActivationError`. Test: real loader with an injected `ModuleNotFoundError` exits non-zero (`test_entry_point_import_failure_stops_the_process`). |
| 2 | blocker | Activation is gated on raw bytes of every module containing a target or dependency (catches conditional redefinitions, constants, CRLF, indentation). Definition hashes are raw bytes incl. decorators and enclosing class headers. Writing a record requires exactly one binding per name along the path (def/class/assign/import/del, through compound statements). Activation checks the live object is the fingerprinted definition (file + first line, or class qualname + file); inherited members are rejected; `unchecked_bindings` is an explicit per-patch opt-out that still keeps module bytes pinned. Tests: five module-edit counterexamples, class-header change, four ambiguous/inherited binding cases, raw monkeypatch before activation. |
| 3 | major | Activation rejects any foreign hook overlapping a target (same path, ancestor class, member of a replaced class). `verify_final` (framework BEFORE hook on `Scheduler.__init__`, after `load_plugins` in every scheduler/TP process) re-checks overlaps, identity of every patched attribute and attachments, so hooks registered by later plugins fail the process. Tests: earlier foreign AROUND and class REPLACE; later foreign class REPLACE. |
| 4 | major | Attachments are applied after HookRegistry (owners resolve to replacement classes) and verified. Method hooks inside a class the plugin replaces are rejected; foreign class replaces are rejected by #3. Tests added. |
| 5 | major | `verify_final` writes per-process activation records (`SGLANG_QSA_ACTIVATION_DIR`). W7 owns `launch.py`: entry-point/`SGLANG_PLUGINS` preflight, private dist-info path inherited by spawned processes, and readiness only after records from every scheduler/TP rank. Real-loader success test added. |
| 6 | major | Inventory F01 → hisparse `after` on `ForwardBatch.init_new` (D3); U01, Z05 → drops (D2, D1); reconciliation note added. |
| 7 | major | PLAN rule 9: target-model scope; generic compat hooks delegate to the original outside the target model; predicate raises if undecidable. W7 owns `scope.py`. |
| 8 | major | Compat-only HTTP comparison on a reduced deterministic profile added to G2-2; W8 defines the profile. |
| 9 | major | W8 owns the byte observer and F-vs-P comparator; W7 owns the service-control port with lifecycle-test deselection. |
| 10 | major | Baseline step 6 and all GPU criteria use fresh same-window F-vs-P evidence; historical hashes are informational. |
| 11 | major | U2+U3 merged into U23 with one owner and a frozen lifecycle contract; branch base and rebase order specified. |
| 12 | minor | `PYTHONDONTWRITEBYTECODE=1` in the runner; existing caches removed from the pin checkout. |

Framework tests: 38 passed (`tools/run_cpu_tests.sh`). `tools/fingerprint.py
--source-root ../qsa-hisparse/python check` fails as intended (the fork's
`Scheduler.__init__` differs from the pin).
