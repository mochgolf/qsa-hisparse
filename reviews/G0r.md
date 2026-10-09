G0 is **not cleared**: 6 findings are resolved and 6 are partially resolved. Review includes commit `797ae34`; the requested framework files are unchanged from `142b2f7`.

| # | Verdict | Evidence and remaining work |
|---|---|---|
| 1 | **Resolved** | [plugin.py:17](/src/sglang_qsa_hisparse/plugin.py:17) wraps enabled activation, including patch-module imports. An injected `ModuleNotFoundError` escaped the isolated pinned loader as `PluginActivationError`, outside `Exception`. |
| 2 | **Partially resolved** | Raw module hashing correctly matches the pin and rejects the fork; the manifest also matches independently regenerated records. However, [patching.py:222](/src/sglang_qsa_hisparse/patching.py:222) blindly unwraps live functions, allowing foreign wrappers through. Binding uniqueness also misses several Python bindings. See new findings 2 and 8. |
| 3 | **Partially resolved** | [patching.py:255](/src/sglang_qsa_hisparse/patching.py:255) checks registered foreign overlaps. Final verification can be bypassed by replacing its scheduler owner, and early registry-wide application still skips later foreign hooks. See findings 1 and 3. |
| 4 | **Resolved** | [patching.py:188](/src/sglang_qsa_hisparse/patching.py:188) rejects hooks inside QSA-replaced classes; line 233 rejects inherited members; line 337 attaches to final owners. Direct staticmethod/classmethod probes preserved descriptors and returned correct results. Property handling has a separate defect below. |
| 5 | **Partially resolved** | Scheduler records exist, and [W7.md:8](/docs/tasks/W7.md:8) assigns launcher preflight and rank readiness checks. `launch.py` remains absent. Missing/excluded entry-point probes still returned successfully with zero hooks. Successful spawned scheduler/TP activation and failed discovery are not tested. |
| 6 | **Partially resolved** | F01, U01 and Z05’s primary rows were updated. But [patch-inventory.md:171](/docs/patch-inventory.md:171) still prescribes the `ForwardBatch` class replacement; §5 and Appendix B retain old replacements/targets. Service ownership and U2/U3 references are also stale. The current table implies **37 REPLACEs: 36 functions/methods and 1 class**. F01’s new dependency is unpinnable; see finding 6. |
| 7 | **Partially resolved** | PLAN rule 9 defines scope, but [scope.py:27](/src/sglang_qsa_hisparse/scope.py:27) is a raising stub. [W4.md:26](/docs/tasks/W4.md:26) permits changes applying to all QSA models, contrary to rule 9’s delegation requirement. DEVIATIONS contains no corresponding scope exception. |
| 8 | **Partially resolved** | G2-2 and W8 now require compat-only HTTP parity. But [baseline.md:358](/docs/baseline.md:358) still offers dropping that comparison, and the reduced profile/fixtures are not yet frozen. Remove the opt-out and complete W8’s profile. |
| 9 | **Resolved — assignment** | [W8.md:6](/docs/tasks/W8.md:6) assigns observer/comparator paths and discriminating CPU tests; [W7.md:18](/docs/tasks/W7.md:18) assigns the service helper/test port and lifecycle deselection. These deliverables remain future implementation work. |
| 10 | **Resolved** | [baseline.md:293](/docs/baseline.md:293) now compares fresh same-environment runs and makes historical hashes informational. The referenced native harness also asserts fixed-input repeatability. GPU evidence has not yet been produced. |
| 11 | **Resolved — planning** | [PLAN.md:150](/docs/PLAN.md:150) consolidates U2/U3 under U23, requires the shared lifecycle contract before code, and specifies common branch bases plus U23’s rebase onto U1. Inventory identifiers still need reconciliation under #6. |
| 12 | **Resolved** | [run_cpu_tests.sh:17](/tools/run_cpu_tests.sh:17) exports `PYTHONDONTWRITEBYTECODE=1`. |

The new findings, ordered by severity, are:

1. **Blocker — final verification can remove itself.**  
   [framework.py:15](/src/sglang_qsa_hisparse/patches/framework.py:15), [test_framework.py:324](/tests/test_framework.py:324).

   A later plugin replaces `Scheduler` with a class defining its own initializer. That removes the BEFORE hook responsible for detecting the replacement. My isolated pinned-loader probe returned successfully, constructed the replacement, and recorded **zero verifier calls**. An explicit `verify_final()` would fail, but the real constructor path never calls it. The existing test invokes verification manually and misses this escape.

   **Smallest fix:** run scheduler final verification at the pinned `configure_scheduler_process` call after `load_plugins()`, outside the replaceable class; test replacement followed by construction.

2. **Blocker — `functools.wraps` hides foreign live behavior.**  
   [patching.py:222](/src/sglang_qsa_hisparse/patching.py:222).

   Replacing pinned `f(x)=2*x` with an unregistered wrapper using `@wraps(f)` passes binding checks. Adding QSA’s `+1` hook then passes activation and final verification, but `f(2)` returns **100 instead of 5**. Unwrapping validates the inner function while accepting arbitrary outer behavior.

   **Smallest fix:** validate the complete permitted decoration/descriptor chain. Reject unexpected outer wrappers; retain explicit support for pinned decorators such as `lru_cache`.

3. **Major — activation prematurely applies other plugins’ hooks.**  
   [patching.py:327](/src/sglang_qsa_hisparse/patching.py:327).

   An earlier plugin registers `+1` on an unrelated function; QSA applies it globally; a later plugin registers `+10` on that function. The loader skips it because `_patched` already contains the target. Final verification passes, and the result is **2 instead of 12**. A late same-source QSA hook likewise remained unapplied while verification passed.

   **Smallest fix:** apply only QSA-owned targets during activation, leaving unrelated entries for the loader’s final application. Freeze and validate the expected QSA hook set.

4. **Major — empty features receive successful activation status.**  
   [patching.py:173](/src/sglang_qsa_hisparse/patching.py:173), [test_framework.py:446](/tests/test_framework.py:446).

   Both `compat` and `compat+hisparse` currently succeed with only `Scheduler.__init__` verification installed. Final verification accepts them although neither feature has implementation patches. Missing feature modules can therefore produce records claiming activation while upstream behavior remains.

   **Smallest fix:** reject requested features without implementations, then validate their required declarations against a completeness manifest. Assert feature coverage in the success test.

5. **Major — property hooks pass verification but destroy property behavior.**  
   [patching.py:225](/src/sglang_qsa_hisparse/patching.py:225).

   The binding check accepts a property through its getter, but the pinned registry wraps the property object as a callable. My probe passed activation/final verification; property access became a bound method, and calling it raised `TypeError("'property' object is not callable")`.

   **Smallest fix:** reject properties as hook targets before registration, while allowing getters to establish dependency provenance. Supporting property hooks requires descriptor-preserving application.

6. **Major — reconciled F01 declares an invalid fingerprint dependency.**  
   [patch-inventory.md:77](/docs/patch-inventory.md:77).

   `ScheduleBatch.req_pool_indices_cpu` is an annotated field, not a function/class definition. Pinning it fails: **“found 1 binding(s), 0 direct”**. Following the row literally blocks fingerprint generation and activation.

   **Smallest fix:** depend on `ScheduleBatch` or the methods producing that field, then generate the records.

7. **Major — GOAL coverage still lacks complete ownership and acceptance mapping.**  
   [PLAN.md:109](/docs/PLAN.md:109), [PLAN.md:129](/docs/PLAN.md:129).

   Parity implementation has W1–W6 owners, off/non-target regression has W7, evidence tooling has W8, fingerprints have the orchestrator, and release ordering has W2. However, integration still merges only W1–W7. Image implementation and acceptance are not assigned to W1–W8; W6’s card explicitly limits Track I to a design note. PLAN does not explicitly gate preservation of the logits tail or deterministic divergent-suffix image outputs.

   **Smallest fix:** include W8 in integration and assign explicit image implementation/evidence ownership, with acceptance criteria covering every GOAL image obligation.

8. **Minor — binding uniqueness counts references and misses bindings.**  
   [fingerprint.py:36](/src/sglang_qsa_hisparse/fingerprint.py:36).

   Probes correctly rejected definitions inside `if`, `try` and match bodies, and ignored nested local bindings. They incorrectly accepted exception aliases, match captures and walrus rebinding in module/class scopes. Conversely, `f.extra = 1` and annotation-only `f: object` incorrectly counted as rebinding `f`.

   **Smallest fix:** count actual bound names, including exception/pattern/named-expression bindings in their proper scopes; exclude attribute/subscript references and annotation-only declarations.

9. **Minor — the CPU runner does not enforce documented exclusions.**  
   [run_cpu_tests.sh:21](/tools/run_cpu_tests.sh:21).

   An echo-interpreter probe showed plain `pytest … tests`, without GPU-marker or `ServiceLifecycleTests` deselection. Registering a `gpu` marker does not skip it. After the planned ports, automated runs can execute the explicitly excluded lifecycle class or GPU tests enabled by inherited switches.

   **Smallest fix:** enforce the automated exclusions in the runner and clear inherited GPU-test switches.

Verification was read-only and used in-memory probes. The scheduler positional/keyword rank extraction is correct. All **196 hunks map exactly once**. Full pytest and GPU suites were not run; full SGLang import was blocked by a dependency’s writable-temporary-directory probe. No files were modified by this review.