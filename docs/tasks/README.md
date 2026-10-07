# Phase 1 task cards

Each card is the complete brief for one parallel agent. Shared rules,
layout and import mapping: [PLAN.md](../PLAN.md). Row IDs refer to
[patch-inventory.md](../patch-inventory.md); deviations:
[DEVIATIONS.md](../DEVIATIONS.md).

Common to every card:
- Work only in your own git worktree/branch of this repository; commit there
  with clear messages ending with the `Co-Authored-By` line given by the
  orchestrator. Do not merge into `main`.
- Implement exactly your rows. If an inventory equivalence argument for a
  narrow hook fails, use the listed fallback REPLACE and report it.
- Fingerprints: `tools/fingerprint.py write <your-module> <targets...>` for
  every target and `depends` name, against the pin only.
- Run `tools/run_cpu_tests.sh` (all tests, including the framework suite).
  A ported test that needs another workstream's patches stays unchanged,
  gets `@pytest.mark.integration`, and is listed in your report; the
  orchestrator runs those after merging.
- Prohibited actions: PLAN.md rule 6. Use `-k 'not ServiceLifecycleTests'`
  when running fork service tests (rule 8).
