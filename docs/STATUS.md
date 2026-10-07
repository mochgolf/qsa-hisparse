# Status

Progress log for [PLAN.md](PLAN.md). Newest entries last.

| Gate | State | Evidence |
| --- | --- | --- |
| P0-D framework | done (`f8f0958`) | `tests/test_framework.py` 15 passed on the pin |
| P0-A inventory | running | `docs/patch-inventory.md` |
| P0-B baseline | running | `docs/baseline.md` |
| P0-C upstream status | running | `docs/upstream-status.md` |
| G0 review | pending | `reviews/G0.md` |
| Phase 1 W1–W7 | not started | |
| G1 review | pending | |
| Phase 2 GPU | needs owner window | |
| Phase 3 tracks I/U | not started | |

## Log

- 2026-10-07: Goal and decisions recorded. Pinned worktree
  `../.worktrees/sglang-pin-76e06febab` created. Found that SGLang's
  `load_plugins`/`apply_hooks` swallow `Exception`; activation errors derive
  from `BaseException` and every applied target is verified. Fork runtime
  package (9 modules) and `stable_align.py` import cleanly on the pin.
- Open item (W7): if `SGLANG_PLUGINS` excludes `qsa_hisparse`, or the
  dist-info is not on the path of a spawned process, switches are ignored and
  SGLang serves pristine upstream silently. Add a launcher that verifies the
  entry point is discoverable and allowed before starting, and evidence of
  activation in every process (main, scheduler/TP).
