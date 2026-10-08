# Phase 5 task cards (common)

PLAN.md "Phase 4" rules P1–P4 and "Phase 5" rules Q1–Q3 apply, plus the
common rules in this folder's README. Survey: `docs/phase5-survey.md`. CPU
only. Environment (export in every shell; relative defaults do not resolve
from an agent worktree):

    export QSA_PIN_ROOT=/home/zyk/projects/interests/ai-video/qwen/.worktrees/sglang-main-35f3c96ff4
    export QSA_PYTHON=/home/zyk/projects/interests/ai-video/qwen/results/dsh-maintenance-20261004/upstream-runtime-env/bin/python
    export QSA_FORK_ROOT=/home/zyk/projects/interests/ai-video/qwen/qsa-hisparse

Production source (read-only): `/home/zyk/projects/interests/ai-video/qwen/.worktrees/sglang-dsh-production-20261004`
(`897286b12a`); previous pin: `.worktrees/sglang-v0.5.21`. Read the fork
repository only with `git show`/`git diff`. Production's own history
(`git log 35f3c96ff4..897286b12a`, e.g. `773f3c2d84`, `bdb935d70f`,
`eec9df4723`, `0c0e30dcdd`, `0ffe23cc63`) explains its changes and has their
tests.

Report: per REPLACE row keep/narrow/drop; every `?` hunk you mapped (row,
existing or new, and how implemented); new rows with hook types; D1–D7
findings in your area; files changed; tests run with results (and failures
only due to other tasks); open questions.
