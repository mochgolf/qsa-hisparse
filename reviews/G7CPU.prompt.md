Review the Phase 7 CPU maintenance candidate read-only, using the repository's
GOAL.md/PLAN.md review protocol. Do not edit files or delegate; do not use
GPUs, start/stop services, modify environments, contact anyone or publish.
Report concrete correctness/compatibility defects, not style or hypothetical
threats outside PLAN.md. Do not inspect credential/configuration files.

Repository is this checkout, branch maintenance/sglang-main-20261102; previous
plugin HEAD is 983603dc7c. Review `git diff HEAD -- src tests tools docs` plus
new untracked test files listed by git status. Source pin is unmodified
sglang main 582168dd7c3821198c92766f0319b9e1c8c17bd3 at
../sources/sglang-main-582168dd7c. Previous pin is ../.worktrees/sglang-v0.5.22.
Historical reference is production 897286b12a, base35f3c96ff4, available with
git show in ../qsa-hisparse (partial clone: do not git log -S/-G/-p or blame
over ranges). This port preserves Phase 6 numerical/behavioral contracts;
GPU qualification is explicitly pending, production is unchanged.

Read docs/phase7-survey.md, docs/phase7-results.md, relevant GOAL.md/PLAN.md
constraints and the changed inventory/dependency entries. Final merged CPU
runner is underway at ../results/qsa-maintenance-20261102/cpu.log; inspect
existing results without launching duplicate full tests. Baseline old pin
is frozen and green (725 +4 +11 passed), source delta checks and module
checks have passed separately. New native versions are in environment.lock.

Prioritize:
1. Scheduler/row/state ownership: copied upstream changes are retained;
   rollback and release use StaticKVStorage/StateStorage without double free,
   leaks, missing lease drains or altered cache-budget/cached_tokens behavior.
2. Graph captures/replays preserve main's forward context, early metadata,
   capture hooks and the plugin's begin/finish/restore ordering; K02 forwards
   the actual state_pool; inherited prepare_replay can't bypass preparation.
3. QSA/GPTQ/Marlin/model changes retain single FP8 descale, stable selection,
   numerical dispatch and native/custom-op ABI. Z02 drop correctly inherits
   #35955. E08 selectively incorporates #38900's closure lifetime fix. Every
   manual merge in tests/regression/test_replace_deltas.py RESOLVED is valid;
   no source or numerical regression is hidden by a permissive test change.
4. All relevant changed/new dependencies are fingerprinted, stale/moved names
   removed, activation remains fail-closed and non-target/off paths delegate.
5. Evidence/reporting: no GPU claims from CPU tests, no numerical tolerances
   loosened, prior qualification not transferred silently. Pristine main
   plugin-off outcomes were independently measured without the plugin on
   PYTHONPATH (../results/qsa-maintenance-20261102/pristine-upstream.json).

If needed, small CPU checks may use tools/run_cpu_tests.sh <specific paths>;
its defaults select the new immutable pin and isolated maintenance env. Do
not install dependencies or run model/GPU tests. Return findings with severity,
verified file:line, concrete triggering scenario, and smallest fix; otherwise
state no actionable findings. End with G7-CPU: cleared or G7-CPU: not cleared.
