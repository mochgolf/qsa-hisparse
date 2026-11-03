Read-only review of one small incremental compatibility fix, D02. Follow
GOAL.md/PLAN.md's sol/xhigh review protocol. No edits, delegation, GPU/service
work, environment changes, credential/config reads or publication. Scope is
only D02; reuse reviews/G7CPU-remediation-2.md's cleared prior source review.
Do not review the whole port, other hooks or large raw tensor/observer files.

Inspect these bounded sources: new patches/model_compat/shared_gate.py;
tests/model_compat/test_shared_gate_launch.py; the five records in
fingerprints/model_compat_shared_gate.json; manifest D02 and its section9
patch-inventory entry (not the entire inventory). Native source is pristine
main582168dd7c ../sources/sglang-main-582168dd7c/python. Reference is
../.worktrees/sglang-dsh-production-20261004/python, revision897286b12a.
Compare only the old/new _launch_fused_gate_sigmoid_mul definitions and the
native kernel's GATE_ONLY=False default. The local launcher is an exact
43-line F copy (frozen source SHA in the test). It changes launch geometry,
not native kernel math; other models/platforms/global-OFF calls delegate.

Context: main#42087 caps the gate at8warps under global batch-invariant mode.
Reference at hidden2560 uses16 for rows<1024 and8 otherwise. Earlier corrected
candidate matches all96 KV/recurrent checkpoints through65536 on both ranks,
but token17 differs. Fresh F/P65548/max32 diagnostic controls reproduce each
arm's original outputs/checkpoints. Layer0 same-input gate is a retained
negative control: output identical despite warp difference. Layer3 has exact
same H, weight, shared output and pre-add routed output on both ranks;
rank0 output equal, rank1 has3 BF16 differences. A separate frozen layer3
GPU experiment reproduces both original outputs and restores F by changing
ONLY the global-mode cap condition back to the historical row threshold.
Raw proof: ../results/qsa-maintenance-20261102/gate-op-diag-layer3/run/result.json
(inspect summary flags/rank results, not embedded source repeatedly).
This local delta is masked by subsequent BF16 TP sum for this sample, so it
does not prove sole cause or full-model sufficiency. Earlier MoE.return
differences are semantic boundaries: F already TP-reduced, P defers that sum.
No fixture, tolerance, golden or full-stage comparator has changed.

The declared profile disables whole-model torch.compile and uses eager
prefill/fullstatic decode graphs; launcher M is concrete int. A separate
agent already checked scope/alias propagation/native ABI/fingerprints.
15 draft CPU cases pass; the actual native launcher fails6 small-row cases
and passes4 larger-row cases. Targeted result is d02-targeted-cpu.log; full
CPU runner is cpu-d02-final.log (both in the maintenance results directory).
No need to rerun whole tests. Limit inspection to directly related code and
evidence; end with concrete findings or no findings, then
G7-CPU-D02: cleared or G7-CPU-D02: not cleared. Full GPU qualification is a
subsequent gate and remains required; no scientific validity claim is made.
