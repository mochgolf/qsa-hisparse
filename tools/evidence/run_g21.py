#!/usr/bin/env python3
"""Run one arm of G2-1 (docs/baseline.md steps 2-8) on GPU 0, no model load.

    run_g21.py --arm {fork,plugin} --output <new arm directory> [--dry-run]

The steps and arms are those of the Phase 2/4 window scripts. Since Phase 5
the fork arm runs production's code (run_g2.FORK_ROOT): its ``python/`` tree,
its ``test/qsa_hisparse`` Marlin test, its ``test/manual`` probes and its
registered kernel tests at the pin's layout
(``test/registered/kernels/ops/...``). The plugin arm runs the plugin on the
pin: the ported Marlin test (which runs production's whole-K script), the same
production probes through ``plugin_probe.py``, the ported step-8 files (8a,
``--runxfail``) and the pin's unmodified files under the probe (8b). Both arms
use run_g2.PYTHON with ``CUDA_VISIBLE_DEVICES=0``; PATH and CUDA_HOME (JIT
builds) come from the caller.

Each step writes ``<output>/<step>.log``; ``steps.json`` records each step's
command, directory and exit status. All steps run; the exit status is 0 only
if every step passed. Compare afterwards: the JSON reports of steps 3-7 per
case, and ``pytest_outcomes.py --fork <F>/step8.log --plugin <P>/step8a.log
<P>/step8b.log --inventory g21_step8_inventory_35f3c96ff4.txt`` (step 8a runs
with ``--runxfail``, so a strict xfail port reports its exception; known
failures, if any, as observed in the fork arm).
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_g2  # noqa: E402

# Registered kernel tests at the pin layout (relative to test/).
QSA_TESTS = "registered/kernels/ops/attention/qsa"
GEMM_TESTS = "registered/kernels/ops/gemm"
UNSET = (
    "TRITON_INTERPRET",
    "SGLANG_QSA_MODEL_COMPAT",
    "SGLANG_QSA_HISPARSE_V3",
    "SGLANG_PLUGINS",
    "SGLANG_CRASH_ON_JIT_COMPILE",
)
PYTEST = ["-m", "pytest", "-p", "no:cacheprovider"]


def probes(run, output):
    """Steps 3-7: ``run`` maps a production test/manual script to a command."""
    graphs = ["--cuda-graphs", "--batch-sizes", "1", "8", "96", "2048",
              "--patterns", "identical", "spread_routes"]  # fmt: skip
    return [
        ("step3", run("marlin_deterministic_alignment.py", "--output", f"{output}/marlin-align.json")),
        ("step4", run("marlin_batch_invariance.py", "--output", f"{output}/marlin-whole-k.json")),
        ("step5", run("marlin_batch_invariance.py", *graphs, "--output", f"{output}/marlin-graphs.json")),
        ("step6", run("marlin_batch_invariance.py", "--native", "--output", f"{output}/marlin-native.json")),
        ("step7", run("qsa_deterministic_topk_probe.py", "--large-prefill", "--output", f"{output}/topk.json")),
    ]  # fmt: skip


def plan(arm, output):
    """[(step, cwd, argv, extra environment)] and the arm's PYTHONPATH."""
    py, fork, pin, plugin = (
        str(run_g2.PYTHON), run_g2.FORK_ROOT, run_g2.PIN_ROOT, run_g2.PLUGIN_ROOT
    )
    if arm == "fork":
        steps = [
            ("step2", fork, [py, *PYTEST, "-q", "test/qsa_hisparse/test_marlin_deterministic_alignment.py"],
             {"SGLANG_TEST_MARLIN_GPU": "1"}),
            *((name, fork, argv, {}) for name, argv in probes(
                lambda script, *args: [py, f"test/manual/{script}", *args], output)),
            ("step8", fork, [py, *PYTEST, "-v", "-rA", f"test/{QSA_TESTS}/test_qsa.py",
                             f"test/{QSA_TESTS}/test_qsa_indexer.py",
                             f"test/{QSA_TESTS}/test_qsa_strided_zero_fill.py",
                             f"test/{GEMM_TESTS}/test_hc_mix.py"], {}),
        ]  # fmt: skip
        return steps, str(fork / "python")
    probe = [py, str(plugin / "tools" / "evidence" / "plugin_probe.py")]
    steps = [
        ("step2", plugin, [py, *PYTEST, "-q", "tests/model_compat/test_marlin_deterministic_alignment.py"],
         {"QSA_GPU_TESTS": "1", "QSA_FORK_ROOT": str(fork)}),
        *((name, plugin, argv, {}) for name, argv in probes(
            lambda script, *args: [*probe, str(fork / "test" / "manual" / script), *args], output)),
        ("step8a", plugin, [py, *PYTEST, "-v", "-rA", "--runxfail", "tests/qsa/test_qsa.py",
                            "tests/model_compat/test_hc_mix_triton.py"], {"QSA_GPU_TESTS": "1"}),
        ("step8b", pin / "test", [*probe, *PYTEST, "-v", "-rA", f"{QSA_TESTS}/test_qsa_indexer.py",
                                  f"{QSA_TESTS}/test_qsa_strided_zero_fill.py",
                                  f"{GEMM_TESTS}/test_hc_mix.py"], {}),
    ]  # fmt: skip
    return steps, os.pathsep.join([str(plugin / "src"), str(pin / "python")])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--arm", choices=("fork", "plugin"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    steps, pythonpath = plan(args.arm, output)
    env = {k: v for k, v in os.environ.items() if k not in UNSET}
    env.update(PYTHONPATH=pythonpath, CUDA_VISIBLE_DEVICES="0", PYTHONDONTWRITEBYTECODE="1")
    record = [
        {"step": name, "cwd": str(cwd), "argv": argv, "env": extra}
        for name, cwd, argv, extra in steps
    ]
    if args.dry_run:
        print(json.dumps({"arm": args.arm, "PYTHONPATH": pythonpath, "steps": record}, indent=2))
        return 0
    output.mkdir(parents=True)  # Refuses an existing arm directory.
    for entry, (name, cwd, argv, extra) in zip(record, steps):
        with (output / f"{name}.log").open("w") as log:
            entry["exit"] = subprocess.run(
                argv, cwd=cwd, env={**env, **extra}, stdout=log, stderr=subprocess.STDOUT
            ).returncode
        print(f"{name} {entry['exit']}", flush=True)
        (output / "steps.json").write_text(json.dumps(record, indent=2) + "\n")
    return 0 if all(entry["exit"] == 0 for entry in record) else 1


if __name__ == "__main__":
    sys.exit(main())
