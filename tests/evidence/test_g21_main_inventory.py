"""The main GPU inventory comes from collection, never observed outcomes."""

import os
import re
import subprocess
import sys
from pathlib import Path

import sglang

REPO = Path(__file__).resolve().parents[2]
EVIDENCE = REPO / "tools" / "evidence"
MAIN_INVENTORY = EVIDENCE / "g21_step8_inventory_582168dd7c.txt"


def test_main_inventory_preserves_previous_outcomes_and_exact_skip(evidence):
    outcomes = evidence("pytest_outcomes")
    old, old_skips = outcomes.read_inventory(EVIDENCE / "g21_step8_inventory_v0.5.22.txt")
    main, main_skips = outcomes.read_inventory(MAIN_INVENTORY)
    assert old <= main
    assert main_skips == old_skips
    assert len(main) == 105 and len(main - old) == 10
    assert {key for kind, key in main_skips if kind == "id"} == {
        "test_qsa.py::test_qsa_sm121_compaction_and_attention_match_sparse_reference"
    }


def test_main_inventory_matches_cpu_collection_of_every_planned_file(evidence):
    outcomes = evidence("pytest_outcomes")
    run_g21 = evidence("run_g21")
    pin = Path(sglang.__file__).resolve().parents[2]
    plans, _ = run_g21.plan("plugin", Path("/unused-output"))
    planned_files = {
        step: (cwd, [arg for arg in argv if arg.endswith(".py") and not arg.endswith("plugin_probe.py")])
        for step, cwd, argv, _ in plans if step in ("step8a", "step8b")
    }
    env = dict(os.environ)
    for name in (*run_g21.UNSET, "QSA_ACTIVATE_FEATURES", "SGLANG_QSA_ACTIVATION_DIR"):
        env.pop(name, None)
    env.update(CUDA_VISIBLE_DEVICES="99", PYTHONDONTWRITEBYTECODE="1", QSA_GPU_TESTS="1")
    collected = []
    for step in ("step8a", "step8b"):
        cwd, files = planned_files[step]
        # 8b collection reads pristine upstream alone; 8a needs plugin imports.
        paths = ([str(REPO / "src")] if step == "step8a" else []) + [str(pin / "python")]
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "--collect-only", "-q", *files],
            cwd=cwd, env={**env, "PYTHONPATH": os.pathsep.join(paths)},
            capture_output=True, text=True, timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        nodes = re.findall(r"^(\S+\.py)::(\S+)$", result.stdout, re.MULTILINE)
        counts = re.findall(r"^(\d+) tests? collected in ", result.stdout, re.MULTILINE)
        assert len(counts) == 1 and int(counts[0]) == len(nodes)
        collected.extend(outcomes.test_key(file, name) for file, name in nodes)
    assert len(collected) == len(set(collected)), "collection contains colliding canonical IDs"
    ids, skips = outcomes.read_inventory(MAIN_INVENTORY)
    skip_ids = {key for kind, key in skips if kind == "id"}
    assert set(collected) == ids | skip_ids
