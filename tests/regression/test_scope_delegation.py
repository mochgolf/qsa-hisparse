"""Compat on with a non-target model: scoped hooks delegate (PLAN.md rule 9).

A fresh process publishes a temporary Llama model directory, so the real
``scope.target_model_active()`` is False, activates model_compat through
``plugin.load()``, and checks each hook of the per-call scoped rows with
``scope_delegation_probe.py``. Those rows are the generic shared-path rows of
``docs/patch-inventory.md`` section 1 (M04, J03, Z01, Z02, Z03, E-rows), the
other AutoRound MoE loading row Z04, and the stable-HC rows (H-rows; W5 card
item 2). QSA attention rows decide scope once at backend construction (W4
card item 4) and are tested with their hooks.

The real check needs every model_compat row merged (activation requires the
full manifest), hence ``integration``; the probe's self-test runs now.
"""

import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from sglang_qsa_hisparse import patching

PROBE = Path(__file__).resolve().parent / "scope_delegation_probe.py"
SCOPED_ROW = re.compile(r"^(M04|J03|Z0[1-4]|E\d\d|H\d\d)$")
LLAMA = {"architectures": ["LlamaForCausalLM"], "model_type": "llama"}


def probe_module():
    spec = importlib.util.spec_from_file_location("scope_delegation_probe", PROBE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_probe(tmp_path, rows, *flags):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps(LLAMA))
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("SGLANG_QSA_") and key != "SGLANG_PLUGINS"
    }
    env["SGLANG_QSA_MODEL_COMPAT"] = "1"
    result = subprocess.run(
        [sys.executable, str(PROBE), str(model), ",".join(rows), *flags],
        env=env,
        capture_output=True,
        text=True,
    )
    marker = probe_module().MARKER
    lines = [l for l in result.stdout.splitlines() if l.startswith(marker)]
    assert len(lines) == 1, result.stdout[-3000:] + result.stderr[-5000:]
    return json.loads(lines[0][len(marker) :])


def test_probe_tells_delegating_from_non_delegating_hooks(tmp_path):
    probe = probe_module()
    result = run_probe(tmp_path, probe.SELFTEST, "--selftest")
    assert result["active"] is False
    assert sorted(result["checked"]) == sorted(t for t, _ in probe.SELFTEST.values())
    assert sorted(result["failures"]) == sorted(
        probe.SELFTEST[row][0] for row in probe.SELFTEST_NON_DELEGATING
    ), result["failures"]


@pytest.mark.integration
def test_scoped_hooks_delegate_for_a_non_target_model(tmp_path):
    manifest = patching.load_manifest()
    rows = sorted(
        row
        for row, entry in manifest.items()
        if entry["feature"] == "model_compat" and entry["patches"] and SCOPED_ROW.match(row)
    )
    assert rows, "no scoped rows in the manifest"
    result = run_probe(tmp_path, rows)
    assert result["active"] is False
    assert sorted(result["checked"]) == sorted(
        patch["target"] for row in rows for patch in manifest[row]["patches"]
    )
    assert result["failures"] == {}
