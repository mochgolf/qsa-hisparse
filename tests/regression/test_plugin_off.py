"""Plugin installed but off is pristine upstream (GOAL.md, "Done means").

A fresh process gets the launcher's environment (private dist-info on
``PYTHONPATH``, ``SGLANG_PLUGINS=qsa_hisparse``) without feature switches,
runs SGLang's plugin loader, and then runs the pinned upstream subset from
``docs/baseline.md`` in-process with pytest. The plugin must be loaded, the
HookRegistry must stay empty, no activation record may appear, and the subset
must give the pinned results (those of pristine 35f3c96ff4 with no plugin on
the path, CPU runner, 2026-10-08): the five unit files 82 passed / 3 skipped
(13 subtests), ``kernels/ops/attention/qsa/test_qsa.py`` (``kernel/qsa/``
before v0.5.21) 35 passed / 2 skipped with exactly the 13 CUDA-only failures
("No CUDA GPUs are available"; at v0.5.21: 81 / 3 / 9 subtests and two
unparametrized CUDA-only failures).
"""

import json
import os
import subprocess
import sys
import textwrap
from collections import Counter
from pathlib import Path

from sglang_qsa_hisparse import launch

UNIT_FILES = [
    "test/registered/unit/model_executor/test_pool_configurator.py",
    "test/registered/unit/mem_cache/test_qsa_kv_pool.py",
    "test/registered/unit/models/test_qwen4_exp_ple_table.py",
    "test/registered/unit/managers/test_batch_result_processor_hidden_states.py",
    "test/registered/unit/managers/test_scheduler_chunked_req_gate.py",
]
QSA_FILE = "test/registered/kernels/ops/attention/qsa/test_qsa.py"
CUDA_ONLY_FAILURES = {
    f"registered/kernels/ops/attention/qsa/test_qsa.py::{name}"
    for name in (
        *(f"test_qsa_draft_metadata_multi_step_graph[{case}]"
          for case in ("0-1", "0-3", "0-128", "1-1", "1-3", "1-128")),
        *(f"test_qsa_graph_layout_covers_speculative_rows_and_padded_tail[{offset}]"
          for offset in (0, 1, 3)),
        *(f"test_qsa_graph_metadata_kernels_match_legacy_host_path[{pages}]"
          for pages in (32, 129, 2048, 8193)),
    )
}  # fmt: skip
MARKER = "QSA_OFF_RESULT "

PROBE = textwrap.dedent(
    f"""
    import collections, json, os, sys
    import pytest
    from sglang.srt.plugins import load_plugins
    from sglang.srt.plugins.hook_registry import HookRegistry

    load_plugins()
    state = {{
        "plugin_loaded": "sglang_qsa_hisparse.plugin" in sys.modules,
        "patching_imported": "sglang_qsa_hisparse.patching" in sys.modules,
        "hooks_after_load": sorted(HookRegistry._hooks),
        "patched_after_load": sorted(HookRegistry._patched),
    }}

    class Collector:
        def __init__(self):
            self.outcomes = {{}}
            self.subtests = collections.Counter()

        def pytest_runtest_logreport(self, report):
            if type(report).__name__ != "TestReport":
                self.subtests[report.outcome] += 1
            elif report.failed:
                self.outcomes[report.nodeid] = "failed"
            elif report.skipped:
                self.outcomes[report.nodeid] = "skipped"
            elif report.when == "call":
                self.outcomes.setdefault(report.nodeid, "passed")

    collector = Collector()
    os.chdir(sys.argv[1])
    state["exit"] = int(pytest.main(["-q", "-p", "no:cacheprovider", *sys.argv[2:]], plugins=[collector]))
    state["outcomes"] = collector.outcomes
    state["subtests"] = dict(collector.subtests)
    state["hooks_after_tests"] = sorted(HookRegistry._hooks)
    print({MARKER!r} + json.dumps(state))
    """
)


def test_installed_but_off_is_pristine_upstream(tmp_path):
    import sglang

    pin = Path(sglang.__file__).resolve().parents[2]
    launch.write_dist_info(tmp_path / "site")
    base = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("SGLANG_QSA_") and key != "SGLANG_PLUGINS"
    }
    env = launch.server_environment(base, tmp_path / "site", tmp_path / "activation")
    result = subprocess.run(
        [sys.executable, "-c", PROBE, str(pin), *UNIT_FILES, QSA_FILE],
        env=env,
        capture_output=True,
        text=True,
    )
    lines = [l for l in result.stdout.splitlines() if l.startswith(MARKER)]
    assert len(lines) == 1, result.stdout[-3000:] + result.stderr[-3000:]
    state = json.loads(lines[0][len(MARKER) :])

    assert state["plugin_loaded"] and not state["patching_imported"]
    assert state["hooks_after_load"] == [] and state["patched_after_load"] == []
    assert state["hooks_after_tests"] == []
    assert not (tmp_path / "activation").exists()

    outcomes = state["outcomes"]
    unit = Counter(v for k, v in outcomes.items() if not k.startswith("registered/kernels/"))
    qsa = Counter(v for k, v in outcomes.items() if k.startswith("registered/kernels/ops/attention/qsa/"))
    failed = {k for k, v in outcomes.items() if v == "failed"}
    assert unit == {"passed": 82, "skipped": 3}, unit
    assert qsa == {"passed": 35, "skipped": 2, "failed": 13}, qsa
    assert failed == CUDA_ONLY_FAILURES
    assert state["subtests"] == {"passed": 13}
