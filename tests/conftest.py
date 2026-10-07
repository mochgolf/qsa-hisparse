"""Automated-run exclusions enforced at collection time (PLAN.md rules 5, 8).

- Tests marked ``gpu`` are skipped unless ``QSA_GPU_TESTS=1``.
- ``ServiceLifecycleTests`` (start real systemd user units) and tests marked
  ``service_lifecycle`` are deselected unless ``QSA_SERVICE_LIFECYCLE_TESTS=1``.
Both switches are cleared by ``tools/run_cpu_tests.sh``.

``QSA_ACTIVATE_FEATURES=1`` (set by the runner's integration pass) activates
the features named by the ``SGLANG_QSA_*`` switches before collection, with
the target-model scope forced on, since unit tests publish no served model.
"""

import os

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "service_lifecycle: starts systemd user units")
    if os.environ.get("QSA_ACTIVATE_FEATURES") == "1":
        from sglang_qsa_hisparse import patching, scope
        from sglang_qsa_hisparse.features import read_features

        scope.target_model_active = lambda: True
        patching.activate(read_features())


def pytest_collection_modifyitems(config, items):
    allow_gpu = os.environ.get("QSA_GPU_TESTS") == "1"
    allow_service = os.environ.get("QSA_SERVICE_LIFECYCLE_TESTS") == "1"
    skip_gpu = pytest.mark.skip(reason="GPU test; set QSA_GPU_TESTS=1")
    kept, deselected = [], []
    for item in items:
        lifecycle = "ServiceLifecycleTests" in item.nodeid or item.get_closest_marker(
            "service_lifecycle"
        )
        if lifecycle and not allow_service:
            deselected.append(item)
            continue
        if item.get_closest_marker("gpu") and not allow_gpu:
            item.add_marker(skip_gpu)
        kept.append(item)
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = kept
