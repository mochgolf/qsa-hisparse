"""Automated-run exclusions enforced at collection time (PLAN.md rules 5, 8).

- Tests marked ``gpu`` are skipped unless ``QSA_GPU_TESTS=1``.
- ``ServiceLifecycleTests`` (start real systemd user units) and tests marked
  ``service_lifecycle`` are deselected unless ``QSA_SERVICE_LIFECYCLE_TESTS=1``.
Both switches are cleared by ``tools/run_cpu_tests.sh``.
"""

import os

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "service_lifecycle: starts systemd user units")


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
