import importlib.util
from pathlib import Path

import pytest

EVIDENCE = Path(__file__).resolve().parents[2] / "tools" / "evidence"


@pytest.fixture
def evidence():
    """Load a fresh copy of tools/evidence/<name>.py (module state per test)."""

    def load(name):
        spec = importlib.util.spec_from_file_location(
            f"qsa_evidence_{name}_test", EVIDENCE / f"{name}.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    return load
