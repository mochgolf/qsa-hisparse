"""Helpers of the pinned upstream ``test_pool_configurator.py``.

The fork left these helpers unchanged, so the plugin's port uses the pinned
checkout's copy instead of duplicating it.
"""

import importlib.util
from pathlib import Path

import sglang

PATH = (
    Path(sglang.__file__).resolve().parents[2]
    / "test/registered/unit/model_executor/test_pool_configurator.py"
)
_spec = importlib.util.spec_from_file_location("pinned_test_pool_configurator", PATH)
module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(module)

make_model_runner = module._make_model_runner
