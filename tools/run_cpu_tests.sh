#!/usr/bin/env bash
# CPU tests of the plugin against the pinned SGLang checkout. Selects sources
# with PYTHONPATH; never installs into or modifies the shared environment.
# Default interpreter: the owner-approved validation env (read-only use).
set -euo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
pin_root=${QSA_PIN_ROOT:-$repo/../.worktrees/sglang-pin-76e06febab}
python=${QSA_PYTHON:-$repo/../service/runtime-env-sglang-20260923/bin/python}

cd "$repo"
export PYTHONPATH="$repo/src:$pin_root/python${PYTHONPATH:+:$PYTHONPATH}"
# Hide CUDA the same way the fork's CPU runner does (numeric, invalid ordinal).
export CUDA_VISIBLE_DEVICES=99
export TRITON_INTERPRET=1
# Never write bytecode into the pinned checkout or the shared interpreter.
export PYTHONDONTWRITEBYTECODE=1
# Never inherit feature switches or opt-ins for GPU/service tests
# (tests/conftest.py enforces the exclusions at collection time).
unset SGLANG_QSA_MODEL_COMPAT SGLANG_QSA_HISPARSE_V3 SGLANG_QSA_ACTIVATION_DIR \
  QSA_GPU_TESTS QSA_SERVICE_LIFECYCLE_TESTS SGLANG_TEST_MARLIN_GPU SGLANG_PLUGINS

exec "$python" -m pytest -q -p no:cacheprovider tests "$@"
