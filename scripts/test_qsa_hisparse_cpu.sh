#!/usr/bin/env bash
# QSA regressions plus the upstream contracts used by the integration.
set -euo pipefail

qsa_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$qsa_root"
export PYTHONPATH="$qsa_root/python${PYTHONPATH:+:$PYTHONPATH}"
# SGLang's test port helper needs a numeric first character. An invalid ordinal
# hides CUDA without its empty-CUDA_VISIBLE_DEVICES parsing failure.
export CUDA_VISIBLE_DEVICES=99
export TRITON_INTERPRET=1

exec "${QSA_PYTHON:-python3}" -m pytest -q \
  test/qsa_hisparse \
  test/registered/unit/model_executor/test_pool_configurator.py \
  test/registered/unit/mem_cache/test_qsa_kv_pool.py \
  test/registered/unit/models/test_qwen4_exp_ple_table.py \
  test/registered/unit/managers/test_batch_result_processor_hidden_states.py \
  test/registered/unit/managers/test_scheduler_chunked_req_gate.py \
  "$@"
