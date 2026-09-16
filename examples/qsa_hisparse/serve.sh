#!/usr/bin/env bash
# Bounded TP2/B8 offload; requires a compatible Qwen3.8 Flash Next checkpoint.
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to a compatible local checkpoint or model repository}"
export SGLANG_QSA_HISPARSE_V3=p2-offload
export SGLANG_QSA_HISPARSE_V3_OBSERVE=light
export SGLANG_QWEN38_GDN_QKVZ_WNA16=0

exec "${QSA_PYTHON:-python3}" -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --model-impl sglang \
  --host "${BIND_HOST:-127.0.0.1}" \
  --port "${PORT:-30000}" \
  --tp-size 2 --pp-size 1 \
  --dtype bfloat16 --kv-cache-dtype fp8_e4m3 \
  --linear-attn-backend triton --mamba-ssm-dtype float32 \
  --mamba-radix-cache-strategy extra_buffer --mamba-track-interval 64 \
  --max-mamba-cache-size 40 \
  --context-length 262144 --max-total-tokens 2097152 \
  --max-running-requests 8 --chunked-prefill-size 2048 \
  --page-size 64 --random-seed 147342228 \
  --disable-radix-cache --disable-overlap-schedule --skip-server-warmup \
  --cuda-graph-backend-decode full --cuda-graph-backend-prefill disabled \
  --disable-cuda-graph-padding \
  --cuda-graph-max-bs-decode 8 --cuda-graph-bs-decode 1 2 3 4 5 6 7 8 \
  --prefill-decode-interval 1 \
  "$@"
