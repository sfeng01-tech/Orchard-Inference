#!/usr/bin/env bash
set -euo pipefail

MODEL_ARCHITECTURE="${MODEL_ARCHITECTURE:?Set the shared architecture identifier}"
MLX_MODEL="${MLX_MODEL:?Set the MLX model ID}"
MPS_MODEL="${MPS_MODEL:?Set the equivalent Hugging Face model ID}"
DURATION="${DURATION:-60}"

# Start each server separately with the documented environment, then run the
# same workload. Separate ports can be used if memory permits both at once.
uv run orchard-bench --base-url "${MLX_URL:-http://127.0.0.1:5000}" \
  --model "$MLX_MODEL" --concurrency 1,2,4,8 --prompt-lengths 32,128,512 \
  --output-lengths 32,128 --duration "$DURATION" --stream \
  --output "benchmarks/results/${MODEL_ARCHITECTURE}-mlx"

uv run orchard-bench --base-url "${MPS_URL:-http://127.0.0.1:8001}" \
  --model "$MPS_MODEL" --concurrency 1,2,4,8 --prompt-lengths 32,128,512 \
  --output-lengths 32,128 --duration "$DURATION" --stream \
  --output "benchmarks/results/${MODEL_ARCHITECTURE}-mps"
