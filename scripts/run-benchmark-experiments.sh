#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:5000}"
MODEL="${MODEL:-mock/orchard-test}"
DURATION="${DURATION:-60}"

uv run orchard-bench --base-url "$BASE_URL" --model "$MODEL" \
  --concurrency 1,2,4,8,16,32 --prompt-lengths 128 --output-lengths 128 \
  --duration "$DURATION" --output benchmarks/results/concurrency

uv run orchard-bench --base-url "$BASE_URL" --model "$MODEL" \
  --concurrency 8 --prompt-lengths 32,128,512,1024 --output-lengths 32,128,256 \
  --duration "$DURATION" --stream --output benchmarks/results/lengths-streaming

uv run orchard-bench --base-url "$BASE_URL" --model "$MODEL" \
  --mode open --arrival-rate 20 --concurrency 32 --prompt-lengths 128 \
  --output-lengths 128 --duration "$DURATION" --output benchmarks/results/overload
