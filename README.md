# Orchard Inference

Orchard Inference is a single-process LLM serving runtime for Apple silicon.
It provides an HTTP API, request admission and scheduling, streaming, batching,
observability, benchmarking, and a local control-room UI around MLX-LM and other
explicitly selected backends.

It is designed for inference-systems experimentation and reliable local serving,
not as a claim of production-scale distributed serving.

## Features

- FastAPI server with liveness, readiness, model, chat-completion, metrics, and UI endpoints.
- OpenAI-compatible chat-completions subset with non-streaming and SSE streaming responses.
- Deterministic mock backend for development, CI, load tests, and fault injection.
- MLX-LM backend for Apple GPU inference, plus an optional PyTorch MPS backend.
- Explicit backend selection with no silent GPU-to-CPU fallback.
- Bounded admission queues, active-request limits, deadlines, cancellation cleanup, and graceful shutdown.
- FIFO, aging-priority, and experimental shortest-job scheduling policies.
- Prompt, output, and total-token admission limits with rejection metrics.
- Compatibility-aware dynamic batching with size, wait, and token budgets.
- Prefix-aware routing, bounded prompt/tokenization caches, and cache instrumentation.
- Prometheus metrics, Grafana dashboard, example alerts, structured JSON logs, and bounded labels.
- Benchmark runners with deterministic workloads, percentile analysis, JSON/CSV artifacts, and chart generation.
- Continuous-batching, chunked-prefill, and paged-KV simulators for evaluating scheduling and memory policies.
- Local Control Room UI for requests, streaming output, metrics, and simulator visualizations.
- Reliability tests covering request lifecycle, streaming cancellation, backend failures, health failures, and memory pressure.

MLX-LM remains responsible for model loading, tokenization, sampling, and
autoregressive generation. Orchard owns the serving layer around those primitives.

## Orchard vs. MLX-LM

| Capability | MLX-LM | Orchard Inference |
|---|---|---|
| Primary role | Model and generation library/CLI | Long-running inference service |
| HTTP API | Not the main abstraction | FastAPI chat-completions API with SSE |
| Request management | Application-owned | Admission limits, queues, deadlines, cancellation, shutdown |
| Scheduling | Model-generation focused | FIFO, aging priority, and experimental shortest-job policies |
| Batching | Library-level generation support | Compatibility-aware batch formation and instrumentation |
| Observability | Application-owned | Structured logs, Prometheus metrics, dashboard, and alerts |
| Testing | Library/model behavior | API, lifecycle, fault-injection, scheduling, and backend tests |
| Benchmarking | Generation-oriented tools | Reproducible load generation, comparisons, and charts |
| UI | None required | Local Control Room |

Orchard is therefore complementary to MLX-LM: it uses MLX-LM as an inference
engine and adds the operational control plane needed to serve it through a
bounded, observable application.

## Requirements

- Apple silicon macOS
- Python 3.12
- [`uv`](https://docs.astral.sh/uv/)

## Install

```shell
uv sync --all-groups
```

To install the optional PyTorch MPS backend:

```shell
uv sync --all-groups --extra mps
```

## Run

Start the deterministic mock backend (no model download required):

```shell
uv run orchard-serve
```

Then check readiness and make a request:

```shell
curl http://127.0.0.1:5000/health/ready
curl http://127.0.0.1:5000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"mock/orchard-test","messages":[{"role":"user","content":"Hello"}]}'
```

Run MLX-LM on a model available to the machine:

```shell
ORCHARD_BACKEND=mlx \
ORCHARD_MODEL=mlx-community/Qwen2.5-3B-Instruct-4bit \
uv run orchard-serve
```

The server fails explicitly when MLX cannot access the Apple GPU or the model
cannot be loaded. It does not silently fall back to CPU execution.

## API surface

- `GET /health/live`
- `GET /health/ready`
- `GET /v1/models`
- `GET /metrics`
- `GET /ui`
- `POST /v1/chat/completions`

The chat endpoint supports `model`, `messages`, `temperature`, `top_p`,
`max_tokens`, `stop`, `client_request_id`, and optional `stream`.

## Benchmarking

With a server running:

```shell
uv run orchard-bench --model mock/orchard-test --concurrency 1,2,4 \
  --prompt-lengths 32,128 --output-lengths 32 --duration 10
```

Generate charts from a benchmark artifact:

```shell
python scripts/generate-benchmark-charts.py \
  benchmarks/results/orchard-benchmark.json \
  --output-dir benchmarks/charts/orchard-benchmark
```

## Development

```shell
uv run make check
```

This runs Ruff, strict mypy, and the full test suite. Tests use the mock backend
by default and do not download models.

## Scope and limitations

This is intentionally a single-process Apple-silicon runtime. It does not
provide distributed serving, multi-process scaling, or full OpenAI API
compatibility. Runtime KV-prefix reuse is not yet implemented; the prefix router
and KV block manager are bounded experiments and simulators. MLX cancellation is
cooperative between token-iterator steps, and an in-flight Metal operation cannot
be preempted by Python.

See [`docs/`](docs/) for architecture, reliability, observability, benchmarking,
backends, and simulator details.
