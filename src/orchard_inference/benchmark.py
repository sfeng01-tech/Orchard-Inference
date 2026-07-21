"""Reproducible asynchronous HTTP benchmark CLI."""

import argparse
import asyncio
import csv
import importlib.metadata
import json
import math
import platform
import random
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import psutil


@dataclass(slots=True)
class RequestSample:
    """Measurements for one benchmark request."""

    sequence: int
    concurrency: int
    prompt_length: int
    output_length: int
    success: bool
    status_code: int | None
    latency_seconds: float
    ttft_seconds: float | None = None
    inter_token_seconds: list[float] = field(default_factory=list)
    queue_seconds: float | None = None
    generated_tokens: int = 0
    batch_size: int | None = None
    timed_out: bool = False
    cancelled: bool = False
    error_type: str | None = None


@dataclass(slots=True)
class ConcurrencyTracker:
    """Track current and peak in-flight requests on one event loop."""

    current: int = 0
    peak: int = 0

    def enter(self) -> None:
        self.current += 1
        self.peak = max(self.peak, self.current)

    def exit(self) -> None:
        self.current -= 1


class Workload:
    """Deterministic cyclic prompt/output bucket selection."""

    def __init__(self, prompt_lengths: list[int], output_lengths: list[int], seed: int) -> None:
        randomizer = random.Random(seed)
        combinations = [(prompt, output) for prompt in prompt_lengths for output in output_lengths]
        randomizer.shuffle(combinations)
        self._combinations = combinations
        self._sequence = 0

    def next(self) -> tuple[int, int, int, str]:
        """Return sequence, buckets, and deterministic prompt text."""

        sequence = self._sequence
        self._sequence += 1
        prompt_length, output_length = self._combinations[sequence % len(self._combinations)]
        words = [f"token{index % 97}" for index in range(prompt_length)]
        return sequence, prompt_length, output_length, " ".join(words)


def percentile(values: list[float], quantile: float) -> float | None:
    """Return a linearly interpolated percentile or None for no observations."""

    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(
    samples: list[RequestSample], elapsed_seconds: float, peak_concurrency: int
) -> dict[str, Any]:
    """Aggregate one benchmark scenario without hiding failed requests."""

    successful = [sample for sample in samples if sample.success]
    latencies = [sample.latency_seconds for sample in successful]
    ttfts = [sample.ttft_seconds for sample in successful if sample.ttft_seconds is not None]
    inter_tokens = [value for sample in successful for value in sample.inter_token_seconds]
    queues = [sample.queue_seconds for sample in successful if sample.queue_seconds is not None]
    batch_sizes: dict[str, int] = {}
    for sample in successful:
        if sample.batch_size is not None:
            key = str(sample.batch_size)
            batch_sizes[key] = batch_sizes.get(key, 0) + 1
    total = len(samples)
    return {
        "requests": total,
        "successful_requests": len(successful),
        "successful_requests_per_second": len(successful) / elapsed_seconds,
        "generated_tokens_per_second": (
            sum(sample.generated_tokens for sample in successful) / elapsed_seconds
        ),
        "latency_p50_seconds": percentile(latencies, 0.50),
        "latency_p95_seconds": percentile(latencies, 0.95),
        "latency_p99_seconds": percentile(latencies, 0.99),
        "ttft_p50_seconds": percentile(ttfts, 0.50),
        "ttft_p95_seconds": percentile(ttfts, 0.95),
        "ttft_p99_seconds": percentile(ttfts, 0.99),
        "inter_token_p50_seconds": percentile(inter_tokens, 0.50),
        "inter_token_p95_seconds": percentile(inter_tokens, 0.95),
        "inter_token_p99_seconds": percentile(inter_tokens, 0.99),
        "queue_p50_seconds": percentile(queues, 0.50),
        "queue_p95_seconds": percentile(queues, 0.95),
        "error_rate": (total - len(successful)) / total if total else 0.0,
        "timeout_rate": sum(sample.timed_out for sample in samples) / total if total else 0.0,
        "cancellation_rate": (
            sum(sample.cancelled for sample in samples) / total if total else 0.0
        ),
        "achieved_concurrency": peak_concurrency,
        "batch_size_distribution": batch_sizes,
        "elapsed_seconds": elapsed_seconds,
    }


async def _one_request(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    workload: Workload,
    concurrency: int,
    stream: bool,
    tracker: ConcurrencyTracker,
    offered_at: float | None = None,
) -> RequestSample:
    sequence, prompt_length, output_length, prompt = workload.next()
    sample = RequestSample(sequence, concurrency, prompt_length, output_length, False, None, 0)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": output_length,
        "stream": stream,
        "client_request_id": f"bench-{sequence}",
    }
    started = time.perf_counter() if offered_at is None else offered_at
    tracker.enter()
    try:
        if stream:
            last_token_at: float | None = None
            async with client.stream(
                "POST", f"{base_url}/v1/chat/completions", json=payload
            ) as response:
                sample.status_code = response.status_code
                async for line in response.aiter_lines():
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    event = json.loads(line[6:])
                    choices = event.get("choices", [])
                    content = choices[0].get("delta", {}).get("content") if choices else None
                    if content:
                        now = time.perf_counter()
                        if sample.ttft_seconds is None:
                            sample.ttft_seconds = now - started
                        elif last_token_at is not None:
                            sample.inter_token_seconds.append(now - last_token_at)
                        last_token_at = now
                    orchard = event.get("orchard")
                    if orchard:
                        sample.queue_seconds = orchard.get("queue_seconds")
                        sample.batch_size = orchard.get("batch_size")
                        sample.generated_tokens = orchard.get("generated_tokens", 0)
                sample.success = response.status_code == 200
        else:
            response = await client.post(f"{base_url}/v1/chat/completions", json=payload)
            sample.status_code = response.status_code
            sample.success = response.status_code == 200
            if sample.success:
                body = response.json()
                sample.generated_tokens = body.get("usage", {}).get("completion_tokens", 0)
                queue = response.headers.get("X-Orchard-Queue-Seconds")
                batch = response.headers.get("X-Orchard-Batch-Size")
                sample.queue_seconds = float(queue) if queue is not None else None
                sample.batch_size = int(batch) if batch is not None else None
    except httpx.TimeoutException:
        sample.timed_out = True
        sample.error_type = "client_timeout"
    except asyncio.CancelledError:
        sample.cancelled = True
        sample.error_type = "client_cancelled"
        raise
    except (httpx.HTTPError, json.JSONDecodeError, ValueError) as exc:
        sample.error_type = type(exc).__name__
    finally:
        tracker.exit()
        sample.latency_seconds = time.perf_counter() - started
    if not sample.success and sample.error_type is None:
        sample.error_type = f"http_{sample.status_code}"
        sample.timed_out = sample.status_code == 504
    return sample


async def run_scenario(
    client: httpx.AsyncClient,
    *,
    base_url: str,
    model: str,
    workload: Workload,
    concurrency: int,
    duration: float,
    stream: bool,
    mode: str,
    arrival_rate: float,
) -> tuple[list[RequestSample], float, int]:
    tracker = ConcurrencyTracker()
    samples: list[RequestSample] = []
    started = time.perf_counter()
    deadline = started + duration
    if mode == "closed":

        async def worker() -> None:
            while time.perf_counter() < deadline:
                samples.append(
                    await _one_request(
                        client, base_url, model, workload, concurrency, stream, tracker
                    )
                )

        await asyncio.gather(*(worker() for _ in range(concurrency)))
    else:
        semaphore = asyncio.Semaphore(concurrency)

        async def offered_request(offered_at: float) -> None:
            async with semaphore:
                samples.append(
                    await _one_request(
                        client,
                        base_url,
                        model,
                        workload,
                        concurrency,
                        stream,
                        tracker,
                        offered_at,
                    )
                )

        tasks: list[asyncio.Task[None]] = []
        interval = 1 / arrival_rate
        next_arrival = started
        while time.perf_counter() < deadline:
            tasks.append(asyncio.create_task(offered_request(time.perf_counter())))
            next_arrival += interval
            await asyncio.sleep(max(0.0, next_arrival - time.perf_counter()))
        await asyncio.gather(*tasks)
    return samples, time.perf_counter() - started, tracker.peak


def _csv_value(value: Any) -> str | int | float | None:
    return json.dumps(value, sort_keys=True) if isinstance(value, dict) else value


async def _main(args: argparse.Namespace) -> None:
    base_url = args.base_url.rstrip("/")
    timeout = httpx.Timeout(args.timeout)
    limits = httpx.Limits(max_connections=max(args.concurrency) * 2)
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        model_response = await client.get(f"{base_url}/v1/models")
        model_response.raise_for_status()
        model_data = model_response.json()["data"]
        loaded = next((item for item in model_data if item["id"] == args.model), None)
        if loaded is None:
            raise RuntimeError(f"model {args.model!r} is not loaded")
        warmup_workload = Workload(args.prompt_lengths, args.output_lengths, args.seed)
        warmup_tracker = ConcurrencyTracker()
        for _ in range(args.warmup_requests):
            await _one_request(
                client,
                base_url,
                args.model,
                warmup_workload,
                1,
                args.stream,
                warmup_tracker,
            )
        runs = []
        for concurrency in args.concurrency:
            workload = Workload(args.prompt_lengths, args.output_lengths, args.seed)
            samples, elapsed, peak = await run_scenario(
                client,
                base_url=base_url,
                model=args.model,
                workload=workload,
                concurrency=concurrency,
                duration=args.duration,
                stream=args.stream,
                mode=args.mode,
                arrival_rate=args.arrival_rate,
            )
            runs.append(
                {
                    "concurrency": concurrency,
                    "summary": summarize(samples, elapsed, peak),
                    "samples": [asdict(sample) for sample in samples],
                }
            )
    result = {
        "metadata": {
            "started_at": datetime.now(UTC).isoformat(),
            "base_url": base_url,
            "model": args.model,
            "server_model": loaded,
            "stream": args.stream,
            "mode": args.mode,
            "arrival_rate": args.arrival_rate if args.mode == "open" else None,
            "duration_seconds": args.duration,
            "warmup_requests": args.warmup_requests,
            "prompt_lengths": args.prompt_lengths,
            "output_lengths": args.output_lengths,
            "seed": args.seed,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "logical_cpu_count": psutil.cpu_count(),
            "memory_bytes": psutil.virtual_memory().total,
            "orchard_inference_version": importlib.metadata.version("orchard-inference"),
            "mlx_version": importlib.metadata.version("mlx"),
            "mlx_lm_version": importlib.metadata.version("mlx-lm"),
        },
        "runs": runs,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    summaries = [dict(concurrency=run["concurrency"], **run["summary"]) for run in runs]
    with output.with_suffix(".csv").open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(
            {key: _csv_value(value) for key, value in row.items()} for row in summaries
        )
    print(json.dumps({"output": str(output), "runs": summaries}, indent=2))


def _comma_ints(value: str) -> list[int]:
    parsed = [int(item) for item in value.split(",")]
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build the orchard-bench argument parser."""

    parser = argparse.ArgumentParser(prog="orchard-bench")
    parser.add_argument("--base-url", default="http://127.0.0.1:5000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--concurrency", type=_comma_ints, default=[1, 2, 4, 8])
    parser.add_argument("--prompt-lengths", type=_comma_ints, default=[32, 128, 512])
    parser.add_argument("--output-lengths", type=_comma_ints, default=[32, 128])
    parser.add_argument("--duration", type=float, default=60)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--mode", choices=("closed", "open"), default="closed")
    parser.add_argument("--arrival-rate", type=float, default=1.0)
    parser.add_argument("--warmup-requests", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--output", default="benchmarks/results/orchard-benchmark")
    return parser


def run() -> None:
    """Run the benchmark CLI."""

    args = build_parser().parse_args()
    if args.duration <= 0 or args.arrival_rate <= 0 or args.warmup_requests < 0:
        raise SystemExit("duration/arrival-rate must be positive and warm-up non-negative")
    asyncio.run(_main(args))


if __name__ == "__main__":
    run()
