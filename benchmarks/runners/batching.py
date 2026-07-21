"""Reproducible mock benchmark for Phase 4 batch policies."""

import argparse
import asyncio
import json
import platform
import statistics
import time
from dataclasses import asdict, dataclass

from orchard_inference.backends.mock import MockBackend
from orchard_inference.lifecycle import LifecycleManager, RequestContext, RequestState
from orchard_inference.models import GenerationRequest
from orchard_inference.scheduler import RequestScheduler, SchedulingPolicy


@dataclass(frozen=True)
class ModeResult:
    """Measured result for one simulated batching mode."""

    mode: str
    requests: int
    elapsed_seconds: float
    requests_per_second: float
    mean_latency_seconds: float
    batch_sizes: list[int]


async def _dynamic(requests: int, batch_size: int, wait_seconds: float) -> ModeResult:
    backend = MockBackend("mock/benchmark", token_delay_seconds=0.002)
    lifecycle = LifecycleManager()
    scheduler = RequestScheduler(
        backend,
        lifecycle,
        policy=SchedulingPolicy.FIFO,
        max_queued=requests,
        max_active=batch_size,
        max_prompt_tokens=128,
        max_output_tokens=32,
        max_total_tokens=160,
        stream_buffer_size=2,
        aging_seconds=1,
        max_batch_size=batch_size,
        max_batch_wait_seconds=wait_seconds,
        batch_token_budget=4096,
    )
    await backend.load()
    await scheduler.start()
    started = time.perf_counter()
    futures = []
    starts = []
    for index in range(requests):
        context = RequestContext(f"bench-{index}", deadline=time.monotonic() + 30)
        context.transition(RequestState.VALIDATED)
        item = await scheduler.submit(
            context,
            GenerationRequest("user: deterministic prompt", 0, 1, 3, ()),
            priority=0,
            stream=False,
        )
        starts.append(time.perf_counter())
        futures.append(item.result)
    latencies = []
    for request_started, future in zip(starts, futures, strict=True):
        await future
        latencies.append(time.perf_counter() - request_started)
    elapsed = time.perf_counter() - started
    await scheduler.shutdown()
    await lifecycle.shutdown(1)
    await backend.unload()
    return ModeResult(
        mode=("batch_1" if batch_size == 1 else "dynamic"),
        requests=requests,
        elapsed_seconds=elapsed,
        requests_per_second=requests / elapsed,
        mean_latency_seconds=statistics.fmean(latencies),
        batch_sizes=scheduler.metrics.batch_sizes,
    )


async def _static(requests: int, batch_size: int) -> ModeResult:
    backend = MockBackend("mock/benchmark", token_delay_seconds=0.002)
    await backend.load()
    request = GenerationRequest("user: deterministic prompt", 0, 1, 3, ())
    latencies = []
    sizes = []
    started = time.perf_counter()
    for offset in range(0, requests, batch_size):
        size = min(batch_size, requests - offset)
        batch_started = time.perf_counter()
        await backend.generate_batch([request] * size)
        latency = time.perf_counter() - batch_started
        latencies.extend([latency] * size)
        sizes.append(size)
    elapsed = time.perf_counter() - started
    await backend.unload()
    return ModeResult(
        mode="static",
        requests=requests,
        elapsed_seconds=elapsed,
        requests_per_second=requests / elapsed,
        mean_latency_seconds=statistics.fmean(latencies),
        batch_sizes=sizes,
    )


async def main() -> None:
    """Run batch-1, static, and dynamic mock workloads and print JSON."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--batch-wait-ms", type=float, default=5)
    args = parser.parse_args()
    results = [
        await _dynamic(args.requests, 1, 0),
        await _static(args.requests, args.batch_size),
        await _dynamic(args.requests, args.batch_size, args.batch_wait_ms / 1000),
    ]
    print(
        json.dumps(
            {
                "metadata": {
                    "backend": "mock",
                    "simulated": True,
                    "python": platform.python_version(),
                    "platform": platform.platform(),
                    "batch_size": args.batch_size,
                    "batch_wait_ms": args.batch_wait_ms,
                },
                "results": [asdict(result) for result in results],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
