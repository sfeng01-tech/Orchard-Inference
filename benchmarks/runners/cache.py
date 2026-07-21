"""Deterministic application-cache workload benchmark."""

import argparse
import asyncio
import json
import random
import time

from orchard_inference.cache import CacheManager
from orchard_inference.observability import MetricMetadata, Metrics


async def run_workload(name: str, prompts: list[str], max_entries: int) -> dict[str, object]:
    """Measure one prompt/token cache access pattern."""

    metrics = Metrics(MetricMetadata("mock", "fifo"))
    manager = CacheManager(
        metrics,
        max_entries=max_entries,
        max_bytes=1024 * 1024,
        ttl_seconds=300,
        prompt_enabled=True,
        tokenization_enabled=True,
    )
    tokenizer_calls = 0

    async def tokenizer(prompt: str) -> tuple[int, ...]:
        nonlocal tokenizer_calls
        tokenizer_calls += 1
        return tuple(range(len(prompt.split())))

    started = time.perf_counter()
    for prompt in prompts:
        messages = [("system", "Shared system policy"), ("user", prompt)]
        rendered = manager.render_prompt("mock/cache-benchmark", messages)
        await manager.tokenize("mock/cache-benchmark", rendered, tokenizer)
    elapsed = time.perf_counter() - started
    return {
        "name": name,
        "requests": len(prompts),
        "elapsed_seconds": elapsed,
        "requests_per_second": len(prompts) / elapsed,
        "tokenizer_calls": tokenizer_calls,
        "estimated_cache_bytes": manager.estimated_bytes,
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    randomizer = random.Random(args.seed)
    unique = [f"unique prompt {index}" for index in range(args.requests)]
    repeated = ["repeated prompt"] * args.requests
    high_hit = [f"hot prompt {randomizer.randrange(10)}" for _ in range(args.requests)]
    low_hit = [f"cold prompt {randomizer.randrange(args.requests)}" for _ in range(args.requests)]
    results = [
        await run_workload("cold", unique, args.requests + 1),
        await run_workload("repeated_prefix", repeated, args.requests + 1),
        await run_workload("high_hit", high_hit, 128),
        await run_workload("low_hit", low_hit, 128),
    ]
    print(
        json.dumps(
            {
                "metadata": {
                    "backend": "mock",
                    "simulated": True,
                    "requests": args.requests,
                    "seed": args.seed,
                },
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
