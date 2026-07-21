"""Compare Orchard serving metrics against direct MLX-LM generation."""

import argparse
import asyncio
import csv
import importlib.metadata
import json
import platform
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import psutil

from orchard_inference.benchmark import (
    ConcurrencyTracker,
    RequestSample,
    Workload,
    _comma_ints,
    _csv_value,
    summarize,
)


def _metric_value(summary: dict[str, Any], metric: str) -> float | None:
    value = summary.get(metric)
    return float(value) if isinstance(value, int | float) else None


def _higher_is_better(metric: str) -> bool:
    return metric.endswith("_per_second")


def improvement_percent(
    orchard_summary: dict[str, Any], baseline_summary: dict[str, Any], metric: str
) -> float | None:
    """Return Orchard improvement percentage for one metric.

    Throughput metrics improve when they increase. Latency/error metrics improve
    when they decrease.
    """

    orchard = _metric_value(orchard_summary, metric)
    baseline = _metric_value(baseline_summary, metric)
    if orchard is None or baseline is None or baseline == 0:
        return None
    if _higher_is_better(metric):
        return ((orchard - baseline) / baseline) * 100
    return ((baseline - orchard) / baseline) * 100


async def _one_mlx_request(
    *,
    model: Any,
    tokenizer: Any,
    workload: Workload,
    concurrency: int,
    tracker: ConcurrencyTracker,
    stream: bool,
) -> RequestSample:
    from mlx_lm import generate, stream_generate
    from mlx_lm.sample_utils import make_sampler

    sequence, prompt_length, output_length, prompt = workload.next()
    sample = RequestSample(sequence, concurrency, prompt_length, output_length, False, None, 0)
    sampler = make_sampler(temp=0.0)
    started = time.perf_counter()
    tracker.enter()
    try:
        if stream:
            iterator = await asyncio.to_thread(
                stream_generate,
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=output_length,
                sampler=sampler,
            )
            text_parts: list[str] = []
            last_token_at: float | None = None
            while True:
                event = await asyncio.to_thread(_next_stream_event, iterator)
                if event is None:
                    break
                if not event.text:
                    continue
                now = time.perf_counter()
                if sample.ttft_seconds is None:
                    sample.ttft_seconds = now - started
                elif last_token_at is not None:
                    sample.inter_token_seconds.append(now - last_token_at)
                last_token_at = now
                text_parts.append(event.text)
            text = "".join(text_parts)
        else:
            text = await asyncio.to_thread(
                generate,
                model,
                tokenizer,
                prompt,
                max_tokens=output_length,
                sampler=sampler,
                verbose=False,
            )
        sample.success = True
        sample.status_code = 200
        sample.generated_tokens = len(tokenizer.encode(text))
    except Exception as exc:  # noqa: BLE001 - benchmark records failure types.
        sample.error_type = type(exc).__name__
    finally:
        tracker.exit()
        sample.latency_seconds = time.perf_counter() - started
    return sample


def _next_stream_event(iterator: Any) -> Any | None:
    try:
        return next(iterator)
    except StopIteration:
        return None


async def _run_mlx_scenario(
    *,
    model: Any,
    tokenizer: Any,
    workload: Workload,
    concurrency: int,
    duration: float,
    stream: bool,
) -> tuple[list[RequestSample], float, int]:
    tracker = ConcurrencyTracker()
    samples: list[RequestSample] = []
    started = time.perf_counter()
    deadline = started + duration

    async def worker() -> None:
        while time.perf_counter() < deadline:
            samples.append(
                await _one_mlx_request(
                    model=model,
                    tokenizer=tokenizer,
                    workload=workload,
                    concurrency=concurrency,
                    tracker=tracker,
                    stream=stream,
                )
            )

    await asyncio.gather(*(worker() for _ in range(concurrency)))
    return samples, time.perf_counter() - started, tracker.peak


async def _run_mlx_baseline(args: argparse.Namespace) -> None:
    from mlx_lm import load

    loaded = await asyncio.to_thread(load, args.model)
    model, tokenizer = loaded[0], loaded[1]
    warmup = Workload(args.prompt_lengths, args.output_lengths, args.seed)
    warmup_tracker = ConcurrencyTracker()
    for _ in range(args.warmup_requests):
        await _one_mlx_request(
            model=model,
            tokenizer=tokenizer,
            workload=warmup,
            concurrency=1,
            tracker=warmup_tracker,
            stream=args.stream,
        )

    runs = []
    for concurrency in args.concurrency:
        workload = Workload(args.prompt_lengths, args.output_lengths, args.seed)
        samples, elapsed, peak = await _run_mlx_scenario(
            model=model,
            tokenizer=tokenizer,
            workload=workload,
            concurrency=concurrency,
            duration=args.duration,
            stream=args.stream,
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
            "runner": "direct_mlx_lm_stream_generate"
            if args.stream
            else "direct_mlx_lm_generate",
            "stream": args.stream,
            "model": args.model,
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
            "mlx_version": importlib.metadata.version("mlx"),
            "mlx_lm_version": importlib.metadata.version("mlx-lm"),
        },
        "runs": runs,
    }
    await asyncio.to_thread(_write_json, Path(args.output), result)
    output = Path(args.output)
    print(json.dumps({"output": str(output), "runs": _summary_rows(result)}, indent=2))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _summary_rows(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    return [dict(concurrency=run["concurrency"], **run["summary"]) for run in artifact["runs"]]


def _load_artifact(path: str) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(Path(path).read_text()))


def compare_artifacts(args: argparse.Namespace) -> dict[str, Any]:
    """Compare an Orchard benchmark artifact with a direct MLX-LM artifact."""

    orchard = _load_artifact(args.orchard)
    baseline = _load_artifact(args.baseline)
    baseline_by_concurrency = {run["concurrency"]: run for run in baseline["runs"]}
    comparisons = []
    for orchard_run in orchard["runs"]:
        concurrency = orchard_run["concurrency"]
        baseline_run = baseline_by_concurrency.get(concurrency)
        if baseline_run is None:
            continue
        row: dict[str, Any] = {"concurrency": concurrency}
        for metric in args.metrics:
            orchard_value = _metric_value(orchard_run["summary"], metric)
            baseline_value = _metric_value(baseline_run["summary"], metric)
            row[f"orchard_{metric}"] = orchard_value
            row[f"baseline_{metric}"] = baseline_value
            row[f"{metric}_improvement_percent"] = improvement_percent(
                orchard_run["summary"], baseline_run["summary"], metric
            )
        comparisons.append(row)
    return {
        "metadata": {
            "started_at": datetime.now(UTC).isoformat(),
            "orchard_artifact": args.orchard,
            "baseline_artifact": args.baseline,
            "baseline_runner": baseline.get("metadata", {}).get("runner", "unknown"),
            "metrics": args.metrics,
        },
        "comparisons": comparisons,
    }


def _write_comparison(result: dict[str, Any], output: str) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    rows = result["comparisons"]
    if rows:
        with path.with_suffix(".csv").open("w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(
                {key: _csv_value(value) for key, value in row.items()} for row in rows
            )
    print(json.dumps({"output": str(path), "comparisons": rows}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    """Build the Orchard vs MLX-LM comparison parser."""

    parser = argparse.ArgumentParser(prog="orchard-compare-mlx")
    commands = parser.add_subparsers(dest="command", required=True)

    baseline = commands.add_parser("baseline", help="Run direct mlx_lm.generate benchmark")
    baseline.add_argument("--model", required=True)
    baseline.add_argument("--concurrency", type=_comma_ints, default=[1, 2, 4, 8])
    baseline.add_argument("--prompt-lengths", type=_comma_ints, default=[32, 128, 512])
    baseline.add_argument("--output-lengths", type=_comma_ints, default=[32, 128])
    baseline.add_argument("--duration", type=float, default=60)
    baseline.add_argument("--stream", action="store_true")
    baseline.add_argument("--warmup-requests", type=int, default=2)
    baseline.add_argument("--seed", type=int, default=42)
    baseline.add_argument("--output", default="benchmarks/results/direct-mlx-baseline.json")

    compare = commands.add_parser("compare", help="Compare Orchard and direct MLX artifacts")
    compare.add_argument("--orchard", required=True)
    compare.add_argument("--baseline", required=True)
    compare.add_argument(
        "--metrics",
        nargs="+",
        default=[
            "successful_requests_per_second",
            "generated_tokens_per_second",
            "latency_p50_seconds",
            "latency_p95_seconds",
            "ttft_p50_seconds",
            "ttft_p95_seconds",
            "inter_token_p50_seconds",
            "inter_token_p95_seconds",
            "error_rate",
        ],
    )
    compare.add_argument("--output", default="benchmarks/results/orchard-vs-mlx")
    return parser


def run() -> None:
    """Run the comparison CLI."""

    args = build_parser().parse_args()
    if args.command == "baseline":
        if args.duration <= 0 or args.warmup_requests < 0:
            raise SystemExit("duration must be positive and warm-up non-negative")
        asyncio.run(_run_mlx_baseline(args))
        return
    result = compare_artifacts(args)
    _write_comparison(result, args.output)


if __name__ == "__main__":
    run()
