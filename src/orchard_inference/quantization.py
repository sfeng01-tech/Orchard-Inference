"""Controlled same-architecture quantization evaluation and comparison."""

import argparse
import asyncio
import importlib.metadata
import importlib.resources
import json
import platform
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import httpx
import psutil

from orchard_inference.benchmark import Workload, percentile, run_scenario, summarize


@dataclass(frozen=True, slots=True)
class QualityResult:
    """Output and measurements for one fixed evaluation prompt."""

    prompt_id: str
    output: str
    keyword_pass: bool
    latency_seconds: float
    ttft_seconds: float | None
    generated_tokens: int
    decode_tokens_per_second: float | None


def _metric_value(text: str, name: str) -> float | None:
    for line in text.splitlines():
        if line.startswith(name + " "):
            return float(line.rsplit(" ", 1)[1])
    return None


async def _quality_request(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    item: dict[str, Any],
    max_tokens: int,
) -> QualityResult:
    started = time.perf_counter()
    first_token: float | None = None
    output_parts: list[str] = []
    generated_tokens = 0
    async with client.stream(
        "POST",
        f"{base_url}/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": item["prompt"]}],
            "temperature": 0,
            "max_tokens": max_tokens,
            "stream": True,
            "client_request_id": f"quant-{item['id']}",
        },
    ) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            choices = event.get("choices", [])
            content = choices[0].get("delta", {}).get("content") if choices else None
            if content:
                if first_token is None:
                    first_token = time.perf_counter()
                output_parts.append(content)
            orchard = event.get("orchard")
            if orchard:
                generated_tokens = orchard.get("generated_tokens", 0)
    finished = time.perf_counter()
    output = "".join(output_parts)
    ttft = first_token - started if first_token is not None else None
    decode_duration = finished - first_token if first_token is not None else None
    decode_tps = (
        generated_tokens / decode_duration
        if decode_duration is not None and decode_duration > 0
        else None
    )
    expected = [str(keyword).lower() for keyword in item["expected_keywords"]]
    return QualityResult(
        prompt_id=item["id"],
        output=output,
        keyword_pass=all(keyword in output.lower() for keyword in expected),
        latency_seconds=finished - started,
        ttft_seconds=ttft,
        generated_tokens=generated_tokens,
        decode_tokens_per_second=decode_tps,
    )


async def _poll_rss(
    client: httpx.AsyncClient,
    base_url: str,
    stopped: asyncio.Event,
    values: list[float],
) -> None:
    while not stopped.is_set():
        try:
            response = await client.get(f"{base_url}/metrics")
            value = _metric_value(response.text, "orchard_process_resident_memory_bytes")
            if value is not None:
                values.append(value)
        except httpx.HTTPError:
            pass
        try:
            await asyncio.wait_for(stopped.wait(), timeout=0.05)
        except TimeoutError:
            continue


def _load_evaluation() -> list[dict[str, Any]]:
    resource = importlib.resources.files("orchard_inference.data").joinpath(
        "quantization_eval.json"
    )
    return cast(list[dict[str, Any]], json.loads(resource.read_text()))


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


async def evaluate(args: argparse.Namespace) -> None:
    """Evaluate one already-running quantized model variant."""

    base_url = args.base_url.rstrip("/")
    async with httpx.AsyncClient(timeout=httpx.Timeout(args.timeout)) as client:
        model_response = await client.get(f"{base_url}/v1/models")
        model_response.raise_for_status()
        server_model = next(
            (item for item in model_response.json()["data"] if item["id"] == args.model),
            None,
        )
        if server_model is None:
            raise RuntimeError(f"model {args.model!r} is not loaded")
        if server_model.get("architecture") != args.architecture:
            raise RuntimeError(
                "server architecture does not match --architecture: "
                f"{server_model.get('architecture')!r} != {args.architecture!r}"
            )
        if server_model.get("quantization") != args.quantization:
            raise RuntimeError(
                "server quantization does not match --quantization: "
                f"{server_model.get('quantization')!r} != {args.quantization!r}"
            )
        metrics_before = (await client.get(f"{base_url}/metrics")).text
        baseline_rss = _metric_value(metrics_before, "orchard_process_resident_memory_bytes")
        load_sum = _metric_value(metrics_before, "orchard_model_load_seconds_sum")
        load_count = _metric_value(metrics_before, "orchard_model_load_seconds_count")
        load_seconds = load_sum / load_count if load_sum is not None and load_count else None
        rss_values = [baseline_rss] if baseline_rss is not None else []
        stopped = asyncio.Event()
        poller = asyncio.create_task(_poll_rss(client, base_url, stopped, rss_values))
        quality = []
        try:
            for item in _load_evaluation():
                quality.append(
                    await _quality_request(
                        client, base_url, args.model, item, args.quality_max_tokens
                    )
                )
            workload = Workload(args.prompt_lengths, args.output_lengths, args.seed)
            samples, elapsed, peak = await run_scenario(
                client,
                base_url=base_url,
                model=args.model,
                workload=workload,
                concurrency=args.concurrency,
                duration=args.duration,
                stream=True,
                mode="closed",
                arrival_rate=1,
            )
        finally:
            stopped.set()
            await poller
    ttfts = [result.ttft_seconds for result in quality if result.ttft_seconds is not None]
    decode_rates = [
        result.decode_tokens_per_second
        for result in quality
        if result.decode_tokens_per_second is not None
    ]
    artifact = {
        "metadata": {
            "captured_at": datetime.now(UTC).isoformat(),
            "variant": args.variant,
            "architecture": args.architecture,
            "quantization": args.quantization,
            "model": args.model,
            "server_model": server_model,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "memory_bytes": psutil.virtual_memory().total,
            "python": platform.python_version(),
            "mlx": importlib.metadata.version("mlx"),
            "mlx_lm": importlib.metadata.version("mlx-lm"),
            "seed": args.seed,
            "concurrency": args.concurrency,
            "duration_seconds": args.duration,
        },
        "model_load_seconds": load_seconds,
        "baseline_process_rss_bytes": baseline_rss,
        "peak_observed_process_rss_bytes": max(rss_values) if rss_values else None,
        "quality_pass_rate": statistics.fmean(result.keyword_pass for result in quality),
        "quality_results": [asdict(result) for result in quality],
        "quality_ttft_p50_seconds": percentile(ttfts, 0.5),
        "quality_decode_tokens_per_second_mean": (
            statistics.fmean(decode_rates) if decode_rates else None
        ),
        "concurrency_summary": summarize(samples, elapsed, peak),
    }
    output = Path(args.output)
    await asyncio.to_thread(_write_json, output, artifact)
    print(json.dumps(artifact, indent=2))


def compare(args: argparse.Namespace) -> None:
    """Validate and compare same-architecture evaluation artifacts."""

    artifacts = [json.loads(Path(path).read_text()) for path in args.artifacts]
    architectures = {artifact["metadata"]["architecture"] for artifact in artifacts}
    if len(architectures) != 1:
        raise SystemExit(f"refusing mixed-architecture comparison: {sorted(architectures)}")
    quantizations = [artifact["metadata"]["quantization"] for artifact in artifacts]
    if len(set(quantizations)) != len(quantizations):
        raise SystemExit("each artifact must have a distinct quantization label")
    rows = []
    for artifact in artifacts:
        summary = artifact["concurrency_summary"]
        rows.append(
            {
                "variant": artifact["metadata"]["variant"],
                "quantization": artifact["metadata"]["quantization"],
                "load_seconds": artifact["model_load_seconds"],
                "peak_rss_bytes": artifact["peak_observed_process_rss_bytes"],
                "quality_pass_rate": artifact["quality_pass_rate"],
                "ttft_p50_seconds": artifact["quality_ttft_p50_seconds"],
                "decode_tokens_per_second": artifact["quality_decode_tokens_per_second_mean"],
                "requests_per_second": summary["successful_requests_per_second"],
                "latency_p95_seconds": summary["latency_p95_seconds"],
            }
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(
        json.dumps({"architecture": architectures.pop(), "variants": rows}, indent=2) + "\n"
    )
    headers = list(rows[0])
    markdown = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    markdown.extend("| " + " | ".join(str(row[key]) for key in headers) + " |" for row in rows)
    output.with_suffix(".md").write_text("\n".join(markdown) + "\n")
    print(json.dumps(rows, indent=2))


def build_parser() -> argparse.ArgumentParser:
    """Build quantization evaluation subcommands."""

    parser = argparse.ArgumentParser(prog="orchard-quantization")
    commands = parser.add_subparsers(dest="command", required=True)
    evaluate_parser = commands.add_parser("evaluate")
    evaluate_parser.add_argument("--base-url", default="http://127.0.0.1:5000")
    evaluate_parser.add_argument("--model", required=True)
    evaluate_parser.add_argument("--variant", required=True)
    evaluate_parser.add_argument("--architecture", required=True)
    evaluate_parser.add_argument("--quantization", required=True)
    evaluate_parser.add_argument("--concurrency", type=int, default=4)
    evaluate_parser.add_argument("--duration", type=float, default=30)
    evaluate_parser.add_argument("--prompt-lengths", type=int, nargs="+", default=[32, 128])
    evaluate_parser.add_argument("--output-lengths", type=int, nargs="+", default=[32, 128])
    evaluate_parser.add_argument("--quality-max-tokens", type=int, default=64)
    evaluate_parser.add_argument("--seed", type=int, default=42)
    evaluate_parser.add_argument("--timeout", type=float, default=120)
    evaluate_parser.add_argument("--output", required=True)
    compare_parser = commands.add_parser("compare")
    compare_parser.add_argument("artifacts", nargs="+")
    compare_parser.add_argument("--output", required=True)
    return parser


def run() -> None:
    """Run quantization evaluation or comparison."""

    args = build_parser().parse_args()
    if args.command == "evaluate":
        asyncio.run(evaluate(args))
    else:
        compare(args)


if __name__ == "__main__":
    run()
