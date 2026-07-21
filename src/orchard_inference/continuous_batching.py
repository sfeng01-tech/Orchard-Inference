"""Transparent simulator and experimental MLX-LM continuous-batch runner."""

import argparse
import importlib.metadata
import json
import math
import platform
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol, cast

import psutil


class _BatchGeneratorProtocol(Protocol):
    def insert(self, prompts: list[list[int]], max_tokens: list[int]) -> list[int]: ...

    def next(self) -> tuple[list[Any], list[Any]]: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class SequenceSpec:
    """A deterministic sequence workload description."""

    request_id: str
    arrival_step: int
    prompt_steps: int
    decode_steps: int


@dataclass(slots=True)
class SequenceState:
    """Mutable simulator state for one sequence."""

    spec: SequenceSpec
    prompt_remaining: int
    decode_remaining: int
    first_token_step: int | None = None
    completed_step: int | None = None


@dataclass(frozen=True, slots=True)
class StepEvent:
    """Inspectable state after one simulator iteration."""

    step: int
    arrived: tuple[str, ...]
    completed: tuple[str, ...]
    prefill_active: int
    decode_active: int
    waiting: int


@dataclass(slots=True)
class SimulationResult:
    """Continuous-batching simulation output and utilization trace."""

    states: dict[str, SequenceState]
    events: list[StepEvent]
    decode_capacity: int


def jains_fairness(values: list[float]) -> float:
    """Return Jain's fairness index for positive service values."""

    if not values or not any(values):
        return 1.0
    total = sum(values)
    return total * total / (len(values) * sum(value * value for value in values))


def percentile(values: list[float], quantile: float) -> float | None:
    """Return a linearly interpolated percentile."""

    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


class ContinuousBatchSimulator:
    """Discrete token-step simulator with live admission and sequence removal."""

    def __init__(self, prefill_capacity: int, decode_capacity: int) -> None:
        if prefill_capacity < 1 or decode_capacity < 1:
            raise ValueError("capacities must be positive")
        self.prefill_capacity = prefill_capacity
        self.decode_capacity = decode_capacity

    def run(self, specs: list[SequenceSpec]) -> SimulationResult:
        """Run until every sequence completes, recording every scheduling step."""

        if not specs:
            return SimulationResult({}, [], self.decode_capacity)
        states = {
            spec.request_id: SequenceState(spec, spec.prompt_steps, spec.decode_steps)
            for spec in specs
        }
        waiting: list[SequenceState] = []
        prefilling: list[SequenceState] = []
        decoding: list[SequenceState] = []
        events = []
        step = 0
        while any(state.completed_step is None for state in states.values()):
            arrived = [
                state
                for state in states.values()
                if state.spec.arrival_step == step and state not in waiting
            ]
            waiting.extend(arrived)
            while waiting and len(prefilling) < self.prefill_capacity:
                prefilling.append(waiting.pop(0))
            completed = []
            decode_processed = decoding[: self.decode_capacity]
            for state in decode_processed:
                if state.first_token_step is None:
                    state.first_token_step = step
                state.decode_remaining -= 1
                if state.decode_remaining == 0:
                    state.completed_step = step + 1
                    completed.append(state)
            decoding = [state for state in decoding if state.completed_step is None]
            for state in prefilling:
                state.prompt_remaining -= 1
            prefill_done = [state for state in prefilling if state.prompt_remaining == 0]
            prefilling = [state for state in prefilling if state.prompt_remaining > 0]
            decoding.extend(prefill_done)
            while waiting and len(prefilling) < self.prefill_capacity:
                prefilling.append(waiting.pop(0))
            events.append(
                StepEvent(
                    step=step,
                    arrived=tuple(state.spec.request_id for state in arrived),
                    completed=tuple(state.spec.request_id for state in completed),
                    prefill_active=len(prefilling),
                    decode_active=len(decode_processed),
                    waiting=len(waiting),
                )
            )
            step += 1
            if step > 1_000_000:
                raise RuntimeError("simulation failed to converge")
        return SimulationResult(states, events, self.decode_capacity)


def summarize_simulation(result: SimulationResult) -> dict[str, Any]:
    """Summarize throughput, latency, TTFT, fairness, and decode utilization."""

    if not result.states:
        return {}
    makespan = len(result.events)
    latencies = [
        state.completed_step - state.spec.arrival_step
        for state in result.states.values()
        if state.completed_step is not None
    ]
    ttfts = [
        state.first_token_step - state.spec.arrival_step
        for state in result.states.values()
        if state.first_token_step is not None
    ]
    inverse_slowdowns = []
    for state in result.states.values():
        if state.completed_step is None:
            continue
        ideal = state.spec.prompt_steps + state.spec.decode_steps
        actual = state.completed_step - state.spec.arrival_step
        inverse_slowdowns.append(ideal / actual)
    utilization = sum(event.decode_active for event in result.events) / (
        makespan * result.decode_capacity
    )
    return {
        "requests": len(result.states),
        "makespan_steps": makespan,
        "requests_per_step": len(result.states) / makespan,
        "latency_mean_steps": sum(latencies) / len(latencies),
        "latency_p95_steps": percentile([float(value) for value in latencies], 0.95),
        "ttft_mean_steps": sum(ttfts) / len(ttfts),
        "ttft_p95_steps": percentile([float(value) for value in ttfts], 0.95),
        "fairness_jain_inverse_slowdown": jains_fairness(inverse_slowdowns),
        "active_sequence_utilization": utilization,
    }


def simulate_static_batches(specs: list[SequenceSpec], batch_size: int) -> dict[str, Any]:
    """Simulate cohort batching where all members finish at the longest member."""

    pending = sorted(specs, key=lambda spec: (spec.arrival_step, spec.request_id))
    now = 0
    latencies = []
    inverse_slowdowns = []
    utilization_numerator = 0
    utilization_denominator = 0
    while pending:
        batch = pending[:batch_size]
        del pending[:batch_size]
        now = max(now, max(spec.arrival_step for spec in batch))
        durations = [spec.prompt_steps + spec.decode_steps for spec in batch]
        duration = max(durations)
        utilization_numerator += sum(durations)
        utilization_denominator += duration * batch_size
        finish = now + duration
        for spec in batch:
            latency = finish - spec.arrival_step
            latencies.append(latency)
            inverse_slowdowns.append((spec.prompt_steps + spec.decode_steps) / latency)
        now = finish
    return {
        "requests": len(specs),
        "makespan_steps": now,
        "requests_per_step": len(specs) / now,
        "latency_mean_steps": sum(latencies) / len(latencies),
        "latency_p95_steps": percentile([float(value) for value in latencies], 0.95),
        "fairness_jain_inverse_slowdown": jains_fairness(inverse_slowdowns),
        "active_sequence_utilization": utilization_numerator / utilization_denominator,
    }


@dataclass(slots=True)
class RuntimeSequence:
    """Measurements collected for one MLX BatchGenerator UID."""

    request_id: str
    uid: int
    arrival_step: int
    inserted_at: float
    first_token_at: float | None = None
    completed_at: float | None = None
    output_tokens: list[int] = field(default_factory=list)
    inter_token_seconds: list[float] = field(default_factory=list)


def run_mlx_experiment(args: argparse.Namespace) -> dict[str, Any]:
    """Run true live insertion/removal through MLX-LM BatchGenerator."""

    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.generate import BatchGenerator

    loaded = load(args.model)
    model, tokenizer = loaded[0], loaded[1]
    generator = cast(
        _BatchGeneratorProtocol,
        BatchGenerator(
            model,
            max_tokens=args.max_tokens,
            stop_tokens=[[token] for token in tokenizer.eos_token_ids],
            completion_batch_size=args.completion_batch_size,
            prefill_batch_size=args.prefill_batch_size,
        ),
    )
    pending = [
        (index * args.arrival_interval_steps, f"request-{index}") for index in range(args.requests)
    ]
    sequences: dict[int, RuntimeSequence] = {}
    active: set[int] = set()
    decode_occupancy: list[int] = []
    step_durations: list[float] = []
    step = 0
    started = time.perf_counter()
    try:
        while pending or active:
            due = [item for item in pending if item[0] <= step]
            pending = [item for item in pending if item[0] > step]
            for arrival_step, request_id in due:
                prompt = " ".join(["benchmark"] * args.prompt_words) + f" {request_id}"
                tokens = tokenizer.encode(prompt)
                uids = generator.insert([tokens], max_tokens=[args.max_tokens])
                uid = uids[0]
                sequences[uid] = RuntimeSequence(
                    request_id=request_id,
                    uid=uid,
                    arrival_step=arrival_step,
                    inserted_at=time.perf_counter(),
                )
                active.add(uid)
            step_started = time.perf_counter()
            _, responses = generator.next()
            step_durations.append(time.perf_counter() - step_started)
            decode_occupancy.append(len(responses))
            now = time.perf_counter()
            for response in responses:
                sequence = sequences[response.uid]
                if sequence.first_token_at is None:
                    sequence.first_token_at = now
                elif sequence.output_tokens:
                    sequence.inter_token_seconds.append(now - (sequence.completed_at or now))
                if response.finish_reason != "stop":
                    sequence.output_tokens.append(response.token)
                sequence.completed_at = now
                if response.finish_reason is not None:
                    active.discard(response.uid)
            step += 1
        elapsed = time.perf_counter() - started
    finally:
        generator.close()
        mx.clear_cache()
    latencies = [
        sequence.completed_at - sequence.inserted_at
        for sequence in sequences.values()
        if sequence.completed_at is not None
    ]
    ttfts = [
        sequence.first_token_at - sequence.inserted_at
        for sequence in sequences.values()
        if sequence.first_token_at is not None
    ]
    service_rates = [
        len(sequence.output_tokens) / latency
        for sequence, latency in zip(sequences.values(), latencies, strict=True)
        if latency > 0
    ]
    return {
        "metadata": {
            "experimental": True,
            "runtime": "mlx_lm.generate.BatchGenerator",
            "model": args.model,
            "mlx": importlib.metadata.version("mlx"),
            "mlx_lm": importlib.metadata.version("mlx-lm"),
            "platform": platform.platform(),
            "memory_bytes": psutil.virtual_memory().total,
            "requests": args.requests,
            "prompt_words": args.prompt_words,
            "max_tokens": args.max_tokens,
            "arrival_interval_steps": args.arrival_interval_steps,
            "prefill_batch_size": args.prefill_batch_size,
            "completion_batch_size": args.completion_batch_size,
        },
        "elapsed_seconds": elapsed,
        "requests_per_second": len(sequences) / elapsed,
        "generated_tokens_per_second": (
            sum(len(sequence.output_tokens) for sequence in sequences.values()) / elapsed
        ),
        "latency_mean_seconds": statistics_mean(latencies),
        "latency_p95_seconds": percentile(latencies, 0.95),
        "ttft_mean_seconds": statistics_mean(ttfts),
        "ttft_p95_seconds": percentile(ttfts, 0.95),
        "inter_token_p95_seconds": percentile(
            [value for sequence in sequences.values() for value in sequence.inter_token_seconds],
            0.95,
        ),
        "fairness_jain_sequence_service_rate": jains_fairness(service_rates),
        "active_sequence_utilization": (
            sum(decode_occupancy) / (len(decode_occupancy) * args.completion_batch_size)
        ),
        "mean_step_seconds": statistics_mean(step_durations),
        "peak_process_rss_bytes": psutil.Process().memory_info().rss,
        "sequences": [
            {
                "request_id": sequence.request_id,
                "uid": sequence.uid,
                "generated_tokens": len(sequence.output_tokens),
                "output": tokenizer.decode(sequence.output_tokens),
            }
            for sequence in sequences.values()
        ],
    }


def statistics_mean(values: list[float]) -> float | None:
    """Return an arithmetic mean or None for an empty list."""

    return sum(values) / len(values) if values else None


def _workload(requests: int, arrival_interval: int) -> list[SequenceSpec]:
    return [
        SequenceSpec(
            request_id=f"request-{index}",
            arrival_step=index * arrival_interval,
            prompt_steps=1 + index % 3,
            decode_steps=2 + (index * 3) % 9,
        )
        for index in range(requests)
    ]


def _write_output(path: str | None, value: object) -> None:
    rendered = json.dumps(value, indent=2) + "\n"
    if path is not None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered)
    print(rendered, end="")


def run() -> None:
    """Run the simulator or opt-in MLX runtime experiment."""

    parser = argparse.ArgumentParser(prog="orchard-continuous-batch")
    commands = parser.add_subparsers(dest="command", required=True)
    simulator = commands.add_parser("simulate")
    simulator.add_argument("--requests", type=int, default=16)
    simulator.add_argument("--arrival-interval-steps", type=int, default=1)
    simulator.add_argument("--prefill-capacity", type=int, default=2)
    simulator.add_argument("--decode-capacity", type=int, default=4)
    simulator.add_argument("--output")
    runtime = commands.add_parser("runtime")
    runtime.add_argument("--model", required=True)
    runtime.add_argument("--requests", type=int, default=8)
    runtime.add_argument("--arrival-interval-steps", type=int, default=1)
    runtime.add_argument("--prompt-words", type=int, default=32)
    runtime.add_argument("--max-tokens", type=int, default=32)
    runtime.add_argument("--prefill-batch-size", type=int, default=2)
    runtime.add_argument("--completion-batch-size", type=int, default=4)
    runtime.add_argument("--output")
    args = parser.parse_args()
    if args.command == "simulate":
        specs = _workload(args.requests, args.arrival_interval_steps)
        result = ContinuousBatchSimulator(args.prefill_capacity, args.decode_capacity).run(specs)
        output = {
            "metadata": {"simulated": True, "token_step_model": True},
            "continuous": summarize_simulation(result),
            "static": simulate_static_batches(specs, args.decode_capacity),
            "events": [asdict(event) for event in result.events],
        }
    else:
        output = run_mlx_experiment(args)
    _write_output(args.output, output)


if __name__ == "__main__":
    run()
