"""Chunked-prefill and mixed prefill/decode scheduling simulator."""

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class ChunkedPrefillPolicy(StrEnum):
    """Policies for choosing between prefill chunks and decode tokens."""

    PREFILL_FIRST = "prefill_first"
    DECODE_FIRST = "decode_first"
    MIXED_SLO = "mixed_slo"


@dataclass(frozen=True, slots=True)
class ChunkedRequestSpec:
    """One synthetic request for chunked-prefill simulation."""

    request_id: str
    arrival_step: int
    prompt_tokens: int
    output_tokens: int
    prefix_saved_tokens: int = 0

    @property
    def effective_prefill_tokens(self) -> int:
        """Return prefill tokens after applying prefix-router savings."""

        return max(0, self.prompt_tokens - self.prefix_saved_tokens)


@dataclass(slots=True)
class ChunkedRequestState:
    """Mutable per-request simulator state."""

    spec: ChunkedRequestSpec
    prefill_remaining: int
    decode_remaining: int
    first_token_step: int | None = None
    completed_step: int | None = None
    last_decode_step: int | None = None
    inter_token_steps: list[int] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class ChunkedStepEvent:
    """Inspectable state after one scheduling step."""

    step: int
    arrived: tuple[str, ...]
    operation: str
    request_id: str | None
    tokens_processed: int
    waiting_prefill: int
    waiting_decode: int


@dataclass(slots=True)
class ChunkedSimulationResult:
    """Complete simulation trace."""

    policy: ChunkedPrefillPolicy
    chunk_size: int
    decode_tpot_slo_steps: int
    states: dict[str, ChunkedRequestState]
    events: list[ChunkedStepEvent]


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


class ChunkedPrefillSimulator:
    """Discrete scheduler where prefill chunks and decode tokens share one budget."""

    def __init__(
        self,
        *,
        chunk_size: int,
        policy: ChunkedPrefillPolicy,
        decode_tpot_slo_steps: int,
    ) -> None:
        if chunk_size < 1 or decode_tpot_slo_steps < 1:
            raise ValueError("chunk_size and decode_tpot_slo_steps must be positive")
        self.chunk_size = chunk_size
        self.policy = policy
        self.decode_tpot_slo_steps = decode_tpot_slo_steps

    def run(self, specs: list[ChunkedRequestSpec]) -> ChunkedSimulationResult:
        """Run the selected policy until every request completes."""

        states = {
            spec.request_id: ChunkedRequestState(
                spec=spec,
                prefill_remaining=spec.effective_prefill_tokens,
                decode_remaining=spec.output_tokens,
            )
            for spec in specs
        }
        waiting_prefill: list[ChunkedRequestState] = []
        waiting_decode: list[ChunkedRequestState] = []
        events: list[ChunkedStepEvent] = []
        step = 0
        while any(state.completed_step is None for state in states.values()):
            arrived = [
                state
                for state in states.values()
                if state.spec.arrival_step == step and state.completed_step is None
            ]
            waiting_prefill.extend(arrived)
            for state in list(waiting_prefill):
                if state.prefill_remaining == 0:
                    waiting_prefill.remove(state)
                    waiting_decode.append(state)
            selected = self._select(waiting_prefill, waiting_decode, step)
            operation = "idle"
            request_id = None
            tokens_processed = 0
            if selected is not None:
                operation, state = selected
                request_id = state.spec.request_id
                if operation == "prefill":
                    tokens_processed = min(self.chunk_size, state.prefill_remaining)
                    state.prefill_remaining -= tokens_processed
                    if state.prefill_remaining == 0:
                        waiting_prefill.remove(state)
                        waiting_decode.append(state)
                else:
                    tokens_processed = 1
                    if state.first_token_step is None:
                        state.first_token_step = step
                    if state.last_decode_step is not None:
                        state.inter_token_steps.append(step - state.last_decode_step)
                    state.last_decode_step = step
                    state.decode_remaining -= 1
                    if state.decode_remaining == 0:
                        waiting_decode.remove(state)
                        state.completed_step = step + 1
            events.append(
                ChunkedStepEvent(
                    step=step,
                    arrived=tuple(state.spec.request_id for state in arrived),
                    operation=operation,
                    request_id=request_id,
                    tokens_processed=tokens_processed,
                    waiting_prefill=len(waiting_prefill),
                    waiting_decode=len(waiting_decode),
                )
            )
            step += 1
            if step > 1_000_000:
                raise RuntimeError("chunked-prefill simulation failed to converge")
        return ChunkedSimulationResult(
            self.policy, self.chunk_size, self.decode_tpot_slo_steps, states, events
        )

    def _select(
        self,
        waiting_prefill: list[ChunkedRequestState],
        waiting_decode: list[ChunkedRequestState],
        step: int,
    ) -> tuple[str, ChunkedRequestState] | None:
        if self.policy is ChunkedPrefillPolicy.PREFILL_FIRST:
            if waiting_prefill:
                return "prefill", waiting_prefill[0]
            if waiting_decode:
                return "decode", waiting_decode[0]
            return None
        if self.policy is ChunkedPrefillPolicy.DECODE_FIRST:
            if waiting_decode:
                return "decode", waiting_decode[0]
            if waiting_prefill:
                return "prefill", waiting_prefill[0]
            return None
        urgent_decode = self._urgent_decode(waiting_decode, step)
        if urgent_decode is not None:
            return "decode", urgent_decode
        if waiting_prefill:
            return "prefill", waiting_prefill[0]
        if waiting_decode:
            return "decode", waiting_decode[0]
        return None

    def _urgent_decode(
        self, waiting_decode: list[ChunkedRequestState], step: int
    ) -> ChunkedRequestState | None:
        urgent = []
        for state in waiting_decode:
            last = state.last_decode_step
            due_step = state.first_token_step if last is None else last + self.decode_tpot_slo_steps
            if due_step is None or step >= due_step:
                urgent.append(state)
        if not urgent:
            return None
        return min(
            urgent,
            key=lambda state: (state.last_decode_step or -1, state.spec.arrival_step),
        )


def summarize(result: ChunkedSimulationResult) -> dict[str, Any]:
    """Summarize TTFT, TPOT, latency, throughput, utilization, and SLO misses."""

    completed = [state for state in result.states.values() if state.completed_step is not None]
    if not completed:
        return {}
    latencies = [
        state.completed_step - state.spec.arrival_step
        for state in completed
        if state.completed_step is not None
    ]
    ttfts = [
        state.first_token_step - state.spec.arrival_step
        for state in completed
        if state.first_token_step is not None
    ]
    tpot = [value for state in completed for value in state.inter_token_steps]
    useful_steps = sum(1 for event in result.events if event.operation != "idle")
    tpot_violations = [value for value in tpot if value > result.decode_tpot_slo_steps]
    return {
        "policy": result.policy.value,
        "chunk_size": result.chunk_size,
        "requests": len(completed),
        "makespan_steps": len(result.events),
        "requests_per_step": len(completed) / len(result.events),
        "latency_mean_steps": sum(latencies) / len(latencies),
        "latency_p95_steps": percentile([float(value) for value in latencies], 0.95),
        "ttft_mean_steps": sum(ttfts) / len(ttfts),
        "ttft_p95_steps": percentile([float(value) for value in ttfts], 0.95),
        "tpot_mean_steps": sum(tpot) / len(tpot) if tpot else None,
        "tpot_p95_steps": percentile([float(value) for value in tpot], 0.95),
        "tpot_violation_rate": len(tpot_violations) / len(tpot) if tpot else 0.0,
        "utilization": useful_steps / len(result.events),
        "effective_prefill_tokens": sum(state.spec.effective_prefill_tokens for state in completed),
        "prefix_saved_tokens": sum(state.spec.prefix_saved_tokens for state in completed),
    }


def deterministic_workload(
    *,
    requests: int,
    arrival_interval_steps: int,
    prompt_tokens: int,
    output_tokens: int,
    prefix_saved_tokens: int,
) -> list[ChunkedRequestSpec]:
    """Create a reproducible workload for CLI comparisons."""

    return [
        ChunkedRequestSpec(
            request_id=f"request-{index}",
            arrival_step=index * arrival_interval_steps,
            prompt_tokens=prompt_tokens + (index % 3) * max(1, prompt_tokens // 4),
            output_tokens=output_tokens + (index % 2) * max(1, output_tokens // 2),
            prefix_saved_tokens=prefix_saved_tokens if index else 0,
        )
        for index in range(requests)
    ]


def run_comparison(args: argparse.Namespace) -> dict[str, Any]:
    """Run every policy over the same deterministic workload."""

    specs = deterministic_workload(
        requests=args.requests,
        arrival_interval_steps=args.arrival_interval_steps,
        prompt_tokens=args.prompt_tokens,
        output_tokens=args.output_tokens,
        prefix_saved_tokens=args.prefix_saved_tokens,
    )
    runs = []
    for policy in ChunkedPrefillPolicy:
        simulator = ChunkedPrefillSimulator(
            chunk_size=args.chunk_size,
            policy=policy,
            decode_tpot_slo_steps=args.decode_tpot_slo_steps,
        )
        result = simulator.run(specs)
        runs.append(
            {
                "policy": policy.value,
                "summary": summarize(result),
                "events": [asdict(event) for event in result.events] if args.include_trace else [],
            }
        )
    return {
        "metadata": {
            "requests": args.requests,
            "arrival_interval_steps": args.arrival_interval_steps,
            "prompt_tokens": args.prompt_tokens,
            "output_tokens": args.output_tokens,
            "prefix_saved_tokens": args.prefix_saved_tokens,
            "chunk_size": args.chunk_size,
            "decode_tpot_slo_steps": args.decode_tpot_slo_steps,
        },
        "runs": runs,
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the chunked-prefill simulator parser."""

    parser = argparse.ArgumentParser(prog="orchard-chunked-prefill")
    parser.add_argument("--requests", type=int, default=16)
    parser.add_argument("--arrival-interval-steps", type=int, default=2)
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument("--prefix-saved-tokens", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--decode-tpot-slo-steps", type=int, default=2)
    parser.add_argument("--include-trace", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/chunked-prefill.json"),
    )
    return parser


def run() -> None:
    """Run the CLI and write JSON output."""

    args = build_parser().parse_args()
    if (
        args.requests <= 0
        or args.arrival_interval_steps < 0
        or args.prompt_tokens <= 0
        or args.output_tokens <= 0
        or args.prefix_saved_tokens < 0
    ):
        raise SystemExit("request counts, token counts, and intervals must be non-negative")
    result = run_comparison(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "runs": result["runs"]}, indent=2))


if __name__ == "__main__":
    run()
