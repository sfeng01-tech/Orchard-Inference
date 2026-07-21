"""Deterministic test backend."""

import asyncio
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass

from orchard_inference.errors import (
    BackendGenerationError,
    BackendLoadError,
    BackendUnhealthyError,
    MemoryPressureError,
)
from orchard_inference.models import (
    BackendHealth,
    GenerationRequest,
    GenerationResult,
    HealthStatus,
    ModelInfo,
    TokenEvent,
)


@dataclass(frozen=True, slots=True)
class MockFaultConfig:
    """Deterministic fault-injection controls for reliability tests."""

    fail_load: bool = False
    fail_generation: bool = False
    fail_after_tokens: int | None = None
    memory_pressure: bool = False
    unhealthy_after_requests: int | None = None


class MockBackend:
    """Fast deterministic backend used by local development and automated tests."""

    def __init__(
        self,
        model_id: str,
        token_delay_seconds: float = 0.0,
        architecture: str = "mock",
        quantization: str = "none",
        faults: MockFaultConfig | None = None,
    ) -> None:
        self._model_id = model_id
        self._token_delay_seconds = token_delay_seconds
        self._status = HealthStatus.UNLOADED
        self._architecture = architecture
        self._quantization = quantization
        self._faults = faults or MockFaultConfig()
        self._requests_seen = 0

    async def load(self) -> None:
        """Mark the in-memory backend ready."""

        self._status = HealthStatus.LOADING
        await asyncio.sleep(0)
        if self._faults.fail_load:
            self._status = HealthStatus.UNHEALTHY
            raise BackendLoadError("mock fault injection: model load failed")
        self._status = HealthStatus.READY

    async def unload(self) -> None:
        """Mark the backend unloaded."""

        self._status = HealthStatus.UNLOADED

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """Return a deterministic response while preserving cancellation points."""

        self._begin_request()
        words = "Orchard mock response".split()
        selected = words[: request.max_tokens]
        for index, _ in enumerate(selected):
            await asyncio.sleep(self._token_delay_seconds)
            self._raise_after_token(index + 1)
        text = " ".join(selected)
        finish_reason = "length" if request.max_tokens < len(words) else "stop"
        return GenerationResult(
            text=text,
            prompt_tokens=request.prompt_token_count,
            generated_tokens=len(selected),
            finish_reason=finish_reason,
        )

    async def stream(self, request: GenerationRequest) -> AsyncIterator[TokenEvent]:
        """Yield deterministic fragments with cancellation points."""

        self._begin_request()
        words = "Orchard mock response".split()[: request.max_tokens]
        for index, word in enumerate(words):
            await asyncio.sleep(self._token_delay_seconds)
            yield TokenEvent(text=("" if index == 0 else " ") + word)
            self._raise_after_token(index + 1)

    async def generate_batch(self, requests: list[GenerationRequest]) -> list[GenerationResult]:
        """Simulate one lockstep batched decode for deterministic scheduler tests."""

        for _request in requests:
            self._begin_request()
        if requests:
            steps = max(min(request.max_tokens, 3) for request in requests)
            for index in range(steps):
                await asyncio.sleep(self._token_delay_seconds)
                self._raise_after_token(index + 1)
        results = []
        words = "Orchard mock response".split()
        for request in requests:
            selected = words[: request.max_tokens]
            results.append(
                GenerationResult(
                    text=" ".join(selected),
                    prompt_tokens=request.prompt_token_count,
                    generated_tokens=len(selected),
                    finish_reason="length" if request.max_tokens < len(words) else "stop",
                )
            )
        return results

    async def tokenize(self, prompt: str) -> tuple[int, ...]:
        """Return deterministic word-level token IDs for tests."""

        await asyncio.sleep(0)
        return tuple(range(len(prompt.split())))

    async def render_prompt(self, messages: Sequence[tuple[str, str]]) -> str:
        """Render the deterministic mock transcript format."""

        await asyncio.sleep(0)
        return "".join(f"{role}: {content}\n" for role, content in messages) + "assistant:"

    def health(self) -> BackendHealth:
        """Return current mock health."""

        return BackendHealth(status=self._status)

    def model_info(self) -> ModelInfo:
        """Return deterministic model metadata."""

        return ModelInfo(
            model_id=self._model_id,
            backend="mock",
            architecture=self._architecture,
            quantization=self._quantization,
        )

    def _begin_request(self) -> None:
        """Apply deterministic per-request fault checks before generation starts."""

        if self._status is not HealthStatus.READY:
            raise BackendUnhealthyError("mock backend is not ready")
        self._requests_seen += 1
        if (
            self._faults.unhealthy_after_requests is not None
            and self._requests_seen > self._faults.unhealthy_after_requests
        ):
            self._status = HealthStatus.UNHEALTHY
            raise BackendUnhealthyError("mock fault injection: backend became unhealthy")
        if self._faults.memory_pressure:
            raise MemoryPressureError("mock fault injection: simulated memory pressure")
        if self._faults.fail_generation and self._faults.fail_after_tokens is None:
            raise BackendGenerationError("mock fault injection: generation failed")

    def _raise_after_token(self, token_count: int) -> None:
        """Raise an injected generation failure after N emitted decode steps."""

        if (
            self._faults.fail_after_tokens is not None
            and token_count >= self._faults.fail_after_tokens
        ):
            raise BackendGenerationError("mock fault injection: generation failed after tokens")
