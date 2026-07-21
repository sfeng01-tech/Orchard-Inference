"""Minimal inference runtime protocol."""

from collections.abc import AsyncIterator, Sequence
from typing import Protocol

from orchard_inference.models import (
    BackendHealth,
    GenerationRequest,
    GenerationResult,
    ModelInfo,
    TokenEvent,
)


class InferenceBackend(Protocol):
    """Capabilities shared by inference runtimes."""

    async def load(self) -> None:
        """Load runtime resources and the configured model."""

    async def unload(self) -> None:
        """Release runtime resources."""

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate one complete response."""

    async def generate_batch(self, requests: list[GenerationRequest]) -> list[GenerationResult]:
        """Generate a compatible request batch using backend-native batching."""

    def stream(self, request: GenerationRequest) -> AsyncIterator[TokenEvent]:
        """Stream decoded text fragments."""

    async def tokenize(self, prompt: str) -> tuple[int, ...]:
        """Tokenize a prompt for estimation and reusable application caching."""

    async def render_prompt(self, messages: Sequence[tuple[str, str]]) -> str:
        """Render chat messages into the backend's preferred prompt format."""

    def health(self) -> BackendHealth:
        """Return current backend health."""

    def model_info(self) -> ModelInfo:
        """Return loaded model metadata."""
