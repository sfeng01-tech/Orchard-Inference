"""Internal inference-domain models."""

from dataclasses import dataclass
from enum import StrEnum


class HealthStatus(StrEnum):
    """Backend health states used by readiness checks."""

    UNLOADED = "unloaded"
    LOADING = "loading"
    READY = "ready"
    UNHEALTHY = "unhealthy"


@dataclass(frozen=True, slots=True)
class BackendHealth:
    """Current backend health snapshot."""

    status: HealthStatus
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """Metadata for the model exposed by a backend."""

    model_id: str
    backend: str
    quantization: str | None = None
    architecture: str = "unknown"


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    """Backend-neutral generation input."""

    prompt: str
    temperature: float
    top_p: float
    max_tokens: int
    stop: tuple[str, ...]
    prompt_token_ids: tuple[int, ...] | None = None

    @property
    def prompt_token_count(self) -> int:
        """Return model token count when cached, otherwise a word estimate."""

        if self.prompt_token_ids is not None:
            return len(self.prompt_token_ids)
        return len(self.prompt.split())


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """Completed generation output and token accounting."""

    text: str
    prompt_tokens: int
    generated_tokens: int
    finish_reason: str


@dataclass(frozen=True, slots=True)
class TokenEvent:
    """One decoded text fragment emitted by a streaming backend."""

    text: str
    token_count: int = 1
