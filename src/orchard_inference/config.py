"""Application configuration."""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed settings loaded from ORCHARD_* environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="ORCHARD_", env_file=".env", extra="ignore", frozen=True
    )

    backend: Literal["mock", "mlx", "mps"] = "mock"
    model: str = "mock/orchard-test"
    model_architecture: str = "mock"
    model_quantization: str = "none"
    mps_dtype: Literal["float16", "float32", "bfloat16"] = "float16"
    host: str = "127.0.0.1"
    port: int = Field(default=5000, ge=1, le=65535)
    log_level: str = "INFO"
    mock_token_delay_seconds: float = Field(default=0.0, ge=0.0)
    mock_fail_load: bool = False
    mock_fail_generation: bool = False
    mock_fail_after_tokens: int | None = Field(default=None, ge=0)
    mock_memory_pressure: bool = False
    mock_unhealthy_after_requests: int | None = Field(default=None, ge=1)
    request_timeout_seconds: float = Field(default=30.0, gt=0.0)
    stream_buffer_size: int = Field(default=8, ge=1, le=1024)
    shutdown_grace_seconds: float = Field(default=10.0, ge=0.0)
    scheduling_policy: Literal["fifo", "priority", "shortest_job"] = "fifo"
    max_queued_requests: int = Field(default=64, ge=1)
    max_active_requests: int = Field(default=4, ge=1)
    max_prompt_tokens: int = Field(default=4096, ge=1)
    max_output_tokens: int = Field(default=1024, ge=1)
    max_total_estimated_tokens: int = Field(default=8192, ge=2)
    scheduler_aging_seconds: float = Field(default=5.0, gt=0.0)
    max_batch_size: int = Field(default=4, ge=1)
    max_batch_wait_seconds: float = Field(default=0.005, ge=0.0)
    batch_token_budget: int = Field(default=8192, ge=1)
    prompt_cache_enabled: bool = True
    tokenization_cache_enabled: bool = True
    cache_max_entries: int = Field(default=1024, ge=1)
    cache_max_bytes: int = Field(default=64 * 1024 * 1024, ge=1024)
    cache_ttl_seconds: float = Field(default=300.0, gt=0.0)
    prefix_router_enabled: bool = True
    prefix_router_min_match_tokens: int = Field(default=8, ge=1)
    prefix_router_max_prefix_tokens: int = Field(default=2048, ge=1)


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide immutable settings instance."""

    return Settings()
