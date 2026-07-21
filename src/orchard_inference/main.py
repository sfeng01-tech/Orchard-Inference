"""Application factory and server entry point."""

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
import uvicorn
from fastapi import FastAPI

from orchard_inference.api import router
from orchard_inference.backends import (
    InferenceBackend,
    MLXBackend,
    MockBackend,
    MockFaultConfig,
    PyTorchMPSBackend,
)
from orchard_inference.cache import CacheManager
from orchard_inference.config import Settings, get_settings
from orchard_inference.lifecycle import LifecycleManager
from orchard_inference.logging import configure_logging
from orchard_inference.observability import MetricMetadata, Metrics
from orchard_inference.prefix_router import PrefixRouter
from orchard_inference.scheduler import RequestScheduler, SchedulingPolicy
from orchard_inference.ui import router as ui_router


def build_backend(settings: Settings) -> InferenceBackend:
    """Create the explicitly configured backend without fallback."""

    if settings.backend == "mock":
        return MockBackend(
            settings.model,
            settings.mock_token_delay_seconds,
            settings.model_architecture,
            settings.model_quantization,
            MockFaultConfig(
                fail_load=settings.mock_fail_load,
                fail_generation=settings.mock_fail_generation,
                fail_after_tokens=settings.mock_fail_after_tokens,
                memory_pressure=settings.mock_memory_pressure,
                unhealthy_after_requests=settings.mock_unhealthy_after_requests,
            ),
        )
    if settings.backend == "mlx":
        return MLXBackend(settings.model, settings.model_architecture, settings.model_quantization)
    if settings.backend == "mps":
        return PyTorchMPSBackend(
            settings.model,
            settings.model_architecture,
            settings.model_quantization,
            settings.mps_dtype,
        )
    raise ValueError(f"unsupported backend: {settings.backend}")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create an application whose lifespan owns backend resources."""

    resolved = settings or get_settings()
    configure_logging(resolved.log_level)
    log = structlog.get_logger(__name__)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        backend = build_backend(resolved)
        lifecycle = LifecycleManager()
        metrics = Metrics(
            MetricMetadata(backend=resolved.backend, policy=resolved.scheduling_policy)
        )

        cache = CacheManager(
            metrics,
            max_entries=resolved.cache_max_entries,
            max_bytes=resolved.cache_max_bytes,
            ttl_seconds=resolved.cache_ttl_seconds,
            prompt_enabled=resolved.prompt_cache_enabled,
            tokenization_enabled=resolved.tokenization_cache_enabled,
        )
        prefix_router = PrefixRouter(
            enabled=resolved.prefix_router_enabled,
            min_match_tokens=resolved.prefix_router_min_match_tokens,
            max_prefix_tokens=resolved.prefix_router_max_prefix_tokens,
        )
        scheduler = RequestScheduler(
            backend,
            lifecycle,
            policy=SchedulingPolicy(resolved.scheduling_policy),
            max_queued=resolved.max_queued_requests,
            max_active=resolved.max_active_requests,
            max_prompt_tokens=resolved.max_prompt_tokens,
            max_output_tokens=resolved.max_output_tokens,
            max_total_tokens=resolved.max_total_estimated_tokens,
            stream_buffer_size=resolved.stream_buffer_size,
            aging_seconds=resolved.scheduler_aging_seconds,
            max_batch_size=resolved.max_batch_size,
            max_batch_wait_seconds=resolved.max_batch_wait_seconds,
            batch_token_budget=resolved.batch_token_budget,
            observability=metrics,
        )
        app.state.backend = backend
        app.state.lifecycle = lifecycle
        app.state.scheduler = scheduler
        app.state.settings = resolved
        app.state.metrics = metrics
        app.state.cache = cache
        app.state.prefix_router = prefix_router
        await log.ainfo("backend_loading", backend=resolved.backend, model=resolved.model)
        load_started = time.perf_counter()
        await backend.load()
        metrics.model_load_duration.observe(time.perf_counter() - load_started)
        metrics.loaded_models.set(1)
        await scheduler.start()
        await log.ainfo("backend_ready", backend=resolved.backend, model=resolved.model)
        try:
            yield
        finally:
            await scheduler.shutdown()
            await lifecycle.shutdown(resolved.shutdown_grace_seconds)
            cache.clear()
            await backend.unload()
            metrics.loaded_models.set(0)
            await log.ainfo("backend_unloaded", backend=resolved.backend, model=resolved.model)

    app = FastAPI(title="Orchard Inference", version="0.1.0", lifespan=lifespan)
    app.include_router(router)
    app.include_router(ui_router)
    return app


app = create_app()


def run() -> None:
    """Run the ASGI server from the console entry point."""

    settings = get_settings()
    uvicorn.run("orchard_inference.main:app", host=settings.host, port=settings.port)
