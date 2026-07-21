"""OpenAI-compatible chat completion API subset."""

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Annotated, Literal, cast

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST
from pydantic import BaseModel, Field
from starlette.responses import Response

from orchard_inference.backends.base import InferenceBackend
from orchard_inference.cache import CacheManager
from orchard_inference.config import Settings
from orchard_inference.errors import ErrorCategory, error_category
from orchard_inference.lifecycle import LifecycleManager, RequestContext, RequestState
from orchard_inference.models import GenerationRequest, HealthStatus
from orchard_inference.observability import Metrics
from orchard_inference.prefix_router import PrefixRoute, PrefixRouter
from orchard_inference.scheduler import (
    AdmissionRejected,
    RejectionReason,
    RequestCancelled,
    RequestScheduler,
    RequestTimedOut,
    WorkItem,
)

router = APIRouter()
logger = structlog.get_logger(__name__)


class ChatMessage(BaseModel):
    """One chat message in the supported OpenAI-compatible subset."""

    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class ChatCompletionRequest(BaseModel):
    """Validated chat completion input."""

    model: str
    messages: list[ChatMessage] = Field(min_length=1)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    max_tokens: int = Field(default=128, ge=1, le=4096)
    stream: bool = False
    stop: str | list[str] | None = None
    client_request_id: str | None = Field(default=None, max_length=128)
    priority: int = Field(default=0, ge=-100, le=100)


def get_backend(request: Request) -> InferenceBackend:
    """Resolve the application-owned backend."""

    return cast(InferenceBackend, request.app.state.backend)


def get_lifecycle(request: Request) -> LifecycleManager:
    """Resolve the application-owned lifecycle manager."""

    return cast(LifecycleManager, request.app.state.lifecycle)


def get_settings_from_app(request: Request) -> Settings:
    """Resolve immutable application settings."""

    return cast(Settings, request.app.state.settings)


def get_scheduler(request: Request) -> RequestScheduler:
    """Resolve the application-owned scheduler."""

    return cast(RequestScheduler, request.app.state.scheduler)


def get_metrics(request: Request) -> Metrics:
    """Resolve the application-owned metrics registry."""

    return cast(Metrics, request.app.state.metrics)


def get_cache(request: Request) -> CacheManager:
    """Resolve the application-owned cache manager."""

    return cast(CacheManager, request.app.state.cache)


def get_prefix_router(request: Request) -> PrefixRouter:
    """Resolve the application-owned prefix router."""

    return cast(PrefixRouter, request.app.state.prefix_router)


BackendDependency = Annotated[InferenceBackend, Depends(get_backend)]
LifecycleDependency = Annotated[LifecycleManager, Depends(get_lifecycle)]
SettingsDependency = Annotated[Settings, Depends(get_settings_from_app)]
SchedulerDependency = Annotated[RequestScheduler, Depends(get_scheduler)]
MetricsDependency = Annotated[Metrics, Depends(get_metrics)]
CacheDependency = Annotated[CacheManager, Depends(get_cache)]
PrefixRouterDependency = Annotated[PrefixRouter, Depends(get_prefix_router)]


def _generation_request(
    body: ChatCompletionRequest, prompt: str, prompt_token_ids: tuple[int, ...]
) -> GenerationRequest:
    stops = (body.stop,) if isinstance(body.stop, str) else tuple(body.stop or ())
    return GenerationRequest(
        prompt=prompt,
        temperature=body.temperature,
        top_p=body.top_p,
        max_tokens=body.max_tokens,
        stop=stops,
        prompt_token_ids=prompt_token_ids,
    )


def _new_context(body: ChatCompletionRequest, settings: Settings) -> RequestContext:
    now = time.monotonic()
    context = RequestContext(
        request_id=f"chatcmpl-{uuid.uuid4().hex}",
        client_request_id=body.client_request_id,
        deadline=now + settings.request_timeout_seconds,
    )
    context.transition(RequestState.VALIDATED)
    return context


def _sse(payload: object) -> bytes:
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n".encode()


def _status_for_category(category: ErrorCategory) -> int:
    """Map bounded internal error categories to HTTP status codes."""

    if category is ErrorCategory.MEMORY_PRESSURE:
        return 503
    if category is ErrorCategory.BACKEND_UNHEALTHY:
        return 503
    if category is ErrorCategory.BACKEND_LOAD_FAILED:
        return 503
    return 500


@router.get("/health/live")
async def live() -> dict[str, str]:
    """Report process liveness independently of model state."""

    return {"status": "live"}


@router.get("/health/ready")
async def ready(backend: BackendDependency, lifecycle: LifecycleDependency) -> dict[str, str]:
    """Report readiness only while the loaded backend accepts requests."""

    health = backend.health()
    if health.status is not HealthStatus.READY or not lifecycle.accepting:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"status": health.status, "detail": health.detail},
        )
    return {"status": "ready"}


@router.get("/v1/models")
async def models(backend: BackendDependency) -> dict[str, object]:
    """Return the single configured model and backend metadata."""

    info = backend.model_info()
    return {
        "object": "list",
        "data": [
            {
                "id": info.model_id,
                "object": "model",
                "owned_by": "orchard-inference",
                "backend": info.backend,
                "quantization": info.quantization,
                "architecture": info.architecture,
            }
        ],
    }


@router.get("/metrics", include_in_schema=False)
async def prometheus_metrics(metrics: MetricsDependency) -> Response:
    """Expose the application-owned Prometheus registry."""

    return Response(content=metrics.render(), media_type=CONTENT_TYPE_LATEST)


async def _complete(
    body: ChatCompletionRequest,
    backend: InferenceBackend,
    scheduler: RequestScheduler,
    item: WorkItem,
    http_response: Response,
    prefix_route: PrefixRoute,
) -> dict[str, object]:
    try:
        result = await item.result
    except RequestTimedOut as exc:
        raise HTTPException(status_code=504, detail="generation timed out") from exc
    except asyncio.CancelledError:
        await scheduler.cancel(item.context.request_id)
        raise
    except RequestCancelled as exc:
        raise HTTPException(status_code=499, detail="generation cancelled") from exc
    except Exception as exc:
        category = error_category(exc)
        await logger.aexception(
            "generation_failed",
            request_id=item.context.request_id,
            error_type=type(exc).__name__,
            error_category=category.value,
        )
        raise HTTPException(
            status_code=_status_for_category(category),
            detail={"error": category.value, "message": "generation failed"},
        ) from exc
    await _log_completion(
        item.context,
        body.model,
        backend.model_info().backend,
        result.generated_tokens,
        item.batch_id,
        item.batch_size,
    )
    queue_seconds = item.context.duration(RequestState.QUEUED, RequestState.SCHEDULED)
    if queue_seconds is not None:
        http_response.headers["X-Orchard-Queue-Seconds"] = str(queue_seconds)
    http_response.headers["X-Orchard-Batch-Size"] = str(item.batch_size)
    http_response.headers["X-Orchard-Prefix-Route"] = prefix_route.route
    http_response.headers["X-Orchard-Prefix-Matched-Tokens"] = str(prefix_route.matched_tokens)
    http_response.headers["X-Orchard-Prefix-Estimated-Saved-Tokens"] = str(
        prefix_route.estimated_prefill_tokens_saved
    )
    if item.batch_id is not None:
        http_response.headers["X-Orchard-Batch-ID"] = item.batch_id
    return {
        "id": item.context.request_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result.text},
                "finish_reason": result.finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.generated_tokens,
            "total_tokens": result.prompt_tokens + result.generated_tokens,
        },
        "orchard": {
            "prefix_route": prefix_route.route,
            "prefix_matched_tokens": prefix_route.matched_tokens,
            "prefix_matched_ratio": prefix_route.matched_ratio,
            "prefix_estimated_saved_tokens": prefix_route.estimated_prefill_tokens_saved,
            "prefix_candidate_count": prefix_route.candidate_count,
        },
    }


async def _log_completion(
    context: RequestContext,
    model: str,
    backend: str,
    generated_tokens: int,
    batch_id: str | None,
    batch_size: int,
) -> None:
    await logger.ainfo(
        "request_finished",
        request_id=context.request_id,
        client_request_id=context.client_request_id,
        model=model,
        backend=backend,
        generated_tokens=generated_tokens,
        batch_id=batch_id,
        batch_size=batch_size,
        final_state=context.state,
        time_to_first_token_seconds=context.time_to_first_token,
        mean_inter_token_seconds=(
            sum(context.inter_token_seconds) / len(context.inter_token_seconds)
            if context.inter_token_seconds
            else None
        ),
        end_to_end_seconds=context.duration(RequestState.RECEIVED, context.state),
    )


def _stream_response(
    http_request: Request,
    body: ChatCompletionRequest,
    backend: InferenceBackend,
    scheduler: RequestScheduler,
    item: WorkItem,
    prefix_route: PrefixRoute,
) -> StreamingResponse:
    async def events() -> AsyncIterator[bytes]:
        generated_tokens = 0
        try:
            yield _sse(
                {
                    "id": item.context.request_id,
                    "object": "chat.completion.chunk",
                    "model": body.model,
                    "choices": [{"index": 0, "delta": {"role": "assistant"}}],
                }
            )
            while True:
                if await http_request.is_disconnected():
                    await scheduler.cancel(item.context.request_id)
                    break
                event = await item.output.get()
                if event is None:
                    break
                if isinstance(event, BaseException):
                    category = error_category(event)
                    yield _sse(
                        {
                            "error": {
                                "type": category.value,
                                "message": "stream terminated before completion",
                            }
                        }
                    )
                    break
                generated_tokens += event.token_count
                yield _sse(
                    {
                        "id": item.context.request_id,
                        "object": "chat.completion.chunk",
                        "model": body.model,
                        "choices": [{"index": 0, "delta": {"content": event.text}}],
                    }
                )
            if item.context.state is RequestState.COMPLETED:
                yield _sse(
                    {
                        "id": item.context.request_id,
                        "object": "chat.completion.chunk",
                        "model": body.model,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                        "orchard": {
                            "queue_seconds": item.context.duration(
                                RequestState.QUEUED, RequestState.SCHEDULED
                            ),
                            "batch_id": item.batch_id,
                            "batch_size": item.batch_size,
                            "generated_tokens": generated_tokens,
                            "prefix_route": prefix_route.route,
                            "prefix_matched_tokens": prefix_route.matched_tokens,
                            "prefix_matched_ratio": prefix_route.matched_ratio,
                            "prefix_estimated_saved_tokens": (
                                prefix_route.estimated_prefill_tokens_saved
                            ),
                            "prefix_candidate_count": prefix_route.candidate_count,
                        },
                    }
                )
            yield b"data: [DONE]\n\n"
        finally:
            if not item.result.done():
                await scheduler.cancel(item.context.request_id)
            elif not item.result.cancelled():
                item.result.exception()
            await _log_completion(
                item.context,
                body.model,
                backend.model_info().backend,
                generated_tokens,
                item.batch_id,
                item.batch_size,
            )

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "X-Request-ID": item.context.request_id,
            "Cache-Control": "no-cache",
            "X-Orchard-Prefix-Route": prefix_route.route,
            "X-Orchard-Prefix-Matched-Tokens": str(prefix_route.matched_tokens),
            "X-Orchard-Prefix-Estimated-Saved-Tokens": str(
                prefix_route.estimated_prefill_tokens_saved
            ),
        },
    )


@router.post("/v1/chat/completions", response_model=None)
async def chat_completion(
    http_request: Request,
    http_response: Response,
    body: ChatCompletionRequest,
    backend: BackendDependency,
    lifecycle: LifecycleDependency,
    settings: SettingsDependency,
    scheduler: SchedulerDependency,
    metrics: MetricsDependency,
    cache: CacheDependency,
    prefix_router: PrefixRouterDependency,
) -> dict[str, object] | StreamingResponse:
    """Generate a streaming or non-streaming chat completion."""

    metrics.requests_received.inc()
    info = backend.model_info()
    if body.model != info.model_id:
        raise HTTPException(status_code=404, detail=f"model {body.model!r} is not loaded")
    if not lifecycle.accepting:
        raise HTTPException(status_code=503, detail="server is shutting down")
    context = _new_context(body, settings)
    messages = [(message.role, message.content) for message in body.messages]
    prompt = await backend.render_prompt(messages)
    prompt_token_ids = await cache.tokenize(body.model, prompt, backend.tokenize)
    prefix_route = prefix_router.observe(prompt_token_ids)
    metrics.record_prefix_route(
        prefix_route.route,
        prefix_route.matched_tokens,
        prefix_route.estimated_prefill_tokens_saved,
        prefix_route.matched_ratio,
    )
    await logger.ainfo(
        "prefix_route_selected",
        request_id=context.request_id,
        route=prefix_route.route,
        matched_tokens=prefix_route.matched_tokens,
        matched_ratio=prefix_route.matched_ratio,
        estimated_saved_tokens=prefix_route.estimated_prefill_tokens_saved,
        candidate_count=prefix_route.candidate_count,
    )
    try:
        item = await scheduler.submit(
            context,
            _generation_request(body, prompt, prompt_token_ids),
            priority=body.priority,
            stream=body.stream,
        )
    except AdmissionRejected as exc:
        if exc.reason in {RejectionReason.PROMPT_TOO_LONG, RejectionReason.TOTAL_TOKENS}:
            code = 413
        elif exc.reason is RejectionReason.OUTPUT_TOO_LONG:
            code = 422
        elif exc.reason is RejectionReason.QUEUE_FULL:
            code = 429
        else:
            code = 503
        raise HTTPException(
            status_code=code,
            detail={"error": "request_rejected", "reason": exc.reason.value},
        ) from exc
    if body.stream:
        return _stream_response(http_request, body, backend, scheduler, item, prefix_route)
    return await _complete(body, backend, scheduler, item, http_response, prefix_route)
