"""MLX-LM inference backend."""

import asyncio
from collections.abc import AsyncIterator, Sequence
from typing import Any

from orchard_inference.models import (
    BackendHealth,
    GenerationRequest,
    GenerationResult,
    HealthStatus,
    ModelInfo,
    TokenEvent,
)


class MLXBackend:
    """Minimal MLX-LM backend with blocking runtime work isolated from the event loop."""

    def __init__(self, model_id: str, architecture: str, quantization: str) -> None:
        self._model_id = model_id
        self._status = HealthStatus.UNLOADED
        self._detail: str | None = None
        self._model: Any = None
        self._tokenizer: Any = None
        self._architecture = architecture
        self._quantization = quantization

    async def load(self) -> None:
        """Load the configured MLX model or fail without falling back to CPU."""

        self._status = HealthStatus.LOADING
        try:
            self._model, self._tokenizer = await asyncio.to_thread(self._load_sync)
        except Exception as exc:
            self._status = HealthStatus.UNHEALTHY
            self._detail = f"model load failed: {type(exc).__name__}: {exc}"
            raise RuntimeError(self._detail) from exc
        self._status = HealthStatus.READY

    def _load_sync(self) -> tuple[Any, Any]:
        import mlx.core as mx
        from mlx_lm import load

        if str(mx.default_device()) != "Device(gpu, 0)":
            raise RuntimeError(f"MLX GPU unavailable; selected device is {mx.default_device()}")
        loaded = load(self._model_id)
        return loaded[0], loaded[1]

    async def unload(self) -> None:
        """Drop model references and clear the MLX cache."""

        self._model = None
        self._tokenizer = None
        if self._status is not HealthStatus.UNLOADED:
            import mlx.core as mx

            mx.clear_cache()
        self._status = HealthStatus.UNLOADED

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate a complete response using MLX-LM."""

        if self._status is not HealthStatus.READY:
            raise RuntimeError("MLX backend is not ready")
        return await asyncio.to_thread(self._generate_sync, request)

    async def stream(self, request: GenerationRequest) -> AsyncIterator[TokenEvent]:
        """Stream MLX-LM generation while keeping blocking iteration off the event loop."""

        if self._status is not HealthStatus.READY:
            raise RuntimeError("MLX backend is not ready")
        iterator = self._stream_sync(request)
        while True:
            event = await asyncio.to_thread(self._next_event, iterator)
            if event is None:
                break
            yield event

    async def generate_batch(self, requests: list[GenerationRequest]) -> list[GenerationResult]:
        """Run compatible requests through MLX-LM's native batch generator."""

        if self._status is not HealthStatus.READY:
            raise RuntimeError("MLX backend is not ready")
        return await asyncio.to_thread(self._generate_batch_sync, requests)

    async def tokenize(self, prompt: str) -> tuple[int, ...]:
        """Tokenize with the loaded MLX-LM tokenizer off the event loop."""

        if self._status is not HealthStatus.READY:
            raise RuntimeError("MLX backend is not ready")
        encoded = await asyncio.to_thread(self._tokenizer.encode, prompt)
        return tuple(encoded)

    async def render_prompt(self, messages: Sequence[tuple[str, str]]) -> str:
        """Render messages using the tokenizer chat template when one exists."""

        if self._status is not HealthStatus.READY:
            raise RuntimeError("MLX backend is not ready")
        return await asyncio.to_thread(self._render_prompt_sync, messages)

    def _render_prompt_sync(self, messages: Sequence[tuple[str, str]]) -> str:
        structured = [{"role": role, "content": content} for role, content in messages]
        if hasattr(self._tokenizer, "apply_chat_template"):
            rendered = self._tokenizer.apply_chat_template(
                structured,
                tokenize=False,
                add_generation_prompt=True,
            )
            if isinstance(rendered, str):
                return rendered
        return "".join(f"{role}: {content}\n" for role, content in messages) + "assistant:"

    def _generate_batch_sync(self, requests: list[GenerationRequest]) -> list[GenerationResult]:
        from mlx_lm import batch_generate
        from mlx_lm.sample_utils import make_sampler

        if not requests:
            return []
        first = requests[0]
        sampler = make_sampler(temp=first.temperature, top_p=first.top_p)
        prompts = [
            list(request.prompt_token_ids)
            if request.prompt_token_ids is not None
            else self._tokenizer.encode(request.prompt)
            for request in requests
        ]
        response = batch_generate(
            self._model,
            self._tokenizer,
            prompts,
            max_tokens=[request.max_tokens for request in requests],
            sampler=sampler,
            verbose=False,
        )
        results = []
        for request, prompt_tokens, text in zip(requests, prompts, response.texts, strict=True):
            stopped = False
            for stop in request.stop:
                if stop and stop in text:
                    text = text.split(stop, 1)[0]
                    stopped = True
            generated_tokens = len(self._tokenizer.encode(text))
            results.append(
                GenerationResult(
                    text=text,
                    prompt_tokens=len(prompt_tokens),
                    generated_tokens=generated_tokens,
                    finish_reason=(
                        "stop" if stopped or generated_tokens < request.max_tokens else "length"
                    ),
                )
            )
        return results

    def _stream_sync(self, request: GenerationRequest) -> Any:
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        sampler = make_sampler(temp=request.temperature, top_p=request.top_p)
        return iter(
            stream_generate(
                self._model,
                self._tokenizer,
                prompt=(
                    list(request.prompt_token_ids)
                    if request.prompt_token_ids is not None
                    else request.prompt
                ),
                max_tokens=request.max_tokens,
                sampler=sampler,
            )
        )

    @staticmethod
    def _next_event(iterator: Any) -> TokenEvent | None:
        try:
            response = next(iterator)
        except StopIteration:
            return None
        return TokenEvent(text=response.text)

    def _generate_sync(self, request: GenerationRequest) -> GenerationResult:
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        sampler = make_sampler(temp=request.temperature, top_p=request.top_p)
        text = generate(
            self._model,
            self._tokenizer,
            prompt=(
                list(request.prompt_token_ids)
                if request.prompt_token_ids is not None
                else request.prompt
            ),
            max_tokens=request.max_tokens,
            sampler=sampler,
            verbose=False,
        )
        prompt_tokens = (
            len(request.prompt_token_ids)
            if request.prompt_token_ids is not None
            else len(self._tokenizer.encode(request.prompt))
        )
        generated_tokens = len(self._tokenizer.encode(text))
        return GenerationResult(text, prompt_tokens, generated_tokens, "stop")

    def health(self) -> BackendHealth:
        """Return current MLX health."""

        return BackendHealth(status=self._status, detail=self._detail)

    def model_info(self) -> ModelInfo:
        """Return configured MLX model metadata."""

        return ModelInfo(
            self._model_id,
            "mlx",
            quantization=self._quantization,
            architecture=self._architecture,
        )
