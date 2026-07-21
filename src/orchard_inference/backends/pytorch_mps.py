"""Optional Hugging Face Transformers backend on PyTorch MPS."""

import asyncio
import queue
import threading
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


class PyTorchMPSBackend:
    """Generate on Apple MPS with strict capability checks and no CPU fallback."""

    def __init__(
        self,
        model_id: str,
        architecture: str,
        quantization: str,
        dtype: str,
    ) -> None:
        self._model_id = model_id
        self._architecture = architecture
        self._quantization = quantization
        self._dtype = dtype
        self._model: Any = None
        self._tokenizer: Any = None
        self._status = HealthStatus.UNLOADED
        self._detail: str | None = None

    @staticmethod
    def _require_mps(torch: Any) -> None:
        if not torch.backends.mps.is_built():
            raise RuntimeError("installed PyTorch was not built with MPS support")
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "PyTorch MPS is unavailable; macOS 12.3+ and an MPS-capable device are required"
            )

    async def load(self) -> None:
        """Load the tokenizer/model directly onto MPS or fail startup."""

        self._status = HealthStatus.LOADING
        try:
            self._model, self._tokenizer = await asyncio.to_thread(self._load_sync)
        except Exception as exc:
            self._status = HealthStatus.UNHEALTHY
            self._detail = f"model load failed: {type(exc).__name__}: {exc}"
            raise RuntimeError(self._detail) from exc
        self._status = HealthStatus.READY

    def _load_sync(self) -> tuple[Any, Any]:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("PyTorch MPS backend requires: uv sync --extra mps") from exc
        self._require_mps(torch)
        dtype = getattr(torch, self._dtype)
        tokenizer = AutoTokenizer.from_pretrained(self._model_id)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model_loader: Any = AutoModelForCausalLM
        model = model_loader.from_pretrained(self._model_id, dtype=dtype)
        model.to("mps")
        model.eval()
        return model, tokenizer

    async def unload(self) -> None:
        """Release model references and the PyTorch MPS allocator cache."""

        self._model = None
        self._tokenizer = None
        if self._status is not HealthStatus.UNLOADED:
            try:
                import torch

                self._require_mps(torch)
                torch.mps.empty_cache()
            except ImportError:
                pass
        self._status = HealthStatus.UNLOADED

    async def tokenize(self, prompt: str) -> tuple[int, ...]:
        """Tokenize with the loaded Hugging Face tokenizer off the event loop."""

        if self._status is not HealthStatus.READY:
            raise RuntimeError("PyTorch MPS backend is not ready")
        encoded = await asyncio.to_thread(self._tokenizer.encode, prompt)
        return tuple(encoded)

    async def render_prompt(self, messages: Sequence[tuple[str, str]]) -> str:
        """Render messages using the Hugging Face chat template when one exists."""

        if self._status is not HealthStatus.READY:
            raise RuntimeError("PyTorch MPS backend is not ready")
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

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate one complete response on MPS."""

        results = await self.generate_batch([request])
        return results[0]

    async def generate_batch(self, requests: list[GenerationRequest]) -> list[GenerationResult]:
        """Generate a padded request batch in one Transformers call on MPS."""

        if self._status is not HealthStatus.READY:
            raise RuntimeError("PyTorch MPS backend is not ready")
        return await asyncio.to_thread(self._generate_batch_sync, requests)

    def _generation_kwargs(
        self, request: GenerationRequest, max_new_tokens: int | None = None
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens or request.max_tokens,
            "do_sample": request.temperature > 0,
            "pad_token_id": self._tokenizer.pad_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
        }
        if request.temperature > 0:
            kwargs["temperature"] = request.temperature
            kwargs["top_p"] = request.top_p
        return kwargs

    def _generate_batch_sync(self, requests: list[GenerationRequest]) -> list[GenerationResult]:
        if not requests:
            return []
        import torch

        prompts: list[str | list[int]] = [
            list(request.prompt_token_ids)
            if request.prompt_token_ids is not None
            else request.prompt
            for request in requests
        ]
        if all(isinstance(prompt, str) for prompt in prompts):
            inputs = self._tokenizer(prompts, padding=True, return_tensors="pt")
        else:
            token_lists = [
                prompt if isinstance(prompt, list) else self._tokenizer.encode(prompt)
                for prompt in prompts
            ]
            inputs = self._tokenizer.pad(
                {"input_ids": token_lists}, padding=True, return_tensors="pt"
            )
        inputs = inputs.to("mps")
        first = requests[0]
        with torch.inference_mode():
            outputs = self._model.generate(
                **inputs,
                **self._generation_kwargs(first, max(request.max_tokens for request in requests)),
            )
        input_width = inputs["input_ids"].shape[1]
        results = []
        for request, output in zip(requests, outputs, strict=True):
            generated_ids = output[input_width:].tolist()[: request.max_tokens]
            text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
            stopped = False
            for stop in request.stop:
                if stop and stop in text:
                    text = text.split(stop, 1)[0]
                    stopped = True
            generated_tokens = len(self._tokenizer.encode(text, add_special_tokens=False))
            results.append(
                GenerationResult(
                    text=text,
                    prompt_tokens=request.prompt_token_count,
                    generated_tokens=generated_tokens,
                    finish_reason=(
                        "stop" if stopped or generated_tokens < request.max_tokens else "length"
                    ),
                )
            )
        return results

    async def stream(self, request: GenerationRequest) -> AsyncIterator[TokenEvent]:
        """Stream decoded fragments and stop generation cooperatively on cancellation."""

        if self._status is not HealthStatus.READY:
            raise RuntimeError("PyTorch MPS backend is not ready")
        from transformers import TextIteratorStreamer

        streamer = TextIteratorStreamer(
            self._tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=None,
        )
        stop_event = threading.Event()
        generation = asyncio.create_task(
            asyncio.to_thread(self._stream_sync, request, streamer, stop_event)
        )
        try:
            while True:
                has_value, text = await asyncio.to_thread(self._stream_next, streamer)
                if not has_value:
                    break
                if text:
                    token_count = max(
                        1, len(self._tokenizer.encode(text, add_special_tokens=False))
                    )
                    yield TokenEvent(text=text, token_count=token_count)
            await generation
        finally:
            stop_event.set()
            await asyncio.gather(generation, return_exceptions=True)

    def _stream_sync(
        self, request: GenerationRequest, streamer: Any, stopped: threading.Event
    ) -> None:
        import torch

        class CancellationCriteria:
            def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
                return stopped.is_set()

        prompt: str | list[int] = (
            list(request.prompt_token_ids)
            if request.prompt_token_ids is not None
            else request.prompt
        )
        if isinstance(prompt, str):
            inputs = self._tokenizer(prompt, return_tensors="pt").to("mps")
        else:
            inputs = self._tokenizer.pad(
                {"input_ids": [prompt]}, padding=True, return_tensors="pt"
            ).to("mps")
        try:
            with torch.inference_mode():
                self._model.generate(
                    **inputs,
                    **self._generation_kwargs(request),
                    streamer=streamer,
                    stopping_criteria=[CancellationCriteria()],
                )
        finally:
            streamer.end()

    @staticmethod
    def _stream_next(streamer: Any) -> tuple[bool, str]:
        try:
            return True, next(streamer)
        except (StopIteration, queue.Empty):
            return False, ""

    def health(self) -> BackendHealth:
        """Return current MPS backend health."""

        return BackendHealth(self._status, self._detail)

    def model_info(self) -> ModelInfo:
        """Return explicit MPS model metadata."""

        return ModelInfo(
            self._model_id,
            "pytorch_mps",
            quantization=self._quantization,
            architecture=self._architecture,
        )
