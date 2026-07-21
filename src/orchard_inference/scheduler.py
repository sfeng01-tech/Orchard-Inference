"""Bounded request scheduler and admission control."""

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from functools import partial

from orchard_inference.backends.base import InferenceBackend
from orchard_inference.lifecycle import LifecycleManager, RequestContext, RequestState
from orchard_inference.models import GenerationRequest, GenerationResult, TokenEvent
from orchard_inference.observability import MetricMetadata, Metrics


class SchedulingPolicy(StrEnum):
    """Supported request ordering policies."""

    FIFO = "fifo"
    PRIORITY = "priority"
    SHORTEST_JOB = "shortest_job"


class RejectionReason(StrEnum):
    """Bounded admission rejection categories."""

    QUEUE_FULL = "queue_full"
    PROMPT_TOO_LONG = "prompt_too_long"
    OUTPUT_TOO_LONG = "output_too_long"
    TOTAL_TOKENS = "total_estimated_tokens"
    SHUTTING_DOWN = "shutting_down"


class AdmissionRejected(RuntimeError):
    """Raised when admission control rejects a request."""

    def __init__(self, reason: RejectionReason) -> None:
        super().__init__(reason.value)
        self.reason = reason


class RequestTimedOut(TimeoutError):
    """Raised when a request deadline expires in queue or generation."""


class RequestCancelled(RuntimeError):
    """Raised when an admitted request is cancelled."""


@dataclass(slots=True)
class SchedulerMetrics:
    """Process-local scheduler instrumentation pending Prometheus export."""

    rejected: dict[RejectionReason, int] = field(default_factory=dict)
    scheduled: int = 0
    queue_depth: int = 0
    active_requests: int = 0
    batch_sizes: list[int] = field(default_factory=list)
    batch_formation_seconds: list[float] = field(default_factory=list)
    batch_tokens: list[int] = field(default_factory=list)
    batch_processing_seconds: list[float] = field(default_factory=list)


@dataclass(slots=True)
class WorkItem:
    """One admitted request and its completion channels."""

    context: RequestContext
    generation: GenerationRequest
    priority: int
    sequence: int
    enqueued_at: float
    stream: bool
    output: asyncio.Queue[TokenEvent | BaseException | None]
    result: asyncio.Future[GenerationResult]
    cancellation_state: RequestState = RequestState.CANCELLED
    batch_id: str | None = None
    batch_size: int = 1
    generated_tokens: int = 0

    @property
    def estimated_tokens(self) -> int:
        """Return the conservative scheduling/admission token estimate."""

        return self.generation.prompt_token_count + self.generation.max_tokens


def select_item(
    items: list[WorkItem],
    policy: SchedulingPolicy,
    *,
    now: float,
    aging_seconds: float,
) -> WorkItem:
    """Return the deterministically highest-ranked item without mutating the input."""

    if not items:
        raise IndexError("scheduler queue is empty")

    def score(item: WorkItem) -> tuple[float, int]:
        age_steps = int(max(0.0, now - item.enqueued_at) / aging_seconds)
        if policy is SchedulingPolicy.FIFO:
            return float(item.sequence), item.sequence
        if policy is SchedulingPolicy.PRIORITY:
            return float(-(item.priority + age_steps)), item.sequence
        aged_cost = max(0, item.estimated_tokens - age_steps)
        return float(aged_cost), item.sequence

    return min(items, key=score)


class RequestScheduler:
    """Schedule admitted requests under bounded single-process resources."""

    def __init__(
        self,
        backend: InferenceBackend,
        lifecycle: LifecycleManager,
        *,
        policy: SchedulingPolicy,
        max_queued: int,
        max_active: int,
        max_prompt_tokens: int,
        max_output_tokens: int,
        max_total_tokens: int,
        stream_buffer_size: int,
        aging_seconds: float,
        max_batch_size: int,
        max_batch_wait_seconds: float,
        batch_token_budget: int,
        observability: Metrics | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._backend = backend
        self._lifecycle = lifecycle
        self._policy = policy
        self._max_queued = max_queued
        self._max_active = max_active
        self._max_prompt_tokens = max_prompt_tokens
        self._max_output_tokens = max_output_tokens
        self._max_total_tokens = max_total_tokens
        self._stream_buffer_size = stream_buffer_size
        self._aging_seconds = aging_seconds
        self._max_batch_size = max_batch_size
        self._max_batch_wait_seconds = max_batch_wait_seconds
        self._batch_token_budget = batch_token_budget
        self._observability = observability or Metrics(
            MetricMetadata(backend=backend.model_info().backend, policy=policy.value)
        )
        self._clock = clock
        self._queue: list[WorkItem] = []
        self._items: dict[str, WorkItem] = {}
        self._active_tasks: dict[str, asyncio.Task[None]] = {}
        self._deadline_tasks: dict[str, asyncio.Task[None]] = {}
        self._sequence = 0
        self._condition = asyncio.Condition()
        self._dispatcher: asyncio.Task[None] | None = None
        self._accepting = False
        self.metrics = SchedulerMetrics()

    async def start(self) -> None:
        """Start the background dispatcher."""

        self._accepting = True
        self._dispatcher = asyncio.create_task(self._dispatch())

    def _reject(self, context: RequestContext, reason: RejectionReason) -> None:
        context.transition(RequestState.REJECTED)
        self.metrics.rejected[reason] = self.metrics.rejected.get(reason, 0) + 1
        self._observability.record_rejection(reason.value)
        raise AdmissionRejected(reason)

    async def submit(
        self,
        context: RequestContext,
        generation: GenerationRequest,
        *,
        priority: int,
        stream: bool,
    ) -> WorkItem:
        """Validate limits and enqueue a request without unbounded waiting."""

        prompt_tokens = generation.prompt_token_count
        if not self._accepting:
            self._reject(context, RejectionReason.SHUTTING_DOWN)
        if prompt_tokens > self._max_prompt_tokens:
            self._reject(context, RejectionReason.PROMPT_TOO_LONG)
        if generation.max_tokens > self._max_output_tokens:
            self._reject(context, RejectionReason.OUTPUT_TOO_LONG)
        if prompt_tokens + generation.max_tokens > self._max_total_tokens:
            self._reject(context, RejectionReason.TOTAL_TOKENS)
        async with self._condition:
            if len(self._queue) >= self._max_queued:
                self._reject(context, RejectionReason.QUEUE_FULL)
            context.transition(RequestState.QUEUED)
            item = WorkItem(
                context=context,
                generation=generation,
                priority=priority,
                sequence=self._sequence,
                enqueued_at=self._clock(),
                stream=stream,
                output=asyncio.Queue(self._stream_buffer_size),
                result=asyncio.get_running_loop().create_future(),
            )
            self._sequence += 1
            self._queue.append(item)
            self._items[context.request_id] = item
            self.metrics.queue_depth = len(self._queue)
            self._observability.queue_depth.set(len(self._queue))
            self._deadline_tasks[context.request_id] = asyncio.create_task(self._expire(item))
            self._condition.notify_all()
            return item

    def select_next(self, now: float | None = None) -> WorkItem:
        """Select and remove the next work item deterministically."""

        if not self._queue:
            raise IndexError("scheduler queue is empty")
        current = self._clock() if now is None else now

        selected = select_item(
            self._queue,
            self._policy,
            now=current,
            aging_seconds=self._aging_seconds,
        )
        self._queue.remove(selected)
        self.metrics.queue_depth = len(self._queue)
        self._observability.queue_depth.set(len(self._queue))
        return selected

    @staticmethod
    def compatible(left: WorkItem, right: WorkItem) -> bool:
        """Return whether two requests can share one backend-native batch."""

        if left.stream or right.stream:
            return False
        return (
            left.generation.temperature == right.generation.temperature
            and left.generation.top_p == right.generation.top_p
            and left.generation.stop == right.generation.stop
        )

    def form_batch(self, capacity: int, now: float | None = None) -> list[WorkItem]:
        """Remove one policy-ordered, compatible, token-bounded batch from the queue."""

        current = self._clock() if now is None else now
        seed = self.select_next(current)
        batch = [seed]
        if seed.stream:
            return batch
        budget = seed.estimated_tokens
        candidates = list(self._queue)
        limit = min(capacity, self._max_batch_size)
        while candidates and len(batch) < limit:
            candidate = select_item(
                candidates,
                self._policy,
                now=current,
                aging_seconds=self._aging_seconds,
            )
            candidates.remove(candidate)
            if not self.compatible(seed, candidate):
                continue
            if budget + candidate.estimated_tokens > self._batch_token_budget:
                continue
            self._queue.remove(candidate)
            batch.append(candidate)
            budget += candidate.estimated_tokens
        self.metrics.queue_depth = len(self._queue)
        self._observability.queue_depth.set(len(self._queue))
        return batch

    async def _dispatch(self) -> None:
        try:
            while True:
                async with self._condition:
                    await self._condition.wait_for(
                        lambda: bool(self._queue) and len(self._active_tasks) < self._max_active
                    )
                if self._max_batch_wait_seconds:
                    await asyncio.sleep(self._max_batch_wait_seconds)
                async with self._condition:
                    if not self._queue or len(self._active_tasks) >= self._max_active:
                        continue
                    capacity = self._max_active - len(self._active_tasks)
                    batch = self.form_batch(capacity)
                    batch_id = f"batch-{batch[0].sequence}"
                    for item in batch:
                        item.batch_id = batch_id
                        item.batch_size = len(batch)
                    task = asyncio.create_task(self._run_batch(batch))
                    for item in batch:
                        self._active_tasks[item.context.request_id] = task
                    self.metrics.active_requests = len(self._active_tasks)
                    self.metrics.scheduled += len(batch)
                    self.metrics.batch_sizes.append(len(batch))
                    self.metrics.batch_tokens.append(sum(item.estimated_tokens for item in batch))
                    self.metrics.batch_formation_seconds.append(
                        self._clock() - min(item.enqueued_at for item in batch)
                    )
                    self._observability.active_requests.set(len(self._active_tasks))
                    self._observability.queue_depth.set(len(self._queue))
                    self._observability.record_batch(
                        len(batch),
                        self.metrics.batch_formation_seconds[-1],
                        self.metrics.batch_tokens[-1],
                    )
                    self._condition.notify_all()
                    for item in batch:
                        self._lifecycle.register(item.context, task)
                    request_ids = tuple(item.context.request_id for item in batch)
                    task.add_done_callback(partial(self._task_done, request_ids))
        except asyncio.CancelledError:
            raise

    def _task_done(self, request_ids: tuple[str, ...], _task: asyncio.Task[None]) -> None:
        asyncio.create_task(self._finished(request_ids))

    async def wait_for_state(
        self,
        *,
        active_requests: int,
        queue_depth: int,
        timeout_seconds: float = 1.0,
    ) -> None:
        """Wait for an exact scheduler state, primarily for deterministic diagnostics."""

        async with asyncio.timeout(timeout_seconds):
            async with self._condition:
                await self._condition.wait_for(
                    lambda: (
                        self.metrics.active_requests == active_requests
                        and self.metrics.queue_depth == queue_depth
                    )
                )

    async def _finished(self, request_ids: tuple[str, ...]) -> None:
        async with self._condition:
            for request_id in request_ids:
                self._active_tasks.pop(request_id, None)
                self._items.pop(request_id, None)
                deadline = self._deadline_tasks.pop(request_id, None)
                if deadline is not None and deadline is not asyncio.current_task():
                    deadline.cancel()
            self.metrics.active_requests = len(self._active_tasks)
            self._observability.active_requests.set(len(self._active_tasks))
            self._condition.notify_all()

    async def _expire(self, item: WorkItem) -> None:
        remaining = item.context.remaining_seconds
        if remaining is None:
            return
        await asyncio.sleep(remaining)
        await self.cancel(item.context.request_id, RequestState.TIMED_OUT)

    async def cancel(self, request_id: str, state: RequestState = RequestState.CANCELLED) -> None:
        """Cancel queued or active work and release its completion channels."""

        item = self._items.get(request_id)
        if item is None or item.result.done():
            return
        item.cancellation_state = state
        task = self._active_tasks.get(request_id)
        if task is not None:
            if item.batch_size == 1:
                task.cancel()
            else:
                await self._terminal_error(item, state)
            return
        async with self._condition:
            if item in self._queue:
                self._queue.remove(item)
                self.metrics.queue_depth = len(self._queue)
                self._observability.queue_depth.set(len(self._queue))
            item.context.transition(state)
            error: BaseException = (
                RequestTimedOut("request timed out while queued")
                if state is RequestState.TIMED_OUT
                else RequestCancelled("request cancelled while queued")
            )
            if state is RequestState.TIMED_OUT:
                self._lifecycle.metrics.timed_out += 1
            else:
                self._lifecycle.metrics.cancelled += 1
            item.result.set_exception(error)
            self._force_output(item, error)
            self._items.pop(request_id, None)
            self._observability.record_finished(
                item.context,
                prompt_tokens=item.generation.prompt_token_count,
                output_tokens=0,
            )
            deadline = self._deadline_tasks.pop(request_id, None)
            if deadline is not None and deadline is not asyncio.current_task():
                deadline.cancel()
            self._condition.notify_all()

    async def _run_batch(self, batch: list[WorkItem]) -> None:
        started = self._clock()
        try:
            if len(batch) == 1:
                item = batch[0]
                if item.stream:
                    await self._run(item)
                else:
                    await self._run_complete(item)
                return
            for item in batch:
                item.context.transition(RequestState.SCHEDULED)
                item.context.transition(RequestState.PREFILL)
            remaining = [item.context.remaining_seconds for item in batch]
            finite = [value for value in remaining if value is not None]
            timeout = min(finite) if finite else None
            async with asyncio.timeout(timeout):
                results = await self._backend.generate_batch([item.generation for item in batch])
            if len(results) != len(batch):
                raise RuntimeError("backend returned the wrong number of batch results")
            for item, result in zip(batch, results, strict=True):
                if item.result.done():
                    continue
                item.context.transition(RequestState.DECODING)
                item.context.transition(RequestState.COMPLETED)
                self._lifecycle.metrics.completed += 1
                item.result.set_result(result)
                self._observability.record_finished(
                    item.context,
                    prompt_tokens=result.prompt_tokens,
                    output_tokens=result.generated_tokens,
                )
        except TimeoutError:
            for item in batch:
                await self._terminal_error(item, RequestState.TIMED_OUT)
        except asyncio.CancelledError:
            for item in batch:
                await self._terminal_error(item, item.cancellation_state)
            raise
        except Exception as exc:
            for item in batch:
                await self._terminal_error(item, RequestState.FAILED, exc)
        finally:
            duration = self._clock() - started
            self.metrics.batch_processing_seconds.append(duration)
            self._observability.batch_processing.observe(duration)

    async def _run_complete(self, item: WorkItem) -> None:
        """Run a non-streaming singleton request through one backend generate call."""

        try:
            item.context.transition(RequestState.SCHEDULED)
            async with asyncio.timeout(item.context.remaining_seconds):
                item.context.transition(RequestState.PREFILL)
                result = await self._backend.generate(item.generation)
                item.context.transition(RequestState.DECODING)
            item.generated_tokens = result.generated_tokens
            item.context.transition(RequestState.COMPLETED)
            self._lifecycle.metrics.completed += 1
            if not item.result.done():
                item.result.set_result(result)
            self._observability.record_finished(
                item.context,
                prompt_tokens=result.prompt_tokens,
                output_tokens=result.generated_tokens,
            )
        except TimeoutError:
            await self._terminal_error(item, RequestState.TIMED_OUT)
        except asyncio.CancelledError:
            await self._terminal_error(item, item.cancellation_state)
            raise
        except Exception as exc:
            await self._terminal_error(item, RequestState.FAILED, exc)

    async def _run(self, item: WorkItem) -> None:
        text: list[str] = []
        generated_tokens = 0
        try:
            item.context.transition(RequestState.SCHEDULED)
            async with asyncio.timeout(item.context.remaining_seconds):
                item.context.transition(RequestState.PREFILL)
                async for event in self._backend.stream(item.generation):
                    if item.context.state is RequestState.PREFILL:
                        item.context.transition(RequestState.DECODING)
                    item.context.record_token()
                    text.append(event.text)
                    generated_tokens += event.token_count
                    item.generated_tokens = generated_tokens
                    if item.stream:
                        await item.output.put(event)
                if item.context.state is RequestState.PREFILL:
                    item.context.transition(RequestState.DECODING)
            item.context.transition(RequestState.COMPLETED)
            self._lifecycle.metrics.completed += 1
            result = GenerationResult(
                text="".join(text),
                prompt_tokens=item.generation.prompt_token_count,
                generated_tokens=generated_tokens,
                finish_reason=(
                    "length" if generated_tokens >= item.generation.max_tokens else "stop"
                ),
            )
            if not item.result.done():
                item.result.set_result(result)
            self._observability.record_finished(
                item.context,
                prompt_tokens=result.prompt_tokens,
                output_tokens=result.generated_tokens,
            )
        except TimeoutError:
            await self._terminal_error(item, RequestState.TIMED_OUT)
        except asyncio.CancelledError:
            await self._terminal_error(item, item.cancellation_state)
            raise
        except Exception as exc:
            await self._terminal_error(item, RequestState.FAILED, exc)
        finally:
            if item.stream and item.context.state is RequestState.COMPLETED:
                await item.output.put(None)

    async def _terminal_error(
        self,
        item: WorkItem,
        state: RequestState,
        cause: Exception | None = None,
    ) -> None:
        if item.context.state in {
            RequestState.COMPLETED,
            RequestState.CANCELLED,
            RequestState.TIMED_OUT,
            RequestState.FAILED,
        }:
            return
        item.context.transition(state)
        if state is RequestState.TIMED_OUT:
            self._lifecycle.metrics.timed_out += 1
            error: BaseException = RequestTimedOut("request deadline exceeded")
        elif state is RequestState.CANCELLED:
            self._lifecycle.metrics.cancelled += 1
            error = RequestCancelled("request cancelled")
        else:
            self._lifecycle.metrics.failed += 1
            error = cause or RuntimeError("backend generation failed")
        if not item.result.done():
            item.result.set_exception(error)
        if item.stream:
            self._force_output(item, error)
        self._observability.record_finished(
            item.context,
            prompt_tokens=item.generation.prompt_token_count,
            output_tokens=item.generated_tokens,
        )

    @staticmethod
    def _force_output(item: WorkItem, value: TokenEvent | BaseException | None) -> None:
        """Deliver terminal stream events without blocking on a full client buffer."""

        try:
            item.output.put_nowait(value)
        except asyncio.QueueFull:
            try:
                item.output.get_nowait()
            except asyncio.QueueEmpty:
                pass
            item.output.put_nowait(value)

    async def shutdown(self) -> None:
        """Stop dispatch and cancel work that remains queued."""

        self._accepting = False
        if self._dispatcher is not None:
            self._dispatcher.cancel()
            await asyncio.gather(self._dispatcher, return_exceptions=True)
        for item in tuple(self._queue):
            await self.cancel(item.context.request_id)
