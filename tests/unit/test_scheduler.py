import asyncio

import pytest

from orchard_inference.backends.mock import MockBackend
from orchard_inference.lifecycle import LifecycleManager, RequestContext, RequestState
from orchard_inference.models import GenerationRequest
from orchard_inference.scheduler import (
    AdmissionRejected,
    RejectionReason,
    RequestCancelled,
    RequestScheduler,
    SchedulingPolicy,
    WorkItem,
    select_item,
)


class CountingBackend(MockBackend):
    def __init__(self) -> None:
        super().__init__("mock/test")
        self.generate_calls = 0
        self.stream_calls = 0

    async def generate(self, request: GenerationRequest):
        self.generate_calls += 1
        return await super().generate(request)

    async def stream(self, request: GenerationRequest):
        self.stream_calls += 1
        async for event in super().stream(request):
            yield event


def _item(
    sequence: int,
    *,
    priority: int = 0,
    enqueued_at: float = 0.0,
    prompt_words: int = 1,
    max_tokens: int = 1,
) -> WorkItem:
    loop = asyncio.get_running_loop()
    return WorkItem(
        context=RequestContext(f"request-{sequence}"),
        generation=GenerationRequest(
            prompt="word " * prompt_words,
            temperature=0,
            top_p=1,
            max_tokens=max_tokens,
            stop=(),
        ),
        priority=priority,
        sequence=sequence,
        enqueued_at=enqueued_at,
        stream=False,
        output=asyncio.Queue(2),
        result=loop.create_future(),
    )


@pytest.mark.asyncio
async def test_fifo_is_stable() -> None:
    items = [_item(2), _item(0), _item(1)]
    assert select_item(items, SchedulingPolicy.FIFO, now=10, aging_seconds=5).sequence == 0


@pytest.mark.asyncio
async def test_priority_uses_fifo_for_ties() -> None:
    items = [_item(0, priority=1), _item(1, priority=5), _item(2, priority=5)]
    assert select_item(items, SchedulingPolicy.PRIORITY, now=0, aging_seconds=5).sequence == 1


@pytest.mark.asyncio
async def test_shortest_job_uses_estimated_tokens() -> None:
    items = [
        _item(0, prompt_words=20, max_tokens=20),
        _item(1, prompt_words=2, max_tokens=3),
    ]
    selected = select_item(items, SchedulingPolicy.SHORTEST_JOB, now=0, aging_seconds=5)
    assert selected.sequence == 1


@pytest.mark.asyncio
async def test_aging_prevents_shortest_job_starvation() -> None:
    old_long = _item(0, enqueued_at=0, prompt_words=10, max_tokens=10)
    new_short = _item(1, enqueued_at=100, prompt_words=1, max_tokens=1)
    selected = select_item(
        [old_long, new_short], SchedulingPolicy.SHORTEST_JOB, now=100, aging_seconds=1
    )
    assert selected is old_long


@pytest.mark.asyncio
async def test_admission_rejects_output_limit() -> None:
    backend = MockBackend("mock/test")
    lifecycle = LifecycleManager()
    scheduler = RequestScheduler(
        backend,
        lifecycle,
        policy=SchedulingPolicy.FIFO,
        max_queued=1,
        max_active=1,
        max_prompt_tokens=10,
        max_output_tokens=2,
        max_total_tokens=12,
        stream_buffer_size=2,
        aging_seconds=1,
        max_batch_size=2,
        max_batch_wait_seconds=0,
        batch_token_budget=20,
    )
    await backend.load()
    await scheduler.start()
    context = RequestContext("request-1")
    context.transition(RequestState.VALIDATED)
    with pytest.raises(AdmissionRejected) as caught:
        await scheduler.submit(
            context,
            GenerationRequest("hello", 0, 1, 3, ()),
            priority=0,
            stream=False,
        )
    assert caught.value.reason is RejectionReason.OUTPUT_TOO_LONG
    await scheduler.shutdown()
    await backend.unload()


@pytest.mark.asyncio
async def test_non_stream_singleton_uses_generate_fast_path() -> None:
    backend = CountingBackend()
    lifecycle = LifecycleManager()
    scheduler = RequestScheduler(
        backend,
        lifecycle,
        policy=SchedulingPolicy.FIFO,
        max_queued=4,
        max_active=1,
        max_prompt_tokens=10,
        max_output_tokens=3,
        max_total_tokens=12,
        stream_buffer_size=2,
        aging_seconds=1,
        max_batch_size=1,
        max_batch_wait_seconds=0,
        batch_token_budget=20,
    )
    await backend.load()
    await scheduler.start()
    context = RequestContext("request-1")
    context.transition(RequestState.VALIDATED)
    item = await scheduler.submit(
        context,
        GenerationRequest("hello", 0, 1, 3, ()),
        priority=0,
        stream=False,
    )

    result = await item.result

    assert result.text == "Orchard mock response"
    assert backend.generate_calls == 1
    assert backend.stream_calls == 0
    await scheduler.shutdown()
    await backend.unload()


@pytest.mark.asyncio
async def test_batch_compatibility_includes_generation_configuration() -> None:
    first = _item(0)
    compatible = _item(1)
    incompatible = _item(2)
    incompatible.generation = GenerationRequest("word", 0.7, 1, 1, ())

    assert RequestScheduler.compatible(first, compatible)
    assert not RequestScheduler.compatible(first, incompatible)
    first.stream = True
    assert not RequestScheduler.compatible(first, compatible)


@pytest.mark.asyncio
async def test_cancellation_stress_cleans_up_queued_and_active_work() -> None:
    backend = MockBackend("mock/test", token_delay_seconds=1.0)
    lifecycle = LifecycleManager()
    scheduler = RequestScheduler(
        backend,
        lifecycle,
        policy=SchedulingPolicy.FIFO,
        max_queued=64,
        max_active=1,
        max_prompt_tokens=10,
        max_output_tokens=3,
        max_total_tokens=12,
        stream_buffer_size=1,
        aging_seconds=1,
        max_batch_size=1,
        max_batch_wait_seconds=0,
        batch_token_budget=20,
    )
    await backend.load()
    await scheduler.start()
    items = []
    for index in range(20):
        context = RequestContext(f"cancel-{index}")
        context.transition(RequestState.VALIDATED)
        items.append(
            await scheduler.submit(
                context,
                GenerationRequest("hello", 0, 1, 3, ()),
                priority=0,
                stream=True,
            )
        )
    await scheduler.wait_for_state(active_requests=1, queue_depth=19)

    await asyncio.gather(*(scheduler.cancel(item.context.request_id) for item in items))
    await scheduler.wait_for_state(active_requests=0, queue_depth=0)

    assert lifecycle.metrics.cancelled == 20
    assert all(item.result.done() for item in items)
    for item in items:
        with pytest.raises(RequestCancelled):
            item.result.result()
    await scheduler.shutdown()
    await backend.unload()
