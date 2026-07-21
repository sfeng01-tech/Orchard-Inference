import asyncio
import time

import pytest

from orchard_inference.lifecycle import (
    InvalidStateTransition,
    LifecycleManager,
    RequestContext,
    RequestState,
)


def test_request_state_happy_path_records_timestamps() -> None:
    context = RequestContext("request-1")
    for state in (
        RequestState.VALIDATED,
        RequestState.QUEUED,
        RequestState.SCHEDULED,
        RequestState.PREFILL,
        RequestState.DECODING,
        RequestState.COMPLETED,
    ):
        context.transition(state)

    assert context.state is RequestState.COMPLETED
    assert context.duration(RequestState.RECEIVED, RequestState.COMPLETED) is not None


def test_invalid_and_terminal_transitions_are_rejected() -> None:
    context = RequestContext("request-1")
    with pytest.raises(InvalidStateTransition):
        context.transition(RequestState.DECODING)
    context.transition(RequestState.FAILED)
    with pytest.raises(InvalidStateTransition):
        context.transition(RequestState.COMPLETED)


def test_remaining_deadline_is_clamped() -> None:
    context = RequestContext("request-1", deadline=time.monotonic() - 1)
    assert context.remaining_seconds == 0.0


def test_token_timing_records_ttft_and_intervals() -> None:
    context = RequestContext("request-1")
    context.record_token()
    context.record_token()

    assert context.time_to_first_token is not None
    assert context.time_to_first_token >= 0
    assert len(context.inter_token_seconds) == 1
    assert context.inter_token_seconds[0] >= 0


@pytest.mark.asyncio
async def test_shutdown_cancels_work_after_grace_period() -> None:
    manager = LifecycleManager()
    context = RequestContext("request-1")
    task = asyncio.create_task(asyncio.sleep(60))
    manager.register(context, task)

    await manager.shutdown(0)

    assert task.cancelled()
    assert manager.accepting is False
