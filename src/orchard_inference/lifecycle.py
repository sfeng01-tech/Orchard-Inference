"""Request lifecycle state, timing, and active-task ownership."""

import asyncio
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class RequestState(StrEnum):
    """Explicit inference request states."""

    RECEIVED = "received"
    VALIDATED = "validated"
    QUEUED = "queued"
    SCHEDULED = "scheduled"
    PREFILL = "prefill"
    DECODING = "decoding"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    REJECTED = "rejected"
    FAILED = "failed"


TERMINAL_STATES = {
    RequestState.COMPLETED,
    RequestState.CANCELLED,
    RequestState.TIMED_OUT,
    RequestState.REJECTED,
    RequestState.FAILED,
}

_NEXT: dict[RequestState, set[RequestState]] = {
    RequestState.RECEIVED: {RequestState.VALIDATED, RequestState.REJECTED},
    RequestState.VALIDATED: {RequestState.QUEUED, RequestState.REJECTED},
    RequestState.QUEUED: {RequestState.SCHEDULED},
    RequestState.SCHEDULED: {RequestState.PREFILL},
    RequestState.PREFILL: {RequestState.DECODING},
    RequestState.DECODING: {RequestState.COMPLETED},
}


class InvalidStateTransition(ValueError):
    """Raised when request code attempts an invalid lifecycle transition."""


@dataclass(slots=True)
class RequestContext:
    """Lifecycle state and monotonic timestamps for one request."""

    request_id: str
    client_request_id: str | None = None
    state: RequestState = RequestState.RECEIVED
    created_at: float = field(default_factory=time.monotonic)
    deadline: float | None = None
    timestamps: dict[RequestState, float] = field(default_factory=dict)
    first_token_at: float | None = None
    last_token_at: float | None = None
    inter_token_seconds: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.timestamps[self.state] = self.created_at

    def transition(self, target: RequestState) -> None:
        """Move to a valid next state and record its timestamp."""

        if self.state in TERMINAL_STATES:
            raise InvalidStateTransition(f"terminal state {self.state} cannot transition")
        if target not in _NEXT.get(self.state, set()) and target not in {
            RequestState.CANCELLED,
            RequestState.TIMED_OUT,
            RequestState.FAILED,
        }:
            raise InvalidStateTransition(f"invalid transition {self.state} -> {target}")
        self.state = target
        self.timestamps[target] = time.monotonic()

    @property
    def remaining_seconds(self) -> float | None:
        """Return remaining deadline time, clamped at zero."""

        if self.deadline is None:
            return None
        return max(0.0, self.deadline - time.monotonic())

    def duration(self, start: RequestState, end: RequestState) -> float | None:
        """Return elapsed seconds between two recorded states."""

        if start not in self.timestamps or end not in self.timestamps:
            return None
        return self.timestamps[end] - self.timestamps[start]

    def record_token(self) -> None:
        """Record first-token and inter-token timing using the monotonic clock."""

        now = time.monotonic()
        if self.first_token_at is None:
            self.first_token_at = now
        elif self.last_token_at is not None:
            self.inter_token_seconds.append(now - self.last_token_at)
        self.last_token_at = now

    @property
    def time_to_first_token(self) -> float | None:
        """Return seconds from receipt to the first emitted token."""

        if self.first_token_at is None:
            return None
        return self.first_token_at - self.created_at


@dataclass(slots=True)
class LifecycleMetrics:
    """Bounded-label lifecycle counters exposed internally until Phase 5."""

    completed: int = 0
    cancelled: int = 0
    timed_out: int = 0
    failed: int = 0


class LifecycleManager:
    """Own active request tasks and cancel them during graceful shutdown."""

    def __init__(self) -> None:
        self._tasks: dict[str, asyncio.Task[Any]] = {}
        self.metrics = LifecycleMetrics()
        self.accepting = True

    def register(self, context: RequestContext, task: asyncio.Task[Any]) -> None:
        """Register one active producer task."""

        if not self.accepting:
            raise RuntimeError("server is shutting down")
        self._tasks[context.request_id] = task
        task.add_done_callback(lambda _task: self._tasks.pop(context.request_id, None))

    @property
    def active_count(self) -> int:
        """Return the current number of owned producer tasks."""

        return len(self._tasks)

    async def shutdown(self, grace_seconds: float) -> None:
        """Stop admission, wait briefly, then cancel remaining active tasks."""

        self.accepting = False
        if not self._tasks:
            return
        _, pending = await asyncio.wait(tuple(self._tasks.values()), timeout=grace_seconds)
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
