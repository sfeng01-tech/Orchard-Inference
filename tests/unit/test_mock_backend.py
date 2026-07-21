import asyncio

import pytest

from orchard_inference.backends.mock import MockBackend, MockFaultConfig
from orchard_inference.errors import (
    BackendGenerationError,
    BackendLoadError,
    BackendUnhealthyError,
    MemoryPressureError,
)
from orchard_inference.models import GenerationRequest, HealthStatus


@pytest.mark.asyncio
async def test_mock_backend_lifecycle_and_generation() -> None:
    backend = MockBackend("mock/test")
    assert backend.health().status is HealthStatus.UNLOADED
    await backend.load()

    result = await backend.generate(GenerationRequest("hello world", 0.0, 1.0, 2, ()))

    assert result.text == "Orchard mock"
    assert result.prompt_tokens == 2
    assert result.generated_tokens == 2
    assert result.finish_reason == "length"
    await backend.unload()
    assert backend.health().status is HealthStatus.UNLOADED


@pytest.mark.asyncio
async def test_mock_generation_is_cancellable() -> None:
    backend = MockBackend("mock/test", token_delay_seconds=1.0)
    await backend.load()
    task = asyncio.create_task(backend.generate(GenerationRequest("hello", 0.0, 1.0, 3, ())))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_mock_stream_yields_incremental_fragments() -> None:
    backend = MockBackend("mock/test")
    await backend.load()
    fragments = [
        event.text async for event in backend.stream(GenerationRequest("hello", 0.0, 1.0, 3, ()))
    ]
    assert fragments == ["Orchard", " mock", " response"]


@pytest.mark.asyncio
async def test_mock_fault_injection_can_fail_load() -> None:
    backend = MockBackend("mock/test", faults=MockFaultConfig(fail_load=True))

    with pytest.raises(BackendLoadError):
        await backend.load()

    assert backend.health().status is HealthStatus.UNHEALTHY


@pytest.mark.asyncio
async def test_mock_fault_injection_can_fail_generation() -> None:
    backend = MockBackend("mock/test", faults=MockFaultConfig(fail_generation=True))
    await backend.load()

    with pytest.raises(BackendGenerationError):
        await backend.generate(GenerationRequest("hello", 0.0, 1.0, 3, ()))


@pytest.mark.asyncio
async def test_mock_fault_injection_can_fail_after_streamed_tokens() -> None:
    backend = MockBackend("mock/test", faults=MockFaultConfig(fail_after_tokens=2))
    await backend.load()
    seen = []

    with pytest.raises(BackendGenerationError):
        async for event in backend.stream(GenerationRequest("hello", 0.0, 1.0, 3, ())):
            seen.append(event.text)

    assert seen == ["Orchard", " mock"]


@pytest.mark.asyncio
async def test_mock_fault_injection_can_simulate_memory_pressure() -> None:
    backend = MockBackend("mock/test", faults=MockFaultConfig(memory_pressure=True))
    await backend.load()

    with pytest.raises(MemoryPressureError):
        await backend.generate(GenerationRequest("hello", 0.0, 1.0, 3, ()))


@pytest.mark.asyncio
async def test_mock_fault_injection_can_become_unhealthy() -> None:
    backend = MockBackend("mock/test", faults=MockFaultConfig(unhealthy_after_requests=1))
    await backend.load()

    await backend.generate(GenerationRequest("hello", 0.0, 1.0, 1, ()))
    with pytest.raises(BackendUnhealthyError):
        await backend.generate(GenerationRequest("hello", 0.0, 1.0, 1, ()))
    assert backend.health().status is HealthStatus.UNHEALTHY
