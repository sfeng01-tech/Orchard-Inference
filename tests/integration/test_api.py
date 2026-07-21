import asyncio

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from orchard_inference.config import Settings
from orchard_inference.main import create_app


@pytest_asyncio.fixture
async def client() -> AsyncClient:
    app = create_app(Settings(backend="mock", model="mock/test"))
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as value:
            yield value


@pytest.mark.asyncio
async def test_health_and_models(client: AsyncClient) -> None:
    assert (await client.get("/health/live")).json() == {"status": "live"}
    assert (await client.get("/health/ready")).json() == {"status": "ready"}
    response = await client.get("/v1/models")
    assert response.status_code == 200
    assert response.json()["data"][0]["id"] == "mock/test"
    assert response.json()["data"][0]["architecture"] == "mock"
    assert response.json()["data"][0]["quantization"] == "none"


@pytest.mark.asyncio
async def test_control_room_ui_assets_are_served(client: AsyncClient) -> None:
    page = await client.get("/ui")
    script = await client.get("/ui/app.js")
    styles = await client.get("/ui/styles.css")

    assert page.status_code == 200
    assert "Orchard Control Room" in page.text
    assert "Observability Dashboard" in page.text
    assert script.status_code == 200
    assert "sendRequest" in script.text
    assert "parseMetrics" in script.text
    assert "runChunkedPrefill" in script.text
    assert "runKvBlocks" in script.text
    assert styles.status_code == 200
    assert "text/css" in styles.headers["content-type"]


@pytest.mark.asyncio
async def test_control_room_simulator_endpoints(client: AsyncClient) -> None:
    chunked = await client.post(
        "/ui/simulate/chunked-prefill",
        json={
            "requests": 4,
            "arrival_interval_steps": 2,
            "prompt_tokens": 32,
            "output_tokens": 4,
            "prefix_saved_tokens": 16,
            "chunk_size": 8,
            "decode_tpot_slo_steps": 2,
        },
    )
    kv = await client.post(
        "/ui/simulate/kv-blocks",
        json={
            "block_size_tokens": 8,
            "sequences": 4,
            "base_prompt_tokens": 32,
            "shared_prefix_tokens": 24,
            "decode_tokens": 4,
        },
    )

    assert chunked.status_code == 200
    assert {run["policy"] for run in chunked.json()["runs"]} == {
        "prefill_first",
        "decode_first",
        "mixed_slo",
    }
    assert kv.status_code == 200
    assert kv.json()["summary"]["estimated_capacity_tokens_saved_vs_dense"] > 0


@pytest.mark.asyncio
async def test_non_streaming_completion(client: AsyncClient) -> None:
    response = await client.post(
        "/v1/chat/completions",
        json={"model": "mock/test", "messages": [{"role": "user", "content": "Hello"}]},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "chat.completion"
    assert payload["choices"][0]["message"]["content"] == "Orchard mock response"
    assert payload["usage"]["completion_tokens"] == 3
    assert response.headers["X-Orchard-Batch-Size"] == "1"
    assert float(response.headers["X-Orchard-Queue-Seconds"]) >= 0
    metrics = await client.get("/metrics")
    assert metrics.status_code == 200
    assert "orchard_requests_completed_total 1.0" in metrics.text
    assert "orchard_output_tokens_total 3.0" in metrics.text
    assert payload["id"] not in metrics.text


@pytest.mark.asyncio
async def test_unknown_model_is_rejected(client: AsyncClient) -> None:
    response = await client.post(
        "/v1/chat/completions",
        json={"model": "missing", "messages": [{"role": "user", "content": "Hello"}]},
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_repeated_prompt_uses_application_caches(client: AsyncClient) -> None:
    payload = {
        "model": "mock/test",
        "messages": [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hello"},
        ],
    }
    assert (await client.post("/v1/chat/completions", json=payload)).status_code == 200
    assert (await client.post("/v1/chat/completions", json=payload)).status_code == 200
    metrics = (await client.get("/metrics")).text
    assert 'orchard_cache_operations_total{layer="tokenization",result="hit"} 1.0' in metrics


@pytest.mark.asyncio
async def test_prefix_router_reports_repeated_prompt_hit() -> None:
    app = create_app(
        Settings(
            backend="mock",
            model="mock/test",
            prefix_router_min_match_tokens=2,
        )
    )
    payload = {
        "model": "mock/test",
        "messages": [
            {"role": "system", "content": "Be precise."},
            {"role": "user", "content": "Explain queues."},
        ],
        "max_tokens": 1,
    }
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as value:
            first = await value.post("/v1/chat/completions", json=payload)
            second = await value.post("/v1/chat/completions", json=payload)
            metrics = (await value.get("/metrics")).text

    assert first.json()["orchard"]["prefix_route"] == "prefix_miss"
    assert second.headers["X-Orchard-Prefix-Route"] == "prefix_hit"
    assert second.json()["orchard"]["prefix_matched_tokens"] >= 2
    assert second.json()["orchard"]["prefix_estimated_saved_tokens"] >= 2
    assert 'orchard_prefix_router_requests_total{route="prefix_hit"} 1.0' in metrics
    assert "orchard_prefix_router_estimated_saved_prefill_tokens_total" in metrics


@pytest.mark.asyncio
async def test_streaming_completion_uses_sse(client: AsyncClient) -> None:
    async with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "mock/test",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        },
    ) as response:
        body = (await response.aread()).decode()

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert '"content":"Orchard"' in body
    assert '"content":" mock"' in body
    assert '"orchard":{' in body
    assert '"generated_tokens":3' in body
    assert '"prefix_route":"prefix_miss"' in body
    assert body.endswith("data: [DONE]\n\n")


@pytest.mark.asyncio
async def test_non_streaming_timeout() -> None:
    app = create_app(
        Settings(
            backend="mock",
            model="mock/test",
            mock_token_delay_seconds=0.1,
            request_timeout_seconds=0.01,
        )
    )
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as value:
            response = await value.post(
                "/v1/chat/completions",
                json={"model": "mock/test", "messages": [{"role": "user", "content": "Hi"}]},
            )
    assert response.status_code == 504
    assert app.state.lifecycle.metrics.timed_out == 1


@pytest.mark.asyncio
async def test_startup_fails_when_model_cannot_load() -> None:
    app = create_app(Settings(backend="mock", model="mock/test", mock_fail_load=True))

    with pytest.raises(RuntimeError, match="model load failed"):
        async with app.router.lifespan_context(app):
            pass


@pytest.mark.asyncio
async def test_backend_generation_failure_returns_structured_error() -> None:
    app = create_app(Settings(backend="mock", model="mock/test", mock_fail_generation=True))
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as value:
            response = await value.post(
                "/v1/chat/completions",
                json={"model": "mock/test", "messages": [{"role": "user", "content": "Hi"}]},
            )

    assert response.status_code == 500
    assert response.json()["detail"]["error"] == "backend_generation_failed"
    assert app.state.lifecycle.metrics.failed == 1


@pytest.mark.asyncio
async def test_memory_pressure_returns_retryable_structured_error() -> None:
    app = create_app(Settings(backend="mock", model="mock/test", mock_memory_pressure=True))
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as value:
            response = await value.post(
                "/v1/chat/completions",
                json={"model": "mock/test", "messages": [{"role": "user", "content": "Hi"}]},
            )

    assert response.status_code == 503
    assert response.json()["detail"]["error"] == "memory_pressure"


@pytest.mark.asyncio
async def test_backend_unhealthy_flips_readiness() -> None:
    app = create_app(Settings(backend="mock", model="mock/test", mock_unhealthy_after_requests=1))
    payload = {"model": "mock/test", "messages": [{"role": "user", "content": "Hi"}]}
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as value:
            assert (await value.post("/v1/chat/completions", json=payload)).status_code == 200
            failed = await value.post("/v1/chat/completions", json=payload)
            readiness = await value.get("/health/ready")

    assert failed.status_code == 503
    assert failed.json()["detail"]["error"] == "backend_unhealthy"
    assert readiness.status_code == 503


@pytest.mark.asyncio
async def test_stream_timeout_sends_terminal_error_event() -> None:
    app = create_app(
        Settings(
            backend="mock",
            model="mock/test",
            mock_token_delay_seconds=0.1,
            request_timeout_seconds=0.01,
        )
    )
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as value:
            response = await value.post(
                "/v1/chat/completions",
                json={
                    "model": "mock/test",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "stream": True,
                },
            )
    assert response.status_code == 200
    assert '"type":"timeout"' in response.text
    assert response.text.endswith("data: [DONE]\n\n")


@pytest.mark.asyncio
async def test_stream_backend_failure_after_tokens_does_not_retry() -> None:
    app = create_app(Settings(backend="mock", model="mock/test", mock_fail_after_tokens=2))
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as value:
            response = await value.post(
                "/v1/chat/completions",
                json={
                    "model": "mock/test",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "stream": True,
                },
            )

    assert response.status_code == 200
    assert response.text.count('"content":"Orchard"') == 1
    assert response.text.count('"content":" mock"') == 1
    assert '"type":"backend_generation_failed"' in response.text
    assert response.text.endswith("data: [DONE]\n\n")


@pytest.mark.asyncio
async def test_queue_overload_is_rejected() -> None:
    app = create_app(
        Settings(
            backend="mock",
            model="mock/test",
            mock_token_delay_seconds=0.05,
            max_active_requests=1,
            max_queued_requests=1,
        )
    )
    payload = {"model": "mock/test", "messages": [{"role": "user", "content": "Hi"}]}
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as value:
            first = asyncio.create_task(value.post("/v1/chat/completions", json=payload))
            await app.state.scheduler.wait_for_state(active_requests=1, queue_depth=0)
            second = asyncio.create_task(value.post("/v1/chat/completions", json=payload))
            await app.state.scheduler.wait_for_state(active_requests=1, queue_depth=1)
            rejected = await value.post("/v1/chat/completions", json=payload)
            assert rejected.status_code == 429
            assert rejected.json()["detail"]["reason"] == "queue_full"
            assert (await first).status_code == 200
            assert (await second).status_code == 200


@pytest.mark.asyncio
async def test_cache_pressure_evicts_without_breaking_requests() -> None:
    app = create_app(
        Settings(
            backend="mock",
            model="mock/test",
            cache_max_entries=1,
            cache_max_bytes=1024,
        )
    )
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as value:
            first = await value.post(
                "/v1/chat/completions",
                json={"model": "mock/test", "messages": [{"role": "user", "content": "one"}]},
            )
            second = await value.post(
                "/v1/chat/completions",
                json={"model": "mock/test", "messages": [{"role": "user", "content": "two"}]},
            )
            metrics = (await value.get("/metrics")).text

    assert first.status_code == 200
    assert second.status_code == 200
    assert "orchard_cache_evictions_total" in metrics


@pytest.mark.asyncio
async def test_compatible_requests_form_runtime_batch() -> None:
    app = create_app(
        Settings(
            backend="mock",
            model="mock/test",
            max_active_requests=2,
            max_batch_size=2,
            max_batch_wait_seconds=0.02,
        )
    )
    payload = {"model": "mock/test", "messages": [{"role": "user", "content": "Hi"}]}
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as value:
            responses = await asyncio.gather(
                value.post("/v1/chat/completions", json=payload),
                value.post("/v1/chat/completions", json=payload),
            )
    assert all(response.status_code == 200 for response in responses)
    assert 2 in app.state.scheduler.metrics.batch_sizes


@pytest.mark.asyncio
async def test_streaming_requests_remain_singleton_batches() -> None:
    app = create_app(
        Settings(
            backend="mock",
            model="mock/test",
            max_active_requests=2,
            max_batch_size=2,
            max_batch_wait_seconds=0.02,
        )
    )
    payload = {
        "model": "mock/test",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": True,
    }
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as value:
            responses = await asyncio.gather(
                value.post("/v1/chat/completions", json=payload),
                value.post("/v1/chat/completions", json=payload),
            )
    assert all(response.status_code == 200 for response in responses)
    assert app.state.scheduler.metrics.batch_sizes == [1, 1]
