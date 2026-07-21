from orchard_inference.lifecycle import RequestContext, RequestState
from orchard_inference.observability import MetricMetadata, Metrics


def test_terminal_metrics_are_recorded_once_with_bounded_labels() -> None:
    metrics = Metrics(MetricMetadata(backend="mock", policy="fifo"))
    context = RequestContext("unique-request-id")
    for state in (
        RequestState.VALIDATED,
        RequestState.QUEUED,
        RequestState.SCHEDULED,
        RequestState.PREFILL,
        RequestState.DECODING,
        RequestState.COMPLETED,
    ):
        context.transition(state)
    context.record_token()
    metrics.record_finished(context, prompt_tokens=4, output_tokens=1)

    rendered = metrics.render().decode()
    assert "orchard_requests_completed_total 1.0" in rendered
    assert "orchard_prompt_tokens_total 4.0" in rendered
    assert "unique-request-id" not in rendered


def test_cache_metrics_exist_but_start_at_zero() -> None:
    metrics = Metrics(MetricMetadata(backend="mock", policy="fifo"))
    rendered = metrics.render().decode()
    assert "orchard_cache_hits_total 0.0" in rendered
    assert "orchard_estimated_active_cache_bytes 0.0" in rendered


def test_metrics_render_survives_process_telemetry_failure() -> None:
    metrics = Metrics(MetricMetadata(backend="mock", policy="fifo"))

    def fail() -> None:
        raise RuntimeError("collector failed")

    metrics.update_process_telemetry = fail  # type: ignore[method-assign]

    rendered = metrics.render().decode()

    assert "orchard_telemetry_collection_failures_total 1.0" in rendered
