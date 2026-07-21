"""Prometheus metrics and safe process telemetry."""

import os
from dataclasses import dataclass

import psutil
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

from orchard_inference.lifecycle import RequestContext, RequestState

LATENCY_BUCKETS = (0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30)
TOKEN_BUCKETS = (1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)


@dataclass(frozen=True, slots=True)
class MetricMetadata:
    """Bounded labels shared by inference metrics."""

    backend: str
    policy: str


class Metrics:
    """Application-owned Prometheus registry and instrumentation hooks."""

    def __init__(self, metadata: MetricMetadata) -> None:
        self.registry = CollectorRegistry()
        self.metadata = metadata
        self.requests_received = Counter(
            "orchard_requests_received_total", "Validated requests received", registry=self.registry
        )
        self.requests_completed = Counter(
            "orchard_requests_completed_total", "Requests completed", registry=self.registry
        )
        self.requests_failed = Counter(
            "orchard_requests_failed_total", "Requests failed", registry=self.registry
        )
        self.requests_cancelled = Counter(
            "orchard_requests_cancelled_total", "Requests cancelled", registry=self.registry
        )
        self.requests_timed_out = Counter(
            "orchard_requests_timed_out_total", "Requests timed out", registry=self.registry
        )
        self.requests_rejected = Counter(
            "orchard_requests_rejected_total",
            "Requests rejected by reason",
            ("reason",),
            registry=self.registry,
        )
        self.telemetry_collection_failures = Counter(
            "orchard_telemetry_collection_failures_total",
            "Process telemetry refresh failures",
            registry=self.registry,
        )
        self.prompt_tokens = Counter(
            "orchard_prompt_tokens_total",
            "Estimated prompt tokens processed",
            registry=self.registry,
        )
        self.output_tokens = Counter(
            "orchard_output_tokens_total", "Output tokens generated", registry=self.registry
        )
        self.cache_hits = Counter(
            "orchard_cache_hits_total", "Cache hits (cache phases only)", registry=self.registry
        )
        self.cache_misses = Counter(
            "orchard_cache_misses_total", "Cache misses (cache phases only)", registry=self.registry
        )
        self.cache_evictions = Counter(
            "orchard_cache_evictions_total",
            "Cache evictions (cache phases only)",
            registry=self.registry,
        )
        self.cache_operations = Counter(
            "orchard_cache_operations_total",
            "Cache operations by bounded cache layer and result",
            ("layer", "result"),
            registry=self.registry,
        )
        self.prefix_router_requests = Counter(
            "orchard_prefix_router_requests_total",
            "Prefix router observations by route",
            ("route",),
            registry=self.registry,
        )
        self.prefix_router_matched_tokens = Counter(
            "orchard_prefix_router_matched_tokens_total",
            "Prompt tokens matched against prior prefixes",
            registry=self.registry,
        )
        self.prefix_router_saved_tokens = Counter(
            "orchard_prefix_router_estimated_saved_prefill_tokens_total",
            "Estimated prefill tokens avoidable with runtime KV reuse",
            registry=self.registry,
        )
        self.prefix_router_match_ratio = Histogram(
            "orchard_prefix_router_match_ratio",
            "Fraction of prompt tokens matched by prefix router",
            buckets=(0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0),
            registry=self.registry,
        )
        self.queue_depth = Gauge(
            "orchard_queue_depth", "Requests waiting for scheduling", registry=self.registry
        )
        self.active_requests = Gauge(
            "orchard_active_requests", "Requests currently executing", registry=self.registry
        )
        self.loaded_models = Gauge(
            "orchard_loaded_models", "Models currently loaded", registry=self.registry
        )
        self.estimated_cache_bytes = Gauge(
            "orchard_estimated_active_cache_bytes",
            "Estimated active cache bytes (not physical memory)",
            registry=self.registry,
        )
        self.current_batch_size = Gauge(
            "orchard_current_batch_size",
            "Most recently dispatched batch size",
            registry=self.registry,
        )
        self.process_rss = Gauge(
            "orchard_process_resident_memory_bytes",
            "Process resident memory",
            registry=self.registry,
        )
        self.process_cpu = Gauge(
            "orchard_process_cpu_percent",
            "Process CPU utilization percentage",
            registry=self.registry,
        )
        self.system_memory = Gauge(
            "orchard_system_memory_used_percent",
            "System memory utilization percentage",
            registry=self.registry,
        )
        self.load_average = Gauge(
            "orchard_system_load_average",
            "System load average",
            ("period",),
            registry=self.registry,
        )
        self.queue_latency = Histogram(
            "orchard_queue_latency_seconds",
            "Time from queue entry to scheduling",
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.time_to_first_token = Histogram(
            "orchard_time_to_first_token_seconds",
            "Time from receipt to first token",
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.inter_token_latency = Histogram(
            "orchard_inter_token_latency_seconds",
            "Time between emitted tokens",
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.prefill_duration = Histogram(
            "orchard_prefill_duration_seconds",
            "Observed prefill-stage duration",
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.decode_duration = Histogram(
            "orchard_decode_duration_seconds",
            "Observed decoding-stage duration",
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.end_to_end = Histogram(
            "orchard_end_to_end_latency_seconds",
            "Request end-to-end duration",
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.batch_size = Histogram(
            "orchard_batch_size",
            "Actual dispatched batch size",
            buckets=(1, 2, 4, 8, 16, 32),
            registry=self.registry,
        )
        self.batch_formation = Histogram(
            "orchard_batch_formation_seconds",
            "Time from oldest batch member enqueue to dispatch",
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.batch_tokens = Histogram(
            "orchard_batch_estimated_tokens",
            "Estimated prompt and requested-output tokens per batch",
            buckets=TOKEN_BUCKETS,
            registry=self.registry,
        )
        self.batch_processing = Histogram(
            "orchard_batch_processing_seconds",
            "Backend batch processing duration",
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.prompt_length = Histogram(
            "orchard_prompt_length_tokens",
            "Estimated prompt length",
            buckets=TOKEN_BUCKETS,
            registry=self.registry,
        )
        self.output_length = Histogram(
            "orchard_output_length_tokens",
            "Generated output length",
            buckets=TOKEN_BUCKETS,
            registry=self.registry,
        )
        self.model_load_duration = Histogram(
            "orchard_model_load_seconds",
            "Configured backend model load duration",
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )
        self._process = psutil.Process()

    def record_rejection(self, reason: str) -> None:
        """Record one bounded admission rejection reason."""

        self.requests_rejected.labels(reason=reason).inc()

    def record_batch(self, size: int, formation_seconds: float, tokens: int) -> None:
        """Record a dispatched batch."""

        self.current_batch_size.set(size)
        self.batch_size.observe(size)
        self.batch_formation.observe(formation_seconds)
        self.batch_tokens.observe(tokens)

    def record_prefix_route(
        self,
        route: str,
        matched_tokens: int,
        saved_tokens: int,
        matched_ratio: float,
    ) -> None:
        """Record one bounded prefix-router decision."""

        self.prefix_router_requests.labels(route=route).inc()
        self.prefix_router_match_ratio.observe(matched_ratio)
        if matched_tokens:
            self.prefix_router_matched_tokens.inc(matched_tokens)
        if saved_tokens:
            self.prefix_router_saved_tokens.inc(saved_tokens)

    def record_finished(
        self,
        context: RequestContext,
        *,
        prompt_tokens: int,
        output_tokens: int,
    ) -> None:
        """Record a terminal request exactly once."""

        if context.state is RequestState.COMPLETED:
            self.requests_completed.inc()
        elif context.state is RequestState.TIMED_OUT:
            self.requests_timed_out.inc()
        elif context.state is RequestState.CANCELLED:
            self.requests_cancelled.inc()
        elif context.state is RequestState.FAILED:
            self.requests_failed.inc()
        if RequestState.PREFILL in context.timestamps:
            self.prompt_tokens.inc(prompt_tokens)
            self.prompt_length.observe(prompt_tokens)
        if output_tokens:
            self.output_tokens.inc(output_tokens)
            self.output_length.observe(output_tokens)
        queue = context.duration(RequestState.QUEUED, RequestState.SCHEDULED)
        prefill = context.duration(RequestState.PREFILL, RequestState.DECODING)
        decode = context.duration(RequestState.DECODING, context.state)
        total = context.duration(RequestState.RECEIVED, context.state)
        if queue is not None:
            self.queue_latency.observe(queue)
        if prefill is not None:
            self.prefill_duration.observe(prefill)
        if decode is not None and decode >= 0:
            self.decode_duration.observe(decode)
        if total is not None:
            self.end_to_end.observe(total)
        if context.time_to_first_token is not None:
            self.time_to_first_token.observe(context.time_to_first_token)
        for interval in context.inter_token_seconds:
            self.inter_token_latency.observe(interval)

    def update_process_telemetry(self) -> None:
        """Refresh unprivileged process and system telemetry."""

        self.process_rss.set(self._process.memory_info().rss)
        self.process_cpu.set(self._process.cpu_percent(interval=None))
        self.system_memory.set(psutil.virtual_memory().percent)
        if hasattr(os, "getloadavg"):
            one, five, fifteen = os.getloadavg()
            self.load_average.labels(period="1m").set(one)
            self.load_average.labels(period="5m").set(five)
            self.load_average.labels(period="15m").set(fifteen)

    def render(self) -> bytes:
        """Refresh telemetry and render this application's registry."""

        try:
            self.update_process_telemetry()
        except Exception:
            self.telemetry_collection_failures.inc()
        return generate_latest(self.registry)
