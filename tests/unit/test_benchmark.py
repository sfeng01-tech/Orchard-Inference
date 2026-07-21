from orchard_inference.benchmark import RequestSample, Workload, percentile, summarize


def test_percentile_interpolates_and_handles_empty_input() -> None:
    assert percentile([], 0.95) is None
    assert percentile([1.0, 2.0, 3.0], 0.5) == 2.0
    assert percentile([1.0, 3.0], 0.5) == 2.0


def test_workload_is_reproducible() -> None:
    left = Workload([2, 4], [1, 3], seed=7)
    right = Workload([2, 4], [1, 3], seed=7)
    assert [left.next() for _ in range(8)] == [right.next() for _ in range(8)]


def test_summary_keeps_failures_and_batch_distribution() -> None:
    samples = [
        RequestSample(
            0,
            2,
            4,
            3,
            True,
            200,
            0.1,
            ttft_seconds=0.02,
            inter_token_seconds=[0.01, 0.02],
            queue_seconds=0.005,
            generated_tokens=3,
            batch_size=2,
        ),
        RequestSample(1, 2, 4, 3, False, 504, 0.2, timed_out=True),
    ]
    result = summarize(samples, elapsed_seconds=1, peak_concurrency=2)
    assert result["successful_requests"] == 1
    assert result["error_rate"] == 0.5
    assert result["timeout_rate"] == 0.5
    assert result["generated_tokens_per_second"] == 3
    assert result["batch_size_distribution"] == {"2": 1}
