from orchard_inference.chunked_prefill import (
    ChunkedPrefillPolicy,
    ChunkedPrefillSimulator,
    ChunkedRequestSpec,
    deterministic_workload,
    summarize,
)


def test_decode_first_protects_tpot_better_than_prefill_first() -> None:
    specs = [
        ChunkedRequestSpec("first", 0, prompt_tokens=4, output_tokens=4),
        ChunkedRequestSpec("second", 1, prompt_tokens=8, output_tokens=1),
    ]
    prefill_first = summarize(
        ChunkedPrefillSimulator(
            chunk_size=2,
            policy=ChunkedPrefillPolicy.PREFILL_FIRST,
            decode_tpot_slo_steps=2,
        ).run(specs)
    )
    decode_first = summarize(
        ChunkedPrefillSimulator(
            chunk_size=2,
            policy=ChunkedPrefillPolicy.DECODE_FIRST,
            decode_tpot_slo_steps=2,
        ).run(specs)
    )

    assert decode_first["tpot_p95_steps"] <= prefill_first["tpot_p95_steps"]


def test_mixed_slo_interleaves_prefill_and_decode() -> None:
    specs = [
        ChunkedRequestSpec("first", 0, prompt_tokens=4, output_tokens=4),
        ChunkedRequestSpec("second", 1, prompt_tokens=6, output_tokens=2),
    ]
    result = ChunkedPrefillSimulator(
        chunk_size=2,
        policy=ChunkedPrefillPolicy.MIXED_SLO,
        decode_tpot_slo_steps=2,
    ).run(specs)
    operations = [event.operation for event in result.events]

    assert "prefill" in operations
    assert "decode" in operations
    assert summarize(result)["tpot_violation_rate"] < 1


def test_prefix_saved_tokens_reduce_effective_prefill_and_ttft() -> None:
    without_prefix = [
        ChunkedRequestSpec("a", 0, prompt_tokens=8, output_tokens=2),
        ChunkedRequestSpec("b", 0, prompt_tokens=8, output_tokens=2),
    ]
    with_prefix = [
        ChunkedRequestSpec("a", 0, prompt_tokens=8, output_tokens=2),
        ChunkedRequestSpec("b", 0, prompt_tokens=8, output_tokens=2, prefix_saved_tokens=6),
    ]
    simulator = ChunkedPrefillSimulator(
        chunk_size=2,
        policy=ChunkedPrefillPolicy.PREFILL_FIRST,
        decode_tpot_slo_steps=2,
    )

    baseline = summarize(simulator.run(without_prefix))
    routed = summarize(simulator.run(with_prefix))

    assert routed["effective_prefill_tokens"] < baseline["effective_prefill_tokens"]
    assert routed["ttft_p95_steps"] < baseline["ttft_p95_steps"]


def test_deterministic_workload_is_reproducible() -> None:
    first = deterministic_workload(
        requests=4,
        arrival_interval_steps=2,
        prompt_tokens=16,
        output_tokens=4,
        prefix_saved_tokens=8,
    )
    second = deterministic_workload(
        requests=4,
        arrival_interval_steps=2,
        prompt_tokens=16,
        output_tokens=4,
        prefix_saved_tokens=8,
    )

    assert first == second
    assert first[0].prefix_saved_tokens == 0
    assert first[1].prefix_saved_tokens == 8
