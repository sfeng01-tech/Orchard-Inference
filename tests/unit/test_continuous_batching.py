from orchard_inference.continuous_batching import (
    ContinuousBatchSimulator,
    SequenceSpec,
    jains_fairness,
    simulate_static_batches,
    summarize_simulation,
)


def test_completed_sequence_leaves_and_late_sequence_enters() -> None:
    specs = [
        SequenceSpec("short", arrival_step=0, prompt_steps=1, decode_steps=1),
        SequenceSpec("long", arrival_step=0, prompt_steps=1, decode_steps=5),
        SequenceSpec("late", arrival_step=2, prompt_steps=1, decode_steps=2),
    ]
    result = ContinuousBatchSimulator(prefill_capacity=2, decode_capacity=2).run(specs)

    assert result.states["short"].completed_step == 2
    assert result.states["late"].first_token_step is not None
    assert result.states["late"].first_token_step < result.states["long"].completed_step
    assert any("short" in event.completed for event in result.events)
    assert any("late" in event.arrived for event in result.events)


def test_simulation_metrics_are_bounded_and_reproducible() -> None:
    specs = [SequenceSpec(f"request-{index}", index, 1, 2 + index % 3) for index in range(6)]
    simulator = ContinuousBatchSimulator(prefill_capacity=2, decode_capacity=3)
    first = summarize_simulation(simulator.run(specs))
    second = summarize_simulation(simulator.run(specs))
    assert first == second
    assert 0 < first["active_sequence_utilization"] <= 1
    assert 0 < first["fairness_jain_inverse_slowdown"] <= 1


def test_static_and_continuous_are_labeled_separately() -> None:
    specs = [
        SequenceSpec("short", 0, 1, 1),
        SequenceSpec("long", 0, 1, 10),
    ]
    continuous = summarize_simulation(
        ContinuousBatchSimulator(prefill_capacity=2, decode_capacity=2).run(specs)
    )
    static = simulate_static_batches(specs, batch_size=2)
    assert continuous["latency_mean_steps"] < static["latency_mean_steps"]


def test_jains_fairness() -> None:
    assert jains_fairness([1, 1, 1]) == 1
    assert 0 < jains_fairness([1, 2, 10]) < 1
