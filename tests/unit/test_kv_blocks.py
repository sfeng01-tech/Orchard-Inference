import pytest

from orchard_inference.kv_blocks import KVBlockManager, deterministic_scenario


def test_fork_shares_full_prefix_blocks_and_saves_capacity() -> None:
    manager = KVBlockManager(block_size_tokens=4)
    manager.allocate("parent", 10)
    child = manager.fork(
        parent_request_id="parent",
        child_request_id="child",
        shared_prefix_tokens=8,
        child_total_tokens=12,
    )
    summary = manager.summary()

    assert child.shared_prefix_tokens == 8
    assert child.copied_prefix_tokens == 0
    assert summary.refcounted_blocks == 2
    assert summary.estimated_capacity_tokens_saved_vs_dense == 8


def test_partial_prefix_requires_private_copy_tokens() -> None:
    manager = KVBlockManager(block_size_tokens=4)
    manager.allocate("parent", 10)
    child = manager.fork(
        parent_request_id="parent",
        child_request_id="child",
        shared_prefix_tokens=6,
        child_total_tokens=10,
    )
    summary = manager.summary()

    assert child.shared_prefix_tokens == 4
    assert child.copied_prefix_tokens == 2
    assert summary.copied_prefix_tokens == 2
    assert summary.refcounted_blocks == 1


def test_release_decrements_refcounts_and_frees_blocks() -> None:
    manager = KVBlockManager(block_size_tokens=4)
    manager.allocate("parent", 8)
    manager.fork(
        parent_request_id="parent",
        child_request_id="child",
        shared_prefix_tokens=8,
        child_total_tokens=8,
    )

    assert manager.summary().physical_blocks == 2
    manager.release("child")
    summary = manager.summary()

    assert summary.active_sequences == 1
    assert summary.physical_blocks == 2
    assert summary.refcounted_blocks == 0
    manager.release("parent")
    assert manager.summary().physical_blocks == 0


def test_append_uses_private_tail_capacity() -> None:
    manager = KVBlockManager(block_size_tokens=4)
    manager.allocate("request", 6)

    manager.append("request", 1)
    summary = manager.summary()

    assert summary.physical_blocks == 2
    assert summary.physical_used_tokens == 7


def test_invalid_inputs_are_rejected() -> None:
    with pytest.raises(ValueError):
        KVBlockManager(block_size_tokens=0)
    manager = KVBlockManager(block_size_tokens=4)
    with pytest.raises(ValueError):
        manager.allocate("bad", -1)


def test_deterministic_scenario_reports_capacity_savings() -> None:
    result = deterministic_scenario(
        block_size_tokens=8,
        sequences=4,
        base_prompt_tokens=32,
        shared_prefix_tokens=24,
        decode_tokens=4,
    )
    summary = result["summary"]

    assert summary["active_sequences"] == 4
    assert summary["shared_logical_tokens"] == 72
    assert summary["estimated_capacity_tokens_saved_vs_dense"] > 0
