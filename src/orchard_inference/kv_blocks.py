"""Paged KV-cache block manager simulator.

This module models the memory-management side of paged attention: block tables,
refcounts, prefix sharing, copy-on-write at block boundaries, and fragmentation.
It does not implement an attention kernel or runtime KV-cache reuse.
"""

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class KVBlock:
    """One physical KV block."""

    block_id: int
    used_tokens: int
    refcount: int = 1


@dataclass(slots=True)
class SequenceAllocation:
    """Logical block table for one sequence."""

    request_id: str
    block_ids: list[int]
    token_count: int
    shared_prefix_tokens: int = 0
    copied_prefix_tokens: int = 0


@dataclass(frozen=True, slots=True)
class KVSummary:
    """Aggregate memory accounting for the simulated KV block manager."""

    block_size_tokens: int
    active_sequences: int
    physical_blocks: int
    physical_capacity_tokens: int
    physical_used_tokens: int
    logical_tokens: int
    dense_capacity_tokens: int
    shared_logical_tokens: int
    copied_prefix_tokens: int
    internal_fragmentation_tokens: int
    estimated_capacity_tokens_saved_vs_dense: int
    refcounted_blocks: int


class KVBlockManager:
    """Block-table allocator with full-block prefix sharing."""

    def __init__(self, block_size_tokens: int) -> None:
        if block_size_tokens < 1:
            raise ValueError("block_size_tokens must be positive")
        self.block_size_tokens = block_size_tokens
        self._blocks: dict[int, KVBlock] = {}
        self._sequences: dict[str, SequenceAllocation] = {}
        self._next_block_id = 0

    def allocate(self, request_id: str, token_count: int) -> SequenceAllocation:
        """Allocate a standalone sequence."""

        if request_id in self._sequences:
            raise ValueError(f"sequence {request_id!r} already exists")
        if token_count < 0:
            raise ValueError("token_count must be non-negative")
        allocation = SequenceAllocation(request_id, [], token_count)
        self._append_private_blocks(allocation, token_count)
        self._sequences[request_id] = allocation
        return allocation

    def fork(
        self,
        *,
        parent_request_id: str,
        child_request_id: str,
        shared_prefix_tokens: int,
        child_total_tokens: int,
    ) -> SequenceAllocation:
        """Create a child sequence sharing full prefix blocks with a parent."""

        if child_request_id in self._sequences:
            raise ValueError(f"sequence {child_request_id!r} already exists")
        parent = self._sequences[parent_request_id]
        shared_prefix_tokens = min(shared_prefix_tokens, parent.token_count, child_total_tokens)
        full_shared_blocks = shared_prefix_tokens // self.block_size_tokens
        copied_prefix_tokens = shared_prefix_tokens % self.block_size_tokens
        child = SequenceAllocation(
            child_request_id,
            [],
            child_total_tokens,
            shared_prefix_tokens=full_shared_blocks * self.block_size_tokens,
            copied_prefix_tokens=copied_prefix_tokens,
        )
        for block_id in parent.block_ids[:full_shared_blocks]:
            self._blocks[block_id].refcount += 1
            child.block_ids.append(block_id)
        remaining = child_total_tokens - child.shared_prefix_tokens
        self._append_private_blocks(child, remaining)
        self._sequences[child_request_id] = child
        return child

    def append(self, request_id: str, token_count: int) -> None:
        """Append decode tokens to an existing sequence using private blocks."""

        if token_count < 0:
            raise ValueError("token_count must be non-negative")
        sequence = self._sequences[request_id]
        if token_count == 0:
            return
        remaining = token_count
        if sequence.block_ids:
            tail = self._blocks[sequence.block_ids[-1]]
            free = self.block_size_tokens - tail.used_tokens
            if tail.refcount == 1 and free > 0:
                added = min(free, remaining)
                tail.used_tokens += added
                sequence.token_count += added
                remaining -= added
        self._append_private_blocks(sequence, remaining)
        sequence.token_count += remaining

    def release(self, request_id: str) -> None:
        """Release one sequence and free unreferenced physical blocks."""

        sequence = self._sequences.pop(request_id)
        for block_id in sequence.block_ids:
            block = self._blocks[block_id]
            block.refcount -= 1
            if block.refcount == 0:
                del self._blocks[block_id]

    def summary(self) -> KVSummary:
        """Return current memory accounting."""

        physical_blocks = len(self._blocks)
        physical_capacity = physical_blocks * self.block_size_tokens
        physical_used = sum(block.used_tokens for block in self._blocks.values())
        logical_tokens = sum(sequence.token_count for sequence in self._sequences.values())
        dense_capacity = sum(
            math.ceil(sequence.token_count / self.block_size_tokens) * self.block_size_tokens
            for sequence in self._sequences.values()
        )
        shared_tokens = sum(sequence.shared_prefix_tokens for sequence in self._sequences.values())
        copied_tokens = sum(sequence.copied_prefix_tokens for sequence in self._sequences.values())
        return KVSummary(
            block_size_tokens=self.block_size_tokens,
            active_sequences=len(self._sequences),
            physical_blocks=physical_blocks,
            physical_capacity_tokens=physical_capacity,
            physical_used_tokens=physical_used,
            logical_tokens=logical_tokens,
            dense_capacity_tokens=dense_capacity,
            shared_logical_tokens=shared_tokens,
            copied_prefix_tokens=copied_tokens,
            internal_fragmentation_tokens=physical_capacity - physical_used,
            estimated_capacity_tokens_saved_vs_dense=dense_capacity - physical_capacity,
            refcounted_blocks=sum(1 for block in self._blocks.values() if block.refcount > 1),
        )

    def _append_private_blocks(self, allocation: SequenceAllocation, token_count: int) -> None:
        remaining = token_count
        while remaining > 0:
            used = min(self.block_size_tokens, remaining)
            block = KVBlock(self._next_block_id, used)
            self._blocks[block.block_id] = block
            allocation.block_ids.append(block.block_id)
            self._next_block_id += 1
            remaining -= used


def deterministic_scenario(
    *,
    block_size_tokens: int,
    sequences: int,
    base_prompt_tokens: int,
    shared_prefix_tokens: int,
    decode_tokens: int,
) -> dict[str, Any]:
    """Run a deterministic prefix-sharing scenario."""

    manager = KVBlockManager(block_size_tokens)
    root = manager.allocate("request-0", base_prompt_tokens)
    allocations = [asdict(root)]
    for index in range(1, sequences):
        child = manager.fork(
            parent_request_id="request-0",
            child_request_id=f"request-{index}",
            shared_prefix_tokens=shared_prefix_tokens,
            child_total_tokens=base_prompt_tokens + index,
        )
        manager.append(child.request_id, decode_tokens)
        allocations.append(asdict(child))
    return {
        "allocations": allocations,
        "summary": asdict(manager.summary()),
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the KV block simulator parser."""

    parser = argparse.ArgumentParser(prog="orchard-kv-blocks")
    parser.add_argument("--block-size-tokens", type=int, default=16)
    parser.add_argument("--sequences", type=int, default=8)
    parser.add_argument("--base-prompt-tokens", type=int, default=128)
    parser.add_argument("--shared-prefix-tokens", type=int, default=96)
    parser.add_argument("--decode-tokens", type=int, default=16)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/kv-blocks.json"))
    return parser


def run() -> None:
    """Run the KV block simulator CLI."""

    args = build_parser().parse_args()
    if (
        args.block_size_tokens <= 0
        or args.sequences <= 0
        or args.base_prompt_tokens < 0
        or args.shared_prefix_tokens < 0
        or args.decode_tokens < 0
    ):
        raise SystemExit(
            "block size and sequence count must be positive; token counts non-negative"
        )
    result = deterministic_scenario(
        block_size_tokens=args.block_size_tokens,
        sequences=args.sequences,
        base_prompt_tokens=args.base_prompt_tokens,
        shared_prefix_tokens=args.shared_prefix_tokens,
        decode_tokens=args.decode_tokens,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "summary": result["summary"]}, indent=2))


if __name__ == "__main__":
    run()
