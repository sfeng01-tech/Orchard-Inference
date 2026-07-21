"""Radix-style prompt prefix analysis for scheduler hints and observability."""

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class PrefixRoute:
    """Result of routing one tokenized prompt through the prefix index."""

    prompt_tokens: int
    matched_tokens: int
    matched_ratio: float
    estimated_prefill_tokens_saved: int
    candidate_count: int
    route: str


@dataclass(slots=True)
class _Node:
    children: dict[int, "_Node"] = field(default_factory=dict)
    visits: int = 0
    terminal_count: int = 0


class PrefixRouter:
    """Bounded radix tree over token IDs.

    This is a control-plane router: it detects reusable prompt prefixes and
    estimates avoided prefill work. It does not claim runtime KV-cache reuse.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        min_match_tokens: int,
        max_prefix_tokens: int,
    ) -> None:
        self._enabled = enabled
        self._min_match_tokens = min_match_tokens
        self._max_prefix_tokens = max_prefix_tokens
        self._root = _Node()
        self.requests = 0
        self.matches = 0
        self.total_matched_tokens = 0
        self.total_saved_tokens = 0

    @property
    def enabled(self) -> bool:
        """Return whether prefix routing is active."""

        return self._enabled

    def observe(self, token_ids: tuple[int, ...]) -> PrefixRoute:
        """Route and insert a prompt token sequence."""

        self.requests += 1
        if not self._enabled or not token_ids:
            return PrefixRoute(len(token_ids), 0, 0.0, 0, 0, "disabled")
        bounded = token_ids[: self._max_prefix_tokens]
        matched, candidates = self._longest_match(bounded)
        saved = matched if matched >= self._min_match_tokens else 0
        route = "prefix_hit" if saved else "prefix_miss"
        if saved:
            self.matches += 1
            self.total_matched_tokens += matched
            self.total_saved_tokens += saved
        self._insert(bounded)
        return PrefixRoute(
            prompt_tokens=len(token_ids),
            matched_tokens=matched,
            matched_ratio=matched / len(token_ids) if token_ids else 0.0,
            estimated_prefill_tokens_saved=saved,
            candidate_count=candidates,
            route=route,
        )

    def _longest_match(self, token_ids: tuple[int, ...]) -> tuple[int, int]:
        node = self._root
        matched = 0
        candidates = 0
        for token in token_ids:
            child = node.children.get(token)
            if child is None:
                break
            matched += 1
            candidates = max(candidates, child.visits)
            node = child
        return matched, candidates

    def _insert(self, token_ids: tuple[int, ...]) -> None:
        node = self._root
        node.visits += 1
        for token in token_ids:
            node = node.children.setdefault(token, _Node())
            node.visits += 1
        node.terminal_count += 1
