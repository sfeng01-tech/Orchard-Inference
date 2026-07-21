"""Bounded prompt-template and tokenization caches."""

import hashlib
import json
import time
import unicodedata
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

from orchard_inference.observability import Metrics


def stable_cache_key(namespace: str, value: object) -> str:
    """Return a versioned, deterministic SHA-256 cache key."""

    canonical = json.dumps(
        {"namespace": namespace, "version": 1, "value": value},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def normalize_prefix(value: str) -> str:
    """Normalize reusable prefix text without changing internal whitespace."""

    normalized = unicodedata.normalize("NFC", value).replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in normalized.split("\n")).strip()


@dataclass(slots=True)
class CacheEntry[ValueT]:
    """One cache value with estimated ownership metadata."""

    value: ValueT
    estimated_bytes: int
    created_at: float
    last_access: float


class BoundedLRU[ValueT]:
    """Entry- and byte-bounded LRU cache with lazy TTL expiration."""

    def __init__(
        self,
        *,
        max_entries: int,
        max_bytes: int,
        ttl_seconds: float,
        on_evict: Callable[[str, CacheEntry[ValueT]], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._max_entries = max_entries
        self._max_bytes = max_bytes
        self._ttl_seconds = ttl_seconds
        self._on_evict = on_evict
        self._clock = clock
        self._entries: OrderedDict[str, CacheEntry[ValueT]] = OrderedDict()
        self.estimated_bytes = 0

    def get(self, key: str) -> ValueT | None:
        """Return and promote a live entry, or remove an expired entry."""

        entry = self._entries.get(key)
        if entry is None:
            return None
        now = self._clock()
        if now - entry.created_at >= self._ttl_seconds:
            self._evict(key)
            return None
        entry.last_access = now
        self._entries.move_to_end(key)
        return entry.value

    def put(self, key: str, value: ValueT, estimated_bytes: int) -> bool:
        """Insert a value and evict LRU entries; return false if it cannot fit."""

        if estimated_bytes > self._max_bytes:
            return False
        if key in self._entries:
            self._evict(key)
        now = self._clock()
        self._entries[key] = CacheEntry(value, estimated_bytes, now, now)
        self.estimated_bytes += estimated_bytes
        while len(self._entries) > self._max_entries or self.estimated_bytes > self._max_bytes:
            self._evict(next(iter(self._entries)))
        return key in self._entries

    def _evict(self, key: str) -> None:
        entry = self._entries.pop(key)
        self.estimated_bytes -= entry.estimated_bytes
        if self._on_evict is not None:
            self._on_evict(key, entry)

    def clear(self) -> None:
        """Release every owned entry."""

        for key in tuple(self._entries):
            self._evict(key)

    def __len__(self) -> int:
        return len(self._entries)


class CacheManager:
    """Coordinate prompt-prefix and full-prompt tokenization caches."""

    def __init__(
        self,
        metrics: Metrics,
        *,
        max_entries: int,
        max_bytes: int,
        ttl_seconds: float,
        prompt_enabled: bool,
        tokenization_enabled: bool,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._metrics = metrics
        self._prompt_enabled = prompt_enabled
        self._tokenization_enabled = tokenization_enabled
        layer_bytes = max(1, max_bytes // 2)
        self._prompt = BoundedLRU[str](
            max_entries=max_entries,
            max_bytes=layer_bytes,
            ttl_seconds=ttl_seconds,
            on_evict=lambda key, entry: self._evicted("prompt_template"),
            clock=clock,
        )
        self._tokens = BoundedLRU[tuple[int, ...]](
            max_entries=max_entries,
            max_bytes=layer_bytes,
            ttl_seconds=ttl_seconds,
            on_evict=lambda key, entry: self._evicted("tokenization"),
            clock=clock,
        )

    def _event(self, layer: str, result: str) -> None:
        self._metrics.cache_operations.labels(layer=layer, result=result).inc()
        if result == "hit":
            self._metrics.cache_hits.inc()
        elif result == "miss":
            self._metrics.cache_misses.inc()

    def _evicted(self, layer: str) -> None:
        self._metrics.cache_evictions.inc()
        self._metrics.cache_operations.labels(layer=layer, result="eviction").inc()
        self._update_bytes()

    def _update_bytes(self) -> None:
        self._metrics.estimated_cache_bytes.set(
            self._prompt.estimated_bytes + self._tokens.estimated_bytes
        )

    def render_prompt(self, model: str, messages: Sequence[tuple[str, str]]) -> str:
        """Render a role transcript while caching a normalized leading system prefix."""

        system_messages = []
        remainder_index = 0
        for role, content in messages:
            if role != "system":
                break
            system_messages.append(normalize_prefix(content))
            remainder_index += 1
        prefix = ""
        if system_messages:
            key = stable_cache_key("prompt-template", [model, system_messages])
            cached_prefix = self._prompt.get(key) if self._prompt_enabled else None
            if cached_prefix is None:
                self._event("prompt_template", "miss")
                prefix = "\n".join(f"system: {content}" for content in system_messages) + "\n"
                if self._prompt_enabled:
                    self._prompt.put(key, prefix, len(prefix.encode()))
                    self._update_bytes()
            else:
                self._event("prompt_template", "hit")
                prefix = cached_prefix
        remainder = "\n".join(f"{role}: {content}" for role, content in messages[remainder_index:])
        return prefix + remainder + "\nassistant:"

    async def tokenize(
        self,
        model: str,
        prompt: str,
        tokenizer: Callable[[str], Awaitable[tuple[int, ...]]],
    ) -> tuple[int, ...]:
        """Return cached full-prompt token IDs or invoke the configured backend tokenizer."""

        key = stable_cache_key("tokenization", [model, prompt])
        cached = self._tokens.get(key) if self._tokenization_enabled else None
        if cached is not None:
            self._event("tokenization", "hit")
            return cached
        self._event("tokenization", "miss")
        tokens = await tokenizer(prompt)
        if self._tokenization_enabled:
            self._tokens.put(key, tokens, len(tokens) * 8)
            self._update_bytes()
        return tokens

    def clear(self) -> None:
        """Release both application-layer caches."""

        self._prompt.clear()
        self._tokens.clear()
        self._update_bytes()

    @property
    def estimated_bytes(self) -> int:
        """Return total application-owned estimated cache bytes."""

        return self._prompt.estimated_bytes + self._tokens.estimated_bytes
