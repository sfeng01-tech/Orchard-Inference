import pytest

from orchard_inference.cache import BoundedLRU, CacheManager, normalize_prefix, stable_cache_key
from orchard_inference.observability import MetricMetadata, Metrics


def test_cache_key_is_stable_and_namespaced() -> None:
    assert stable_cache_key("tokens", {"b": 2, "a": 1}) == stable_cache_key(
        "tokens", {"a": 1, "b": 2}
    )
    assert stable_cache_key("tokens", "same") != stable_cache_key("prompt", "same")
    assert normalize_prefix("Cafe\u0301  \r\nnext ") == "Café\nnext"


def test_lru_eviction_respects_recent_access_and_bytes() -> None:
    evicted: list[str] = []
    cache = BoundedLRU[str](
        max_entries=2,
        max_bytes=6,
        ttl_seconds=60,
        on_evict=lambda key, entry: evicted.append(key),
    )
    assert cache.put("a", "A", 3)
    assert cache.put("b", "B", 3)
    assert cache.get("a") == "A"
    assert cache.put("c", "C", 3)
    assert cache.get("b") is None
    assert evicted == ["b"]
    assert cache.estimated_bytes == 6


def test_ttl_expiration_releases_entry() -> None:
    now = [0.0]
    cache = BoundedLRU[str](
        max_entries=2,
        max_bytes=10,
        ttl_seconds=5,
        clock=lambda: now[0],
    )
    cache.put("key", "value", 5)
    now[0] = 5
    assert cache.get("key") is None
    assert cache.estimated_bytes == 0


@pytest.mark.asyncio
async def test_prompt_and_tokenization_cache_hits() -> None:
    metrics = Metrics(MetricMetadata("mock", "fifo"))
    manager = CacheManager(
        metrics,
        max_entries=4,
        max_bytes=1024,
        ttl_seconds=60,
        prompt_enabled=True,
        tokenization_enabled=True,
    )
    messages = [("system", "Be concise."), ("user", "Hello")]
    first = manager.render_prompt("model", messages)
    second = manager.render_prompt("model", messages)
    calls = 0

    async def tokenizer(prompt: str) -> tuple[int, ...]:
        nonlocal calls
        calls += 1
        return tuple(range(len(prompt.split())))

    assert await manager.tokenize("model", first, tokenizer) == await manager.tokenize(
        "model", second, tokenizer
    )
    assert calls == 1
    rendered = metrics.render().decode()
    assert 'layer="prompt_template",result="hit"' in rendered
    assert 'layer="tokenization",result="hit"' in rendered


def test_oversized_entry_is_not_cached() -> None:
    cache = BoundedLRU[str](max_entries=2, max_bytes=4, ttl_seconds=60)
    assert not cache.put("large", "value", 5)
    assert len(cache) == 0
