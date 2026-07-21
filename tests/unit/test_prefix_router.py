from orchard_inference.prefix_router import PrefixRouter


def test_prefix_router_reports_miss_then_hit_for_repeated_prefix() -> None:
    router = PrefixRouter(enabled=True, min_match_tokens=3, max_prefix_tokens=16)

    first = router.observe((1, 2, 3, 4, 5))
    second = router.observe((1, 2, 3, 9, 10))

    assert first.route == "prefix_miss"
    assert first.estimated_prefill_tokens_saved == 0
    assert second.route == "prefix_hit"
    assert second.matched_tokens == 3
    assert second.estimated_prefill_tokens_saved == 3
    assert second.candidate_count == 1
    assert router.matches == 1
    assert router.total_saved_tokens == 3


def test_prefix_router_respects_minimum_match_threshold() -> None:
    router = PrefixRouter(enabled=True, min_match_tokens=4, max_prefix_tokens=16)

    router.observe((1, 2, 3, 4, 5))
    route = router.observe((1, 2, 3, 9, 10))

    assert route.route == "prefix_miss"
    assert route.matched_tokens == 3
    assert route.estimated_prefill_tokens_saved == 0


def test_prefix_router_can_be_disabled() -> None:
    router = PrefixRouter(enabled=False, min_match_tokens=1, max_prefix_tokens=16)

    assert router.observe((1, 2, 3)).route == "disabled"
    assert router.observe((1, 2, 3)).matched_tokens == 0
