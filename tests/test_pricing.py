"""Cost computation — the number budgets are enforced against."""

from __future__ import annotations

import pytest

from app.pricing import format_cost

async def test_cost_uses_separate_input_and_output_rates(prices):
    # alpha-small: $0.15/1M in, $0.60/1M out
    cost = prices.cost_usd("alpha-small", 1_000_000, 1_000_000)
    assert cost == pytest.approx(0.75)


async def test_cost_scales_linearly_with_tokens(prices):
    assert prices.cost_usd("alpha-small", 1000, 0) == pytest.approx(0.00015)
    assert prices.cost_usd("alpha-small", 2000, 0) == pytest.approx(0.00030)


async def test_output_tokens_cost_more_than_input(prices):
    """Not an accounting detail — it's why routing to the right tier matters."""
    assert prices.cost_usd("alpha-large", 0, 1000) > prices.cost_usd("alpha-large", 1000, 0)


async def test_smart_tier_is_dramatically_more_expensive(prices):
    """The economic premise of the auto router, asserted rather than assumed."""
    cheap = prices.cost_usd("alpha-small", 1000, 1000)
    strong = prices.cost_usd("alpha-large", 1000, 1000)
    assert strong / cheap > 15


async def test_zero_tokens_costs_nothing(prices):
    assert prices.cost_usd("alpha-small", 0, 0) == 0.0


async def test_unknown_model_costs_zero_rather_than_raising(prices):
    """A missing price row must not turn a successful completion into an error."""
    assert prices.cost_usd("model-with-no-price", 1000, 1000) == 0.0


async def test_known_models_are_reported(prices):
    assert prices.has("alpha-small")
    assert not prices.has("nope")
    assert "beta-large" in prices.models()


@pytest.mark.parametrize(
    ("cost", "expected"),
    [(0.0, "0.00000000"), (1.23e-05, "0.00001230"), (12.5, "12.50000000")],
)
def test_cost_header_never_uses_scientific_notation(cost, expected):
    """The load test parses this header with float(); '1.23e-05' would still
    parse, but a header a human has to squint at is a bad contract."""
    assert format_cost(cost) == expected
    assert "e" not in format_cost(cost)
