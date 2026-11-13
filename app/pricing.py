"""Cost accounting.

One rule, applied everywhere:

    cost = prompt_tokens/1M * input_price + completion_tokens/1M * output_price

using the token counts the *provider* reported, never a local estimate. If the
gateway guessed at token counts, its bills would drift from the provider's, and
budgets would be enforced against a number that isn't real money.

Prices are loaded from data/model_pricing.json into the ``model_prices`` table
at startup and cached in memory here, because this runs on every request.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import ModelPrice

TOKENS_PER_PRICE_UNIT = 1_000_000


@dataclass(frozen=True)
class Price:
    input_per_1m: float
    output_per_1m: float


class PriceTable:
    """In-memory snapshot of the price table."""

    def __init__(self, prices: dict[str, Price]):
        self._prices = prices

    @classmethod
    async def load(cls, session: AsyncSession) -> PriceTable:
        rows = (await session.execute(select(ModelPrice))).scalars().all()
        return cls({r.model: Price(r.input_per_1m, r.output_per_1m) for r in rows})

    def has(self, model: str) -> bool:
        return model in self._prices

    def get(self, model: str) -> Price | None:
        return self._prices.get(model)

    def models(self) -> list[str]:
        return sorted(self._prices)

    def cost_usd(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Cost of one request. An unpriced model costs 0 and is reported as such.

        Returning 0 rather than raising is deliberate: a missing price row must
        not turn a successful completion into a client-visible failure. The
        gap shows up in the usage API instead, where an operator will see it.
        """
        price = self._prices.get(model)
        if price is None:
            return 0.0
        return (
            prompt_tokens / TOKENS_PER_PRICE_UNIT * price.input_per_1m
            + completion_tokens / TOKENS_PER_PRICE_UNIT * price.output_per_1m
        )


def format_cost(cost: float) -> str:
    """Header-friendly cost: fixed precision, never scientific notation.

    ``x-prism-cost-usd`` is parsed by the load test with ``float()``, and mock
    prices are small enough that ``repr()`` would emit ``1e-05``.
    """
    return f"{cost:.8f}"
