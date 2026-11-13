"""Loading the provided data files into the store.

Runs on every startup and is idempotent — existing rows are updated in place
rather than duplicated, so restarting the gateway never resets a tenant's
accumulated spend.

Key values and alias names are preserved exactly as they appear in the seed
files, because the demo flow and the provided scripts reference them literally.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.models import ModelPrice, VirtualKey

logger = logging.getLogger(__name__)


def _strip_comments(payload: dict) -> dict:
    """The provided JSON files carry ``_comment`` keys; they aren't data."""
    return {k: v for k, v in payload.items() if not k.startswith("_")}


async def seed_model_prices(session: AsyncSession, path: Path) -> int:
    raw = _strip_comments(json.loads(path.read_text()))
    for model, prices in raw.items():
        existing = await session.get(ModelPrice, model)
        if existing is None:
            session.add(
                ModelPrice(
                    model=model,
                    input_per_1m=float(prices["input_per_1m"]),
                    output_per_1m=float(prices["output_per_1m"]),
                )
            )
        else:
            existing.input_per_1m = float(prices["input_per_1m"])
            existing.output_per_1m = float(prices["output_per_1m"])
    await session.commit()
    return len(raw)


async def seed_virtual_keys(session: AsyncSession, path: Path) -> int:
    raw = json.loads(path.read_text())
    tenants = raw.get("tenants", [])

    for tenant in tenants:
        rate_limit = tenant.get("rate_limit", {})
        cache = tenant.get("semantic_cache", {})
        existing = await session.get(VirtualKey, tenant["virtual_key"])

        fields = {
            "team": tenant["team"],
            "monthly_budget_usd": float(tenant["monthly_budget_usd"]),
            "requests_per_minute": int(rate_limit.get("requests_per_minute", 60)),
            "tokens_per_minute": rate_limit.get("tokens_per_minute"),
            "model_allowlist": list(tenant.get("model_allowlist", [])),
            "cache_enabled": bool(cache.get("enabled", False)),
            "cache_similarity_threshold": cache.get("similarity_threshold"),
        }

        if existing is None:
            session.add(VirtualKey(virtual_key=tenant["virtual_key"], **fields))
        else:
            # Policy is refreshed from the seed file; accumulated usage is not
            # touched, since it lives in usage_records.
            for name, value in fields.items():
                setattr(existing, name, value)

    await session.commit()
    return len(tenants)


async def seed_all(session: AsyncSession, settings: Settings) -> None:
    prices = await seed_model_prices(session, settings.model_pricing_path)
    keys = await seed_virtual_keys(session, settings.seed_keys_path)
    logger.info("Seeded %d model prices and %d virtual keys", prices, keys)
