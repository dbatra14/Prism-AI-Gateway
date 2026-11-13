"""The control plane: usage, logs, cache stats, provider health.

Separate from the data plane in both URL space and auth. Callers authenticate
with a *virtual key* and can only ever affect their own tenant; an operator
authenticates with the admin token and can see across tenants. Handing the two
audiences the same credential is how cross-tenant data leaks happen.

The admin token here is a single shared secret, which is enough for a
single-operator demo and is called out as a limitation in the README — a real
deployment wants per-operator identities and an audit trail.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from sqlalchemy import case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.db import get_session
from app.deps import get_app_settings, get_config, get_registry
from app.limits import current_period
from app.models import CacheEntry, RequestLog, UsageRecord, VirtualKey, utcnow

router = APIRouter(prefix="/admin", tags=["admin"])


async def require_admin(
    request: Request,
    x_admin_token: str | None = Header(default=None),
) -> None:
    """Guard every admin route."""
    settings: Settings = request.app.state.settings
    if x_admin_token != settings.admin_token:
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid X-Admin-Token header.",
        )


def _iso(value: datetime | None) -> str | None:
    """Serialise a stored naive-UTC timestamp as an explicit UTC instant."""
    return None if value is None else value.isoformat() + "Z"


# --------------------------------------------------------------------------- #
# Usage
# --------------------------------------------------------------------------- #


@router.get("/usage", dependencies=[Depends(require_admin)])
async def usage_summary(
    key: str | None = Query(default=None, description="Virtual key; omit for all keys"),
    from_: str | None = Query(default=None, alias="from", description="ISO date, inclusive"),
    to: str | None = Query(default=None, description="ISO date, inclusive"),
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Spend and token totals for a key over a window.

    Computed from ``request_logs`` when a date window is given, and from the
    pre-aggregated ``usage_records`` row otherwise. The aggregate is the one the
    budget check reads on every request, so the fast path stays a single row
    read; the log scan is only for ad-hoc windows.

    These totals are what the load test's client-side sums reconcile against.
    """
    if from_ or to:
        return await _usage_from_logs(session, key, from_, to)

    period = current_period()
    stmt = select(UsageRecord).where(UsageRecord.period == period)
    if key:
        stmt = stmt.where(UsageRecord.virtual_key == key)
    records = list((await session.execute(stmt)).scalars().all())

    return {
        "key": key,
        "period": period,
        "from": f"{period}-01",
        "to": _iso(utcnow()),
        "requests": sum(r.requests for r in records),
        "prompt_tokens": sum(r.prompt_tokens for r in records),
        "completion_tokens": sum(r.completion_tokens for r in records),
        "cost_usd": round(sum(r.cost_usd for r in records), 8),
        "cache_hits": sum(r.cache_hits for r in records),
        "by_key": [
            {
                "key": r.virtual_key,
                "requests": r.requests,
                "prompt_tokens": r.prompt_tokens,
                "completion_tokens": r.completion_tokens,
                "cost_usd": round(r.cost_usd, 8),
                "cache_hits": r.cache_hits,
            }
            for r in records
        ],
    }


async def _usage_from_logs(
    session: AsyncSession, key: str | None, from_: str | None, to: str | None
) -> dict:
    stmt = select(
        func.count(RequestLog.request_id),
        func.coalesce(func.sum(RequestLog.prompt_tokens), 0),
        func.coalesce(func.sum(RequestLog.completion_tokens), 0),
        func.coalesce(func.sum(RequestLog.cost_usd), 0.0),
        # Portable conditional count: CASE works identically on SQLite and Postgres.
        func.coalesce(func.sum(case((RequestLog.cache == "hit", 1), else_=0)), 0),
    ).where(RequestLog.status.in_(("ok", "cache_hit")))

    if key:
        stmt = stmt.where(RequestLog.virtual_key == key)
    if from_:
        stmt = stmt.where(RequestLog.created_at >= datetime.fromisoformat(from_))
    if to:
        # Inclusive of the whole end day.
        stmt = stmt.where(RequestLog.created_at < datetime.fromisoformat(to) + timedelta(days=1))

    requests, prompt_tokens, completion_tokens, cost, cache_hits = (
        await session.execute(stmt)
    ).one()

    return {
        "key": key,
        "from": from_,
        "to": to,
        "requests": int(requests),
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "cost_usd": round(float(cost), 8),
        "cache_hits": int(cache_hits),
    }


# --------------------------------------------------------------------------- #
# Logs
# --------------------------------------------------------------------------- #


@router.get("/logs", dependencies=[Depends(require_admin)])
async def request_logs(
    key: str | None = Query(default=None),
    status: str | None = Query(default=None),
    limit: int = Query(default=50, le=500),
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Recent requests, newest first."""
    stmt = select(RequestLog).order_by(RequestLog.created_at.desc()).limit(limit)
    if key:
        stmt = stmt.where(RequestLog.virtual_key == key)
    if status:
        stmt = stmt.where(RequestLog.status == status)

    rows = list((await session.execute(stmt)).scalars().all())
    return {
        "count": len(rows),
        "logs": [
            {
                "request_id": r.request_id,
                "virtual_key": r.virtual_key,
                "requested_model": r.requested_model,
                "resolved_provider": r.resolved_provider,
                "resolved_model": r.resolved_model,
                "status": r.status,
                "prompt_tokens": r.prompt_tokens,
                "completion_tokens": r.completion_tokens,
                "cost_usd": r.cost_usd,
                "cache": r.cache,
                "fallback": r.fallback,
                "route_reason": r.route_reason,
                "retries": r.retries,
                "latency_ms": r.latency_ms,
                "streamed": r.streamed,
                "error_type": r.error_type,
                "created_at": _iso(r.created_at),
            }
            for r in rows
        ],
    }


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #


@router.get("/cache/stats", dependencies=[Depends(require_admin)])
async def cache_stats(
    key: str | None = Query(default=None),
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Hits, misses and hit rate, from the request log.

    Derived from the log rather than kept as a counter so the number can't drift
    away from what actually happened.
    """
    stmt = select(RequestLog.cache, func.count()).where(
        RequestLog.status.in_(("ok", "cache_hit"))
    )
    if key:
        stmt = stmt.where(RequestLog.virtual_key == key)
    counts = dict((await session.execute(stmt.group_by(RequestLog.cache))).all())

    hits, misses = int(counts.get("hit", 0)), int(counts.get("miss", 0))
    total = hits + misses

    entries_stmt = select(func.count()).select_from(CacheEntry)
    if key:
        entries_stmt = entries_stmt.where(CacheEntry.virtual_key == key)
    entries = int((await session.execute(entries_stmt)).scalar_one())

    return {
        "key": key,
        "hits": hits,
        "misses": misses,
        "hit_rate": round(hits / total, 4) if total else 0.0,
        "entries": entries,
    }


# --------------------------------------------------------------------------- #
# Keys & providers
# --------------------------------------------------------------------------- #


@router.get("/keys", dependencies=[Depends(require_admin)])
async def list_keys(session: AsyncSession = Depends(get_session)) -> dict:
    """Tenants and their policy, with this month's spend against budget."""
    keys = list((await session.execute(select(VirtualKey))).scalars().all())
    period = current_period()
    usage = {
        r.virtual_key: r
        for r in (
            await session.execute(select(UsageRecord).where(UsageRecord.period == period))
        ).scalars()
    }

    return {
        "keys": [
            {
                "virtual_key": k.virtual_key,
                "team": k.team,
                "status": k.status,
                "monthly_budget_usd": k.monthly_budget_usd,
                "requests_per_minute": k.requests_per_minute,
                "model_allowlist": k.model_allowlist,
                "cache_enabled": k.cache_enabled,
                "cache_similarity_threshold": k.cache_similarity_threshold,
                "spent_usd": round(usage[k.virtual_key].cost_usd, 8) if k.virtual_key in usage else 0.0,
                "requests_this_period": usage[k.virtual_key].requests if k.virtual_key in usage else 0,
            }
            for k in keys
        ]
    }


@router.get("/providers/health", dependencies=[Depends(require_admin)])
async def provider_health(
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Per-provider status: a live probe plus what recent traffic looked like.

    "Reachable" and "healthy" are different questions. A provider can answer
    /health while failing every completion, so the error rate and latency come
    from real traffic in the degradation window, not from the probe.
    """
    registry = get_registry(request)
    config = get_config(request)
    settings = get_app_settings(request)
    window_start = utcnow() - timedelta(seconds=config.degradation.window_seconds)

    stmt = (
        select(
            RequestLog.resolved_provider,
            func.count(),
            func.coalesce(func.sum(case((RequestLog.status == "ok", 0), else_=1)), 0),
            func.coalesce(func.avg(RequestLog.latency_ms), 0),
            func.coalesce(func.max(RequestLog.latency_ms), 0),
        )
        .where(RequestLog.created_at >= window_start, RequestLog.resolved_provider.isnot(None))
        .group_by(RequestLog.resolved_provider)
    )
    traffic = {
        row[0]: {"requests": int(row[1]), "errors": int(row[2]), "avg": float(row[3]), "max": int(row[4])}
        for row in (await session.execute(stmt)).all()
    }

    results = []
    for provider in registry.all():
        reachable = await provider.health()
        stats = traffic.get(provider.name, {"requests": 0, "errors": 0, "avg": 0.0, "max": 0})
        error_rate = stats["errors"] / stats["requests"] if stats["requests"] else 0.0
        degraded = (
            stats["requests"] > 0
            and (
                error_rate >= config.degradation.error_rate_threshold
                or stats["max"] >= config.degradation.p95_latency_ms
            )
        )
        results.append(
            {
                "provider": provider.name,
                "reachable": reachable,
                "requests_in_window": stats["requests"],
                "error_rate": round(error_rate, 4),
                "avg_latency_ms": int(stats["avg"]),
                "max_latency_ms": stats["max"],
                "status": "unreachable" if not reachable else ("degraded" if degraded else "healthy"),
            }
        )

    return {"window_seconds": config.degradation.window_seconds, "providers": results,
            "timeout_seconds": settings.provider_timeout_seconds}


@router.get("/config", dependencies=[Depends(require_admin)])
async def gateway_config(request: Request) -> dict:
    """The active routing config — aliases, chains, retry policy.

    Provider credentials are deliberately absent from this response.
    """
    config = get_config(request)
    return {
        "providers": [{"name": p.name, "base_url": p.base_url} for p in config.providers],
        "aliases": {
            name: {
                "primary": a.primary,
                "fallbacks": list(a.fallbacks),
                "route_by_difficulty": a.route_by_difficulty or None,
            }
            for name, a in config.aliases.items()
        },
        "retry": {
            "max_attempts": config.retry.max_attempts,
            "initial_backoff_ms": config.retry.initial_backoff_ms,
            "backoff_multiplier": config.retry.backoff_multiplier,
        },
    }
