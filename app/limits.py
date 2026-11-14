"""Admission control: who is calling, may they call this model, and can they afford it.

This is the part of the gateway most likely to be *subtly* wrong, so it is worth
reading closely. The pipeline runs in a fixed, documented order:

    1. authenticate      — is this a real, active virtual key?
    2. allowlist         — is this key permitted to use this model/alias?
    3. rate limit        — has it used up this minute's request budget?
    4. budget            — has it used up this month's dollar budget?

The order matters. Cheap, purely-local checks come first, so a request with a
bogus key never touches a counter. The rate limit is checked *before* the cache,
not after: a cache hit is still a request against the gateway, and a tenant that
can hammer the cache without limit can still hammer the gateway.

The race condition this module exists to avoid
----------------------------------------------
The naive rate limiter reads the counter, compares it to the limit, then writes
the counter back. Under concurrency, ten requests can all read "9 used, limit
10" before any of them writes — and all ten get admitted. That's over-admission,
and the provided load test checks for exactly it.

The fix is to never read-then-write. The check and the increment are a *single*
SQL statement whose WHERE clause carries the limit, so the database decides.
If the statement updates no row, the request is over the limit. This is atomic
on both SQLite and Postgres.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import DateTime, bindparam, delete, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.errors import (
    AuthenticationError,
    BudgetExceededError,
    ModelNotAllowedError,
    RateLimitError,
)
from app.models import RateLimitWindow, UsageRecord, VirtualKey, utcnow

# How long a rejected-by-budget key must wait: until next month.
BUDGET_PERIOD_FORMAT = "%Y-%m"


def current_period(now: datetime | None = None) -> str:
    """The monthly bucket usage is accumulated into, e.g. ``2026-07``."""
    return (now or datetime.now(timezone.utc)).strftime(BUDGET_PERIOD_FORMAT)


def current_window(now: float | None = None) -> int:
    """The unix minute the rate limiter is counting in."""
    return int((now if now is not None else time.time()) // 60)


# --------------------------------------------------------------------------- #
# 1. Authentication
# --------------------------------------------------------------------------- #


async def authenticate(session: AsyncSession, authorization: str | None) -> VirtualKey:
    """Resolve ``Authorization: Bearer <virtual-key>`` to a tenant.

    Every failure mode here returns the same 401 shape on purpose — telling a
    caller the difference between "no such key" and "that key is disabled" is
    free information for anyone probing the gateway.
    """
    if not authorization or not authorization.lower().startswith("bearer "):
        raise AuthenticationError("Missing or malformed Authorization header. Expected 'Bearer <virtual-key>'.")

    presented = authorization[len("bearer ") :].strip()
    if not presented:
        raise AuthenticationError("Missing virtual key.")

    key = await session.get(VirtualKey, presented)
    if key is None or key.status != "active":
        raise AuthenticationError("Invalid virtual key.")
    return key


# --------------------------------------------------------------------------- #
# 2. Allowlist
# --------------------------------------------------------------------------- #


def check_allowlist(key: VirtualKey, requested_model: str) -> None:
    """The allowlist is checked against what the *caller asked for*.

    That is, the alias (``fast``/``smart``/``auto``), not the concrete model it
    resolves to. A tenant granted ``fast`` should keep working if an operator
    later repoints ``fast`` at a different model — and a tenant *not* granted
    ``smart`` must not reach a smart model by way of ``auto``. That second case
    is enforced again after auto-routing picks a tier (see app/api/chat.py).
    """
    if not key.allows_model(requested_model):
        allowed = ", ".join(key.model_allowlist) or "none"
        raise ModelNotAllowedError(
            f"Model '{requested_model}' is not allowed for this key. Allowed: {allowed}."
        )


# --------------------------------------------------------------------------- #
# 3. Rate limit
# --------------------------------------------------------------------------- #

# INSERT-or-conditionally-INCREMENT, in one statement.
#
#   - First request of a minute: the INSERT succeeds with count = 1.
#   - Later requests: the insert conflicts, and DO UPDATE fires *only while*
#     the stored count is still below the limit.
#   - At the limit: the WHERE fails, no row is updated, RETURNING is empty.
#
# The database serialises these, so no two requests can both see the last free
# slot. Same text works on SQLite and Postgres.
_RATE_LIMIT_SQL = text(
    """
    INSERT INTO rate_limit_windows (virtual_key, window_start, count)
    VALUES (:virtual_key, :window_start, 1)
    ON CONFLICT (virtual_key, window_start) DO UPDATE
        SET count = rate_limit_windows.count + 1
        WHERE rate_limit_windows.count < :limit
    RETURNING count
    """
)


async def consume_rate_limit(
    session: AsyncSession, key: VirtualKey, *, now: float | None = None
) -> int:
    """Claim one request slot in the current minute, or raise.

    Returns the caller's position in the window (1-based), which is handy in
    tests and in the request log.
    """
    limit = key.requests_per_minute
    if limit <= 0:
        raise RateLimitError("This key is not permitted to make requests.")

    window = current_window(now)
    result = await session.execute(
        _RATE_LIMIT_SQL,
        {"virtual_key": key.virtual_key, "window_start": window, "limit": limit},
    )
    row = result.first()
    await session.commit()  # commit immediately: concurrent requests must see this

    if row is None:
        raise RateLimitError(
            f"Rate limit of {limit} requests per minute exceeded for this key. Retry next minute."
        )
    return int(row[0])


async def prune_rate_limit_windows(session: AsyncSession, *, keep_minutes: int = 5) -> int:
    """Drop counters for minutes that have passed. Called periodically, not per request."""
    cutoff = current_window() - keep_minutes
    result = await session.execute(
        delete(RateLimitWindow).where(RateLimitWindow.window_start < cutoff)
    )
    await session.commit()
    return result.rowcount or 0


# --------------------------------------------------------------------------- #
# 4. Budget
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BudgetStatus:
    period: str
    spent_usd: float
    budget_usd: float

    @property
    def remaining_usd(self) -> float:
        return self.budget_usd - self.spent_usd


async def check_budget(session: AsyncSession, key: VirtualKey) -> BudgetStatus:
    """Reject the request if this month's spend has already reached the cap.

    Known limitation, stated plainly rather than papered over: the exact cost of
    a request is not knowable until the provider reports its token counts, which
    happens *after* the request runs. So this is a pre-check against spend so
    far, and the actual cost is added afterwards by ``record_usage``.

    The consequence: requests admitted while spend is just under the cap can push
    the total over it, by at most the cost of the requests in flight at that
    moment. For streaming this window is wider, because the stream may run for
    seconds. This is the standard trade-off — the alternative is holding every
    request until all in-flight ones settle, which serialises the gateway. It is
    bounded and documented (see docs/ARCHITECTURE.md), not silent.
    """
    period = current_period()
    usage = await _get_usage_row(session, key.virtual_key, period)
    spent = usage.cost_usd if usage else 0.0

    if spent >= key.monthly_budget_usd:
        raise BudgetExceededError(
            f"Monthly budget of ${key.monthly_budget_usd:.5f} exhausted for this key "
            f"(spent ${spent:.6f} in {period})."
        )
    return BudgetStatus(period=period, spent_usd=spent, budget_usd=key.monthly_budget_usd)


async def _get_usage_row(session: AsyncSession, virtual_key: str, period: str) -> UsageRecord | None:
    stmt = select(UsageRecord).where(
        UsageRecord.virtual_key == virtual_key, UsageRecord.period == period
    )
    return (await session.execute(stmt)).scalar_one_or_none()


# Same insert-or-increment shape as the rate limiter, minus the conditional:
# usage is always recorded, it just has to be recorded *atomically* so two
# concurrent requests can't both read the old total and clobber each other.
_USAGE_INCREMENT_SQL = text(
    """
    INSERT INTO usage_records
        (virtual_key, period, requests, prompt_tokens, completion_tokens, cost_usd, cache_hits, updated_at)
    VALUES
        (:virtual_key, :period, 1, :prompt_tokens, :completion_tokens, :cost_usd, :cache_hit, :updated_at)
    ON CONFLICT (virtual_key, period) DO UPDATE SET
        requests          = usage_records.requests + 1,
        prompt_tokens     = usage_records.prompt_tokens + :prompt_tokens,
        completion_tokens = usage_records.completion_tokens + :completion_tokens,
        cost_usd          = usage_records.cost_usd + :cost_usd,
        cache_hits        = usage_records.cache_hits + :cache_hit,
        updated_at        = :updated_at
    """
).bindparams(bindparam("updated_at", type_=DateTime()))
# The typed bindparam matters: without it the raw datetime goes to the sqlite3
# driver's default adapter, which is deprecated in Python 3.12. Declaring the
# type lets SQLAlchemy's own dialect render it, on either backend.


async def record_usage(
    session: AsyncSession,
    virtual_key: str,
    *,
    prompt_tokens: int,
    completion_tokens: int,
    cost_usd: float,
    cache_hit: bool,
    period: str | None = None,
) -> None:
    """Add one request's real cost to the month's running total."""
    await session.execute(
        _USAGE_INCREMENT_SQL,
        {
            "virtual_key": virtual_key,
            "period": period or current_period(),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost_usd": cost_usd,
            "cache_hit": 1 if cache_hit else 0,
            "updated_at": utcnow(),
        },
    )
    await session.commit()
