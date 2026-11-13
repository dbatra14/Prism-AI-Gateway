"""Admission control: the tests that matter most.

The rate limiter and the usage counter are the two places where a plausible
implementation is silently wrong under concurrency. These tests run real
concurrent tasks against a real database rather than asserting on mocks,
because the bug being guarded against is a race, and a mock cannot have one.
"""

from __future__ import annotations

import asyncio

import pytest

from app import limits
from app.errors import (
    AuthenticationError,
    BudgetExceededError,
    ModelNotAllowedError,
    RateLimitError,
)
from app.models import VirtualKey
from tests.conftest import make_key

# --------------------------------------------------------------------------- #
# Authentication
# --------------------------------------------------------------------------- #


async def test_valid_key_authenticates(session, key):
    resolved = await limits.authenticate(session, f"Bearer {key.virtual_key}")
    assert resolved.virtual_key == key.virtual_key


@pytest.mark.parametrize(
    "header",
    [None, "", "not-bearer", "Bearer ", "Bearer wrong-key", "Basic abc"],
)
async def test_bad_credentials_are_rejected(session, key, header):
    with pytest.raises(AuthenticationError):
        await limits.authenticate(session, header)


async def test_bearer_scheme_is_case_insensitive(session, key):
    resolved = await limits.authenticate(session, f"bearer {key.virtual_key}")
    assert resolved.virtual_key == key.virtual_key


async def test_disabled_key_is_rejected(session):
    disabled = make_key("k-disabled")
    disabled.status = "disabled"
    session.add(disabled)
    await session.commit()

    with pytest.raises(AuthenticationError):
        await limits.authenticate(session, "Bearer k-disabled")


# --------------------------------------------------------------------------- #
# Allowlist
# --------------------------------------------------------------------------- #


def test_allowlist_permits_listed_model():
    limits.check_allowlist(make_key(allowlist=["fast"]), "fast")


def test_allowlist_rejects_unlisted_model():
    with pytest.raises(ModelNotAllowedError):
        limits.check_allowlist(make_key(allowlist=["fast"]), "smart")


def test_allowlist_error_does_not_leak_other_tenants_models():
    key = make_key(allowlist=["fast"])
    with pytest.raises(ModelNotAllowedError) as exc:
        limits.check_allowlist(key, "smart")
    assert "fast" in str(exc.value)  # tells the caller what they *can* use


# --------------------------------------------------------------------------- #
# Rate limiting
# --------------------------------------------------------------------------- #


async def test_rate_limit_admits_up_to_the_limit(session):
    key = make_key("k-rl", rpm=5)
    session.add(key)
    await session.commit()

    for expected in range(1, 6):
        assert await limits.consume_rate_limit(session, key) == expected

    with pytest.raises(RateLimitError):
        await limits.consume_rate_limit(session, key)


async def test_rate_limit_does_not_over_admit_under_concurrency(session_factory):
    """The core anti-race test: 50 simultaneous requests against a limit of 10.

    A read-then-write limiter passes the sequential test above and fails this
    one. Exactly 10 must be admitted — not 11, not 50.
    """
    async with session_factory() as setup:
        setup.add(make_key("k-burst", rpm=10))
        await setup.commit()

    async def attempt() -> bool:
        async with session_factory() as s:
            key = await s.get(VirtualKey, "k-burst")
            try:
                await limits.consume_rate_limit(s, key)
                return True
            except RateLimitError:
                return False

    results = await asyncio.gather(*[attempt() for _ in range(50)])
    assert sum(results) == 10


async def test_rate_limit_windows_are_independent(session):
    key = make_key("k-window", rpm=2)
    session.add(key)
    await session.commit()

    now = 1_000_000.0
    await limits.consume_rate_limit(session, key, now=now)
    await limits.consume_rate_limit(session, key, now=now)
    with pytest.raises(RateLimitError):
        await limits.consume_rate_limit(session, key, now=now)

    # A minute later the quota is fresh again.
    assert await limits.consume_rate_limit(session, key, now=now + 60) == 1


async def test_rate_limit_is_per_key(session):
    session.add_all([make_key("k-a", rpm=1), make_key("k-b", rpm=1)])
    await session.commit()
    a = await session.get(VirtualKey, "k-a")
    b = await session.get(VirtualKey, "k-b")

    await limits.consume_rate_limit(session, a)
    # b has its own counter and must be unaffected by a exhausting theirs.
    assert await limits.consume_rate_limit(session, b) == 1


async def test_zero_rpm_key_cannot_call(session):
    key = make_key("k-zero", rpm=0)
    session.add(key)
    await session.commit()
    with pytest.raises(RateLimitError):
        await limits.consume_rate_limit(session, key)


async def test_pruning_removes_only_stale_windows(session):
    key = make_key("k-prune", rpm=10)
    session.add(key)
    await session.commit()

    await limits.consume_rate_limit(session, key, now=0)  # ancient window
    await limits.consume_rate_limit(session, key)  # current window

    removed = await limits.prune_rate_limit_windows(session)
    assert removed == 1
    # The current window survived, so the key keeps its consumed quota.
    assert await limits.consume_rate_limit(session, key) == 2


# --------------------------------------------------------------------------- #
# Budget
# --------------------------------------------------------------------------- #


async def test_budget_allows_spending_under_the_cap(session):
    key = make_key("k-budget", budget=1.0)
    session.add(key)
    await session.commit()

    await limits.record_usage(
        session, "k-budget", prompt_tokens=10, completion_tokens=10,
        cost_usd=0.25, cache_hit=False,
    )
    status = await limits.check_budget(session, key)
    assert status.spent_usd == pytest.approx(0.25)
    assert status.remaining_usd == pytest.approx(0.75)


async def test_budget_rejects_once_the_cap_is_reached(session):
    key = make_key("k-broke", budget=0.10)
    session.add(key)
    await session.commit()

    await limits.record_usage(
        session, "k-broke", prompt_tokens=1, completion_tokens=1,
        cost_usd=0.10, cache_hit=False,
    )
    with pytest.raises(BudgetExceededError):
        await limits.check_budget(session, key)


async def test_usage_increments_are_not_lost_under_concurrency(session_factory):
    """Concurrent cost recording must sum exactly.

    A read-modify-write on the usage row would drop increments here, and the
    gateway would under-bill — the failure mode that lets a tenant blow through
    their budget while the dashboard says they're fine.
    """
    async with session_factory() as setup:
        setup.add(make_key("k-sum", budget=100.0))
        await setup.commit()

    async def add_one() -> None:
        async with session_factory() as s:
            await limits.record_usage(
                s, "k-sum", prompt_tokens=7, completion_tokens=3,
                cost_usd=0.01, cache_hit=False,
            )

    await asyncio.gather(*[add_one() for _ in range(60)])

    async with session_factory() as s:
        key = await s.get(VirtualKey, "k-sum")
        status = await limits.check_budget(s, key)
        assert status.spent_usd == pytest.approx(0.60)

        row = await limits._get_usage_row(s, "k-sum", limits.current_period())
        assert row.requests == 60
        assert row.prompt_tokens == 420
        assert row.completion_tokens == 180


async def test_cache_hits_are_counted_separately(session):
    session.add(make_key("k-cache"))
    await session.commit()

    await limits.record_usage(
        session, "k-cache", prompt_tokens=5, completion_tokens=5, cost_usd=0.0, cache_hit=True
    )
    await limits.record_usage(
        session, "k-cache", prompt_tokens=5, completion_tokens=5, cost_usd=0.02, cache_hit=False
    )

    row = await limits._get_usage_row(session, "k-cache", limits.current_period())
    assert row.requests == 2
    assert row.cache_hits == 1
    # The cache hit contributed tokens but no cost — that gap is the saving.
    assert row.cost_usd == pytest.approx(0.02)
    assert row.prompt_tokens == 10
