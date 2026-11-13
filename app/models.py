"""Database schema — the six entities Prism persists.

    VirtualKey       one per tenant: budget, rate limit, allowlist, cache policy
    ModelPrice       price table, loaded from data/model_pricing.json
    UsageRecord      per key, per month: the running spend/token totals
    RateLimitWindow  per key, per minute: the request counter
    RequestLog       one row per data-plane request, including rejected ones
    CacheEntry       per key: a prompt embedding and the response it produced

UsageRecord and RateLimitWindow are deliberately *aggregates*, not derived from
RequestLog. Budget and rate-limit checks happen on every request, so they must
be a single indexed row read — never a scan over the log table.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utcnow() -> datetime:
    """Naive UTC — every timestamp in this database is UTC, with no offset stored.

    SQLite's DATETIME type discards tzinfo on write and hands back naive values
    on read, so a ``timezone=True`` column would quietly mean different things on
    SQLite and Postgres. Storing naive UTC everywhere makes the two backends
    behave identically; the API appends the ``Z`` when serialising.
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass


class VirtualKey(Base):
    """A tenant. Seeded from data/seed_keys.json; key values preserved verbatim."""

    __tablename__ = "virtual_keys"

    virtual_key: Mapped[str] = mapped_column(String(128), primary_key=True)
    team: Mapped[str] = mapped_column(String(64), nullable=False)
    monthly_budget_usd: Mapped[float] = mapped_column(Float, nullable=False)
    requests_per_minute: Mapped[int] = mapped_column(Integer, nullable=False)
    tokens_per_minute: Mapped[int | None] = mapped_column(Integer, nullable=True)
    model_allowlist: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    cache_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    cache_similarity_threshold: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="active")
    created_at: Mapped[datetime] = mapped_column(DateTime(), default=utcnow)

    def allows_model(self, requested_model: str) -> bool:
        return requested_model in self.model_allowlist


class ModelPrice(Base):
    """USD per 1M tokens. The single source of truth for cost accounting."""

    __tablename__ = "model_prices"

    model: Mapped[str] = mapped_column(String(64), primary_key=True)
    input_per_1m: Mapped[float] = mapped_column(Float, nullable=False)
    output_per_1m: Mapped[float] = mapped_column(Float, nullable=False)


class UsageRecord(Base):
    """Running totals per key per month — the row the budget check reads and writes."""

    __tablename__ = "usage_records"
    __table_args__ = (UniqueConstraint("virtual_key", "period", name="uq_usage_key_period"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    virtual_key: Mapped[str] = mapped_column(
        String(128), ForeignKey("virtual_keys.virtual_key"), nullable=False, index=True
    )
    period: Mapped[str] = mapped_column(String(7), nullable=False)  # "YYYY-MM"
    requests: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    prompt_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    cache_hits: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    updated_at: Mapped[datetime] = mapped_column(DateTime(), default=utcnow, onupdate=utcnow)


class RateLimitWindow(Base):
    """One counter row per key per minute — a fixed-window rate limiter.

    The counter is incremented by a conditional UPDATE that only matches while
    the count is below the limit, which is what makes it race-free. See
    ``app/limits.py``. Old windows are pruned rather than kept forever.
    """

    __tablename__ = "rate_limit_windows"
    __table_args__ = (UniqueConstraint("virtual_key", "window_start", name="uq_rl_key_window"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    virtual_key: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    window_start: Mapped[int] = mapped_column(Integer, nullable=False)  # unix epoch minute
    count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class RequestLog(Base):
    """One row per data-plane request — the trace needed to debug AI traffic.

    Rejections are logged too: a request that never reached a provider is
    exactly the kind of thing you need to see when a tenant reports errors.
    """

    __tablename__ = "request_logs"
    __table_args__ = (Index("ix_logs_key_created", "virtual_key", "created_at"),)

    request_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    virtual_key: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    requested_model: Mapped[str | None] = mapped_column(String(64), nullable=True)
    resolved_provider: Mapped[str | None] = mapped_column(String(64), nullable=True)
    resolved_model: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    prompt_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    cache: Mapped[str] = mapped_column(String(8), nullable=False, default="miss")
    fallback: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    route_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    retries: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    streamed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    error_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(), default=utcnow, index=True)


class CacheEntry(Base):
    """A cached response, scoped to one virtual key.

    ``virtual_key`` is part of every lookup query — that scoping is what stops
    one tenant's answer being served to another. The embedding is stored as raw
    float32 bytes, which both backends can hold; on Postgres with pgvector the
    same vector is also written to an indexed vector column.
    """

    __tablename__ = "cache_entries"
    __table_args__ = (Index("ix_cache_scope", "virtual_key", "model", "prefix_hash"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    virtual_key: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    model: Mapped[str] = mapped_column(String(64), nullable=False)
    # Hash of the conversation turns *before* the last user message. Only
    # requests with an identical prefix can match each other — see
    # app/cache/semantic_cache.py for why a follow-up question can't be matched
    # on its own text.
    prefix_hash: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    prompt_text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    response_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    prompt_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    hit_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(), default=utcnow, index=True)
