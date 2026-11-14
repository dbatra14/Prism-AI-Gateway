"""Database engine and session management.

Prism uses one store for everything: virtual keys, usage counters, request logs
and cache entries. Fewer moving parts, and — more importantly — it lets the
rate-limit and budget checks be single atomic SQL statements rather than a
read-then-write across two systems (see ``app/limits.py``).

SQLite is the default so the gateway runs with no setup. Postgres is supported
by pointing ``PRISM_DATABASE_URL`` at it; the correctness-critical SQL is
written to behave identically on both.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from app.config import get_settings
from app.models import Base

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def _configure_sqlite(engine: AsyncEngine) -> None:
    """Make SQLite safe for concurrent writers.

    WAL lets readers run while a writer holds the lock, and busy_timeout makes
    a competing writer wait instead of immediately raising "database is locked".
    Without these, the load test's concurrent burst would fail on lock errors
    rather than on anything meaningful.
    """

    @event.listens_for(engine.sync_engine, "connect")
    def _set_pragmas(dbapi_connection, _record):  # noqa: ANN001
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=10000")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


# Connection pool sizing. The default (5 + 10 overflow) is too small for this
# workload and fails in a confusing way: under a concurrent burst, requests
# block waiting for a connection and then surface as slow 500s that look like
# provider timeouts but aren't. Two things drive the requirement up:
#
#   * a streaming request holds its request-scoped session for the whole
#     stream, and its finaliser opens a second one to record usage, so a
#     long stream can occupy two connections for seconds at a time;
#   * admission control commits several times per request.
#
# pool_timeout is short on purpose. If the pool really is exhausted, failing in
# a couple of seconds is far more useful than a 30-second hang.
POOL_SIZE = 20
MAX_OVERFLOW = 40
POOL_TIMEOUT_SECONDS = 5


def get_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        url = get_settings().database_url
        _engine = create_async_engine(
            url,
            echo=False,
            pool_pre_ping=True,
            pool_size=POOL_SIZE,
            max_overflow=MAX_OVERFLOW,
            pool_timeout=POOL_TIMEOUT_SECONDS,
        )
        if url.startswith("sqlite"):
            _configure_sqlite(_engine)
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(get_engine(), expire_on_commit=False)
    return _session_factory


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency yielding a session per request."""
    async with get_session_factory()() as session:
        yield session


def is_postgres() -> bool:
    return get_settings().database_url.startswith("postgresql")


async def init_db() -> None:
    """Create tables. Enables pgvector first when running on Postgres."""
    engine = get_engine()
    async with engine.begin() as conn:
        if is_postgres():
            from sqlalchemy import text

            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)


async def dispose_engine() -> None:
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _session_factory = None
