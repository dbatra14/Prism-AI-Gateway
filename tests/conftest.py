"""Shared fixtures.

Each test gets its own SQLite file in a temp directory rather than an in-memory
database. In-memory SQLite is per-connection, so a pooled async engine would
hand different tests — and different concurrent tasks inside one test — entirely
different databases, which would quietly invalidate the concurrency tests.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import GatewayConfig, RetryConfig, DegradationConfig, ProviderConfig, AliasConfig
from app.models import Base, ModelPrice, VirtualKey
from app.pricing import PriceTable


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest_asyncio.fixture
async def engine(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}", pool_size=10)

    from sqlalchemy import event

    @event.listens_for(engine.sync_engine, "connect")
    def _pragmas(dbapi_connection, _record):  # noqa: ANN001
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=10000")
        cursor.close()

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def session_factory(engine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)


@pytest_asyncio.fixture
async def session(session_factory) -> AsyncIterator[AsyncSession]:
    async with session_factory() as s:
        yield s


@pytest_asyncio.fixture
async def prices(session) -> PriceTable:
    session.add_all(
        [
            ModelPrice(model="alpha-small", input_per_1m=0.15, output_per_1m=0.60),
            ModelPrice(model="alpha-large", input_per_1m=3.00, output_per_1m=15.00),
            ModelPrice(model="beta-small", input_per_1m=0.10, output_per_1m=0.40),
            ModelPrice(model="beta-large", input_per_1m=2.50, output_per_1m=10.00),
        ]
    )
    await session.commit()
    return await PriceTable.load(session)


def make_key(
    virtual_key: str = "k-test",
    *,
    budget: float = 10.0,
    rpm: int = 60,
    allowlist: list[str] | None = None,
    cache_enabled: bool = True,
    threshold: float = 0.85,
) -> VirtualKey:
    return VirtualKey(
        virtual_key=virtual_key,
        team="test",
        monthly_budget_usd=budget,
        requests_per_minute=rpm,
        model_allowlist=allowlist if allowlist is not None else ["fast", "smart", "auto"],
        cache_enabled=cache_enabled,
        cache_similarity_threshold=threshold,
        status="active",
    )


@pytest_asyncio.fixture
async def key(session) -> VirtualKey:
    k = make_key()
    session.add(k)
    await session.commit()
    return k


class FakeProviderFactory:
    """Stand-in providers for the end-to-end tests.

    Behaves like the mock HTTP providers — same model names, same OpenAI shapes,
    same word-count token accounting — but in-process and with failure injection
    that can be flipped mid-test. No sockets, so the suite stays fast and can't
    fail because a port was busy.
    """

    def __init__(self) -> None:
        self.providers = {name: _InProcessProvider(name) for name in ("alpha", "beta")}

    def fail(self, provider: str, error: Exception | None) -> None:
        self.providers[provider].injected_error = error

    def fail_all(self, error: Exception | None) -> None:
        for provider in self.providers.values():
            provider.injected_error = error


class _InProcessProvider:
    """Implements the Provider interface without any HTTP."""

    def __init__(self, name: str):
        self.name = name
        self.injected_error: Exception | None = None
        self.calls = 0

    def _reply(self, messages: list[dict], model: str) -> str:
        last_user = next(
            (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"),
            "your request",
        )
        return f"[{self.name}:{model}] Here is a concise answer about {str(last_user)[:40]}."

    async def chat(self, request):
        from app.providers.base import ChatResponse, Usage

        self.calls += 1
        if self.injected_error:
            raise self.injected_error

        reply = self._reply(request.messages, request.model)
        usage = Usage(
            prompt_tokens=sum(len(str(m.get("content", "")).split()) for m in request.messages),
            completion_tokens=len(reply.split()),
        )
        return ChatResponse(
            body={
                "id": "chatcmpl-fake",
                "object": "chat.completion",
                "model": request.model,
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": reply},
                     "finish_reason": "stop"}
                ],
                "usage": {
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens,
                },
            },
            usage=usage,
        )

    @asynccontextmanager
    async def open_stream(self, request):
        import json as _json

        from app.providers.base import StreamHandle, Usage

        self.calls += 1
        if self.injected_error:
            raise self.injected_error

        reply = self._reply(request.messages, request.model)
        usage = Usage(
            prompt_tokens=sum(len(str(m.get("content", "")).split()) for m in request.messages),
            completion_tokens=len(reply.split()),
        )
        handle = StreamHandle(request.model)

        async def lines():
            for word in reply.split(" "):
                chunk = {
                    "id": "chatcmpl-fake",
                    "object": "chat.completion.chunk",
                    "model": request.model,
                    "choices": [{"index": 0, "delta": {"content": word + " "}, "finish_reason": None}],
                }
                handle.completion_text.append(word + " ")
                yield f"data: {_json.dumps(chunk)}".encode()
                yield b""
            handle.usage = usage
            yield b"data: [DONE]"
            yield b""

        yield handle.bind(lines())

    async def health(self) -> bool:
        return self.injected_error is None

    async def aclose(self) -> None:
        return None


@pytest.fixture
def gateway_config() -> GatewayConfig:
    return GatewayConfig(
        providers=(
            ProviderConfig("alpha", "http://localhost:9001/v1", "secret-alpha"),
            ProviderConfig("beta", "http://localhost:9002/v1", "secret-beta"),
        ),
        aliases={
            "fast": AliasConfig("fast", "alpha-small", ("beta-small",)),
            "smart": AliasConfig("smart", "alpha-large", ("beta-large",)),
            "auto": AliasConfig(
                "auto", None, (), {"simple": "fast", "complex": "smart"}
            ),
        },
        retry=RetryConfig(max_attempts=3, initial_backoff_ms=1, backoff_multiplier=1.0),
        degradation=DegradationConfig(),
    )
