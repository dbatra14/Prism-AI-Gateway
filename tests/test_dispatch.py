"""Routing, retries and failover.

These use fake providers rather than the mock HTTP servers, so each failure mode
(timeout, 429, 5xx, malformed request) can be produced exactly and on demand.
The adapter's job of *translating* real HTTP into those failures is covered in
test_providers.py.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest

from app.dispatch import Dispatcher
from app.errors import ModelNotFoundError, UpstreamError
from app.providers.base import (
    ChatRequest,
    ChatResponse,
    Provider,
    ProviderBadRequest,
    ProviderRateLimited,
    ProviderTimeout,
    ProviderUnavailable,
    StreamHandle,
    Usage,
)
from app.providers.registry import ProviderRegistry


class FakeProvider(Provider):
    """A provider whose failures are scripted.

    ``failures`` is consumed one entry per call: an exception is raised, or
    ``None`` means answer normally. So ``[ProviderTimeout(...), None]`` is
    "fail once, then succeed".
    """

    def __init__(self, name: str, failures: list[Exception | None] | None = None,
                 *, stream_fail_after: int | None = None):
        self.name = name
        self.failures = list(failures or [])
        self.calls = 0
        self.stream_fail_after = stream_fail_after

    def _next_failure(self) -> Exception | None:
        self.calls += 1
        return self.failures.pop(0) if self.failures else None

    async def chat(self, request: ChatRequest) -> ChatResponse:
        failure = self._next_failure()
        if failure:
            raise failure
        return ChatResponse(
            body={
                "id": "chatcmpl-fake",
                "model": request.model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20},
            },
            usage=Usage(prompt_tokens=10, completion_tokens=20),
        )

    @asynccontextmanager
    async def open_stream(self, request: ChatRequest) -> AsyncIterator[StreamHandle]:
        failure = self._next_failure()
        if failure:
            raise failure

        fail_after = self.stream_fail_after

        async def lines() -> AsyncIterator[bytes]:
            for index in range(5):
                if fail_after is not None and index == fail_after:
                    raise ProviderUnavailable("connection dropped mid-stream")
                yield f"data: chunk-{index}".encode()

        yield StreamHandle(request.model).bind(lines())

    async def health(self) -> bool:
        return True


def build(config, providers: dict[str, Provider], prices) -> Dispatcher:
    return Dispatcher(ProviderRegistry(providers), config, prices)


# --------------------------------------------------------------------------- #
# Alias resolution
# --------------------------------------------------------------------------- #


def test_alias_resolves_to_primary_then_fallbacks(gateway_config, prices):
    dispatcher = build(gateway_config, {}, prices)
    assert dispatcher.resolve_chain("fast") == ("alpha-small", "beta-small")
    assert dispatcher.resolve_chain("smart") == ("alpha-large", "beta-large")


def test_concrete_model_gets_no_fallback_chain(gateway_config, prices):
    """A caller who pinned a model gets that model or an error — never a
    silent substitution they didn't ask for."""
    assert build(gateway_config, {}, prices).resolve_chain("alpha-large") == ("alpha-large",)


def test_unknown_model_raises_not_found(gateway_config, prices):
    with pytest.raises(ModelNotFoundError):
        build(gateway_config, {}, prices).resolve_chain("no-such-model")


def test_is_known_covers_aliases_and_models(gateway_config, prices):
    dispatcher = build(gateway_config, {}, prices)
    assert dispatcher.is_known("fast") and dispatcher.is_known("beta-large")
    assert not dispatcher.is_known("nope")


def test_auto_must_be_resolved_before_dispatch(gateway_config, prices):
    """`auto` is a routing decision, not a destination."""
    with pytest.raises(ValueError):
        build(gateway_config, {}, prices).resolve_chain("auto")


# --------------------------------------------------------------------------- #
# Retries
# --------------------------------------------------------------------------- #


async def test_transient_failure_is_retried_on_the_same_model(gateway_config, prices):
    alpha = FakeProvider("alpha", [ProviderTimeout("slow"), None])
    dispatcher = build(gateway_config, {"alpha": alpha, "beta": FakeProvider("beta")}, prices)

    result = await dispatcher.complete(("alpha-small", "beta-small"), [], {})

    assert result.meta.model == "alpha-small"
    assert result.meta.retries == 1
    assert result.meta.fallback is False  # recovered without leaving the primary
    assert alpha.calls == 2


@pytest.mark.parametrize(
    "error",
    [ProviderTimeout("timeout"), ProviderRateLimited("429"), ProviderUnavailable("503")],
)
async def test_all_transient_error_types_are_retried(gateway_config, prices, error):
    alpha = FakeProvider("alpha", [error, None])
    dispatcher = build(gateway_config, {"alpha": alpha, "beta": FakeProvider("beta")}, prices)

    await dispatcher.complete(("alpha-small", "beta-small"), [], {})
    assert alpha.calls == 2


async def test_retries_are_capped(gateway_config, prices):
    alpha = FakeProvider("alpha", [ProviderTimeout("t")] * 10)
    beta = FakeProvider("beta")
    dispatcher = build(gateway_config, {"alpha": alpha, "beta": beta}, prices)

    await dispatcher.complete(("alpha-small", "beta-small"), [], {})
    assert alpha.calls == gateway_config.retry.max_attempts  # then it gave up and moved on


async def test_non_retryable_error_skips_straight_to_fallback(gateway_config, prices):
    """Asking a provider twice for a model it doesn't serve is just latency."""
    alpha = FakeProvider("alpha", [ProviderBadRequest("no such model", 404)])
    beta = FakeProvider("beta")
    dispatcher = build(gateway_config, {"alpha": alpha, "beta": beta}, prices)

    result = await dispatcher.complete(("alpha-small", "beta-small"), [], {})

    assert alpha.calls == 1  # not retried
    assert result.meta.provider == "beta"


# --------------------------------------------------------------------------- #
# Failover
# --------------------------------------------------------------------------- #


async def test_fails_over_when_the_primary_is_down(gateway_config, prices):
    alpha = FakeProvider("alpha", [ProviderUnavailable("down")] * 5)
    beta = FakeProvider("beta")
    dispatcher = build(gateway_config, {"alpha": alpha, "beta": beta}, prices)

    result = await dispatcher.complete(("alpha-small", "beta-small"), [], {})

    assert result.meta.provider == "beta"
    assert result.meta.model == "beta-small"
    assert result.meta.fallback is True
    assert result.meta.provider_header == "beta/beta-small"


async def test_upstream_error_when_every_provider_fails(gateway_config, prices):
    dispatcher = build(
        gateway_config,
        {"alpha": FakeProvider("alpha", [ProviderUnavailable("down")] * 5),
         "beta": FakeProvider("beta", [ProviderUnavailable("down")] * 5)},
        prices,
    )

    with pytest.raises(UpstreamError) as exc:
        await dispatcher.complete(("alpha-small", "beta-small"), [], {})
    assert exc.value.status_code == 502


async def test_failure_message_never_leaks_credentials(gateway_config, prices):
    dispatcher = build(
        gateway_config,
        {"alpha": FakeProvider("alpha", [ProviderUnavailable("down")] * 5),
         "beta": FakeProvider("beta", [ProviderUnavailable("down")] * 5)},
        prices,
    )

    with pytest.raises(UpstreamError) as exc:
        await dispatcher.complete(("alpha-small", "beta-small"), [], {})

    message = str(exc.value)
    assert "secret-alpha" not in message and "secret-beta" not in message


async def test_no_provider_configured_is_skipped(gateway_config, prices):
    """A model with no registered provider must not stall the chain."""
    dispatcher = build(gateway_config, {"beta": FakeProvider("beta")}, prices)
    result = await dispatcher.complete(("alpha-small", "beta-small"), [], {})
    assert result.meta.provider == "beta"


# --------------------------------------------------------------------------- #
# Streaming
# --------------------------------------------------------------------------- #


async def test_stream_fails_over_before_the_first_byte(gateway_config, prices):
    alpha = FakeProvider("alpha", [ProviderUnavailable("down")] * 5)
    dispatcher = build(gateway_config, {"alpha": alpha, "beta": FakeProvider("beta")}, prices)

    async with dispatcher.open_stream(("alpha-small", "beta-small"), [], {}) as (handle, meta):
        chunks = [line async for line in handle]

    assert meta.fallback is True
    assert meta.provider == "beta"
    assert len(chunks) == 5


async def test_stream_does_not_fail_over_after_bytes_are_sent(gateway_config, prices):
    """The safety rule: once the client has part of an answer, switching
    providers would splice two different answers together."""
    alpha = FakeProvider("alpha", stream_fail_after=2)
    beta = FakeProvider("beta")
    dispatcher = build(gateway_config, {"alpha": alpha, "beta": beta}, prices)

    received = []
    with pytest.raises(ProviderUnavailable):
        async with dispatcher.open_stream(("alpha-small", "beta-small"), [], {}) as (handle, _):
            async for line in handle:
                received.append(line)

    assert len(received) == 2  # partial answer delivered
    assert beta.calls == 0  # and beta was never asked to finish it


async def test_stream_usage_is_available_after_completion(gateway_config, prices):
    dispatcher = build(gateway_config, {"alpha": FakeProvider("alpha")}, prices)
    async with dispatcher.open_stream(("alpha-small",), [], {}) as (handle, _):
        async for _ in handle:
            pass
        assert handle.chunks_sent == 5
