"""Alias resolution, retries, and failover — getting the request answered.

By the time a request reaches here it has been authenticated, allowed, rate
limited, budget checked, and missed the cache. This module's only job is to turn
"the caller asked for ``fast``" into an actual completion, surviving whatever
the providers are doing today.

Two independent recovery mechanisms, and the difference matters
---------------------------------------------------------------
**Retry** is for transient trouble — a timeout, a 429, a one-off 500. The same
model is asked again after a backoff, because the request was probably fine and
the provider was momentarily not.

**Failover** is for a provider that is actually broken. After a model has used
up its retries, the next model in the alias's chain is tried instead — a
different provider entirely. ``fast`` is ``alpha-small`` with ``beta-small``
behind it, so alpha being down degrades latency, not availability.

Backoff is exponential (200ms, 400ms, 800ms) because retrying a struggling
provider immediately, in a tight loop, is how you turn its bad minute into its
bad afternoon.

Errors are classified by the adapter, not here (see ``app/providers/base.py``).
Non-retryable failures skip straight to the next model — asking a provider a
second time to serve a model it doesn't have is just latency.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from app.config import GatewayConfig
from app.errors import ModelNotFoundError, UpstreamError
from app.pricing import PriceTable
from app.providers.base import ChatRequest, ChatResponse, ProviderError, StreamHandle
from app.providers.registry import ProviderRegistry

logger = logging.getLogger(__name__)


@dataclass
class Attempt:
    model: str
    provider: str
    error: str | None = None


@dataclass
class DispatchMeta:
    """What happened on the way to an answer — everything the log needs."""

    provider: str
    model: str
    fallback: bool
    retries: int
    attempts: list[Attempt] = field(default_factory=list)

    @property
    def provider_header(self) -> str:
        """Value of ``x-prism-provider``, e.g. ``alpha/alpha-small``."""
        return f"{self.provider}/{self.model}"


@dataclass
class DispatchResult:
    response: ChatResponse
    meta: DispatchMeta


class Dispatcher:
    def __init__(self, registry: ProviderRegistry, config: GatewayConfig, prices: PriceTable):
        self._registry = registry
        self._config = config
        self._prices = prices

    # -- resolution -------------------------------------------------------- #

    def alias(self, name: str):
        """The configured alias by name, or ``None`` if it isn't one."""
        return self._config.aliases.get(name)

    def is_known(self, requested_model: str) -> bool:
        """Does this gateway serve this alias or model at all?

        Checked before the allowlist so that a nonexistent model gets 404
        ("no such thing") rather than 403 ("not for you"), which would be a
        confusing thing to tell someone about a model that doesn't exist.
        """
        return requested_model in self._config.aliases or self._prices.has(requested_model)

    def known_names(self) -> list[str]:
        return [*sorted(self._config.aliases), *self._prices.models()]

    def resolve_chain(self, requested_model: str) -> tuple[str, ...]:
        """Turn what the caller asked for into an ordered list of models to try.

        Accepts an alias (``fast``) or a concrete model name (``alpha-small``).
        A concrete model gets no fallback chain — the caller pinned it on
        purpose, and silently answering from a different model would be a
        surprise they didn't ask for.
        """
        alias = self._config.aliases.get(requested_model)
        if alias is not None:
            if alias.is_auto:
                raise ValueError("'auto' must be resolved to a tier before dispatch")
            chain = tuple(m for m in alias.chain if m)
            if not chain:
                raise ModelNotFoundError(f"Alias '{requested_model}' has no models configured.")
            return chain

        if self._prices.has(requested_model):
            return (requested_model,)

        known = ", ".join([*sorted(self._config.aliases), *self._prices.models()])
        raise ModelNotFoundError(f"Unknown model or alias '{requested_model}'. Known: {known}.")

    def _provider_for(self, model: str):
        spec = self._config.provider_for_model(model)
        if spec is None:
            return None
        return self._registry.get(spec.name)

    async def _backoff(self, attempt_index: int) -> None:
        retry = self._config.retry
        delay_ms = retry.initial_backoff_ms * (retry.backoff_multiplier**attempt_index)
        await asyncio.sleep(delay_ms / 1000)

    # -- non-streaming ----------------------------------------------------- #

    async def complete(self, chain: tuple[str, ...], messages: list[dict], extra: dict) -> DispatchResult:
        """Try each model in the chain until one answers."""
        attempts: list[Attempt] = []
        retries = 0
        max_attempts = self._config.retry.max_attempts

        for chain_index, model in enumerate(chain):
            provider = self._provider_for(model)
            if provider is None:
                attempts.append(Attempt(model, "unknown", "no provider configured for this model"))
                continue

            request = ChatRequest(model=model, messages=messages, extra=extra)

            for attempt_index in range(max_attempts):
                try:
                    response = await provider.chat(request)
                except ProviderError as exc:
                    attempts.append(Attempt(model, provider.name, exc.message))
                    logger.warning(
                        "provider %s model %s failed (attempt %d/%d): %s",
                        provider.name, model, attempt_index + 1, max_attempts, exc.message,
                    )
                    if not exc.retryable:
                        break  # a different model may work; a repeat won't
                    if attempt_index + 1 < max_attempts:
                        retries += 1
                        await self._backoff(attempt_index)
                    continue

                attempts.append(Attempt(model, provider.name))
                return DispatchResult(
                    response=response,
                    meta=DispatchMeta(
                        provider=provider.name,
                        model=model,
                        fallback=chain_index > 0,
                        retries=retries,
                        attempts=attempts,
                    ),
                )

        raise UpstreamError(_exhausted_message(attempts, chain))

    # -- streaming --------------------------------------------------------- #

    @asynccontextmanager
    async def open_stream(
        self, chain: tuple[str, ...], messages: list[dict], extra: dict
    ) -> AsyncIterator[tuple[StreamHandle, DispatchMeta]]:
        """Open a stream, failing over only while that is still safe to do.

        The moment the first byte reaches the client, this provider is the one
        answering. A failure after that point is re-raised rather than retried:
        switching providers mid-response would splice two different answers
        together, which is worse than a truncated stream the client can detect.
        """
        attempts: list[Attempt] = []
        retries = 0
        max_attempts = self._config.retry.max_attempts

        for chain_index, model in enumerate(chain):
            provider = self._provider_for(model)
            if provider is None:
                attempts.append(Attempt(model, "unknown", "no provider configured for this model"))
                continue

            request = ChatRequest(model=model, messages=messages, extra=extra)

            for attempt_index in range(max_attempts):
                handle: StreamHandle | None = None
                try:
                    async with provider.open_stream(request) as opened:
                        handle = opened
                        attempts.append(Attempt(model, provider.name))
                        meta = DispatchMeta(
                            provider=provider.name,
                            model=model,
                            fallback=chain_index > 0,
                            retries=retries,
                            attempts=attempts,
                        )
                        yield handle, meta
                        return
                except ProviderError as exc:
                    if handle is not None and handle.chunks_sent > 0:
                        raise  # committed - the client already has part of this answer
                    attempts.append(Attempt(model, provider.name, exc.message))
                    logger.warning(
                        "provider %s model %s failed to open stream (attempt %d/%d): %s",
                        provider.name, model, attempt_index + 1, max_attempts, exc.message,
                    )
                    if not exc.retryable:
                        break
                    if attempt_index + 1 < max_attempts:
                        retries += 1
                        await self._backoff(attempt_index)
                    continue

        raise UpstreamError(_exhausted_message(attempts, chain))


def _exhausted_message(attempts: list[Attempt], chain: tuple[str, ...]) -> str:
    """Explain the failure without leaking provider credentials or URLs."""
    tried = "; ".join(f"{a.provider}/{a.model}: {a.error}" for a in attempts if a.error) or "none"
    return f"All providers failed for models [{', '.join(chain)}]. Attempts: {tried}"
