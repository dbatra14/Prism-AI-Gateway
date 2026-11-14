"""The provider abstraction.

Core gateway code never imports an SDK or knows a provider's URL shape. It only
ever calls ``Provider.chat()`` or ``Provider.open_stream()``. Adding a real
provider means writing one adapter class — nothing in routing, metering,
caching, or limits changes.

The other job of this layer is to translate every upstream failure into one of
the ``ProviderError`` subclasses below, each tagged ``retryable`` or not. The
dispatcher retries and fails over purely on that flag, so it never has to know
whether a given provider signals overload with a 429, a 503, or a dropped
connection.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class ProviderError(Exception):
    """Base class for anything that went wrong upstream."""

    retryable: bool = False

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class ProviderTimeout(ProviderError):
    """The provider did not answer inside the configured timeout."""

    retryable = True


class ProviderUnavailable(ProviderError):
    """Connection refused, reset, or a 5xx response. The provider is down or sick."""

    retryable = True


class ProviderRateLimited(ProviderError):
    """Upstream 429. Retryable with backoff, and a good reason to fail over."""

    retryable = True


class ProviderBadRequest(ProviderError):
    """4xx that retrying cannot fix — a malformed body or an unknown model."""

    retryable = False


# --------------------------------------------------------------------------- #
# Value objects
# --------------------------------------------------------------------------- #


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @classmethod
    def from_payload(cls, payload: dict | None) -> Usage:
        payload = payload or {}
        return cls(
            prompt_tokens=int(payload.get("prompt_tokens", 0) or 0),
            completion_tokens=int(payload.get("completion_tokens", 0) or 0),
        )


@dataclass
class ChatRequest:
    """A provider-agnostic completion request."""

    model: str
    messages: list[dict]
    extra: dict = field(default_factory=dict)  # temperature, max_tokens, ...

    def to_payload(self, *, stream: bool = False) -> dict:
        payload = {"model": self.model, "messages": self.messages, **self.extra}
        if stream:
            payload["stream"] = True
        return payload


@dataclass
class ChatResponse:
    """A non-streaming completion, already in OpenAI response shape."""

    body: dict
    usage: Usage


class StreamHandle:
    """A live stream, handed out only once the provider has accepted the request.

    Usage is filled in as chunks flow past — the mock providers (like the real
    ones) put the ``usage`` object in the final chunk, so cost is only known
    after the last token. ``app/api/chat.py`` reads ``.usage`` in a finalizer
    once the stream closes.
    """

    def __init__(self, model: str):
        self.model = model
        self.usage = Usage()
        self.completion_text: list[str] = []
        # Counts bytes actually handed to the client. Once this is non-zero the
        # response is committed to this provider: the dispatcher must not fail
        # over mid-stream, because the client already has part of the answer.
        self.chunks_sent = 0
        self._lines: AsyncIterator[bytes] | None = None

    def bind(self, lines: AsyncIterator[bytes]) -> StreamHandle:
        """Attach the byte source. Separate from ``__init__`` so the adapter can
        hand the handle to its own line iterator, which writes usage back here."""
        self._lines = lines
        return self

    async def __aiter__(self) -> AsyncIterator[bytes]:
        if self._lines is None:
            raise RuntimeError("StreamHandle used before bind()")
        async for line in self._lines:
            self.chunks_sent += 1
            yield line


# --------------------------------------------------------------------------- #
# Interface
# --------------------------------------------------------------------------- #


class Provider(ABC):
    """One upstream model provider."""

    name: str

    @abstractmethod
    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Non-streaming completion. Must raise a ``ProviderError`` on failure."""

    @abstractmethod
    def open_stream(self, request: ChatRequest) -> AbstractAsyncContextManager[StreamHandle]:
        """Open a streaming completion, as an async context manager.

        The context manager must not yield until the provider has returned a
        success status, so that a failed provider can still be retried or failed
        over *before* the client has been committed to a response.
        """

    @abstractmethod
    async def health(self) -> bool:
        """Cheap liveness probe, used by the admin health endpoint."""

    async def aclose(self) -> None:
        """Release connections held by the adapter."""
