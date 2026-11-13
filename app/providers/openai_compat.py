"""Adapter for any OpenAI-compatible HTTP provider.

The two mock providers speak this protocol, and so do OpenAI, Together, Groq,
Fireworks, vLLM and most others — which is the point: one adapter covers the
mocks used in the demo *and* a real provider, with only the base URL and API key
changing.

Two things this adapter is responsible for:

1. **Timeouts.** Every call is bounded. A provider that accepts a connection and
   then goes silent is the worst failure mode there is, because without a
   timeout it consumes a gateway worker forever. ``httpx.Timeout`` bounds the
   connect, read, and write phases separately.
2. **Error translation.** Upstream status codes become ``ProviderError``
   subclasses tagged retryable or not, so the dispatcher's retry logic stays
   provider-agnostic.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx

from app.providers.base import (
    ChatRequest,
    ChatResponse,
    Provider,
    ProviderBadRequest,
    ProviderError,
    ProviderRateLimited,
    ProviderTimeout,
    ProviderUnavailable,
    StreamHandle,
    Usage,
)


class OpenAICompatibleProvider(Provider):
    def __init__(
        self,
        name: str,
        base_url: str,
        api_key: str,
        *,
        timeout_seconds: float = 30.0,
        connect_timeout_seconds: float = 5.0,
    ):
        self.name = name
        self._base_url = base_url.rstrip("/")
        # The provider credential lives here and nowhere else. It is never put
        # into a response, a log row, or a client-visible error message.
        self._api_key = api_key
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                timeout_seconds, connect=connect_timeout_seconds, read=timeout_seconds
            ),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )

    # -- helpers ----------------------------------------------------------- #

    @property
    def _completions_url(self) -> str:
        return f"{self._base_url}/chat/completions"

    @staticmethod
    def _raise_for_status(status: int, body_text: str) -> None:
        """Map an upstream status onto the gateway's retryable/terminal taxonomy."""
        if status < 400:
            return
        detail = _extract_message(body_text)
        if status == 429:
            raise ProviderRateLimited(f"upstream rate limited: {detail}", status)
        if status >= 500:
            raise ProviderUnavailable(f"upstream server error: {detail}", status)
        # 4xx other than 429: the request itself is wrong. Retrying an unknown
        # model or a malformed body against the same provider cannot help.
        raise ProviderBadRequest(f"upstream rejected the request: {detail}", status)

    # -- Provider interface ------------------------------------------------ #

    async def chat(self, request: ChatRequest) -> ChatResponse:
        try:
            response = await self._client.post(
                self._completions_url, json=request.to_payload(stream=False)
            )
        except httpx.TimeoutException as exc:
            raise ProviderTimeout(f"{self.name} timed out") from exc
        except httpx.HTTPError as exc:
            raise ProviderUnavailable(f"{self.name} unreachable: {type(exc).__name__}") from exc

        self._raise_for_status(response.status_code, response.text)

        try:
            body = response.json()
        except json.JSONDecodeError as exc:
            raise ProviderUnavailable(f"{self.name} returned a non-JSON body") from exc

        return ChatResponse(body=body, usage=Usage.from_payload(body.get("usage")))

    @asynccontextmanager
    async def open_stream(self, request: ChatRequest) -> AsyncIterator[StreamHandle]:
        """Stream tokens through without buffering the completion.

        Note where the ``yield`` sits: the upstream status is checked *before*
        the handle is handed back, so a provider that fails at connect time can
        still be failed over. Once we yield, the client is committed to this
        provider for the rest of the response.
        """
        payload = request.to_payload(stream=True)
        try:
            async with self._client.stream("POST", self._completions_url, json=payload) as response:
                if response.status_code >= 400:
                    body_text = (await response.aread()).decode(errors="replace")
                    self._raise_for_status(response.status_code, body_text)

                handle = StreamHandle(request.model)
                yield handle.bind(self._iter_lines(response, handle))
        except httpx.TimeoutException as exc:
            raise ProviderTimeout(f"{self.name} timed out while streaming") from exc
        except httpx.HTTPError as exc:
            raise ProviderUnavailable(f"{self.name} unreachable: {type(exc).__name__}") from exc

    async def _iter_lines(
        self, response: httpx.Response, handle: StreamHandle
    ) -> AsyncIterator[bytes]:
        """Yield raw SSE lines as they arrive, sniffing usage from the final chunk.

        The bytes are forwarded untouched — we only *read* each chunk to pick up
        the token counts the provider reports at the end. Nothing is held back.
        """
        async for raw_line in response.aiter_lines():
            if raw_line.startswith("data: "):
                _sniff_chunk(raw_line[len("data: ") :], handle)
            yield raw_line.encode()

    async def health(self) -> bool:
        try:
            response = await self._client.get(f"{self._base_url.rsplit('/v1', 1)[0]}/health", timeout=2.0)
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    async def aclose(self) -> None:
        await self._client.aclose()


def _sniff_chunk(data: str, handle: StreamHandle) -> None:
    """Extract usage and text from one SSE chunk, without ever blocking it."""
    if data.strip() == "[DONE]":
        return
    try:
        chunk = json.loads(data)
    except json.JSONDecodeError:
        return  # not our business — forward whatever the provider sent
    if chunk.get("usage"):
        handle.usage = Usage.from_payload(chunk["usage"])
    for choice in chunk.get("choices") or []:
        content = (choice.get("delta") or {}).get("content")
        if content:
            handle.completion_text.append(content)


def _extract_message(body_text: str) -> str:
    """Pull a human-readable reason out of an upstream error body."""
    try:
        parsed = json.loads(body_text)
    except (json.JSONDecodeError, TypeError):
        return (body_text or "").strip()[:200]
    if isinstance(parsed, dict) and isinstance(parsed.get("error"), dict):
        return str(parsed["error"].get("message", ""))[:200]
    return str(parsed)[:200]


__all__ = ["OpenAICompatibleProvider", "ProviderError"]
