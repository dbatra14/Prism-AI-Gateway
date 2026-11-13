"""The data plane: ``POST /v1/chat/completions``.

This is the whole gateway in one request lifecycle. Reading top to bottom:

     1. parse and validate      400 on a malformed body
     2. authenticate            401 on a bad virtual key
     3. allowlist               403 if this key may not use this model
     4. rate limit              429 if this minute's quota is spent
     5. budget                  402 if this month's dollars are spent
     6. route                   resolve the alias; for `auto`, classify first
     7. cache                   a semantic hit ends the request here, for $0
     8. dispatch                call upstream, with retries and failover
     9. meter                   compute cost, add to usage, write the log

Steps 2-5 are admission control and live in ``app/limits.py``. Step 6 is
``app/routing/`` plus ``app/dispatch.py``. Step 7 is ``app/cache/``. This module
is the sequencing and the HTTP contract, and tries to hold no policy of its own.

Every exit path — success, cache hit, or any rejection — writes exactly one
request log row. That is what makes the log trustworthy enough to debug from:
if a caller says "my request vanished", it is in here with a reason.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from app import limits
from app.cache.semantic_cache import CacheScope, SemanticCache, build_scope
from app.db import get_session, get_session_factory
from app.deps import get_cache, get_config, get_dispatcher, get_prices
from app.dispatch import DispatchMeta, Dispatcher
from app.errors import InvalidRequestError, ModelNotAllowedError, ModelNotFoundError, PrismError
from app.models import RequestLog, VirtualKey
from app.pricing import PriceTable, format_cost
from app.providers.base import ProviderError, StreamHandle
from app.routing.classifier import RoutingDecision, classify_messages
from app.schemas import ChatCompletionRequest

logger = logging.getLogger(__name__)

router = APIRouter(tags=["data-plane"])


# --------------------------------------------------------------------------- #
# Endpoint
# --------------------------------------------------------------------------- #


# response_model=None: this endpoint returns either a JSON body or an SSE
# stream, which is not a shape FastAPI can derive a response model from.
@router.post("/v1/chat/completions", response_model=None)
async def chat_completions(
    request: Request,
    session: AsyncSession = Depends(get_session),
    dispatcher: Dispatcher = Depends(get_dispatcher),
    cache: SemanticCache = Depends(get_cache),
    prices: PriceTable = Depends(get_prices),
) -> JSONResponse | StreamingResponse:
    started = time.monotonic()
    request_id = f"req_{uuid.uuid4().hex[:24]}"
    key: VirtualKey | None = None
    requested_model: str | None = None

    try:
        body = await _parse_body(request)
        requested_model = body.model

        # --- admission control (order documented in app/limits.py) ---------- #
        key = await limits.authenticate(session, request.headers.get("authorization"))
        if not dispatcher.is_known(requested_model):
            raise ModelNotFoundError(
                f"Unknown model or alias '{requested_model}'. "
                f"Known: {', '.join(dispatcher.known_names())}."
            )
        limits.check_allowlist(key, requested_model)
        await limits.consume_rate_limit(session, key)
        await limits.check_budget(session, key)

        # --- routing -------------------------------------------------------- #
        target_alias, decision = _resolve_target(body, key, dispatcher)
        chain = dispatcher.resolve_chain(target_alias)
        route_reason = decision.reason if decision else None

        # --- cache ---------------------------------------------------------- #
        scope = build_scope(key.virtual_key, requested_model, body.messages_as_dicts())
        embedding = None
        if scope is not None and key.cache_enabled:
            embedding = await cache.embed(scope.query_text)
            hit = await cache.lookup(
                session, scope, embedding, key.cache_similarity_threshold or 0.92
            )
            if hit is not None:
                return await _serve_cache_hit(
                    hit=hit,
                    key=key,
                    body=body,
                    request_id=request_id,
                    requested_model=requested_model,
                    route_reason=route_reason,
                    started=started,
                )

        # --- upstream ------------------------------------------------------- #
        if body.stream:
            return await _serve_stream(
                dispatcher=dispatcher,
                chain=chain,
                body=body,
                key=key,
                prices=prices,
                cache=cache,
                scope=scope if key.cache_enabled else None,
                embedding=embedding,
                request_id=request_id,
                requested_model=requested_model,
                route_reason=route_reason,
                started=started,
            )

        return await _serve_completion(
            dispatcher=dispatcher,
            chain=chain,
            body=body,
            key=key,
            prices=prices,
            cache=cache,
            scope=scope if key.cache_enabled else None,
            embedding=embedding,
            session=session,
            request_id=request_id,
            requested_model=requested_model,
            route_reason=route_reason,
            started=started,
        )

    except PrismError as exc:
        await _log_rejection(
            request_id=request_id,
            key=key,
            requested_model=requested_model,
            exc=exc,
            started=started,
        )
        return exc.to_response()


# --------------------------------------------------------------------------- #
# Steps
# --------------------------------------------------------------------------- #


async def _parse_body(request: Request) -> ChatCompletionRequest:
    """Validate into our own error shape.

    FastAPI's default 422 body doesn't look like an OpenAI error, and callers of
    an OpenAI-compatible endpoint parse OpenAI errors.
    """
    try:
        raw = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise InvalidRequestError("Request body is not valid JSON.") from exc

    if not isinstance(raw, dict):
        raise InvalidRequestError("Request body must be a JSON object.")

    try:
        return ChatCompletionRequest.model_validate(raw)
    except ValidationError as exc:
        first = exc.errors()[0]
        location = ".".join(str(p) for p in first.get("loc", ())) or "body"
        raise InvalidRequestError(f"Invalid request at '{location}': {first.get('msg')}") from exc


def _resolve_target(
    body: ChatCompletionRequest, key: VirtualKey, dispatcher: Dispatcher
) -> tuple[str, RoutingDecision | None]:
    """Decide which alias actually serves this request.

    For anything but ``auto`` this is a no-op. For ``auto`` the classifier picks
    a tier, and that tier is then re-checked against the allowlist: a key
    permitted to use ``auto`` but not ``smart`` must not reach a smart model
    through the back door. Rather than reject — the caller asked for "whatever
    is appropriate", not for a specific model — the request is served on the
    best tier the key *is* allowed to use, and the log says so.
    """
    alias = dispatcher.alias(body.model)
    if alias is None or not alias.is_auto:
        return body.model, None

    decision = classify_messages(body.messages_as_dicts())
    target = alias.route_by_difficulty.get(
        "complex" if decision.tier == "smart" else "simple", decision.tier
    )

    if not key.allows_model(target):
        downgrade = next((t for t in ("fast", "smart") if key.allows_model(t)), None)
        if downgrade is None:
            raise ModelNotAllowedError(
                f"Auto-routing selected '{target}', which is not allowed for this key, "
                f"and no alternative tier is allowed either."
            )
        decision.matched.append(f"downgraded to '{downgrade}' (key not allowed '{target}')")
        target = downgrade

    return target, decision


async def _serve_completion(
    *,
    dispatcher: Dispatcher,
    chain: tuple[str, ...],
    body: ChatCompletionRequest,
    key: VirtualKey,
    prices: PriceTable,
    cache: SemanticCache,
    scope: CacheScope | None,
    embedding,
    session: AsyncSession,
    request_id: str,
    requested_model: str,
    route_reason: str | None,
    started: float,
) -> JSONResponse:
    result = await dispatcher.complete(chain, body.messages_as_dicts(), body.passthrough_params())
    usage = result.response.usage
    cost = prices.cost_usd(result.meta.model, usage.prompt_tokens, usage.completion_tokens)

    if scope is not None and embedding is not None:
        await cache.store(
            session, scope, embedding, result.response.body,
            prompt_tokens=usage.prompt_tokens, completion_tokens=usage.completion_tokens,
        )

    await _finalize(
        request_id=request_id,
        key=key,
        requested_model=requested_model,
        meta=result.meta,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        cost=cost,
        cache_result="miss",
        route_reason=route_reason,
        started=started,
        streamed=False,
        status="ok",
    )

    return JSONResponse(
        content=result.response.body,
        headers=_headers(result.meta, cache_result="miss", cost=cost),
    )


async def _serve_cache_hit(
    *, hit, key: VirtualKey, body: ChatCompletionRequest, request_id: str,
    requested_model: str, route_reason: str | None, started: float,
) -> JSONResponse | StreamingResponse:
    """Answer from cache — no provider call, no cost.

    Tokens are still recorded (they were served to the caller, and that keeps
    the usage API reconcilable with response bodies) but cost is zero, because
    no money was spent. The two numbers are meant to differ for cache hits;
    that difference is the cache's value, and the console shows it.
    """
    reason = ((route_reason + "; ") if route_reason else "") + f"cache hit (similarity {hit.similarity:.3f})"
    meta = DispatchMeta(provider="cache", model=requested_model, fallback=False, retries=0)

    await _finalize(
        request_id=request_id, key=key, requested_model=requested_model, meta=meta,
        prompt_tokens=hit.prompt_tokens, completion_tokens=hit.completion_tokens,
        cost=0.0, cache_result="hit", route_reason=reason, started=started,
        streamed=body.stream, status="cache_hit",
    )

    headers = _headers(meta, cache_result="hit", cost=0.0)
    if not body.stream:
        return JSONResponse(content=hit.response, headers=headers)

    # A streaming caller asked for SSE, so a cache hit is replayed as SSE. The
    # caller's parsing path stays identical whether or not the cache answered.
    return StreamingResponse(
        _replay_as_sse(hit.response),
        media_type="text/event-stream",
        headers={**headers, **_SSE_HEADERS},
    )


async def _serve_stream(
    *,
    dispatcher: Dispatcher,
    chain: tuple[str, ...],
    body: ChatCompletionRequest,
    key: VirtualKey,
    prices: PriceTable,
    cache: SemanticCache,
    scope: CacheScope | None,
    embedding,
    request_id: str,
    requested_model: str,
    route_reason: str | None,
    started: float,
) -> StreamingResponse:
    """Relay tokens upstream-to-client with nothing held back.

    The stream is opened *before* the response is constructed, because
    ``x-prism-provider`` has to be on the headers and headers go out before the
    first byte of body. That ordering is also why ``x-prism-cost-usd`` cannot be
    a streaming header: the provider only reports token counts in its final
    chunk, long after the headers are gone. The cost still lands in the request
    log and the usage API — it just arrives late.
    """
    stack = AsyncExitStack()
    handle, meta = await stack.enter_async_context(
        dispatcher.open_stream(chain, body.messages_as_dicts(), body.passthrough_params())
    )

    async def relay() -> AsyncIterator[bytes]:
        status, error_type = "ok", None
        try:
            async for line in handle:
                # Forwarded the instant it arrives. Nothing accumulates here;
                # the only thing read out of the chunks is the usage object,
                # and that happens in the adapter as the bytes go past.
                yield line + b"\n"
        except ProviderError as exc:
            status, error_type = "upstream_error", exc.__class__.__name__
            logger.warning("stream failed mid-response for %s: %s", request_id, exc.message)
            # The client already has a partial answer and the headers are long
            # gone, so there is no way to turn this into an HTTP error. Close
            # the SSE stream cleanly and let the record show it was truncated.
            yield b"data: [DONE]\n\n"
        finally:
            await stack.aclose()
            await _finalize_stream(
                handle=handle, meta=meta, key=key, prices=prices, cache=cache,
                scope=scope, embedding=embedding, request_id=request_id,
                requested_model=requested_model, route_reason=route_reason,
                started=started, status=status, error_type=error_type,
            )

    return StreamingResponse(
        relay(),
        media_type="text/event-stream",
        headers={**_headers(meta, cache_result="miss", cost=None), **_SSE_HEADERS},
    )


async def _finalize_stream(
    *, handle: StreamHandle, meta: DispatchMeta, key: VirtualKey, prices: PriceTable,
    cache: SemanticCache, scope: CacheScope | None, embedding, request_id: str,
    requested_model: str, route_reason: str | None, started: float,
    status: str, error_type: str | None,
) -> None:
    """Account for a stream once it has finished.

    Runs in the response generator's ``finally``, after the request-scoped
    session is long closed, so it opens its own.
    """
    usage = handle.usage
    cost = prices.cost_usd(meta.model, usage.prompt_tokens, usage.completion_tokens)

    async with get_session_factory()() as session:
        if scope is not None and embedding is not None and status == "ok":
            # Only a complete stream is worth caching; a truncated one would be
            # served whole to the next caller who asked the same thing.
            await cache.store(
                session, scope, embedding,
                _rebuild_completion(handle, meta),
                prompt_tokens=usage.prompt_tokens, completion_tokens=usage.completion_tokens,
            )
        await _finalize(
            request_id=request_id, key=key, requested_model=requested_model, meta=meta,
            prompt_tokens=usage.prompt_tokens, completion_tokens=usage.completion_tokens,
            cost=cost, cache_result="miss", route_reason=route_reason, started=started,
            streamed=True, status=status, error_type=error_type, session=session,
        )


def _rebuild_completion(handle: StreamHandle, meta: DispatchMeta) -> dict:
    """Assemble the streamed deltas into a normal completion, for the cache.

    Reassembling here rather than while relaying keeps the hot path clean: the
    text was collected chunk-by-chunk in the adapter, and this only runs once
    the client has already been served.
    """
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "model": meta.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "".join(handle.completion_text)},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": handle.usage.prompt_tokens,
            "completion_tokens": handle.usage.completion_tokens,
            "total_tokens": handle.usage.total_tokens,
        },
    }


async def _replay_as_sse(response: dict) -> AsyncIterator[bytes]:
    """Emit a stored completion in the same SSE shape a provider would."""
    content = ""
    for choice in response.get("choices") or []:
        content = (choice.get("message") or {}).get("content") or ""
        break

    completion_id = response.get("id", f"chatcmpl-{uuid.uuid4().hex[:24]}")
    model = response.get("model", "cache")

    def frame(delta: dict, finish_reason: str | None = None) -> bytes:
        payload = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        return f"data: {json.dumps(payload)}\n\n".encode()

    yield frame({"role": "assistant", "content": ""})
    for word in content.split(" "):
        yield frame({"content": word + " "})
    yield frame({}, finish_reason="stop")
    yield b"data: [DONE]\n\n"


# --------------------------------------------------------------------------- #
# Headers, logging, metering
# --------------------------------------------------------------------------- #

_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    # Tells nginx and friends not to buffer, which would defeat the whole point.
    "X-Accel-Buffering": "no",
}


def _headers(meta: DispatchMeta, *, cache_result: str, cost: float | None) -> dict[str, str]:
    """The ``x-prism-*`` contract every response carries."""
    headers = {
        "x-prism-provider": meta.provider_header,
        "x-prism-cache": cache_result,
        "x-prism-fallback": "true" if meta.fallback else "false",
    }
    if cost is not None:
        headers["x-prism-cost-usd"] = format_cost(cost)
    return headers


async def _finalize(
    *, request_id: str, key: VirtualKey, requested_model: str, meta: DispatchMeta,
    prompt_tokens: int, completion_tokens: int, cost: float, cache_result: str,
    route_reason: str | None, started: float, streamed: bool, status: str,
    error_type: str | None = None, session: AsyncSession | None = None,
) -> None:
    """Record one finished request: usage totals first, then the log row."""
    latency_ms = int((time.monotonic() - started) * 1000)

    async def write(db: AsyncSession) -> None:
        await limits.record_usage(
            db, key.virtual_key,
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
            cost_usd=cost, cache_hit=(cache_result == "hit"),
        )
        db.add(
            RequestLog(
                request_id=request_id,
                virtual_key=key.virtual_key,
                requested_model=requested_model,
                resolved_provider=None if cache_result == "hit" else meta.provider,
                resolved_model=None if cache_result == "hit" else meta.model,
                status=status,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost_usd=cost,
                cache=cache_result,
                fallback=meta.fallback,
                route_reason=route_reason,
                retries=meta.retries,
                latency_ms=latency_ms,
                streamed=streamed,
                error_type=error_type,
            )
        )
        await db.commit()

    if session is not None:
        await write(session)
    else:
        async with get_session_factory()() as db:
            await write(db)


async def _log_rejection(
    *, request_id: str, key: VirtualKey | None, requested_model: str | None,
    exc: PrismError, started: float,
) -> None:
    """Rejections are logged too — a request that never reached a provider is
    exactly what an operator needs to see when a tenant reports errors.

    This runs on its own session: the request session may have been left in a
    failed state by whatever raised.
    """
    latency_ms = int((time.monotonic() - started) * 1000)
    try:
        async with get_session_factory()() as db:
            db.add(
                RequestLog(
                    request_id=request_id,
                    virtual_key=key.virtual_key if key else None,
                    requested_model=requested_model,
                    status=exc.log_status,
                    cache="miss",
                    fallback=False,
                    latency_ms=latency_ms,
                    error_type=exc.error_type,
                )
            )
            await db.commit()
    except Exception:  # noqa: BLE001
        # Never let a logging failure change what the caller sees.
        logger.exception("failed to write rejection log for %s", request_id)
