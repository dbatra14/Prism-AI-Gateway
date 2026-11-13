"""End-to-end tests through the real ASGI app.

These drive the actual FastAPI application — routing, dependencies, exception
handlers, response headers and all — with the providers swapped for fakes and
the database pointed at a temp file. That combination is what makes it possible
to assert on things the unit tests can't reach: the exact status code and body
of each rejection, the ``x-prism-*`` header contract, SSE framing, and that
every request (rejections included) leaves exactly one log row behind.
"""

from __future__ import annotations

import json

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import func, select

from app.cache.embedder import HashingEmbedder
from app.cache.semantic_cache import SemanticCache
from app.dispatch import Dispatcher
from app.models import RequestLog
from app.providers.base import ProviderUnavailable
from app.providers.registry import ProviderRegistry
from tests.conftest import FakeProviderFactory

ADMIN = {"X-Admin-Token": "prism-admin-token"}
SEARCH_KEY = "prism-sk-search-1a2b3c"       # allowlist: fast, cache on  (0.92)
RESEARCH_KEY = "prism-sk-research-4d5e6f"   # allowlist: fast/smart/auto, cache off
FREE_KEY = "prism-sk-free-7g8h9i"           # allowlist: fast, cache on  (0.85)
BUDGET_KEY = "prism-sk-budget-demo-0j1k2l"  # budget of $0.00001


def body(model: str = "fast", content: str = "What is a message queue?", **extra) -> dict:
    return {"model": model, "messages": [{"role": "user", "content": content}], **extra}


def auth(key: str) -> dict:
    return {"Authorization": f"Bearer {key}"}


@pytest_asyncio.fixture
async def client(tmp_path, monkeypatch) -> AsyncClient:
    """The app, wired to a temp database and fake providers."""
    monkeypatch.setenv("PRISM_DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'api.db'}")
    monkeypatch.setenv("PRISM_EMBEDDING_BACKEND", "hash")

    from app import config, db

    config.get_settings.cache_clear()
    config.get_gateway_config.cache_clear()
    await db.dispose_engine()

    from app.main import create_app

    app = create_app()
    async with app.router.lifespan_context(app):
        # Swap in fakes: no network, deterministic token counts and failures.
        factory = FakeProviderFactory()
        app.state.registry = ProviderRegistry(factory.providers)
        app.state.dispatcher = Dispatcher(
            app.state.registry, app.state.gateway_config, app.state.prices
        )
        app.state.cache = SemanticCache(HashingEmbedder(), ttl_seconds=3600)
        app.state.fakes = factory

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            c.app = app
            yield c

    await db.dispose_engine()
    config.get_settings.cache_clear()
    config.get_gateway_config.cache_clear()


async def log_rows(client, **filters) -> list[dict]:
    response = await client.get("/admin/logs", headers=ADMIN, params={"limit": 500, **filters})
    return response.json()["logs"]


# --------------------------------------------------------------------------- #
# The happy path and the header contract
# --------------------------------------------------------------------------- #


async def test_completion_returns_openai_shape(client):
    response = await client.post("/v1/chat/completions", json=body(), headers=auth(RESEARCH_KEY))

    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "chat.completion"
    assert payload["choices"][0]["message"]["role"] == "assistant"
    assert payload["usage"]["prompt_tokens"] > 0


async def test_response_carries_the_full_header_contract(client):
    response = await client.post("/v1/chat/completions", json=body(), headers=auth(RESEARCH_KEY))

    assert response.headers["x-prism-provider"] == "alpha/alpha-small"
    assert response.headers["x-prism-cache"] == "miss"
    assert response.headers["x-prism-fallback"] == "false"
    assert float(response.headers["x-prism-cost-usd"]) > 0


async def test_alias_resolves_to_the_configured_model(client):
    response = await client.post("/v1/chat/completions", json=body("smart"), headers=auth(RESEARCH_KEY))
    assert response.headers["x-prism-provider"] == "alpha/alpha-large"


async def test_provider_credentials_never_reach_the_caller(client):
    response = await client.post("/v1/chat/completions", json=body(), headers=auth(RESEARCH_KEY))
    blob = response.text + json.dumps(dict(response.headers))
    assert "secret" not in blob and "mock-key" not in blob


# --------------------------------------------------------------------------- #
# Rejections — each distinguishable from the body alone
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("headers", "payload", "status", "error_type"),
    [
        ({}, body(), 401, "authentication_error"),
        (auth("not-a-real-key"), body(), 401, "authentication_error"),
        (auth(SEARCH_KEY), body("smart"), 403, "model_not_allowed"),
        (auth(RESEARCH_KEY), body("no-such-model-xyz"), 404, "not_found_error"),
        (auth(RESEARCH_KEY), {"model": "fast", "messages": []}, 400, "invalid_request_error"),
        (auth(RESEARCH_KEY), {"messages": [{"role": "user", "content": "x"}]}, 400, "invalid_request_error"),
    ],
)
async def test_rejections_are_distinct_and_openai_shaped(client, headers, payload, status, error_type):
    response = await client.post("/v1/chat/completions", json=payload, headers=headers)

    assert response.status_code == status
    error = response.json()["error"]
    assert error["type"] == error_type
    assert error["code"] == error_type
    assert error["message"]


async def test_unknown_model_is_404_not_403(client):
    """A model that doesn't exist is "no such thing", not "not for you" —
    telling a caller they lack permission for a typo is a confusing answer."""
    response = await client.post(
        "/v1/chat/completions", json=body("typo-model"), headers=auth(RESEARCH_KEY)
    )
    assert response.status_code == 404


async def test_rate_limit_rejects_with_429(client):
    # free-tier allows 10 requests per minute.
    for _ in range(10):
        await client.post("/v1/chat/completions", json=body(), headers=auth(FREE_KEY))

    response = await client.post("/v1/chat/completions", json=body(), headers=auth(FREE_KEY))
    assert response.status_code == 429
    assert response.json()["error"]["type"] == "rate_limit_exceeded"


async def test_budget_exhaustion_is_distinct_from_rate_limiting(client):
    """Both are "you can't do that right now", but only one is fixed by waiting.

    The budget-demo key's cap is smaller than a single request's cost, so it
    must start rejecting almost immediately. The loop avoids pinning the test to
    an exact token count — what matters is that spending stops, and that it
    stops with ``budget_exceeded`` rather than a rate-limit error.
    """
    statuses = []
    for _ in range(4):
        response = await client.post("/v1/chat/completions", json=body(), headers=auth(BUDGET_KEY))
        statuses.append(response.status_code)
        if response.status_code != 200:
            assert response.json()["error"]["type"] == "budget_exceeded"
            break

    assert statuses[0] == 200, "the first request should fit inside the budget"
    assert statuses[-1] == 402, f"budget was never enforced: {statuses}"


async def test_all_providers_down_returns_502(client):
    client.app.state.fakes.fail_all(ProviderUnavailable("down"))

    response = await client.post("/v1/chat/completions", json=body(), headers=auth(RESEARCH_KEY))

    assert response.status_code == 502
    assert response.json()["error"]["type"] == "upstream_error"


# --------------------------------------------------------------------------- #
# Failover
# --------------------------------------------------------------------------- #


async def test_failover_is_reported_in_the_header(client):
    client.app.state.fakes.fail("alpha", ProviderUnavailable("down"))

    response = await client.post("/v1/chat/completions", json=body(), headers=auth(RESEARCH_KEY))

    assert response.status_code == 200
    assert response.headers["x-prism-fallback"] == "true"
    assert response.headers["x-prism-provider"] == "beta/beta-small"


# --------------------------------------------------------------------------- #
# Streaming
# --------------------------------------------------------------------------- #


async def test_stream_returns_sse_terminated_by_done(client):
    async with client.stream(
        "POST", "/v1/chat/completions", json=body(stream=True), headers=auth(RESEARCH_KEY)
    ) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        assert response.headers["x-prism-provider"] == "alpha/alpha-small"
        lines = [line async for line in response.aiter_lines() if line.startswith("data: ")]

    assert len(lines) >= 3
    assert lines[-1] == "data: [DONE]"


async def test_stream_omits_cost_header_but_still_records_cost(client):
    """Headers are sent before the provider reports tokens, so the cost can't be
    a header — but it must not be lost."""
    async with client.stream(
        "POST", "/v1/chat/completions", json=body(stream=True), headers=auth(RESEARCH_KEY)
    ) as response:
        assert "x-prism-cost-usd" not in response.headers
        async for _ in response.aiter_lines():
            pass

    logs = await log_rows(client, key=RESEARCH_KEY)
    streamed = [row for row in logs if row["streamed"]]
    assert streamed and streamed[0]["cost_usd"] > 0
    assert streamed[0]["completion_tokens"] > 0


async def test_stream_chunks_are_valid_json(client):
    async with client.stream(
        "POST", "/v1/chat/completions", json=body(stream=True), headers=auth(RESEARCH_KEY)
    ) as response:
        payloads = [
            line[len("data: "):]
            async for line in response.aiter_lines()
            if line.startswith("data: ")
        ]

    for payload in payloads[:-1]:
        assert json.loads(payload)["object"] == "chat.completion.chunk"


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #


async def test_repeat_prompt_is_served_from_cache_for_free(client):
    payload = body(content="How do I reset my password on the dashboard?")

    first = await client.post("/v1/chat/completions", json=payload, headers=auth(SEARCH_KEY))
    second = await client.post("/v1/chat/completions", json=payload, headers=auth(SEARCH_KEY))

    assert first.headers["x-prism-cache"] == "miss"
    assert second.headers["x-prism-cache"] == "hit"
    assert float(second.headers["x-prism-cost-usd"]) == 0.0
    assert second.json()["choices"] == first.json()["choices"]


async def test_cache_is_not_shared_between_tenants(client):
    payload = body(content="What is our internal runbook?")

    await client.post("/v1/chat/completions", json=payload, headers=auth(SEARCH_KEY))
    other = await client.post("/v1/chat/completions", json=payload, headers=auth(FREE_KEY))

    assert other.headers["x-prism-cache"] == "miss"


async def test_cache_disabled_key_always_misses(client):
    payload = body(content="Repeated question for a key with caching off")

    await client.post("/v1/chat/completions", json=payload, headers=auth(RESEARCH_KEY))
    second = await client.post("/v1/chat/completions", json=payload, headers=auth(RESEARCH_KEY))

    assert second.headers["x-prism-cache"] == "miss"


async def test_streaming_cache_hit_is_replayed_as_sse(client):
    """A streaming caller gets SSE whether or not the cache answered."""
    payload = body(content="Explain how server-sent events work.")
    await client.post("/v1/chat/completions", json=payload, headers=auth(SEARCH_KEY))

    async with client.stream(
        "POST", "/v1/chat/completions", json={**payload, "stream": True}, headers=auth(SEARCH_KEY)
    ) as response:
        assert response.headers["x-prism-cache"] == "hit"
        assert "text/event-stream" in response.headers["content-type"]
        lines = [line async for line in response.aiter_lines() if line.startswith("data: ")]

    assert lines[-1] == "data: [DONE]"


# --------------------------------------------------------------------------- #
# Auto routing
# --------------------------------------------------------------------------- #


async def test_auto_sends_a_trivial_prompt_to_the_cheap_tier(client):
    response = await client.post(
        "/v1/chat/completions",
        json=body("auto", "What is the capital of France?"),
        headers=auth(RESEARCH_KEY),
    )
    assert response.headers["x-prism-provider"] == "alpha/alpha-small"


async def test_auto_escalates_a_hard_prompt(client):
    response = await client.post(
        "/v1/chat/completions",
        json=body("auto", "Prove that the square root of 2 is irrational."),
        headers=auth(RESEARCH_KEY),
    )
    assert response.headers["x-prism-provider"] == "alpha/alpha-large"


async def test_auto_records_why_it_chose_a_tier(client):
    await client.post(
        "/v1/chat/completions",
        json=body("auto", "Design a billing schema and explain your trade-offs."),
        headers=auth(RESEARCH_KEY),
    )

    logs = await log_rows(client, key=RESEARCH_KEY)
    assert "smart" in logs[0]["route_reason"]
    assert "design" in logs[0]["route_reason"].lower()


# --------------------------------------------------------------------------- #
# Metering and logging
# --------------------------------------------------------------------------- #


async def test_every_request_including_rejections_is_logged(client, session_factory):
    await client.post("/v1/chat/completions", json=body(), headers=auth(RESEARCH_KEY))
    await client.post("/v1/chat/completions", json=body("smart"), headers=auth(SEARCH_KEY))
    await client.post("/v1/chat/completions", json=body(), headers=auth("bogus-key"))

    logs = await log_rows(client)
    statuses = {row["status"] for row in logs}
    assert {"ok", "rejected_allowlist", "rejected_auth"} <= statuses


async def test_exactly_one_log_row_per_request(client):
    for _ in range(5):
        await client.post("/v1/chat/completions", json=body(), headers=auth(RESEARCH_KEY))

    logs = await log_rows(client, key=RESEARCH_KEY)
    assert len(logs) == 5
    assert len({row["request_id"] for row in logs}) == 5


async def test_usage_api_reconciles_with_response_headers(client):
    """The property the load test checks from the outside."""
    costs = []
    for index in range(6):
        response = await client.post(
            "/v1/chat/completions",
            json=body(content=f"unique prompt {index}"),
            headers=auth(RESEARCH_KEY),
        )
        costs.append(float(response.headers["x-prism-cost-usd"]))

    usage = (await client.get("/admin/usage", headers=ADMIN,
                              params={"key": RESEARCH_KEY})).json()

    assert usage["requests"] == 6
    assert usage["cost_usd"] == pytest.approx(sum(costs), rel=1e-6)


async def test_cache_hits_cost_nothing_but_still_count_as_requests(client):
    payload = body(content="A question worth caching")
    await client.post("/v1/chat/completions", json=payload, headers=auth(SEARCH_KEY))
    paid = (await client.get("/admin/usage", headers=ADMIN, params={"key": SEARCH_KEY})).json()

    await client.post("/v1/chat/completions", json=payload, headers=auth(SEARCH_KEY))
    after = (await client.get("/admin/usage", headers=ADMIN, params={"key": SEARCH_KEY})).json()

    assert after["requests"] == paid["requests"] + 1
    assert after["cache_hits"] == 1
    assert after["cost_usd"] == pytest.approx(paid["cost_usd"])  # the hit was free


# --------------------------------------------------------------------------- #
# Admin plane
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "path",
    ["/admin/usage", "/admin/logs", "/admin/cache/stats", "/admin/keys",
     "/admin/config", "/admin/providers/health"],
)
async def test_admin_routes_require_the_admin_token(client, path):
    assert (await client.get(path)).status_code == 401
    assert (await client.get(path, headers={"X-Admin-Token": "wrong"})).status_code == 401
    assert (await client.get(path, headers=ADMIN)).status_code == 200


async def test_a_virtual_key_is_not_an_admin_token(client):
    """The two credentials are for different audiences and must not overlap."""
    response = await client.get("/admin/usage", headers={"X-Admin-Token": SEARCH_KEY})
    assert response.status_code == 401


async def test_admin_config_does_not_expose_provider_credentials(client):
    response = await client.get("/admin/config", headers=ADMIN)
    assert "api_key" not in response.text
    assert "secret" not in response.text


async def test_cache_stats_report_hit_rate(client):
    payload = body(content="A cache-stats question")
    await client.post("/v1/chat/completions", json=payload, headers=auth(SEARCH_KEY))
    await client.post("/v1/chat/completions", json=payload, headers=auth(SEARCH_KEY))

    stats = (await client.get("/admin/cache/stats", headers=ADMIN,
                              params={"key": SEARCH_KEY})).json()

    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["hit_rate"] == pytest.approx(0.5)
    assert stats["entries"] == 1


async def test_logs_can_be_filtered_by_key(client):
    await client.post("/v1/chat/completions", json=body(), headers=auth(RESEARCH_KEY))
    await client.post("/v1/chat/completions", json=body(), headers=auth(FREE_KEY))

    logs = await log_rows(client, key=FREE_KEY)
    assert logs and all(row["virtual_key"] == FREE_KEY for row in logs)


async def test_keys_endpoint_shows_spend_against_budget(client):
    await client.post("/v1/chat/completions", json=body(), headers=auth(RESEARCH_KEY))

    keys = (await client.get("/admin/keys", headers=ADMIN)).json()["keys"]
    research = next(k for k in keys if k["virtual_key"] == RESEARCH_KEY)

    assert research["monthly_budget_usd"] == 500
    assert research["spent_usd"] > 0
    assert research["requests_this_period"] == 1


async def test_health_endpoint_is_public(client):
    assert (await client.get("/health")).json()["status"] == "ok"
