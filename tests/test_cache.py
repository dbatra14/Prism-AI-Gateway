"""The semantic cache.

Most of these use the deterministic ``HashingEmbedder`` so the assertions are
about *cache logic* — scoping, thresholds, TTL — and don't depend on a model
download or on a particular model's similarity scores. The one thing that does
need the real model is the claim that matching is semantic rather than lexical,
and that test says so and skips if the model isn't available.
"""

from __future__ import annotations

import numpy as np
import pytest
from sqlalchemy import select

from app.cache.embedder import HashingEmbedder, get_embedder, normalize
from app.cache.semantic_cache import SemanticCache, build_scope
from app.models import CacheEntry, utcnow


@pytest.fixture
def cache() -> SemanticCache:
    return SemanticCache(HashingEmbedder(), ttl_seconds=3600, max_candidates=100)


def response(text: str = "hello") -> dict:
    return {
        "id": "chatcmpl-test",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 7},
    }


# --------------------------------------------------------------------------- #
# Scoping — what may be compared with what
# --------------------------------------------------------------------------- #


def test_scope_embeds_the_last_user_message():
    scope = build_scope("k1", "fast", [{"role": "user", "content": "What is a queue?"}])
    assert scope.query_text == "What is a queue?"
    assert scope.prefix_hash == ""  # single turn


def test_scope_ignores_a_trailing_assistant_turn():
    scope = build_scope(
        "k1", "fast",
        [{"role": "user", "content": "Question?"}, {"role": "assistant", "content": "Answer."}],
    )
    assert scope.query_text == "Question?"


def test_multi_turn_prefix_is_part_of_the_scope():
    """The same follow-up after different conversations must not collide."""
    a = build_scope("k1", "fast", [
        {"role": "user", "content": "Tell me about connection pooling."},
        {"role": "assistant", "content": "It reuses connections."},
        {"role": "user", "content": "What happens when it is exhausted?"},
    ])
    b = build_scope("k1", "fast", [
        {"role": "user", "content": "Tell me about thread pools."},
        {"role": "assistant", "content": "They reuse threads."},
        {"role": "user", "content": "What happens when it is exhausted?"},
    ])
    assert a.query_text == b.query_text
    assert a.prefix_hash != b.prefix_hash  # so they can never match each other


def test_system_prompt_changes_the_scope():
    plain = build_scope("k1", "fast", [{"role": "user", "content": "Explain X."}])
    with_system = build_scope("k1", "fast", [
        {"role": "system", "content": "Answer only in French."},
        {"role": "user", "content": "Explain X."},
    ])
    assert plain.prefix_hash != with_system.prefix_hash


@pytest.mark.parametrize(
    "messages",
    [
        [],
        [{"role": "assistant", "content": "no user turn"}],
        [{"role": "user", "content": "   "}],
        [{"role": "user", "content": None}],
    ],
)
def test_uncacheable_requests_are_skipped(messages):
    assert build_scope("k1", "fast", messages) is None


# --------------------------------------------------------------------------- #
# Tenant isolation — the security-relevant property
# --------------------------------------------------------------------------- #


async def test_identical_prompt_from_another_tenant_is_a_miss(session, cache):
    prompt = "What is our internal deployment runbook?"
    scope_a = build_scope("tenant-a", "fast", [{"role": "user", "content": prompt}])
    scope_b = build_scope("tenant-b", "fast", [{"role": "user", "content": prompt}])

    vector = await cache.embed(prompt)
    await cache.store(session, scope_a, vector, response("secret answer"),
                      prompt_tokens=5, completion_tokens=7)

    # Same key: hit.
    assert await cache.lookup(session, scope_a, vector, 0.9) is not None
    # Different key, byte-identical prompt: must miss.
    assert await cache.lookup(session, scope_b, vector, 0.9) is None


async def test_entries_do_not_cross_models(session, cache):
    prompt = "What is a load balancer?"
    fast = build_scope("k1", "fast", [{"role": "user", "content": prompt}])
    smart = build_scope("k1", "smart", [{"role": "user", "content": prompt}])

    vector = await cache.embed(prompt)
    await cache.store(session, fast, vector, response(), prompt_tokens=1, completion_tokens=1)

    assert await cache.lookup(session, fast, vector, 0.9) is not None
    assert await cache.lookup(session, smart, vector, 0.9) is None


# --------------------------------------------------------------------------- #
# Hit / miss decisions
# --------------------------------------------------------------------------- #


async def test_empty_cache_is_a_miss(session, cache):
    scope = build_scope("k1", "fast", [{"role": "user", "content": "anything"}])
    assert await cache.lookup(session, scope, await cache.embed("anything"), 0.9) is None


async def test_exact_repeat_hits_and_returns_the_stored_response(session, cache):
    scope = build_scope("k1", "fast", [{"role": "user", "content": "How do I reset my password?"}])
    vector = await cache.embed(scope.query_text)
    await cache.store(session, scope, vector, response("Use the reset link."),
                      prompt_tokens=9, completion_tokens=4)

    hit = await cache.lookup(session, scope, vector, 0.9)
    assert hit is not None
    assert hit.similarity == pytest.approx(1.0, abs=1e-5)
    assert hit.response["choices"][0]["message"]["content"] == "Use the reset link."
    assert (hit.prompt_tokens, hit.completion_tokens) == (9, 4)


async def test_threshold_is_respected(session, cache):
    """The same pair of prompts hits at a loose threshold and misses at a strict
    one — the threshold is a real knob, not decoration."""
    stored = build_scope("k1", "fast", [{"role": "user", "content": "how do I reset my password"}])
    await cache.store(session, stored, await cache.embed(stored.query_text), response(),
                      prompt_tokens=1, completion_tokens=1)

    query = build_scope("k1", "fast", [{"role": "user", "content": "how do I reset my password today"}])
    vector = await cache.embed(query.query_text)

    assert await cache.lookup(session, query, vector, 0.5) is not None
    assert await cache.lookup(session, query, vector, 0.999) is None


async def test_unrelated_prompt_misses(session, cache):
    stored = build_scope("k1", "fast", [{"role": "user", "content": "How do I reset my password?"}])
    await cache.store(session, stored, await cache.embed(stored.query_text), response(),
                      prompt_tokens=1, completion_tokens=1)

    other = build_scope("k1", "fast", [{"role": "user", "content": "Compare TCP and UDP for games."}])
    assert await cache.lookup(session, other, await cache.embed(other.query_text), 0.85) is None


async def test_hit_count_is_tracked(session, cache):
    scope = build_scope("k1", "fast", [{"role": "user", "content": "popular question"}])
    vector = await cache.embed(scope.query_text)
    await cache.store(session, scope, vector, response(), prompt_tokens=1, completion_tokens=1)

    for _ in range(3):
        await cache.lookup(session, scope, vector, 0.9)

    entry = (await session.execute(select(CacheEntry))).scalar_one()
    assert entry.hit_count == 3


async def test_best_match_wins_when_several_are_close(session, cache):
    for text, answer in [
        ("how do I reset my password", "password answer"),
        ("how do I reset my billing address", "billing answer"),
    ]:
        scope = build_scope("k1", "fast", [{"role": "user", "content": text}])
        await cache.store(session, scope, await cache.embed(text), response(answer),
                          prompt_tokens=1, completion_tokens=1)

    query = build_scope("k1", "fast", [{"role": "user", "content": "how do I reset my password"}])
    hit = await cache.lookup(session, query, await cache.embed(query.query_text), 0.8)
    assert hit.response["choices"][0]["message"]["content"] == "password answer"


# --------------------------------------------------------------------------- #
# Expiry
# --------------------------------------------------------------------------- #


async def test_expired_entries_are_not_served(session):
    cache = SemanticCache(HashingEmbedder(), ttl_seconds=0)
    scope = build_scope("k1", "fast", [{"role": "user", "content": "stale question"}])
    vector = await cache.embed(scope.query_text)
    await cache.store(session, scope, vector, response(), prompt_tokens=1, completion_tokens=1)

    entry = (await session.execute(select(CacheEntry))).scalar_one()
    entry.created_at = utcnow().replace(year=utcnow().year - 1)
    await session.commit()

    assert await cache.lookup(session, scope, vector, 0.5) is None


async def test_purge_removes_expired_entries(session):
    cache = SemanticCache(HashingEmbedder(), ttl_seconds=0)
    scope = build_scope("k1", "fast", [{"role": "user", "content": "stale"}])
    await cache.store(session, scope, await cache.embed("stale"), response(),
                      prompt_tokens=1, completion_tokens=1)

    entry = (await session.execute(select(CacheEntry))).scalar_one()
    entry.created_at = utcnow().replace(year=utcnow().year - 1)
    await session.commit()

    assert await cache.purge_expired(session) == 1


async def test_dimension_change_degrades_to_a_miss(session, cache):
    """Swapping the embedding model must not crash lookups on old rows."""
    scope = build_scope("k1", "fast", [{"role": "user", "content": "question"}])
    await cache.store(session, scope, await cache.embed("question"), response(),
                      prompt_tokens=1, completion_tokens=1)

    wrong_size = normalize(np.ones((1, 16), dtype=np.float32))[0]
    assert await cache.lookup(session, scope, wrong_size, 0.5) is None


# --------------------------------------------------------------------------- #
# Embedders
# --------------------------------------------------------------------------- #


async def test_embeddings_are_unit_length():
    """Cosine similarity is a plain dot product only if this holds."""
    vectors = await HashingEmbedder().embed(["one", "another prompt entirely"])
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_hashing_embedder_is_deterministic():
    a = HashingEmbedder().embed_sync(["same text"])
    b = HashingEmbedder().embed_sync(["same text"])
    assert np.allclose(a, b)


def test_semantic_matching_beats_lexical_matching():
    """The reason a real embedding model is the default.

    These two prompts are the same question and share almost no words. A lexical
    embedder scores them near zero; a sentence model puts them close together.
    """
    a = "What is the refund policy for annual plans?"
    b = "If I bought an annual plan, can I get my money back?"

    lexical = HashingEmbedder().embed_sync([a, b])
    assert float(lexical[0] @ lexical[1]) < 0.4  # lexical matching fails here

    semantic_embedder = get_embedder("fastembed")
    if semantic_embedder.name == "hashing":
        pytest.skip("embedding model unavailable (offline); lexical fallback in use")

    semantic = semantic_embedder.embed_sync([a, b])
    assert float(semantic[0] @ semantic[1]) > 0.8
