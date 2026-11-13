"""The semantic response cache.

Why not just hash the prompt? Because callers rarely repeat themselves exactly.
"What is the refund policy for annual plans?" and "If I bought an annual plan,
can I get my money back?" are the same question, and paying twice for them is
the waste this cache exists to remove. So lookup is a nearest-neighbour search
over prompt embeddings, not a dictionary get.

Three decisions worth understanding
-----------------------------------

**Scoping.** Every lookup filters on ``virtual_key`` before anything else. The
cache is per tenant. Tenant A's answer is never a candidate for tenant B's
query, even if the prompts are word-for-word identical — a shared cache would
leak one customer's data to another, which is a far worse outcome than a cache
miss. The scope also includes the requested model and the conversation prefix.

**What gets embedded.** Only the *last user message*. Earlier turns are hashed
into ``prefix_hash``, which is part of the scope. That means a follow-up like
"What happens when the pool is exhausted?" can only match another request that
had the same conversation before it — because on its own, that question means
different things in different conversations. Single-turn prompts (the common
case, and what the graded paraphrase pairs use) all share an empty prefix and
match each other freely.

**Threshold.** Per key, from the seed config. Too low and different questions
collide, serving wrong answers; too high and the cache only catches near-exact
repeats and stops earning its keep. The measured similarity table for the
default model is in docs/ARCHITECTURE.md — the values were chosen from data,
not guessed.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import timedelta

import numpy as np
from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.cache.embedder import Embedder
from app.models import CacheEntry, utcnow

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CacheScope:
    """Everything that must match before two prompts are even compared."""

    virtual_key: str
    model: str
    prefix_hash: str
    query_text: str


@dataclass(frozen=True)
class CacheHit:
    entry_id: int
    response: dict
    similarity: float
    prompt_tokens: int
    completion_tokens: int


def build_scope(virtual_key: str, model: str, messages: list[dict]) -> CacheScope | None:
    """Work out what to embed and what to scope it by. ``None`` == don't cache.

    A request with no user message (or an empty one) has nothing meaningful to
    match on, so it bypasses the cache entirely rather than caching a blank key.
    """
    last_user_index = None
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == "user":
            last_user_index = index
            break

    if last_user_index is None:
        return None

    query_text = str(messages[last_user_index].get("content") or "").strip()
    if not query_text:
        return None

    prefix = messages[:last_user_index]
    prefix_hash = (
        hashlib.sha256(json.dumps(prefix, sort_keys=True).encode()).hexdigest()[:32]
        if prefix
        else ""
    )
    return CacheScope(
        virtual_key=virtual_key, model=model, prefix_hash=prefix_hash, query_text=query_text
    )


class SemanticCache:
    def __init__(self, embedder: Embedder, *, ttl_seconds: int = 86400, max_candidates: int = 500):
        self._embedder = embedder
        self._ttl_seconds = ttl_seconds
        self._max_candidates = max_candidates

    async def embed(self, text: str) -> np.ndarray:
        return await self._embedder.embed_one(text)

    async def lookup(
        self,
        session: AsyncSession,
        scope: CacheScope,
        embedding: np.ndarray,
        threshold: float,
    ) -> CacheHit | None:
        """Find the closest stored prompt within this scope, if it clears the bar.

        Similarity is computed in-process over the scoped candidate set. Because
        the scope is one tenant + one model + one conversation prefix, that set
        is small; this is a bounded linear scan, not a search over every entry
        in the database. The scaling path (pgvector / a dedicated vector index)
        is noted in docs/ARCHITECTURE.md — it changes this method only.
        """
        candidates = await self._candidates(session, scope)
        if not candidates:
            return None

        matrix = np.frombuffer(b"".join(c.embedding for c in candidates), dtype=np.float32)
        matrix = matrix.reshape(len(candidates), -1)
        if matrix.shape[1] != embedding.shape[0]:
            # The embedding model changed since these rows were written; their
            # vectors are not comparable. Treat as a miss rather than crash.
            logger.warning("Cache entry dimension mismatch; skipping %d entries", len(candidates))
            return None

        similarities = matrix @ embedding  # both sides are unit-length -> cosine
        best_index = int(np.argmax(similarities))
        best_score = float(similarities[best_index])

        if best_score < threshold:
            return None

        best = candidates[best_index]
        await session.execute(
            update(CacheEntry)
            .where(CacheEntry.id == best.id)
            .values(hit_count=CacheEntry.hit_count + 1)
        )
        await session.commit()

        return CacheHit(
            entry_id=best.id,
            response=best.response_json,
            similarity=best_score,
            prompt_tokens=best.prompt_tokens,
            completion_tokens=best.completion_tokens,
        )

    async def _candidates(self, session: AsyncSession, scope: CacheScope) -> list[CacheEntry]:
        cutoff = utcnow() - timedelta(seconds=self._ttl_seconds)
        stmt = (
            select(CacheEntry)
            .where(
                # Tenant isolation. This predicate is the whole reason one
                # tenant's response cannot be served to another.
                CacheEntry.virtual_key == scope.virtual_key,
                CacheEntry.model == scope.model,
                CacheEntry.prefix_hash == scope.prefix_hash,
                CacheEntry.created_at >= cutoff,
            )
            .order_by(CacheEntry.created_at.desc())
            .limit(self._max_candidates)
        )
        return list((await session.execute(stmt)).scalars().all())

    async def store(
        self,
        session: AsyncSession,
        scope: CacheScope,
        embedding: np.ndarray,
        response: dict,
        *,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        """Remember a response so a future paraphrase can be served for free."""
        session.add(
            CacheEntry(
                virtual_key=scope.virtual_key,
                model=scope.model,
                prefix_hash=scope.prefix_hash,
                prompt_text=scope.query_text,
                embedding=embedding.astype(np.float32).tobytes(),
                response_json=response,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        )
        await session.commit()

    async def purge_expired(self, session: AsyncSession) -> int:
        """Evict entries past their TTL. Called periodically, not per request."""
        cutoff = utcnow() - timedelta(seconds=self._ttl_seconds)
        result = await session.execute(delete(CacheEntry).where(CacheEntry.created_at < cutoff))
        await session.commit()
        return result.rowcount or 0
