"""Turning prompts into vectors.

The semantic cache needs "do these two prompts mean the same thing?", which is a
different question from "are these two strings equal". An embedding model
answers it by mapping text to a vector whose direction encodes meaning, so
"What is the refund policy for annual plans?" and "If I bought an annual plan,
can I get my money back?" land close together despite sharing almost no words.

Two implementations sit behind one interface:

* ``FastEmbedEmbedder``  — a real sentence-embedding model (ONNX, runs locally,
                           no API key, no PyTorch). This is the default.
* ``HashingEmbedder``    — a dependency-free lexical fallback for environments
                           that can't download a model. It matches reworded
                           prompts that share vocabulary, but genuinely
                           different wordings of the same question will miss.
                           Honest about being the weaker option.

All vectors are L2-normalised on the way out, which makes cosine similarity a
plain dot product — cheap, and it means the cache's similarity threshold has the
same meaning regardless of which backend produced the vector.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
from abc import ABC, abstractmethod
from functools import lru_cache

import numpy as np

_WORD_RE = re.compile(r"[a-z0-9']+")


def normalize(vectors: np.ndarray) -> np.ndarray:
    """Scale each row to unit length so dot product == cosine similarity."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vectors / norms).astype(np.float32)


class Embedder(ABC):
    """Maps text to unit-length vectors."""

    name: str
    dimension: int

    @abstractmethod
    def embed_sync(self, texts: list[str]) -> np.ndarray:
        """Embed a batch. Blocking — call via ``embed`` from async code."""

    async def embed(self, texts: list[str]) -> np.ndarray:
        """Embed off the event loop.

        Embedding is CPU-bound and sits in the request hot path; running it
        inline would stall every other in-flight request, including streams.
        """
        return await asyncio.to_thread(self.embed_sync, texts)

    async def embed_one(self, text: str) -> np.ndarray:
        return (await self.embed([text]))[0]


class FastEmbedEmbedder(Embedder):
    """Local ONNX sentence-transformer via the ``fastembed`` package."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        from fastembed import TextEmbedding

        self.name = f"fastembed:{model_name}"
        # First construction downloads the model (~130MB) and caches it on disk.
        self._model = TextEmbedding(model_name=model_name)
        self.dimension = len(next(iter(self._model.embed(["dimension probe"]))))

    def embed_sync(self, texts: list[str]) -> np.ndarray:
        vectors = np.array(list(self._model.embed(texts)), dtype=np.float32)
        return normalize(vectors)


class HashingEmbedder(Embedder):
    """Lexical fallback: hashed bag of words plus character trigrams.

    Character trigrams give it partial credit for morphology ("refund"/"refunds")
    and typos, which a pure word-level hash would miss. It is still lexical: it
    cannot know that "get my money back" means "refund".
    """

    def __init__(self, dimension: int = 512):
        self.name = "hashing"
        self.dimension = dimension

    def _bucket(self, token: str) -> int:
        digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
        return int.from_bytes(digest, "big") % self.dimension

    def embed_sync(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            words = _WORD_RE.findall(text.lower())
            for word in words:
                vectors[row, self._bucket(word)] += 1.0
            padded = f" {' '.join(words)} "
            for i in range(len(padded) - 2):
                vectors[row, self._bucket(padded[i : i + 3])] += 0.3
        return normalize(vectors)


@lru_cache
def get_embedder(backend: str = "fastembed", model_name: str = "BAAI/bge-small-en-v1.5") -> Embedder:
    """Build the configured embedder once per process.

    Falls back to the lexical embedder rather than failing startup if the model
    can't be loaded (no network on first run, for example) — a degraded cache is
    better than a gateway that won't boot.
    """
    if backend == "hash":
        return HashingEmbedder()
    try:
        return FastEmbedEmbedder(model_name)
    except Exception as exc:  # noqa: BLE001 - any load failure degrades the same way
        import logging

        logging.getLogger(__name__).warning(
            "Could not load embedding model %s (%s). Falling back to the lexical "
            "embedder; paraphrase matching will be weaker.",
            model_name,
            exc,
        )
        return HashingEmbedder()
