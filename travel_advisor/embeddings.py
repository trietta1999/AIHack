"""Embedding clients: Azure OpenAI (production) and deterministic fake (tests).

Both expose the same `embed(texts)` interface returning a `numpy.ndarray`
of shape `(n_texts, dim)`, dtype `float32`, **L2-normalised row-wise** so
the cosine similarity reduces to a dot product downstream.
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Iterable, Sequence

import numpy as np
from openai import (
    APIConnectionError,
    APITimeoutError,
    AzureOpenAI,
    InternalServerError,
    RateLimitError,
)
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from .config import settings

logger = logging.getLogger(__name__)

EMBED_BATCH_SIZE = 64
MAX_INPUT_CHARS = 6000
MAX_ATTEMPTS = 5
BACKOFF_MIN = 1
BACKOFF_MAX = 20

RETRYABLE_ERRORS: tuple[type[Exception], ...] = (
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
    InternalServerError,
)


def _l2_normalise(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (arr / norms).astype(np.float32)


class AzureEmbedder:
    """Azure OpenAI text-embedding client with batching + retry."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        deployment: str | None = None,
        endpoint: str | None = None,
        api_version: str | None = None,
    ):
        key = api_key or settings.api_key or os.getenv("AZURE_OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "AZURE_OPENAI_API_KEY missing — copy .env.example to .env "
                "and paste the key from the course staff."
            )
        self.deployment = deployment or settings.embedding_deployment
        self.client = AzureOpenAI(
            api_key=key,
            azure_endpoint=endpoint or settings.endpoint,
            api_version=api_version or settings.api_version,
        )

    @retry(
        retry=retry_if_exception_type(RETRYABLE_ERRORS),
        stop=stop_after_attempt(MAX_ATTEMPTS),
        wait=wait_random_exponential(min=BACKOFF_MIN, max=BACKOFF_MAX),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        resp = self.client.embeddings.create(model=self.deployment, input=batch)
        return [d.embedding for d in resp.data]

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        rows: list[list[float]] = []
        for i in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = [str(t)[:MAX_INPUT_CHARS] for t in texts[i : i + EMBED_BATCH_SIZE]]
            rows.extend(self._embed_batch(batch))
        return _l2_normalise(np.asarray(rows, dtype=np.float32))


class HashingEmbedder:
    """Deterministic bag-of-words embedder used by offline tests.

    Hashes whitespace-tokenised words into a fixed-width vector via MD5.
    Same text -> same vector; semantically related words rarely collide so
    this is **not** suitable for production retrieval. It is suitable for
    tests that just need stable similarity-by-overlap behaviour.
    """

    def __init__(self, dim: int = 256):
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.deployment = "hashing-fake"
        self.dim = dim

    def _vectorise(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for token in text.lower().split():
            h = int.from_bytes(hashlib.md5(token.encode("utf-8")).digest()[:4], "big")
            vec[h % self.dim] += 1.0
        return vec

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        arr = np.vstack([self._vectorise(t) for t in texts])
        return _l2_normalise(arr)


def make_embedder() -> AzureEmbedder:
    """Default factory for production use."""
    return AzureEmbedder()
