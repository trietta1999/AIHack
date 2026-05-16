"""FAISS-backed vector retriever for the travel knowledge base.

Uses `faiss.IndexFlatIP` (inner product). Because every vector in this
project is L2-normalised at embed time, inner product equals cosine
similarity in `[-1, 1]`.

The index plus per-chunk metadata persist to a directory containing two
files: `index.faiss` (binary) and `chunks.json` (sidecar). Loading
reconstructs both into a `Retriever` instance ready for `.search(...)`.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import faiss
import numpy as np

from .models import Chunk, RetrievedChunk

INDEX_FILENAME = "index.faiss"
META_FILENAME = "chunks.json"


class Retriever:
    """In-memory FAISS vector store + metadata, embedder kept as live ref."""

    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, embedder: Any):
        if vectors.ndim != 2:
            raise ValueError("vectors must be 2-D")
        if len(chunks) != vectors.shape[0]:
            raise ValueError(
                f"chunk count ({len(chunks)}) must match vector rows "
                f"({vectors.shape[0]})"
            )
        self.chunks = chunks
        self.embedder = embedder
        self.dim = int(vectors.shape[1])
        self._index = faiss.IndexFlatIP(self.dim)
        if vectors.shape[0] > 0:
            self._index.add(vectors.astype(np.float32))

    # ----- query API -----

    def search(
        self,
        query: str,
        *,
        top_k: int = 4,
        filter_fn: Callable[[Chunk], bool] | None = None,
    ) -> list[RetrievedChunk]:
        if not query or not query.strip():
            return []
        if self._index.ntotal == 0:
            return []
        qvec = self.embedder.embed([query]).astype(np.float32)
        if qvec.size == 0:
            return []
        # Over-fetch when filtering so we still have top_k after filtering.
        fetch = max(top_k * 5, top_k) if filter_fn else top_k
        fetch = min(fetch, self._index.ntotal)
        scores, idxs = self._index.search(qvec, fetch)
        out: list[RetrievedChunk] = []
        for score, idx in zip(scores[0].tolist(), idxs[0].tolist()):
            if idx < 0:
                continue
            chunk = self.chunks[idx]
            if filter_fn is not None and not filter_fn(chunk):
                continue
            out.append(RetrievedChunk(chunk=chunk, score=float(score)))
            if len(out) >= top_k:
                break
        return out

    def list_titles(self) -> list[str]:
        seen: dict[str, None] = {}
        for c in self.chunks:
            seen.setdefault(c.title, None)
        return list(seen.keys())

    def list_tags(self) -> list[str]:
        tags: set[str] = set()
        for c in self.chunks:
            raw = c.metadata.get("tags") or []
            if isinstance(raw, str):
                tags.add(raw)
            else:
                tags.update(str(t) for t in raw)
        return sorted(tags)

    # ----- persistence -----

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(directory / INDEX_FILENAME))
        payload = {
            "dim": self.dim,
            "chunks": [asdict(c) for c in self.chunks],
        }
        (directory / META_FILENAME).write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: Path, embedder: Any) -> "Retriever":
        index_path = directory / INDEX_FILENAME
        meta_path = directory / META_FILENAME
        if not index_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"Missing FAISS index files in {directory}. "
                f"Run `python scripts/build_index.py` first."
            )
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        chunks = [Chunk(**c) for c in payload["chunks"]]
        index = faiss.read_index(str(index_path))
        # Reconstruct vectors so we keep a single source of truth.
        vectors = (
            index.reconstruct_n(0, index.ntotal)
            if index.ntotal > 0
            else np.zeros((0, payload["dim"]), dtype=np.float32)
        )
        return cls(chunks=chunks, vectors=vectors, embedder=embedder)


def build_retriever(chunks: list[Chunk], embedder: Any) -> Retriever:
    """Embed all chunks in batches and return a fresh retriever."""
    if not chunks:
        return Retriever(chunks=[], vectors=np.zeros((0, 1), dtype=np.float32), embedder=embedder)
    vectors = embedder.embed([c.text for c in chunks])
    if vectors.shape[0] == 0:
        return Retriever(chunks=[], vectors=np.zeros((0, 1), dtype=np.float32), embedder=embedder)
    return Retriever(chunks=chunks, vectors=vectors, embedder=embedder)


# ----- module-level cache so @tool wrappers share one instance -----

_RETRIEVER: Retriever | None = None


def get_retriever() -> Retriever:
    if _RETRIEVER is None:
        raise RuntimeError(
            "Retriever not initialised. Call `set_retriever(...)` "
            "(tests) or `load_default_retriever()` (production)."
        )
    return _RETRIEVER


def set_retriever(retriever: Retriever | None) -> None:
    global _RETRIEVER
    _RETRIEVER = retriever


def load_default_retriever() -> Retriever:
    """Production helper: lazy-load FAISS from disk + OpenAIEmbedder."""
    from .config import settings as _settings
    from .embeddings import OpenAIEmbedder

    retr = Retriever.load(_settings.index_dir, OpenAIEmbedder())
    set_retriever(retr)
    return retr
