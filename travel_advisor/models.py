"""Domain dataclasses shared across ingestion, retrieval, and tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Chunk:
    """A single retrievable text chunk derived from a knowledge-base document."""

    chunk_id: str           # unique within the corpus, e.g. "dest-hanoi#2"
    doc_id: str             # source document, e.g. "dest-hanoi"
    title: str              # document title for citation
    source_file: str        # filename within knowledge_base/
    text: str               # the chunk content actually embedded
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievedChunk:
    """A Chunk plus its similarity score from the most recent search."""

    chunk: Chunk
    score: float
