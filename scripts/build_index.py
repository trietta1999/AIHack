"""Build the FAISS vector index from the markdown knowledge base.

    python scripts/build_index.py

Run once after cloning, or any time you edit `data/knowledge_base/*.md`.
Requires OPENAI_API_KEY in env or .env file.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from travel_advisor.config import settings
from travel_advisor.embeddings import OpenAIEmbedder
from travel_advisor.ingestion import chunks_from_dir
from travel_advisor.retriever import build_retriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("build_index")


def main(kb_dir: Path | None = None, index_dir: Path | None = None) -> None:
    kb = kb_dir or settings.knowledge_base_dir
    idx = index_dir or settings.index_dir

    logger.info("Reading knowledge base from %s", kb)
    chunks = chunks_from_dir(kb, max_chars=settings.chunk_max_chars)
    if not chunks:
        raise RuntimeError(f"No chunks produced from {kb} — is the directory empty?")
    logger.info("Produced %d chunks", len(chunks))

    embedder = OpenAIEmbedder()
    start = time.perf_counter()
    retriever = build_retriever(chunks, embedder=embedder)
    retriever.save(idx)
    elapsed = time.perf_counter() - start
    logger.info(
        "Saved FAISS index to %s (dim=%d, n=%d, elapsed=%.2fs, model=%s)",
        idx,
        retriever.dim,
        len(retriever.chunks),
        elapsed,
        embedder.model,
    )


if __name__ == "__main__":
    main()
