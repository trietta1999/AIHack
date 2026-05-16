"""Runtime configuration (OpenAI models, paths, tunables).

Reads `OPENAI_API_KEY` from env. Override model names via env vars when
you want to swap models without code changes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    # OpenAI
    api_key: str | None
    chat_model: str
    embedding_model: str

    # Retrieval tunables
    chunk_max_chars: int
    top_k_default: int

    # Agent tunables
    recursion_limit: int
    temperature: float

    # Filesystem
    knowledge_base_dir: Path
    index_dir: Path
    bookings_db_path: Path
    chats_db_path: Path
    sample_queries_path: Path


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def load_settings() -> Settings:
    return Settings(
        api_key=os.getenv("OPENAI_API_KEY"),
        chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        chunk_max_chars=int(os.getenv("CHUNK_MAX_CHARS", "1200")),
        top_k_default=int(os.getenv("TOP_K_DEFAULT", "4")),
        recursion_limit=int(os.getenv("AGENT_RECURSION_LIMIT", "12")),
        temperature=float(os.getenv("AGENT_TEMPERATURE", "0.2")),
        knowledge_base_dir=BASE_DIR / "data" / "knowledge_base",
        index_dir=BASE_DIR / "data" / "faiss_index",
        bookings_db_path=BASE_DIR / "data" / "bookings.sqlite",
        chats_db_path=BASE_DIR / "data" / "chats.sqlite",
        sample_queries_path=BASE_DIR / "sample_queries.json",
    )


settings = load_settings()
