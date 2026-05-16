"""Runtime configuration (Azure deployments, paths, tunables).

Course-staff-issued endpoint and deployment names are baked in as defaults
because they are not secrets. Override via environment variables when
running against a different Azure tenant.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    # Azure OpenAI
    api_key: str | None
    endpoint: str
    api_version: str
    chat_deployment: str
    embedding_deployment: str

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
    sample_queries_path: Path


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def load_settings() -> Settings:
    return Settings(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        endpoint=os.getenv(
            "AZURE_OPENAI_ENDPOINT",
            "https://aiportalapi.stu-platform.live/jpe",
        ),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-07-01-preview"),
        chat_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o-mini"),
        embedding_deployment=os.getenv(
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small"
        ),
        chunk_max_chars=int(os.getenv("CHUNK_MAX_CHARS", "1200")),
        top_k_default=int(os.getenv("TOP_K_DEFAULT", "4")),
        recursion_limit=int(os.getenv("AGENT_RECURSION_LIMIT", "12")),
        temperature=float(os.getenv("AGENT_TEMPERATURE", "0.2")),
        knowledge_base_dir=BASE_DIR / "data" / "knowledge_base",
        index_dir=BASE_DIR / "data" / "faiss_index",
        bookings_db_path=BASE_DIR / "data" / "bookings.sqlite",
        sample_queries_path=BASE_DIR / "sample_queries.json",
    )


settings = load_settings()
