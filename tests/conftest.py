"""Shared pytest fixtures: hashing embedder + a real FAISS retriever
seeded from the actual knowledge base, plus a faked tool-calling LLM
so the LangGraph agent can be driven end-to-end without an API key.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4

import pytest

# Ensure the project root is importable when running `pytest` from any cwd.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from travel_advisor.config import settings  # noqa: E402
from travel_advisor.embeddings import HashingEmbedder  # noqa: E402
from travel_advisor.ingestion import chunks_from_dir  # noqa: E402
from travel_advisor.retriever import Retriever, build_retriever, set_retriever  # noqa: E402

from langchain_core.language_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402


# ---------------------------------------------------------------------------
# Knowledge base fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def hashing_embedder() -> HashingEmbedder:
    return HashingEmbedder(dim=256)


@pytest.fixture(scope="session")
def kb_chunks():
    return chunks_from_dir(settings.knowledge_base_dir, max_chars=settings.chunk_max_chars)


@pytest.fixture(scope="session")
def real_retriever(hashing_embedder: HashingEmbedder, kb_chunks) -> Retriever:
    return build_retriever(kb_chunks, embedder=hashing_embedder)


@pytest.fixture
def installed_retriever(real_retriever: Retriever) -> Iterator[Retriever]:
    """Install retriever globally for tool-under-test; clean up afterwards."""
    set_retriever(real_retriever)
    try:
        yield real_retriever
    finally:
        set_retriever(None)


# ---------------------------------------------------------------------------
# Seeded SQLite bookings DB
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def _seeded_bookings_db():
    """Make sure the bookings.sqlite exists before any SQL-tool test runs."""
    from scripts.seed_db import seed

    if not settings.bookings_db_path.exists():
        seed()
    yield


# ---------------------------------------------------------------------------
# Fake tool-calling chat model for LangGraph agent tests
# ---------------------------------------------------------------------------


class FakeToolCallingChatModel(BaseChatModel):
    """A scripted chat model that emits a queue of AIMessage replies.

    Each entry in `replies` is either:
    - an AIMessage with `.tool_calls = [{name, args, id}]` (forces a tool call), or
    - an AIMessage with plain content (terminal answer).

    `create_react_agent` always invokes the LLM at least twice when a tool
    is called (once to plan, once to read the observation). The fake LLM
    pops from `replies` in order so tests stay deterministic.
    """

    replies: list[AIMessage] = []

    @property
    def _llm_type(self) -> str:
        return "fake-tool-calling"

    def bind_tools(self, tools, **_kwargs):  # noqa: ARG002
        # LangGraph's create_react_agent calls bind_tools; we ignore the
        # binding because the scripted replies already specify tool_calls.
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):  # noqa: ARG002
        if not self.replies:
            msg = AIMessage(content="(no more scripted replies)")
        else:
            msg = self.replies.pop(0)
        return ChatResult(generations=[ChatGeneration(message=msg)])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):  # noqa: ARG002
        return self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def make_tool_call(name: str, args: dict[str, Any]) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": f"call_{uuid4().hex[:8]}"}],
    )


def make_final(text: str) -> AIMessage:
    return AIMessage(content=text)
