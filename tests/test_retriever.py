"""FAISS retriever build / search / persist round-trip."""

from __future__ import annotations

from pathlib import Path

from travel_advisor.embeddings import HashingEmbedder
from travel_advisor.models import Chunk
from travel_advisor.retriever import Retriever, build_retriever


def _toy_chunks() -> list[Chunk]:
    return [
        Chunk(
            chunk_id="t-hanoi#0",
            doc_id="t-hanoi",
            title="Hanoi guide",
            source_file="t.md",
            text="Hanoi pho bun cha old quarter hoan kiem",
            metadata={"region": "north", "tags": ["destinations"]},
        ),
        Chunk(
            chunk_id="t-halong#0",
            doc_id="t-halong",
            title="Halong cruise",
            source_file="t.md",
            text="Halong bay limestone karst cruise overnight kayak",
            metadata={"region": "north", "tags": ["cruise"]},
        ),
        Chunk(
            chunk_id="t-saigon#0",
            doc_id="t-saigon",
            title="Saigon city",
            source_file="t.md",
            text="Ho Chi Minh saigon district 1 mekong delta",
            metadata={"region": "south", "tags": ["destinations"]},
        ),
    ]


def test_build_and_search_returns_relevant_first():
    embedder = HashingEmbedder(dim=128)
    retriever = build_retriever(_toy_chunks(), embedder=embedder)
    hits = retriever.search("hoan kiem old quarter pho", top_k=2)
    assert hits, "expected at least 1 hit"
    assert hits[0].chunk.doc_id == "t-hanoi"


def test_search_respects_filter_fn():
    embedder = HashingEmbedder(dim=128)
    retriever = build_retriever(_toy_chunks(), embedder=embedder)
    hits = retriever.search(
        "destinations",
        top_k=3,
        filter_fn=lambda c: c.metadata.get("region") == "south",
    )
    assert hits, "expected at least 1 hit after filter"
    for h in hits:
        assert h.chunk.metadata["region"] == "south"


def test_search_empty_query_returns_empty():
    embedder = HashingEmbedder(dim=64)
    retriever = build_retriever(_toy_chunks(), embedder=embedder)
    assert retriever.search("", top_k=3) == []
    assert retriever.search("   ", top_k=3) == []


def test_save_and_load_roundtrip(tmp_path: Path):
    embedder = HashingEmbedder(dim=128)
    retriever = build_retriever(_toy_chunks(), embedder=embedder)
    retriever.save(tmp_path)
    loaded = Retriever.load(tmp_path, embedder)
    assert len(loaded.chunks) == 3
    assert loaded.dim == 128
    hits = loaded.search("halong cruise overnight", top_k=1)
    assert hits[0].chunk.doc_id == "t-halong"


def test_load_missing_files_raises(tmp_path: Path):
    import pytest

    with pytest.raises(FileNotFoundError):
        Retriever.load(tmp_path, HashingEmbedder(dim=32))


def test_list_titles_unique():
    embedder = HashingEmbedder(dim=64)
    retriever = build_retriever(_toy_chunks(), embedder=embedder)
    titles = retriever.list_titles()
    assert set(titles) == {"Hanoi guide", "Halong cruise", "Saigon city"}
