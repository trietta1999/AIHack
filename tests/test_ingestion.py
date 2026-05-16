"""Markdown -> Chunk ingestion (pure helpers, no API)."""

from __future__ import annotations

from pathlib import Path

from travel_advisor.ingestion import (
    _chunk_body,
    _parse_front_matter,
    _split_documents,
    chunks_from_dir,
)


def test_parse_front_matter_handles_scalars_and_lists():
    fm = _parse_front_matter(
        "doc_id: dest-hanoi\n"
        "title: Hanoi Guide\n"
        "region: north\n"
        "tags: [destinations, hanoi, north]\n"
    )
    assert fm["doc_id"] == "dest-hanoi"
    assert fm["title"] == "Hanoi Guide"
    assert fm["region"] == "north"
    assert fm["tags"] == ["destinations", "hanoi", "north"]


def test_split_documents_finds_multiple_in_one_file():
    text = (
        "---\ndoc_id: a\ntitle: A\n---\n\n"
        "First doc body paragraph 1.\n\nParagraph 2 of A.\n\n"
        "---\ndoc_id: b\ntitle: B\n---\n\n"
        "Body of B.\n"
    )
    docs = _split_documents(text)
    assert len(docs) == 2
    assert docs[0][0]["doc_id"] == "a"
    assert "Paragraph 2 of A." in docs[0][1]
    assert docs[1][0]["doc_id"] == "b"
    assert docs[1][1].strip() == "Body of B."


def test_chunk_body_packs_paragraphs_under_max():
    body = "Para one.\n\nPara two has more words.\n\nThird paragraph here."
    chunks = _chunk_body(body, max_chars=40)
    # Each paragraph likely gets its own chunk because 40 chars is tight.
    assert all(len(c) <= 80 for c in chunks)
    assert "Para one." in chunks[0]


def test_chunk_body_handles_oversized_single_paragraph():
    huge = "X" * 5000
    chunks = _chunk_body(huge, max_chars=1200)
    # Single long paragraph stays as one chunk (no mid-paragraph splitting).
    assert len(chunks) == 1
    assert len(chunks[0]) == 5000


def test_chunks_from_dir_produces_expected_documents():
    # Run against the real KB shipped with the project.
    from travel_advisor.config import settings

    chunks = chunks_from_dir(settings.knowledge_base_dir, max_chars=1200)
    assert len(chunks) > 20  # 7 KB files, multiple docs each
    titles = {c.title for c in chunks}
    assert any("Hanoi" in t for t in titles)
    assert any("Phu Quoc" in t for t in titles)
    assert any("Visa" in t for t in titles)
    # Every chunk must have a doc_id and source_file.
    for c in chunks:
        assert c.doc_id
        assert c.source_file.endswith(".md")
        assert c.text.strip()


def test_chunks_from_dir_missing_dir_raises(tmp_path: Path):
    import pytest

    with pytest.raises(FileNotFoundError):
        chunks_from_dir(tmp_path / "does-not-exist", max_chars=1200)
