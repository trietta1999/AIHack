"""Markdown -> Chunk ingestion.

Knowledge base files follow a simple convention: one or more YAML
front-matter blocks per file, each immediately followed by the document
body until the next `---` separator. This lets us pack multiple short
documents into one file for readability without rebuilding the schema.

The chunker splits each document body by paragraph, then greedily packs
paragraphs into chunks no larger than `max_chars`. This keeps embeddings
focused and is independent of any LLM library.
"""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path
from typing import Iterable

from .models import Chunk

_FRONT_MATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL | re.MULTILINE)
_PARAGRAPH_RE = re.compile(r"\n\s*\n")


def _parse_front_matter(block: str) -> dict[str, object]:
    """Tiny YAML subset parser: `key: value` and `key: [a, b]` only.

    Avoiding PyYAML keeps the package dependency-light. The KB files only
    use scalar strings and one-line lists, so this is enough.
    """
    out: dict[str, object] = {}
    for raw_line in block.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            inner = value[1:-1].strip()
            items = [v.strip() for v in inner.split(",") if v.strip()]
            out[key] = items
        else:
            out[key] = value
    return out


def _split_documents(text: str) -> list[tuple[dict[str, object], str]]:
    """Yield (front_matter, body) pairs from a single markdown file."""
    documents: list[tuple[dict[str, object], str]] = []
    cursor = 0
    while cursor < len(text):
        match = _FRONT_MATTER_RE.search(text, cursor)
        if match is None:
            tail = text[cursor:].strip()
            if tail and documents:
                # Append leftover to the last body if no front matter exists.
                last_meta, last_body = documents[-1]
                documents[-1] = (last_meta, (last_body + "\n\n" + tail).strip())
            break
        fm = _parse_front_matter(match.group(1))
        body_start = match.end()
        next_match = _FRONT_MATTER_RE.search(text, body_start)
        body_end = next_match.start() if next_match else len(text)
        body = text[body_start:body_end].strip()
        if body:
            documents.append((fm, body))
        cursor = body_end
    return documents


def _chunk_body(body: str, max_chars: int) -> list[str]:
    """Greedy paragraph packing. Single paragraphs > max_chars stay as-is."""
    if not body.strip():
        return []
    paragraphs = [p.strip() for p in _PARAGRAPH_RE.split(body) if p.strip()]
    chunks: list[str] = []
    buffer: list[str] = []
    buffer_len = 0
    for para in paragraphs:
        if buffer and buffer_len + 2 + len(para) > max_chars:
            chunks.append("\n\n".join(buffer))
            buffer = [para]
            buffer_len = len(para)
        else:
            buffer.append(para)
            buffer_len += 2 + len(para)
    if buffer:
        chunks.append("\n\n".join(buffer))
    return chunks


def chunks_from_file(path: Path, *, max_chars: int) -> list[Chunk]:
    text = path.read_text(encoding="utf-8")
    out: list[Chunk] = []
    for fm, body in _split_documents(text):
        doc_id = str(fm.get("doc_id") or path.stem)
        title = str(fm.get("title") or doc_id)
        meta = {k: v for k, v in fm.items() if k not in {"doc_id", "title"}}
        for idx, piece in enumerate(_chunk_body(body, max_chars)):
            out.append(
                Chunk(
                    chunk_id=f"{doc_id}#{idx}",
                    doc_id=doc_id,
                    title=title,
                    source_file=path.name,
                    text=piece,
                    metadata=meta,
                )
            )
    return out


def chunks_from_dir(directory: Path, *, max_chars: int) -> list[Chunk]:
    """Walk a directory of .md files (alphabetical) and produce chunks."""
    if not directory.exists():
        raise FileNotFoundError(f"Knowledge base directory not found: {directory}")
    files = sorted(p for p in directory.glob("*.md") if p.is_file())
    out: list[Chunk] = []
    for f in files:
        out.extend(chunks_from_file(f, max_chars=max_chars))
    return out


def attach_chunk_text_prefix(chunks: Iterable[Chunk], prefix_fn) -> list[Chunk]:
    """Rebuild chunks with an extra leading line (e.g. title) for embedding context."""
    return [replace(c, text=f"{prefix_fn(c)}\n\n{c.text}") for c in chunks]
