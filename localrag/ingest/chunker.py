"""Section-aware chunking.

Paragraphs are grouped under their heading path and packed into chunks that never
cross a section or page boundary. Chunk IDs are content hashes, so re-ingesting
unchanged documents is idempotent.
"""
from __future__ import annotations

import hashlib
import re

from localrag.types import Block, Chunk

_SENTENCE = re.compile(r"(?<=[.!?;])\s+(?=[A-Z(\"'§0-9])")


def chunk_id(source: str, page: int, text: str) -> str:
    return hashlib.sha1(f"{source}|{page}|{text}".encode()).hexdigest()[:16]


def split_text(text: str, size: int, overlap: int) -> list[str]:
    """Split on sentence boundaries into pieces of at most ~size chars, with sentence overlap."""
    if len(text) <= size:
        return [text]
    sentences = [s for s in _SENTENCE.split(text) if s.strip()]
    pieces, current = [], []
    for sent in sentences:
        # a single sentence longer than the budget is hard-split
        while len(sent) > size:
            if current:
                pieces.append(" ".join(current))
                current = []
            pieces.append(sent[:size])
            sent = sent[size - overlap:]
        if current and len(" ".join(current)) + len(sent) + 1 > size:
            pieces.append(" ".join(current))
            # carry trailing sentences as overlap
            tail, n = [], 0
            for s in reversed(current):
                if n + len(s) > overlap:
                    break
                tail.insert(0, s)
                n += len(s) + 1
            current = tail
        current.append(sent)
    if current:
        pieces.append(" ".join(current))
    return pieces


def split_table(markdown: str, size: int) -> list[str]:
    """Split a markdown table by rows, repeating the header row in every piece."""
    lines = markdown.splitlines()
    if len(markdown) <= size or len(lines) < 3:
        return [markdown]
    header, rows = lines[:2], lines[2:]
    pieces, current = [], []
    for row in rows:
        if current and len("\n".join(header + current + [row])) > size:
            pieces.append("\n".join(header + current))
            current = []
        current.append(row)
    if current:
        pieces.append("\n".join(header + current))
    return pieces


def chunk_blocks(blocks: list[Block], size: int = 1200, overlap: int = 200) -> list[Chunk]:
    chunks: list[Chunk] = []
    path: list[tuple[int, str]] = []   # stack of (level, heading)
    buffer: list[str] = []
    buf_key: tuple[str, int, str] | None = None  # (source, page, section)

    def flush():
        nonlocal buffer
        if buffer and buf_key:
            source, page, section = buf_key
            for piece in split_text("\n".join(buffer), size, overlap):
                chunks.append(Chunk(chunk_id(source, page, piece), piece, source, page, section))
        buffer = []

    current_source = None
    for b in blocks:
        if b.source != current_source:
            flush()
            path, current_source = [], b.source
        if b.kind == "heading":
            flush()
            while path and path[-1][0] >= b.level:
                path.pop()
            path.append((b.level, b.text))
            continue
        key = (b.source, b.page, " > ".join(h for _, h in path))
        if key != buf_key or len("\n".join(buffer)) + len(b.text) > size:
            flush()
            buf_key = key
        if b.kind == "table":
            # tables are kept whole (split by rows only if too long), never merged with prose
            flush()
            buf_key = key
            for piece in split_table(b.text, size):
                chunks.append(Chunk(chunk_id(b.source, b.page, piece), piece, b.source, b.page, key[2]))
            continue
        buffer.append(b.text)
    flush()

    # identical boilerplate passages can occur twice on a page; keep the first
    seen, unique = set(), []
    for c in chunks:
        if c.id not in seen:
            seen.add(c.id)
            unique.append(c)
    return unique


def make_children(chunks: list[Chunk], size: int = 300, overlap: int = 50) -> list[Chunk]:
    """Parent-child ("small-to-big") retrieval: small, focused children are searched, and their
    parent chunk is what the LLM reads. Children inherit the parent's section and context."""
    children = []
    for p in chunks:
        for piece in split_text(p.text, size, overlap):
            children.append(Chunk(chunk_id(p.source, p.page, "child|" + p.id + piece), piece, p.source, p.page,
                                  p.section, p.context, parent_id=p.id))
    return children
