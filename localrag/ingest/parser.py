"""Layout-aware PDF parsing with PyMuPDF.

Classifies text blocks into headings (with a hierarchy level) and paragraphs using
font statistics, and strips running headers/footers that repeat across pages.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pymupdf as fitz

from localrag.types import Block

_NUMBERED = re.compile(r"^(\(?[0-9ivxIVX]+[.)]|[A-Z][.)]|§|Sec\.|Subpart|Part|Section|Chapter|Appendix)\s*\S")


def _block_text(block: dict) -> str:
    """Join a block's lines into flowing text, undoing end-of-line hyphenation ("manu-" + "facturer")."""
    text = ""
    for line in block.get("lines", []):
        part = "".join(span["text"] for span in line["spans"]).strip()
        if not part:
            continue
        if text.endswith("-") and part[:1].islower():
            text = text[:-1] + part
        else:
            text = f"{text} {part}" if text else part
    return re.sub(r"\s+", " ", text).strip()


def _reading_order(page: fitz.Page) -> list[dict]:
    """Text blocks in column-major reading order: left column top-to-bottom, then right.

    Blocks starting left of the page centre count as the left column (this includes
    full-width blocks), so single-column pages are simply sorted top-to-bottom.
    """
    mid = (page.rect.x0 + page.rect.x1) / 2
    blocks = [b for b in page.get_text("dict")["blocks"] if b["type"] == 0 and _block_text(b)]
    return sorted(blocks, key=lambda b: (b["bbox"][0] >= mid - 5, b["bbox"][1]))


def _block_font(block: dict) -> tuple[float, bool]:
    """Dominant (by character count) font size and boldness of a block."""
    sizes: Counter = Counter()
    bold_chars = total = 0
    for line in block.get("lines", []):
        for span in line["spans"]:
            n = len(span["text"].strip())
            sizes[round(span["size"], 1)] += n
            total += n
            if "bold" in span["font"].lower() or span.get("flags", 0) & 16:
                bold_chars += n
    if not total:
        return 0.0, False
    return sizes.most_common(1)[0][0], bold_chars / total > 0.6


def _body_font_size(doc: fitz.Document) -> float:
    """Most common font size weighted by characters: the body text size."""
    sizes: Counter = Counter()
    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            if block["type"] != 0:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    sizes[round(span["size"], 1)] += len(span["text"].strip())
    return sizes.most_common(1)[0][0] if sizes else 11.0


def _normalise(text: str) -> str:
    # digits vary between pages (page numbers, section numbers), so abstract them away
    return re.sub(r"\d+", "#", text)


def _running_text(doc: fitz.Document, min_share: float = 0.3) -> set[str]:
    """Headers/footers: text among the first or last two blocks (by height) of a page that repeats
    on many pages. Using relative position works even when the page has wide scan margins."""
    counts: Counter = Counter()
    for page in doc:
        blocks = sorted((b for b in page.get_text("dict")["blocks"] if b["type"] == 0 and _block_text(b)),
                        key=lambda b: b["bbox"][1])
        counts.update({_normalise(_block_text(b)) for b in blocks[:2] + blocks[-2:]})
    return {t for t, c in counts.items() if len(doc) >= 3 and c >= 2 and c / len(doc) >= min_share}


def is_heading(text: str, size: float, bold: bool, body_size: float) -> bool:
    if len(text) > 150 or text.endswith((".", ",", ";", ":")) and not _NUMBERED.match(text):
        return False
    if size >= body_size + 1:
        return True
    return bold and len(text) < 120


def parse_pdf(path: str | Path, min_block_chars: int = 20) -> list[Block]:
    path = Path(path)
    doc = fitz.open(path)
    body = _body_font_size(doc)
    running = _running_text(doc)

    raw = []  # (text, size, bold, page)
    for page_no, page in enumerate(doc, start=1):
        for block in _reading_order(page):
            text = _block_text(block)
            if _normalise(text) in running:
                continue
            size, bold = _block_font(block)
            raw.append((text, size, bold, page_no))
    doc.close()

    # Heading levels: larger font -> higher level. Bold body-size headings get the lowest level.
    heading_sizes = sorted({s for t, s, b, _ in raw if is_heading(t, s, b, body) and s >= body + 1},
                           reverse=True)
    level_of = {s: i + 1 for i, s in enumerate(heading_sizes)}
    bold_level = len(heading_sizes) + 1

    blocks = []
    for text, size, bold, page_no in raw:
        if is_heading(text, size, bold, body):
            blocks.append(Block("heading", text, path.name, page_no,
                                level_of.get(size, bold_level)))
        elif len(text) >= min_block_chars:
            blocks.append(Block("paragraph", text, path.name, page_no))
    return blocks


def parse_folder(pdf_dir: str | Path, min_block_chars: int = 20) -> list[Block]:
    blocks = []
    for pdf in sorted(Path(pdf_dir).glob("*.pdf")):
        try:
            parsed = parse_pdf(pdf, min_block_chars)
            print(f"  parsed {pdf.name}: {len(parsed)} blocks")
            blocks.extend(parsed)
        except Exception as e:  # a single bad PDF should not stop ingestion
            print(f"  could not parse {pdf.name}: {e}")
    return blocks
