"""Alternative parser: Docling's layout model + TableFormer, mapped onto our Block type.

Docling classifies page furniture (headers/footers) itself and reconstructs tables, which the
font-heuristic parser cannot. Output feeds the same section-aware chunker.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from localrag.types import Block


@lru_cache(maxsize=1)
def _converter():
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    opts = PdfPipelineOptions(do_ocr=False, do_table_structure=True)  # born-digital PDFs: no OCR needed
    return DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})


def parse_pdf(path: str | Path, min_block_chars: int = 20) -> list[Block]:
    from docling_core.types.doc import DocItemLabel, SectionHeaderItem, TableItem, TextItem

    path = Path(path)
    doc = _converter().convert(path).document
    blocks: list[Block] = []
    for item, _depth in doc.iterate_items():   # body only: page headers/footers are excluded
        if not getattr(item, "prov", None):
            continue
        page = item.prov[0].page_no
        if isinstance(item, TableItem):
            md = item.export_to_markdown(doc).strip()
            if md:
                blocks.append(Block("table", md, path.name, page))
        elif isinstance(item, SectionHeaderItem):
            blocks.append(Block("heading", item.text.strip(), path.name, page, level=item.level + 1))
        elif isinstance(item, TextItem):
            text = item.text.strip()
            if item.label == DocItemLabel.TITLE:
                blocks.append(Block("heading", text, path.name, page, level=1))
            elif item.label not in (DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER) and len(text) >= min_block_chars:
                blocks.append(Block("paragraph", text, path.name, page))
    return blocks


def parse_folder(pdf_dir: str | Path, min_block_chars: int = 20) -> list[Block]:
    blocks = []
    for pdf in sorted(Path(pdf_dir).glob("*.pdf")):
        try:
            parsed = parse_pdf(pdf, min_block_chars)
            print(f"  parsed {pdf.name}: {len(parsed)} blocks ({sum(b.kind == 'table' for b in parsed)} tables)")
            blocks.extend(parsed)
        except Exception as e:
            print(f"  could not parse {pdf.name}: {e}")
    return blocks
