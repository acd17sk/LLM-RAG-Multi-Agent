from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal, Optional


@dataclass
class Block:
    """A text block from the PDF parser, classified as a heading (with level) or body text."""
    kind: Literal["heading", "paragraph", "table"]
    text: str
    source: str
    page: int
    level: int = 0  # 1 = top-level heading; 0 for paragraphs


@dataclass
class Chunk:
    id: str
    text: str                 # the passage shown to the LLM
    source: str
    page: int
    section: str = ""         # heading path, e.g. "Subpart C > 820.30 Design controls"
    context: str = ""         # optional LLM-written situating context (contextual retrieval)
    parent_id: str = ""       # set on child chunks (parent-child retrieval)
    page_end: int = 0         # last page, when a passage spans pages (expanded context)

    @property
    def pages(self) -> set[tuple[str, int]]:
        return {(self.source, p) for p in range(self.page, max(self.page, self.page_end) + 1)}

    @property
    def index_text(self) -> str:
        """Text used for embedding and BM25: section path + situating context + passage."""
        return "\n".join(p for p in (self.section, self.context, self.text) if p)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Hit:
    chunk: Chunk
    score: float
    scores: dict[str, float] = field(default_factory=dict)  # per-stage scores for tracing


@dataclass
class Claim:
    text: str
    citations: list[int]          # 1-based indices into Answer.passages
    supported: Optional[bool] = None
    support_score: Optional[float] = None


@dataclass
class Answer:
    query: str
    action: str
    text: str
    claims: list[Claim] = field(default_factory=list)
    passages: list[Hit] = field(default_factory=list)
    subqueries: list[str] = field(default_factory=list)
    insufficient_context: bool = False
    timings: dict[str, float] = field(default_factory=dict)
    trace: dict = field(default_factory=dict)   # CRAG rewrites, self-correction outcome, ...

    def render(self) -> str:
        """Markdown answer with numbered citations and a reference list."""
        claims = [c for c in self.claims if c.supported is not False]
        if not claims:
            return self.text
        body = " ".join(
            c.text.rstrip() + "".join(f" [{i}]" for i in c.citations) for c in claims)
        used = sorted({i for c in claims for i in c.citations})
        refs = "\n".join(
            f"[{i}] {self.passages[i - 1].chunk.source}, p. {self.passages[i - 1].chunk.page}"
            + (f"–{self.passages[i - 1].chunk.page_end}"
               if self.passages[i - 1].chunk.page_end > self.passages[i - 1].chunk.page else "")
            + (f" — {self.passages[i - 1].chunk.section}" if self.passages[i - 1].chunk.section else "")
            for i in used)
        return f"{body}\n\n{refs}" if refs else body
