"""Synthetic evaluation set: a stronger local 'teacher' model writes questions from sampled chunks.

Question types:
  single        one passage answers it (three styles: direct / paraphrased / practical)
  comparison    needs two related passages from different documents
  bridge        needs two passages from different sections of the same document
  unanswerable  plausible, but not answered by the corpus (filtered by a judge afterwards)

Each item records its gold (source, page) set so retrieval can be scored, plus a reference answer.
Splits are assigned by the gold page, not the question, so no page is gold in both train and test.
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from localrag.llm import LLM
from localrag.types import Chunk

SPLITS = (("train", 0.50), ("dev", 0.15), ("test", 0.35))

QA_SCHEMA = {
    "type": "object",
    "properties": {
        "usable": {"type": "boolean"},
        "question": {"type": "string", "maxLength": 300},
        "answer": {"type": "string", "maxLength": 600},
    },
    "required": ["usable", "question", "answer"],
}

QA_PROMPT = """You are building a test set for a question-answering system over regulatory documents.

Document: {source}
Section: {section}
Passage:
\"\"\"{text}\"\"\"

Write ONE question that a regulatory or quality engineer might realistically ask, which this passage answers.
Rules:
- Style: {style}
- The question must make sense on its own: never say "the passage", "this section", "this regulation" or "this document"; name the regulation, guidance or topic instead.
- The answer must be fully supported by the passage. Keep it to 1-3 sentences.
- Set usable to false if the passage is boilerplate (table of contents, contact info, references list, legal notices) or has no substantive content."""

STYLES = [
    "a direct factual question using the passage's own terminology.",
    "a paraphrased question that avoids reusing the passage's distinctive words (use synonyms).",
    "a practical 'what must a manufacturer do when...' or 'what should be included in...' question.",
]

MULTI_SCHEMA = {
    "type": "object",
    "properties": {
        "usable": {"type": "boolean"},
        "question": {"type": "string", "maxLength": 350},
        "answer": {"type": "string", "maxLength": 800},
        "needs_both": {"type": "boolean"},
    },
    "required": ["usable", "question", "answer", "needs_both"],
}

MULTI_PROMPT = """You are building a hard test set for a question-answering system over regulatory documents.

Passage A ({source_a}; section: {section_a}):
\"\"\"{text_a}\"\"\"

Passage B ({source_b}; section: {section_b}):
\"\"\"{text_b}\"\"\"

Write ONE realistic question that can only be answered by combining information from BOTH passages.
Type: {kind}
Rules:
- The question must make sense on its own: name the documents or topics, never "passage A" or "the passage".
- The answer must be fully supported by the two passages together. 2-4 sentences.
- Set needs_both to true only if neither passage alone is enough to answer it.
- Set usable to false if either passage is boilerplate or the passages share nothing meaningful to combine."""

KINDS = {
    "comparison": "a comparison: how do the two documents differ or agree on a related requirement or concept?",
    "bridge": "a multi-part question whose parts are answered by different passages (e.g. what is X, and what must be done about it).",
}

UNANSWERABLE_SCHEMA = {
    "type": "object",
    "properties": {"questions": {"type": "array", "minItems": 5, "maxItems": 5,
                                 "items": {"type": "string", "maxLength": 300}}},
    "required": ["questions"],
}

UNANSWERABLE_PROMPT = """The knowledge base contains only these documents: {sources}.
Write 5 realistic questions a user of this knowledge base might ask that these documents do NOT answer
(e.g. about other jurisdictions, clauses of other standards, pricing, or neighbouring topics the documents don't cover).
Make them sound similar in style to real questions about these documents.{avoid}"""


def assign_split(key: str) -> str:
    h = int(hashlib.sha1(key.encode()).hexdigest(), 16) % 1000 / 1000
    acc = 0.0
    for name, share in SPLITS:
        acc += share
        if h < acc:
            return name
    return SPLITS[-1][0]


def split_of(item: dict) -> str:
    return item.get("split") or assign_split(item["id"])


def _page_key(source: str, page: int) -> str:
    return f"{source}|{page}"


def _substantive(chunks: list[Chunk], min_chars: int) -> list[Chunk]:
    return [c for c in chunks if len(c.text) >= min_chars and c.text.count("|") < 10]


def _single(chunks, teacher, n, rng, min_chars) -> list[dict]:
    by_doc = defaultdict(list)
    for c in _substantive(chunks, min_chars):
        by_doc[c.source].append(c)
    for v in by_doc.values():
        rng.shuffle(v)
    pool, docs = [], sorted(by_doc)
    while any(by_doc.values()) and len(pool) < n * 2:   # round-robin: every document represented
        for d in docs:
            if by_doc[d]:
                pool.append(by_doc[d].pop())
    items = []
    for c in pool:
        if len(items) >= n:
            break
        style = STYLES[len(items) % len(STYLES)]
        out = teacher.json([{"role": "user", "content": QA_PROMPT.format(
            source=c.source, section=c.section or "-", text=c.text, style=style)}],
            QA_SCHEMA, max_tokens=500, temperature=0.3)
        if not out["usable"] or len(out["question"]) < 15:
            continue
        items.append({"id": f"s{len(items):04d}", "type": "single", "style": style.split(" ")[1],
                      "question": out["question"].strip(), "answer": out["answer"].strip(), "answerable": True,
                      "gold": [[c.source, c.page]], "evidence": [c.text],
                      "split": assign_split(_page_key(c.source, c.page))})
        print(f"  single [{len(items)}/{n}] {out['question'][:90]}")
    return items


def _pairs(chunks: list[Chunk], kind: str, n: int, rng, embedder) -> list[tuple[Chunk, Chunk]]:
    """Related-but-different passage pairs, found by embedding similarity."""
    vecs = embedder.encode([c.text for c in chunks], normalize_embeddings=True, batch_size=64)
    sims = vecs @ vecs.T
    order = list(range(len(chunks)))
    rng.shuffle(order)
    pairs, used = [], set()
    for i in order:
        if len(pairs) >= n or i in used:
            continue
        a = chunks[i]
        ok = [j for j in np.argsort(-sims[i])[:40]
              if j != i and j not in used and 0.55 <= sims[i, j] <= 0.9
              and ((chunks[j].source != a.source) if kind == "comparison"
                   else (chunks[j].source == a.source and chunks[j].section != a.section
                         and abs(chunks[j].page - a.page) > 1))]
        if ok:
            j = ok[0]
            pairs.append((a, chunks[j]))
            used.update((i, j))
    return pairs


def _multi(chunks, teacher, n, rng, min_chars, embedder) -> list[dict]:
    items = []
    subst = _substantive(chunks, min_chars)
    for kind in ("comparison", "bridge"):
        target = n // 2
        made = 0
        for a, b in _pairs(subst, kind, target * 2, rng, embedder):
            if made >= target:
                break
            out = teacher.json([{"role": "user", "content": MULTI_PROMPT.format(
                source_a=a.source, section_a=a.section or "-", text_a=a.text,
                source_b=b.source, section_b=b.section or "-", text_b=b.text, kind=KINDS[kind])}],
                MULTI_SCHEMA, max_tokens=700, temperature=0.3)
            if not (out["usable"] and out["needs_both"]) or len(out["question"]) < 20:
                continue
            made += 1
            items.append({"id": f"m{len(items):04d}", "type": kind, "style": kind,
                          "question": out["question"].strip(), "answer": out["answer"].strip(), "answerable": True,
                          "gold": [[a.source, a.page], [b.source, b.page]], "evidence": [a.text, b.text],
                          "split": assign_split(_page_key(a.source, a.page))})
            print(f"  {kind} [{made}/{target}] {out['question'][:90]}")
    return items


def _unanswerable(teacher, sources: list[str], n: int) -> list[dict]:
    items, seen = [], set()
    while len(items) < n:
        avoid = ("\nAvoid these already-used themes: " + "; ".join(list(seen)[-8:])) if seen else ""
        out = teacher.json([{"role": "user", "content": UNANSWERABLE_PROMPT.format(
            sources=", ".join(sources), avoid=avoid)}], UNANSWERABLE_SCHEMA, max_tokens=600, temperature=1.0)
        for q in out["questions"]:
            if q[:60] not in seen and len(items) < n:
                seen.add(q[:60])
                qid = f"u{len(items):04d}"
                items.append({"id": qid, "type": "unanswerable", "style": "unanswerable", "question": q.strip(),
                              "answer": "", "answerable": False, "gold": [], "evidence": [],
                              "split": assign_split(qid)})
    return items


def generate_dataset(chunks: list[Chunk], teacher: LLM, n_single: int = 300, n_multi: int = 150,
                     n_unanswerable: int = 60, seed: int = 0, min_chars: int = 400,
                     embedding_model: str = "BAAI/bge-small-en-v1.5") -> list[dict]:
    from localrag.retrieval.index import load_embedder
    rng = random.Random(seed)
    items = _single(chunks, teacher, n_single, rng, min_chars)
    items += _multi(chunks, teacher, n_multi, rng, min_chars, load_embedder(embedding_model))
    items += _unanswerable(teacher, sorted({c.source for c in chunks}), n_unanswerable)
    return items


FILTER_SCHEMA = {"type": "object",
                 "properties": {"answered": {"type": "string", "enum": ["fully", "partially", "no"]}},
                 "required": ["answered"]}


def filter_unanswerable(items: list[dict], retriever, judge: LLM, k: int = 10) -> list[dict]:
    """Drop 'unanswerable' questions that the corpus answers at least partially (label noise)."""
    keep = []
    for it in items:
        if it["answerable"]:
            keep.append(it)
            continue
        ctx = "\n\n".join(f"- {h.chunk.text}" for h in retriever.retrieve(it["question"], k=k))
        verdict = judge.json([{"role": "user", "content":
                               f"Passages:\n{ctx}\n\nQuestion: {it['question']}\n\nDo these passages answer the "
                               "question? 'fully', 'partially' (they answer the core of it), or 'no'."}],
                             FILTER_SCHEMA, max_tokens=20, temperature=0)["answered"]
        if verdict == "no":
            keep.append(it)
        else:
            print(f"  drop unanswerable ({verdict}): {it['question'][:100]}")
    return keep


def save(items: list[dict], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def load(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
