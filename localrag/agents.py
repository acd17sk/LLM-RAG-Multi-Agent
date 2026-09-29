"""The LLM-driven steps of the pipeline. Every structured decision is grammar-constrained,
so even a sub-1B model cannot emit malformed JSON or cite a passage that doesn't exist."""
from __future__ import annotations

import re
from dataclasses import replace

from localrag.llm import LLM
from localrag.retrieval.bm25 import tokenize
from localrag.types import Claim, Hit

ROUTE_SCHEMA = {
    "type": "object",
    "properties": {"action": {"type": "string", "enum": ["SEARCH", "ANSWER_DIRECTLY"]}},
    "required": ["action"],
}


def route(llm: LLM, query: str, corpus: str) -> str:
    prompt = f"""You route questions for an assistant with a document knowledge base.
Knowledge base: {corpus}

Choose SEARCH if answering needs facts, definitions, requirements or details that could be in the knowledge base.
Choose ANSWER_DIRECTLY only for greetings, small talk, or questions about the conversation itself.
When unsure, choose SEARCH.

Question: {query}"""
    out = llm.json([{"role": "user", "content": prompt}], ROUTE_SCHEMA, max_tokens=20, temperature=0)
    return out["action"]


def decompose_request(query: str, max_subqueries: int = 3) -> tuple[list[dict], dict]:
    """Messages + JSON schema for decomposition (shared by inference and RL training)."""
    schema = {
        "type": "object",
        "properties": {"queries": {"type": "array", "minItems": 1, "maxItems": max_subqueries,
                                   "items": {"type": "string", "maxLength": 200}}},
        "required": ["queries"],
    }
    prompt = f"""Rewrite the question into 1 to {max_subqueries} short, self-contained search queries.
- A simple question: return one query that restates it with its key terms.
- A comparison or multi-part question: one query per entity or part.
- Expand abbreviations when helpful (e.g. "V&V" -> "verification and validation").
Respond with JSON only, in this format: {{"queries": ["<query>", ...]}}

Question: {query}"""
    return [{"role": "user", "content": prompt}], schema


def decompose(llm: LLM, query: str, max_subqueries: int = 3) -> list[str]:
    messages, schema = decompose_request(query, max_subqueries)
    out = llm.json(messages, schema, max_tokens=300, temperature=0, adapter="decompose")
    seen, queries = {query.strip().lower()}, []
    for q in out["queries"]:
        if q.strip() and q.strip().lower() not in seen:
            seen.add(q.strip().lower())
            queries.append(q.strip())
    return queries


REWRITE_SCHEMA = {"type": "object", "properties": {"query": {"type": "string", "maxLength": 200}},
                  "required": ["query"]}


def rewrite_query(llm: LLM, query: str, hits: list[Hit]) -> str:
    """Corrective RAG step: the first search found nothing convincing, so reformulate it."""
    seen = "; ".join(sorted({h.chunk.section.split(" > ")[-1] for h in hits[:5] if h.chunk.section}))[:400]
    prompt = f"""A search over regulatory documents for the question below returned passages that do not answer it.
Sections that came back: {seen or "unrelated text"}

Write ONE different search query that is more likely to find the answer: use the formal terminology a regulation or guidance document would use, expand abbreviations, and drop conversational words.

Question: {query}"""
    return llm.json([{"role": "user", "content": prompt}], REWRITE_SCHEMA, max_tokens=120, temperature=0)["query"].strip()


NEXT_SEARCH_SCHEMA = {
    "type": "object",
    "properties": {"missing": {"type": "string", "maxLength": 200},
                   "query": {"type": "string", "maxLength": 200}},
    "required": ["missing", "query"],
}
_NOTHING = re.compile(r"^\W*(nothing|none|n/?a|no(thing)? (is )?missing)\b", re.I)


def next_search(llm: LLM, query: str, hits: list[Hit]) -> str | None:
    """Iterative retrieval step: which part of the question do the passages NOT cover yet?
    Returns a follow-up search query, or None when the passages suffice.

    The model first writes what is missing, and only then a query: asking for a yes/no
    "enough?" decision up front makes small models answer "yes" without reading.
    """
    listing = "\n\n".join(f"[{i}] {h.chunk.text[:700]}" for i, h in enumerate(hits, start=1))
    prompt = f"""Passages found so far:
{listing}

Question: {query}

In "missing", write which specific information needed to fully answer the question is NOT in the passages (for comparisons: is every side covered?). Write "nothing" if the passages already cover it.
In "query", write one short search query that would find the missing information, or "" if nothing is missing.
Respond with JSON only: {{"missing": "...", "query": "..."}}"""
    out = llm.json([{"role": "user", "content": prompt}], NEXT_SEARCH_SCHEMA, max_tokens=150, temperature=0)
    q = out["query"].strip()
    if _NOTHING.match(out["missing"].strip()) or not q or q.lower() == query.strip().lower():
        return None
    return q


_SENT_SPLIT = re.compile(r"(?<=[.!?;])\s+(?=[A-Z(\"'§0-9])")


def compress(hits: list[Hit], query: str, scorer, keep: int) -> list[Hit]:
    """Extractive compression (RECOMP-style, with the reranker as sentence scorer): keep the `keep`
    most query-relevant sentences of each passage, in their original order, marking gaps with "…"."""
    out, pairs, spans = [], [], []
    for h in hits:
        sents = [x for x in _SENT_SPLIT.split(h.chunk.text) if x.strip()]
        spans.append((len(pairs), sents))
        pairs.extend((query, x) for x in sents)
    scores = scorer.score(pairs) if pairs else []
    for h, (start, sents) in zip(hits, spans):
        if len(sents) <= keep:
            out.append(h)
            continue
        ranked = sorted(range(len(sents)), key=lambda i: scores[start + i], reverse=True)[:keep]
        parts, prev = [], None
        for i in sorted(ranked):
            if prev is not None and i != prev + 1:
                parts.append("…")
            parts.append(sents[i])
            prev = i
        chunk = replace(h.chunk, text=" ".join(parts))
        out.append(Hit(chunk, h.score, h.scores))
    return out


def self_correction_feedback(claims: list[Claim]) -> str:
    bad = [c.text for c in claims if c.supported is False]
    listing = "\n".join(f"- {t}" for t in bad[:5])
    return ("Your previous answer contained claims that the cited passages do not state:\n" + listing +
            "\nWrite the answer again. Only state what the passages say explicitly, and cite the passage that says it.")


def format_passages(hits: list[Hit]) -> list[str]:
    parts = []
    for i, h in enumerate(hits, start=1):
        c = h.chunk
        where = f"{c.source}, page {c.page}" + (f", section: {c.section}" if c.section else "")
        parts.append(f"[{i}] ({where})\n{c.text}")
    return parts


GROUNDED_SYSTEM = """You answer questions using ONLY the numbered passages provided.
Write the answer as a list of short factual claims, each citing the number(s) of the passage(s) that state it.
If no passage answers the question, return an empty list of claims.
Respond with JSON only, in this format: {"claims": [{"text": "<claim>", "citations": [<passage number>]}]}"""

EVIDENCE_SYSTEM = """You answer questions using ONLY the numbered passages provided.
First, copy into "evidence" the sentences from the passages that answer the question, with their passage number. If no passage answers the question, leave evidence empty.
Then write the answer as short factual claims based only on that evidence, each citing its passage number(s). With no evidence, return no claims.
Respond with JSON only, in this format: {"evidence": [{"passage": <number>, "quote": "<sentence>"}], "claims": [{"text": "<claim>", "citations": [<passage number>]}]}"""

_REFUSAL = re.compile(r"\b(not|no)\b.{0,40}\b(mention|mentioned|detailed|provided|contain|contained|found|covered|specified)\b", re.I)


def _dedupe(claims: list[Claim], threshold: float = 0.8) -> list[Claim]:
    """Drop claims whose word set overlaps an earlier claim's by >= threshold (Jaccard); merge citations."""
    kept: list[tuple[set, Claim]] = []
    for c in claims:
        words = {w.rstrip("s") for w in tokenize(c.text)}
        for kw, k in kept:
            if len(words & kw) / max(len(words | kw), 1) >= threshold:
                k.citations = sorted(set(k.citations) | set(c.citations))
                break
        else:
            kept.append((words, c))
    return [c for _, c in kept]


def grounded_request(query: str, passages: list[str], evidence_first: bool = False,
                     feedback: str = "") -> tuple[list[dict], dict]:
    """Messages + JSON schema for the grounded answer (shared by inference and RL training).
    `passages` are already formatted with their numbers (see format_passages)."""
    cite = {"type": "integer", "enum": list(range(1, len(passages) + 1))}
    props: dict = {
        "claims": {
            "type": "array", "maxItems": 8,
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "maxLength": 400},
                    "citations": {"type": "array", "minItems": 1, "maxItems": 3, "items": cite},
                },
                "required": ["text", "citations"],
            },
        },
    }
    if evidence_first:
        props = {"evidence": {"type": "array", "maxItems": 4, "items": {
            "type": "object", "properties": {"passage": cite, "quote": {"type": "string", "maxLength": 300}},
            "required": ["passage", "quote"]}}, **props}
    schema = {"type": "object", "properties": props, "required": list(props)}
    user = "Passages:\n" + "\n\n".join(passages) + f"\n\nQuestion: {query}"
    if feedback:
        user += f"\n\n{feedback}"
    messages = [{"role": "system", "content": EVIDENCE_SYSTEM if evidence_first else GROUNDED_SYSTEM},
                {"role": "user", "content": user}]
    return messages, schema


def answer_grounded(llm: LLM, query: str, hits: list[Hit], max_tokens: int, temperature: float,
                    evidence_first: bool = False, feedback: str = "") -> list[Claim]:
    """Structured answer: citations are an enum over the passage numbers actually given.

    There is deliberately no up-front "insufficient context" flag: small models decide
    it before reading and refuse everything. An empty claim list is the refusal.
    """
    messages, schema = grounded_request(query, format_passages(hits), evidence_first, feedback)
    out = llm.json(messages, schema, max_tokens=max_tokens, temperature=temperature, adapter="answer")
    claims = [Claim(c["text"].strip(), sorted(set(c["citations"]))) for c in out["claims"]
              if c["text"].strip() and not _REFUSAL.search(c["text"])]
    return _dedupe(claims)


FREETEXT_SYSTEM = """You are a precise assistant. Answer the question based ONLY on the provided context.
When you use information from the context, copy its citation tag, like [Source: document.pdf, Page X], right after the information.
If the context is insufficient, say so clearly."""

_TAG = re.compile(r"\[Source:\s*([^,\]]+),\s*Page:?\s*(\d+)\]", re.I)
_SENT = re.compile(r"(?<=[.!?])\s+")


def answer_freetext(llm: LLM, query: str, hits: list[Hit], max_tokens: int, temperature: float
                    ) -> tuple[str, list[Claim]]:
    """Baseline: the original prompt-only citation style, parsed back into claims for evaluation."""
    context = "\n\n---\n\n".join(f"[Source: {h.chunk.source}, Page: {h.chunk.page}] {h.chunk.text}" for h in hits)
    text = llm.chat([{"role": "system", "content": FREETEXT_SYSTEM},
                     {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}],
                    max_tokens=max_tokens, temperature=temperature)
    claims = []
    for sent in _SENT.split(text.strip()):
        cites = sorted({i for i, h in enumerate(hits, start=1)
                        for src, page in _TAG.findall(sent)
                        if src.strip() == h.chunk.source and int(page) == h.chunk.page})
        clean = _TAG.sub("", sent).strip()
        if clean:
            claims.append(Claim(clean, cites))
    return text, claims


def answer_direct(llm: LLM, query: str, max_tokens: int) -> str:
    return llm.chat([{"role": "user", "content": query}], max_tokens=max_tokens, temperature=0.3)


def verify_claims(claims: list[Claim], hits: list[Hit], scorer, threshold: float) -> None:
    """Self-check: score each claim against its cited passages (reranker relevance or NLI entailment).

    A claim counts as supported if one cited passage, or for multi-citation claims (comparisons,
    syntheses) all cited passages taken together, score above `threshold`. Uncited claims are
    unsupported by definition.
    """
    pairs, owners = [], []
    for ci, c in enumerate(claims):
        texts = [hits[idx - 1].chunk.text for idx in c.citations]
        if len(texts) > 1:
            texts.append("\n".join(texts))
        for t in texts:
            pairs.append((c.text, t))
            owners.append(ci)
    best = [0.0] * len(claims)
    if pairs:
        for ci, s in zip(owners, scorer.score(pairs)):
            best[ci] = max(best[ci], float(s))
    for c, s in zip(claims, best):
        c.support_score = s
        c.supported = bool(c.citations) and s >= threshold


CONTEXT_PROMPT = """<document>{doc}</document>
Here is a passage from the document above:
<passage>{passage}</passage>
Write one or two sentences that situate this passage within the overall document (what document, which topic or requirement it belongs to), to improve search retrieval of the passage. Answer only with the sentences."""


def situate(llm: LLM, doc_excerpt: str, passage: str) -> str:
    """Contextual retrieval: an LLM-written preamble that is indexed together with the chunk."""
    return llm.chat([{"role": "user", "content": CONTEXT_PROMPT.format(doc=doc_excerpt, passage=passage)}],
                    max_tokens=100, temperature=0).strip()
