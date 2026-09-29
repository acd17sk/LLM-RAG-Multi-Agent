"""End-to-end ingestion and question answering."""
from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Optional

from localrag import agents
from localrag.config import Config
from localrag.ingest.chunker import chunk_blocks, make_children
from localrag.llm import LLM, LLMOutputError, get_llm
from localrag.retrieval.index import Index
from localrag.retrieval.rerank import make_verifier
from localrag.retrieval.retriever import Retriever
from localrag.types import Answer, Claim, Hit

REFUSAL = "The provided documents do not contain enough information to answer this question."


def parse(cfg: Config):
    if cfg.ingest.parser == "docling":
        from localrag.ingest.docling_parser import parse_folder
    else:
        from localrag.ingest.parser import parse_folder
    return parse_folder(cfg.ingest.pdf_dir, cfg.ingest.min_block_chars)


def ingest(cfg: Config, llm: Optional[LLM] = None) -> int:
    print(f"Parsing PDFs in {cfg.ingest.pdf_dir} with {cfg.ingest.parser} ...")
    blocks = parse(cfg)
    chunks = chunk_blocks(blocks, cfg.ingest.chunk_size, cfg.ingest.chunk_overlap)
    print(f"{len(blocks)} blocks -> {len(chunks)} chunks")

    if cfg.ingest.contextualize:
        llm = llm or get_llm(cfg.llm)
        by_doc = defaultdict(list)
        for c in chunks:
            by_doc[c.source].append(c)
        for i, c in enumerate(chunks):
            # document opening (title, scope, purpose) as the "whole document" stand-in
            excerpt = "\n".join(x.text for x in by_doc[c.source][:4])[:3000]
            c.context = agents.situate(llm, excerpt, c.text)
            if i % 50 == 0:
                print(f"  contextualized {i}/{len(chunks)}")

    children = make_children(chunks, cfg.ingest.child_size, cfg.ingest.child_overlap) if cfg.ingest.child_size else []
    if children:
        print(f"parent-child: {len(children)} child chunks")
    Index(cfg.index_path, cfg.retrieval.embedding_model, cfg.retrieval.query_prompt).build(
        chunks, children, cfg.retrieval.bm25_k1, cfg.retrieval.bm25_b)
    print(f"Index written to {cfg.index_path}/")
    return len(chunks)


class RAGPipeline:
    def __init__(self, cfg: Config, llm: Optional[LLM] = None):
        self.cfg = cfg
        self.llm = llm or get_llm(cfg.llm)
        self.index = Index(cfg.index_path, cfg.retrieval.embedding_model, cfg.retrieval.query_prompt)
        if cfg.retrieval.mode != "sparse" and not self.index.has_dense():
            # e.g. an ablation with a different embedding model: embed the existing chunks
            print(f"Building dense index for {cfg.retrieval.embedding_model} ...")
            self.index.build_dense()
        self.retriever = Retriever(self.index, cfg.retrieval)
        self.verifier = make_verifier(cfg.agent, self.retriever.reranker)
        sources = sorted({c.source for c in self.index.chunks.values()})
        self.corpus = (cfg.agent.corpus_description + " Documents: " + ", ".join(sources)).strip()

    @staticmethod
    def _top(hits: list[Hit]) -> Optional[float]:
        return hits[0].scores.get("rerank") if hits else None

    def _generate(self, query: str, hits: list[Hit], feedback: str = "") -> tuple[str, list[Claim]]:
        a = self.cfg.agent
        if a.answer_mode == "freetext":
            return agents.answer_freetext(self.llm, query, hits, a.max_new_tokens, a.temperature)
        claims = agents.answer_grounded(self.llm, query, hits, a.max_new_tokens, a.temperature,
                                        evidence_first=a.answer_mode == "grounded_evidence", feedback=feedback)
        return "", claims

    def _verify(self, claims: list[Claim], hits: list[Hit]) -> float:
        """Mark claims supported/unsupported; returns the supported share."""
        if self.verifier is None or not claims:
            return 1.0
        agents.verify_claims(claims, hits, self.verifier, self.cfg.agent.support_threshold)
        return sum(bool(c.supported) for c in claims) / len(claims)

    def _iterate(self, query: str, hits: list[Hit], subqueries: list[str], trace: dict) -> list[Hit]:
        """Up to max_hops follow-up searches for whatever the passages don't cover yet. The best new
        hit of each hop is pinned into the final set, so second-hop evidence (which can rank low
        against the original question) is not reranked away."""
        a, k = self.cfg.agent, self.cfg.retrieval.top_k
        pinned: list[Hit] = []
        for _ in range(a.max_hops):
            follow_up = agents.next_search(self.llm, query, hits)
            asked = {q.strip().lower() for q in trace.get("hops", [])}
            if follow_up is None or follow_up.strip().lower() in asked:
                break   # nothing missing, or the model is repeating itself
            trace.setdefault("hops", []).append(follow_up)
            subqueries.append(follow_up)
            seen = {h.chunk.id for h in hits}
            new = [h for h in self.retriever.retrieve(follow_up) if h.chunk.id not in seen]
            if not new:
                break
            pinned.append(new[0])
            merged = self.retriever.rerank(query, [Hit(h.chunk, 0.0, {}) for h in hits + new],
                                           limit=len(hits) + len(new))
            keep = [h for h in merged if h.chunk.id not in {p.chunk.id for p in pinned}][:k - len(pinned)]
            hits = keep + [next(m for m in merged if m.chunk.id == p.chunk.id) for p in pinned]
        return hits

    def _expand(self, hits: list[Hit]) -> list[Hit]:
        """Replace each hit by its window / section; hits that land in an already-included span merge."""
        a = self.cfg.agent
        out, covered = [], set()
        for h in hits:
            if h.chunk.id in covered:
                continue
            big = self.index.expand(h.chunk, a.context, a.context_window, a.context_chars)
            span = {cid for cid, c in self.index.chunks.items()
                    if c.source == big.source and c.section == big.section and big.page <= c.page <= max(big.page, big.page_end)
                    and c.text in big.text}
            covered |= span
            out.append(Hit(big, h.score, h.scores))
        return out

    def answer(self, query: str) -> Answer:
        a = self.cfg.agent
        timings: dict[str, float] = {}
        trace: dict = {}

        @contextmanager
        def timed(name):
            t = time.perf_counter()
            yield
            timings[name] = round(timings.get(name, 0) + time.perf_counter() - t, 3)

        action = "SEARCH"
        if a.route:
            with timed("route"):
                action = agents.route(self.llm, query, self.corpus)
        if action == "ANSWER_DIRECTLY":
            with timed("generate"):
                text = agents.answer_direct(self.llm, query, a.max_new_tokens)
            return Answer(query, action, text, timings=timings)

        subqueries: list[str] = []
        if a.decompose:
            with timed("decompose"):
                subqueries = agents.decompose(self.llm, query, a.max_subqueries)
        with timed("retrieve"):
            hits = self.retriever.retrieve(query, subqueries)

        # Corrective RAG: weak evidence -> reformulate and search again, reranking old + new together
        if a.crag:
            for _ in range(a.crag_retries):
                top = self._top(hits)
                if top is None or top >= a.crag_threshold:
                    break
                with timed("crag"):
                    rewritten = agents.rewrite_query(self.llm, query, hits)
                    subqueries.append(rewritten)
                    hits = self.retriever.retrieve(query, [rewritten], extra=hits)
                trace.setdefault("crag_rewrites", []).append(rewritten)

        # Iterative retrieval: the LLM names what is still missing and searches for it
        if a.retrieval_strategy == "iterative":
            with timed("iterate"):
                hits = self._iterate(query, hits, subqueries, trace)

        # Retrieval gate: even the best passage looks irrelevant -> refuse rather than improvise
        top = self._top(hits)
        if not hits or (top is not None and top < a.min_relevance):
            return Answer(query, action, REFUSAL, [], hits, subqueries, True, timings, trace)

        # What the LLM reads: expanded context (verification and citations use the same passages),
        # optionally compressed for the prompt only
        if a.context != "chunk":
            hits = self._expand(hits)
        shown = hits
        if a.compression == "sentences" and self.retriever.reranker is not None:
            with timed("compress"):
                shown = agents.compress(hits, query, self.retriever.reranker, a.compress_sentences)
            trace["compression"] = round(sum(len(h.chunk.text) for h in shown) /
                                         max(1, sum(len(h.chunk.text) for h in hits)), 3)

        with timed("generate"):
            try:
                text, claims = self._generate(query, shown)
            except LLMOutputError as e:
                # one malformed generation must not take the whole request down: treat as "no claims"
                text, claims = "", []
                trace["generation_error"] = str(e)
        grounded = a.answer_mode != "freetext"
        with timed("verify"):
            support = self._verify(claims, hits)

        # Self-correction: regenerate once with the unsupported claims as feedback; keep the better answer
        if a.self_correct and grounded and claims and support < a.self_correct_below:
            with timed("self_correct"):
                _, retry = self._generate(query, shown, agents.self_correction_feedback(claims))
                retry_support = self._verify(retry, hits)
            trace["self_correct"] = {"before": round(support, 3), "after": round(retry_support, 3)}
            if sum(bool(c.supported) for c in retry) > sum(bool(c.supported) for c in claims):
                claims = retry

        insufficient = False
        if grounded and not any(c.supported is not False for c in claims):
            insufficient, text = True, REFUSAL
        ans = Answer(query, action, text, claims, hits, subqueries, insufficient, timings, trace)
        if grounded and not insufficient:
            ans.text = ans.render()
        return ans
