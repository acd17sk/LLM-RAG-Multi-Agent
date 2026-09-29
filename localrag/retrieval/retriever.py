"""Multi-query hybrid retrieval: (dense ∪ sparse) per query → RRF → parent mapping → rerank."""
from __future__ import annotations

from localrag.config import RetrievalConfig
from localrag.retrieval.fusion import reciprocal_rank_fusion
from localrag.retrieval.index import Index
from localrag.retrieval.rerank import PairScorer, make_reranker
from localrag.types import Hit


class Retriever:
    def __init__(self, index: Index, cfg: RetrievalConfig, reranker: PairScorer | None = None):
        self.index, self.cfg = index, cfg
        self.reranker = reranker if reranker is not None else make_reranker(cfg)

    def _sparse(self, q: str) -> list[str]:
        if self.cfg.sparse == "splade":
            return [i for i, _ in self.index.splade_search(self.cfg.splade_model, q, self.cfg.fetch_k)]
        return [i for i, _ in self.index.sparse_search(q, self.cfg.fetch_k)]

    def candidates(self, queries: list[str]) -> list[Hit]:
        """Every (query, retriever) pair contributes one ranking; RRF fuses them all.

        The first query is the user's original question and gets double weight, so
        decomposed sub-queries broaden recall without drowning out the original intent.
        With parent-child indexing, child hits are mapped to their parent chunk, which
        keeps the rank of its best child.
        """
        rankings, weights = [], []
        for qi, q in enumerate(queries):
            w = 2.0 if qi == 0 else 1.0
            if self.cfg.mode in ("dense", "hybrid"):
                rankings.append([i for i, _ in self.index.dense_search(q, self.cfg.fetch_k)])
                weights.append(w)
            if self.cfg.mode in ("sparse", "hybrid"):
                rankings.append(self._sparse(q))
                weights.append(w)
        hits, seen = [], set()
        for unit_id, s in reciprocal_rank_fusion(rankings, k=self.cfg.rrf_k, weights=weights):
            chunk = self.index.parent(unit_id)
            if chunk.id not in seen:
                seen.add(chunk.id)
                hits.append(Hit(chunk, s, {"rrf": s}))
        return hits

    def rerank(self, query: str, hits: list[Hit], limit: int | None = None) -> list[Hit]:
        if self.reranker is None or not hits:
            return hits
        hits = hits[:limit or self.cfg.rerank_k]
        for h, s in zip(hits, self.reranker.score([(query, h.chunk.index_text) for h in hits])):
            h.scores["rerank"] = s
            h.score = s
        return sorted(hits, key=lambda h: h.score, reverse=True)

    def retrieve(self, query: str, subqueries: list[str] | None = None, k: int | None = None,
                 extra: list[Hit] | None = None) -> list[Hit]:
        """`extra`: candidates from an earlier retrieval round (CRAG) to rerank together with the new ones."""
        queries = [query] + [q for q in (subqueries or []) if q.strip().lower() != query.strip().lower()]
        cands = self.candidates(queries)[:self.cfg.rerank_k]
        if extra:
            ids = {h.chunk.id for h in cands}
            cands += [Hit(h.chunk, 0.0, {}) for h in extra if h.chunk.id not in ids]
        return self.rerank(query, cands, limit=len(cands))[:k or self.cfg.top_k]
