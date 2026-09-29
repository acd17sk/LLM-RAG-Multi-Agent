"""Rerankers and claim verifiers behind one interface: score (query, text) pairs."""
from __future__ import annotations

from typing import Protocol

from localrag.config import AgentConfig, RetrievalConfig
from localrag.llm import get_llm
from localrag.retrieval.index import load_cross_encoder


class PairScorer(Protocol):
    def score(self, pairs: list[tuple[str, str]]) -> list[float]: ...


class CrossEncoderScorer:
    def __init__(self, model: str):
        self.model = model

    def score(self, pairs):
        if not pairs:
            return []
        return [float(s) for s in load_cross_encoder(self.model).predict(pairs, batch_size=16)]


class LlamaRerankScorer:
    """An LLM-based reranker (e.g. Qwen3-Reranker) served by llama-server's /v1/rerank."""

    def __init__(self, cfg):
        self.llm = get_llm(cfg)

    def score(self, pairs):
        # the endpoint takes one query with many documents, so group pairs by query
        out = [0.0] * len(pairs)
        groups: dict[str, list[int]] = {}
        for i, (q, _) in enumerate(pairs):
            groups.setdefault(q, []).append(i)
        for q, idx in groups.items():
            for i, s in zip(idx, self.llm.rerank(q, [pairs[i][1] for i in idx])):
                out[i] = s
        return out


class NLIScorer:
    """Entailment probability P(passage ⊨ claim) from an NLI cross-encoder."""

    def __init__(self, model: str):
        self.model = model

    def score(self, pairs):
        if not pairs:
            return []
        m = load_cross_encoder(self.model)
        labels = {v.lower(): int(k) for k, v in m.model.config.id2label.items()}
        # NLI convention: (premise, hypothesis) = (passage, claim)
        probs = m.predict([(passage, claim) for claim, passage in pairs], batch_size=16, apply_softmax=True)
        return [float(p[labels["entailment"]]) for p in probs]


def make_reranker(cfg: RetrievalConfig) -> PairScorer | None:
    if cfg.reranker_backend == "llama":
        return LlamaRerankScorer(cfg.llama_reranker)
    return CrossEncoderScorer(cfg.reranker_model) if cfg.reranker_model else None


def make_verifier(agent: AgentConfig, reranker: PairScorer | None) -> PairScorer | None:
    if agent.verifier == "nli":
        return NLIScorer(agent.nli_model)
    if agent.verifier == "reranker":
        return reranker
    return None
