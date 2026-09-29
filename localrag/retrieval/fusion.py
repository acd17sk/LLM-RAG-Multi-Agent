"""Reciprocal Rank Fusion (Cormack et al., 2009)."""
from __future__ import annotations


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60,
                           weights: list[float] | None = None) -> list[tuple[str, float]]:
    """Fuse ranked ID lists: score(d) = sum_i w_i / (k + rank_i(d)), with 1-based ranks.

    RRF only uses ranks, so it needs no score calibration between BM25 and cosine similarity.
    """
    weights = weights or [1.0] * len(rankings)
    scores: dict[str, float] = {}
    for ranking, w in zip(rankings, weights):
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + w / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
