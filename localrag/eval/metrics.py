"""Retrieval metrics with page-level relevance.

Relevance is judged at the (source, page) level rather than the chunk ID, so the same
gold set stays valid when chunking parameters change between ablations.
"""
from __future__ import annotations

import math
from typing import Iterable

Page = tuple[str, int]


def _rels(retrieved: list[Page], gold: set[Page]) -> list[int]:
    return [1 if p in gold else 0 for p in retrieved]


def recall_at_k(retrieved: list[Page], gold: set[Page], k: int) -> float:
    """1 if any gold page appears in the top k (hit rate); the standard for single-evidence QA."""
    return float(any(_rels(retrieved[:k], gold)))


def mrr(retrieved: list[Page], gold: set[Page]) -> float:
    for rank, rel in enumerate(_rels(retrieved, gold), start=1):
        if rel:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[Page], gold: set[Page], k: int) -> float:
    # a page can occur in several chunks; count it relevant only the first time
    seen, gains = set(), []
    for p in retrieved[:k]:
        gains.append(1 if p in gold and p not in seen else 0)
        seen.add(p)
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    idcg = sum(1 / math.log2(i + 2) for i in range(min(len(gold), k)))
    return dcg / idcg if idcg else 0.0


def mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")
