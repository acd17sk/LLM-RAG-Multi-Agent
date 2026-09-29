"""Verifiable rewards for grounded answering and for query decomposition.

Every term is computed by code or a small NLI model, not a learned reward model, which makes
the reward hard to game: a claim only earns credit if the passage it cites actually entails it.

Answer reward, for a completion in the grounded-answer JSON format ({"claims": [...]}):
    format     -1.0 if the output is not JSON; -0.5 if it is JSON but malformed (missing or
               out-of-range citations): partial credit gives a gradient toward the format
    support    + share of claims entailed by their cited passages (NLI)            weight 1.0
    gold       + cites a passage from the page the question was written from      weight 0.5
    coverage   + share of reference-answer sentences entailed by the claims       weight 0.5
               (without it, the cheapest way to be "faithful" is to say less: in the ablations,
               structured answers already score lower on judged correctness than free text)
    refusal    unanswerable: +1 for no claims, else -1 ; answerable: -1 for no claims
    length     -0.1 per claim beyond `max_claims` (discourages padding)
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable, Optional

from localrag.agents import _REFUSAL, _dedupe
from localrag.types import Claim

Scorer = Callable[[list[tuple[str, str]]], list[float]]


@dataclass
class AnswerExample:
    """Everything the reward needs about one training prompt."""
    passages: list[str]
    gold: list[int]          # 1-based indices of passages from the gold page(s); [] if none shown
    answerable: bool
    reference: str = ""      # reference answer, for the coverage term


@dataclass
class RewardWeights:
    support: float = 1.0
    gold: float = 0.5
    coverage: float = 0.5
    refusal: float = 1.0
    max_claims: int = 6
    extra_claim: float = 0.1
    support_threshold: float = 0.5


def _json_object(completion: str) -> Optional[dict]:
    match = re.search(r"\{.*\}", completion, re.S)   # tolerate text around the object
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def parse_claims(completion: str, n_passages: int) -> Optional[list[Claim]]:
    """Parse the grounded-answer JSON; None if malformed or citing out-of-range passages."""
    obj = _json_object(completion)
    if obj is None:
        return None
    try:
        claims = []
        for c in obj["claims"]:
            cites = [int(i) for i in c["citations"]]
            if not cites or any(i < 1 or i > n_passages for i in cites):
                return None
            text = str(c["text"]).strip()
            if text and not _REFUSAL.search(text):
                claims.append(Claim(text, sorted(set(cites))))
        return _dedupe(claims)
    except (KeyError, TypeError, ValueError):
        return None


def entailment(claims: list[Claim], passages: list[str], scorer: Scorer) -> list[float]:
    """Best entailment score per claim over its cited passages, and over them combined."""
    pairs, owners = [], []
    for ci, c in enumerate(claims):
        texts = [passages[i - 1] for i in c.citations]
        if len(texts) > 1:
            texts.append("\n".join(texts))
        for t in texts:
            pairs.append((c.text, t))
            owners.append(ci)
    best = [0.0] * len(claims)
    for ci, s in zip(owners, scorer(pairs) if pairs else []):
        best[ci] = max(best[ci], s)
    return best


_SENTENCES = re.compile(r"(?<=[.!?;])\s+")


def coverage(claims: list[Claim], reference: str, scorer: Scorer, threshold: float) -> float:
    """Share of reference sentences entailed by the answer's claims taken together."""
    ref = [s for s in _SENTENCES.split(reference.strip()) if len(s.split()) >= 4]
    if not ref or not claims:
        return 0.0
    answer = " ".join(c.text for c in claims)
    return sum(s >= threshold for s in scorer([(r, answer) for r in ref])) / len(ref)


def answer_reward(completion: str, ex: AnswerExample, scorer: Scorer,
                  w: RewardWeights = RewardWeights()) -> tuple[float, dict]:
    claims = parse_claims(completion, len(ex.passages))
    if claims is None:
        return (-0.5, {"format": 0.5}) if _json_object(completion) is not None else (-1.0, {"format": 0.0})
    parts = {"format": 1.0}
    if not claims:
        parts["refusal"] = w.refusal if not ex.answerable else -w.refusal
        return parts["refusal"], parts
    if not ex.answerable:
        parts["refusal"] = -w.refusal
    scores = entailment(claims, ex.passages, scorer)
    parts["support"] = sum(s >= w.support_threshold for s in scores) / len(scores)
    if ex.answerable and ex.gold:
        parts["gold"] = float(any(i in ex.gold for c in claims for i in c.citations))
    if ex.answerable and ex.reference:
        parts["coverage"] = coverage(claims, ex.reference, scorer, w.support_threshold)
    parts["length"] = -w.extra_claim * max(0, len(claims) - w.max_claims)
    total = (w.support * parts["support"] + w.gold * parts.get("gold", 0.0)
             + w.coverage * parts.get("coverage", 0.0) + parts.get("refusal", 0.0) + parts["length"])
    return total, parts


def parse_queries(completion: str) -> Optional[list[str]]:
    match = re.search(r"\{.*\}", completion, re.S)
    if not match:
        return None
    try:
        queries = [str(q).strip() for q in json.loads(match.group(0))["queries"] if str(q).strip()]
        return queries or None
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def decompose_reward(completion: str, question: str, gold: set[tuple[str, int]], retrieve,
                     k: int = 5, max_queries: int = 3) -> tuple[float, dict]:
    """Share of gold pages retrieved in the top k with the proposed sub-queries (+ original).
    `retrieve(question, subqueries, k)` returns hits. Extra queries cost latency, so they're capped."""
    queries = parse_queries(completion)
    if queries is None:
        return -1.0, {"format": 0.0}
    hits = retrieve(question, queries[:max_queries], k)
    pages = {(h.chunk.source, h.chunk.page) for h in hits}
    recall = len(gold & pages) / len(gold)
    penalty = 0.1 * max(0, len(queries) - max_queries)
    return recall - penalty, {"format": 1.0, "recall": recall}
