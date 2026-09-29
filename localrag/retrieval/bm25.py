"""Okapi BM25 over chunk text, implemented from scratch with an inverted index."""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

# Keep section numbers like "820.30" or "21 CFR" intact: regulatory queries hinge on them.
_TOKEN = re.compile(r"\d+(?:\.\d+)+|[a-z0-9]+")
_STOP = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the this to was were "
    "will with what which who how when where why do does can should shall may must".split())


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.ids: list[str] = []
        self.doc_len: list[int] = []
        self.postings: dict[str, list[tuple[int, int]]] = {}  # term -> [(doc_idx, tf)]
        self.idf: dict[str, float] = {}
        self.avgdl = 0.0

    def fit(self, ids: list[str], texts: list[str]) -> "BM25":
        self.ids = list(ids)
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.doc_len = []
        for i, text in enumerate(texts):
            tf = Counter(tokenize(text))
            self.doc_len.append(sum(tf.values()))
            for term, n in tf.items():
                postings[term].append((i, n))
        self.postings = dict(postings)
        n_docs = len(texts)
        self.avgdl = sum(self.doc_len) / max(n_docs, 1)
        # BM25+ style non-negative idf
        self.idf = {t: math.log(1 + (n_docs - len(p) + 0.5) / (len(p) + 0.5))
                    for t, p in self.postings.items()}
        return self

    def search(self, query: str, k: int = 20) -> list[tuple[str, float]]:
        scores: dict[int, float] = defaultdict(float)
        for term in set(tokenize(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, tf in self.postings[term]:
                norm = self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                scores[i] += idf * tf * (self.k1 + 1) / (tf + norm)
        top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        return [(self.ids[i], s) for i, s in top]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({
            "k1": self.k1, "b": self.b, "ids": self.ids, "doc_len": self.doc_len,
            "postings": self.postings, "idf": self.idf, "avgdl": self.avgdl}))

    @classmethod
    def load(cls, path: str | Path) -> "BM25":
        d = json.loads(Path(path).read_text())
        bm = cls(d["k1"], d["b"])
        bm.ids, bm.doc_len, bm.idf, bm.avgdl = d["ids"], d["doc_len"], d["idf"], d["avgdl"]
        bm.postings = {t: [tuple(p) for p in ps] for t, ps in d["postings"].items()}
        return bm
