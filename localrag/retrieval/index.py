"""On-disk index: chunk store (JSONL) + BM25 / SPLADE (JSON) + dense vectors (ChromaDB).

Layout under `index_dir`:
    chunks.jsonl           all chunks, the source of truth (what the LLM reads)
    children.jsonl         optional small child chunks (parent-child retrieval): what gets searched
    bm25.json              sparse lexical index
    splade-<model>.json    learned-sparse index, built lazily per model
    chroma/                one dense collection per embedding model, built lazily
"""
from __future__ import annotations

import gc
import json
import re
from functools import lru_cache
from pathlib import Path

import chromadb

from localrag.retrieval.bm25 import BM25
from localrag.types import Chunk


@lru_cache(maxsize=2)
def load_embedder(name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name)


@lru_cache(maxsize=2)
def load_cross_encoder(name: str):
    import torch
    from sentence_transformers import CrossEncoder
    kw = {"model_kwargs": {"torch_dtype": torch.float16}} if torch.cuda.is_available() else {}
    return CrossEncoder(name, **kw)


@lru_cache(maxsize=1)
def load_splade(name: str):
    from sentence_transformers import SparseEncoder
    return SparseEncoder(name)


def release_models() -> None:
    """Free GPU memory held by cached torch models (between eval variants)."""
    for f in (load_embedder, load_cross_encoder, load_splade):
        f.cache_clear()
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except ImportError:
        pass


def _slug(model: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", model)[-60:].strip("_-")


class SpladeIndex:
    """Inverted index over SPLADE term weights; score(q, d) = sum_t w_q(t) * w_d(t)."""

    def __init__(self, ids: list[str], postings: dict[int, list[tuple[int, float]]]):
        self.ids, self.postings = ids, postings

    @staticmethod
    def _rows(sparse) -> list[dict[int, float]]:
        sparse = sparse.coalesce().cpu()
        rows: list[dict[int, float]] = [{} for _ in range(sparse.shape[0])]
        (r, t), v = sparse.indices(), sparse.values()
        for i, term, w in zip(r.tolist(), t.tolist(), v.tolist()):
            rows[i][term] = w
        return rows

    @classmethod
    def build(cls, model_name: str, ids: list[str], texts: list[str]) -> "SpladeIndex":
        model = load_splade(model_name)
        postings: dict[int, list[tuple[int, float]]] = {}
        for start in range(0, len(texts), 64):
            batch = model.encode_document(texts[start:start + 64], convert_to_sparse_tensor=True)
            for offset, row in enumerate(cls._rows(batch)):
                for term, w in row.items():
                    postings.setdefault(term, []).append((start + offset, w))
        return cls(ids, postings)

    def score(self, query_vec: dict[int, float], k: int) -> list[tuple[str, float]]:
        scores: dict[int, float] = {}
        for term, wq in query_vec.items():
            for doc, wd in self.postings.get(term, ()):
                scores[doc] = scores.get(doc, 0.0) + wq * wd
        top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        return [(self.ids[i], s) for i, s in top]

    def search(self, model_name: str, query: str, k: int) -> list[tuple[str, float]]:
        q = self._rows(load_splade(model_name).encode_query([query], convert_to_sparse_tensor=True))[0]
        return self.score(q, k)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"ids": self.ids, "postings": {str(t): p for t, p in self.postings.items()}}))

    @classmethod
    def load(cls, path: Path) -> "SpladeIndex":
        d = json.loads(path.read_text())
        return cls(d["ids"], {int(t): [tuple(x) for x in p] for t, p in d["postings"].items()})


class Index:
    def __init__(self, index_dir: str | Path, embedding_model: str, query_prompt: str | None = None):
        self.dir = Path(index_dir)
        self.embedding_model = embedding_model
        self.query_prompt = query_prompt
        self._client = None
        self._chunks: dict[str, Chunk] | None = None
        self._children: dict[str, Chunk] | None = None
        self._bm25: BM25 | None = None
        self._splade: dict[str, SpladeIndex] = {}

    # ---------- building ----------
    def build(self, chunks: list[Chunk], children: list[Chunk] | None = None,
              k1: float = 1.5, b: float = 0.75) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        for name, items in (("chunks.jsonl", chunks), ("children.jsonl", children or [])):
            with open(self.dir / name, "w") as f:
                for c in items:
                    f.write(json.dumps(c.to_dict()) + "\n")
        self._chunks = self._children = None
        units = self.units
        BM25(k1, b).fit([c.id for c in units], [c.index_text for c in units]).save(self.dir / "bm25.json")
        for p in self.dir.glob("splade-*.json"):
            p.unlink()
        self.build_dense()
        self._bm25, self._splade = None, {}

    def build_dense(self, batch: int = 128) -> None:
        units = self.units
        name = _slug(self.embedding_model)
        try:
            self.client.delete_collection(name)
        except Exception:
            pass
        col = self.client.create_collection(name, metadata={"hnsw:space": "cosine"})
        vecs = load_embedder(self.embedding_model).encode(
            [c.index_text for c in units], batch_size=32, normalize_embeddings=True, show_progress_bar=True)
        for i in range(0, len(units), batch):
            part = units[i:i + batch]
            col.add(ids=[c.id for c in part], embeddings=vecs[i:i + batch].tolist(),
                    metadatas=[{"source": c.source, "page": c.page} for c in part])

    # ---------- reading ----------
    @property
    def client(self):
        if self._client is None:
            self._client = chromadb.PersistentClient(path=str(self.dir / "chroma"))
        return self._client

    def _load(self, name: str) -> dict[str, Chunk]:
        path = self.dir / name
        if not path.exists():
            return {}
        with open(path) as f:
            return {c.id: c for c in (Chunk(**json.loads(line)) for line in f)}

    @property
    def chunks(self) -> dict[str, Chunk]:
        if self._chunks is None:
            self._chunks = self._load("chunks.jsonl")
        return self._chunks

    @property
    def children(self) -> dict[str, Chunk]:
        if self._children is None:
            self._children = self._load("children.jsonl")
        return self._children

    @property
    def units(self) -> list[Chunk]:
        """What gets searched: child chunks if the index has them, else the chunks themselves."""
        return list((self.children or self.chunks).values())

    def parent(self, unit_id: str) -> Chunk:
        child = self.children.get(unit_id)
        return self.chunks[child.parent_id] if child else self.chunks[unit_id]

    @property
    def bm25(self) -> BM25:
        if self._bm25 is None:
            self._bm25 = BM25.load(self.dir / "bm25.json")
        return self._bm25

    def has_dense(self) -> bool:
        return _slug(self.embedding_model) in [c.name for c in self.client.list_collections()]

    def dense_search(self, query: str, k: int) -> list[tuple[str, float]]:
        col = self.client.get_collection(_slug(self.embedding_model))
        kw = {"prompt_name": self.query_prompt} if self.query_prompt else {}
        q = load_embedder(self.embedding_model).encode(query, normalize_embeddings=True, **kw)
        res = col.query(query_embeddings=[q.tolist()], n_results=k)
        return [(i, 1 - d) for i, d in zip(res["ids"][0], res["distances"][0])]

    def sparse_search(self, query: str, k: int) -> list[tuple[str, float]]:
        return self.bm25.search(query, k)

    def splade_search(self, model: str, query: str, k: int) -> list[tuple[str, float]]:
        if model not in self._splade:
            path = self.dir / f"splade-{_slug(model)}.json"
            if path.exists():
                self._splade[model] = SpladeIndex.load(path)
            else:
                print(f"Building SPLADE index with {model} ...")
                units = self.units
                self._splade[model] = SpladeIndex.build(model, [c.id for c in units], [c.index_text for c in units])
                self._splade[model].save(path)
        return self._splade[model].search(model, query, k)
