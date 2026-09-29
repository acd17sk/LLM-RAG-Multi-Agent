"""Late-interaction retrieval (ColBERT): one vector per token, scored with MaxSim.

    score(q, d) = sum over query tokens i of  max over document tokens j of  <q_i, d_j>

Encoding follows the model's own configuration (PyLate format): "[Q] "/"[D] " prefixes,
a bias-free linear projection of the last hidden states, per-token L2 normalisation, and
punctuation tokens dropped from documents. Search is exact brute-force MaxSim on the GPU,
which is fast for corpora of a few thousand chunks; PLAID-style centroid pruning is the
standard next step at larger scale.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file


class ColBERTEncoder:
    def __init__(self, name: str, device: str | None = None):
        from transformers import AutoModel, AutoTokenizer

        path = Path(snapshot_download(name))
        cfg = json.loads((path / "config_sentence_transformers.json").read_text())
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModel.from_pretrained(path, dtype=dtype).to(self.device).eval()
        dense = load_file(str(path / "1_Dense" / "model.safetensors"))
        self.proj = next(iter(dense.values())).to(self.device, dtype)      # (out_dim, hidden)
        self.q_prefix, self.d_prefix = cfg["query_prefix"], cfg["document_prefix"]
        self.q_len, self.d_len = cfg["query_length"], cfg["document_length"]
        self.skip_ids = {i for w in cfg.get("skiplist_words", [])
                         for i in self.tok(w, add_special_tokens=False)["input_ids"]}

    @torch.inference_mode()
    def encode(self, texts: list[str], is_query: bool, batch_size: int = 32) -> list[torch.Tensor]:
        """Per-text (n_tokens, dim) normalised token embeddings, on CPU in float16."""
        prefix, max_len = (self.q_prefix, self.q_len) if is_query else (self.d_prefix, self.d_len)
        out = []
        for start in range(0, len(texts), batch_size):
            batch = self.tok([prefix + t for t in texts[start:start + batch_size]], padding=True,
                             truncation=True, max_length=max_len, return_tensors="pt").to(self.device)
            hidden = self.model(**batch).last_hidden_state
            emb = torch.nn.functional.normalize(hidden @ self.proj.T, dim=-1)
            for i in range(emb.shape[0]):
                keep = batch["attention_mask"][i].bool()
                if not is_query and self.skip_ids:
                    keep &= ~torch.isin(batch["input_ids"][i],
                                        torch.tensor(sorted(self.skip_ids), device=self.device))
                out.append(emb[i][keep].to("cpu", torch.float16))
        return out


@lru_cache(maxsize=1)
def load_colbert(name: str) -> ColBERTEncoder:
    return ColBERTEncoder(name)


def pad_docs(docs: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack variable-length token matrices into (n_docs, max_len, dim) + a validity mask."""
    max_len = max(d.shape[0] for d in docs)
    dim = docs[0].shape[1]
    padded = torch.zeros(len(docs), max_len, dim, dtype=docs[0].dtype)
    mask = torch.zeros(len(docs), max_len, dtype=torch.bool)
    for i, d in enumerate(docs):
        padded[i, :d.shape[0]] = d
        mask[i, :d.shape[0]] = True
    return padded, mask


def maxsim(query: torch.Tensor, docs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """MaxSim scores of one query (m, dim) against padded docs (n, L, dim) -> (n,)."""
    sim = torch.einsum("nld,md->nlm", docs.float(), query.float())      # token-level similarities
    sim = sim.masked_fill(~mask[:, :, None], float("-inf"))
    return sim.max(dim=1).values.sum(dim=1)                             # best doc token per query token


class LateInteractionIndex:
    def __init__(self, ids: list[str], docs: torch.Tensor, mask: torch.Tensor):
        self.ids, self.docs, self.mask = ids, docs, mask
        self._device_docs = None

    @classmethod
    def build(cls, model_name: str, ids: list[str], texts: list[str]) -> "LateInteractionIndex":
        docs, mask = pad_docs(load_colbert(model_name).encode(texts, is_query=False))
        return cls(ids, docs, mask)

    def search(self, model_name: str, query: str, k: int) -> list[tuple[str, float]]:
        enc = load_colbert(model_name)
        if self._device_docs is None:
            self._device_docs = (self.docs.to(enc.device), self.mask.to(enc.device))
        q = enc.encode([query], is_query=True)[0].to(enc.device)
        scores = maxsim(q, *self._device_docs)
        top = torch.topk(scores, min(k, len(self.ids)))
        return [(self.ids[i], float(s)) for s, i in zip(top.values.tolist(), top.indices.tolist())]

    def save(self, path: Path) -> None:
        torch.save({"ids": self.ids, "docs": self.docs, "mask": self.mask}, path)

    @classmethod
    def load(cls, path: Path) -> "LateInteractionIndex":
        d = torch.load(path)
        return cls(d["ids"], d["docs"], d["mask"])
