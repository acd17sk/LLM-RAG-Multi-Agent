"""Training prompts for the RL/preference stage, built only from the TRAIN split.

Prompts are produced by the same builders the pipeline uses at inference
(agents.grounded_request / agents.decompose_request), so the model is trained on exactly
the input format it will see in production.
"""
from __future__ import annotations

from localrag import agents
from localrag.config import Config
from localrag.eval import dataset as ds
from localrag.retrieval.index import Index
from localrag.retrieval.retriever import Retriever


def _retriever(cfg: Config) -> Retriever:
    return Retriever(Index(cfg.index_path, cfg.retrieval.embedding_model, cfg.retrieval.query_prompt),
                     cfg.retrieval)


def answer_examples(cfg: Config, split: str = "train", limit: int | None = None) -> list[dict]:
    """One row per question: chat prompt + what the reward needs (passages, gold indices, answerable)."""
    retriever = _retriever(cfg)
    rows = []
    for it in ds.load(cfg.eval.dataset):
        if ds.split_of(it) != split:
            continue
        hits = retriever.retrieve(it["question"])
        gold_pages = {tuple(g) for g in it["gold"]}
        passages = agents.format_passages(hits)
        messages, _ = agents.grounded_request(it["question"], passages)
        rows.append({
            "id": it["id"], "type": it.get("type", "single"), "prompt": messages,
            "passages": [h.chunk.text for h in hits],
            "gold": [i for i, h in enumerate(hits, start=1) if (h.chunk.source, h.chunk.page) in gold_pages],
            "answerable": it["answerable"], "reference": it["answer"],
        })
        if limit and len(rows) >= limit:
            break
    return rows


def decompose_examples(cfg: Config, split: str = "train", limit: int | None = None) -> list[dict]:
    rows = []
    for it in ds.load(cfg.eval.dataset):
        if ds.split_of(it) != split or not it["answerable"]:
            continue
        messages, _ = agents.decompose_request(it["question"], cfg.agent.max_subqueries)
        rows.append({"id": it["id"], "type": it.get("type", "single"), "prompt": messages,
                     "question": it["question"], "gold": [list(g) for g in it["gold"]]})
        if limit and len(rows) >= limit:
            break
    return rows
