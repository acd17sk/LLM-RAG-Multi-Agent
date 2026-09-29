"""Two-phase evaluation.

Phase 1 (generate): run each pipeline variant over the eval split and store answers, citations
and retrieval results. Only the small models under test occupy the GPU.
Phase 2 (judge): load the judge model once and grade every stored answer. Grading is cached in
the result files, so re-judging or adding variants never regenerates anything.
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Optional

import yaml

from localrag.config import Config, apply_overrides, load_config
from localrag.eval import dataset as ds
from localrag.eval import judge as J
from localrag.eval.metrics import mean, mrr, ndcg_at_k, recall_at_k
from localrag.llm import get_llm, release_llms
from localrag.pipeline import RAGPipeline, ingest
from localrag.retrieval.index import release_models

MULTI = ("comparison", "bridge")


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flatten(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def _all_at_k(pages: list, gold: set, k: int) -> float:
    return float(gold <= set(pages[:k]))


# ---------------------------------------------------------------- phase 1
def generate_variant(cfg: Config, items: list[dict], out_path: Path) -> None:
    pipe = RAGPipeline(cfg)
    ks = cfg.eval.ks
    rows = []
    for n, it in enumerate(items, start=1):
        t0 = time.perf_counter()
        ans = pipe.answer(it["question"])
        latency = time.perf_counter() - t0
        row = {"id": it["id"], "type": it.get("type", "single"), "answerable": it["answerable"],
               "question": it["question"], "reference": it["answer"],
               "action": ans.action, "answer": ans.text, "insufficient": ans.insufficient_context,
               "subqueries": ans.subqueries, "trace": ans.trace, "latency": round(latency, 3),
               "timings": ans.timings,
               "claims": [{"text": c.text, "citations": c.citations, "supported": c.supported,
                           "support_score": c.support_score,
                           "passages": [ans.passages[i - 1].chunk.text for i in c.citations]}
                          for c in ans.claims]}
        kept = [c for c in ans.claims if c.supported is not False]
        if kept:
            row["cited_claim_rate"] = mean(1.0 if c.citations else 0.0 for c in kept)
        if it["answerable"]:
            gold = {tuple(g) for g in it["gold"]}
            # retrieval is scored independently of routing, with the subqueries the agent produced
            hits = pipe.retriever.retrieve(it["question"], ans.subqueries, k=max(ks))
            pages = [(h.chunk.source, h.chunk.page) for h in hits]
            row["retrieved"] = pages
            for k in ks:
                row[f"recall@{k}"] = recall_at_k(pages, gold, k)
                row[f"all@{k}"] = _all_at_k(pages, gold, k)
            row["mrr"] = mrr(pages, gold)
            row["ndcg@10"] = ndcg_at_k(pages, gold, 10)
            used = {(ans.passages[i - 1].chunk.source, ans.passages[i - 1].chunk.page)
                    for c in kept for i in c.citations}
            row["cites_gold"] = float(bool(used & gold))
            row["cites_all_gold"] = float(gold <= used)
        rows.append(row)
        print(f"  [{n}/{len(items)}] {latency:5.1f}s  {it['question'][:70]}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- phase 2
def judge_file(path: Path, judge) -> None:
    rows = [json.loads(line) for line in open(path)]
    changed = False
    for r in rows:
        if r["answerable"] and "correctness" not in r:
            r["correctness"] = J.correctness(judge, r["question"], r["reference"], r["answer"])
            changed = True
        kept = [c for c in r["claims"] if c["supported"] is not False]
        if kept and "faithfulness" not in r:
            r["faithfulness"] = mean(float(v) for v in J.faithfulness(judge, kept))
            changed = True
    if changed:
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- summary
def _ci(values: list[float], n_boot: int, seed: int = 0) -> Optional[float]:
    """Half-width of the 95% bootstrap confidence interval of the mean."""
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    means = sorted(mean(rng.choices(values, k=len(values))) for _ in range(n_boot))
    return round((means[int(0.975 * n_boot)] - means[int(0.025 * n_boot)]) / 2, 3)


def summarize(rows: list[dict], n_boot: int = 1000) -> dict:
    ans = [r for r in rows if r["answerable"]]
    single = [r for r in ans if r.get("type", "single") == "single"]
    multi = [r for r in ans if r.get("type") in MULTI]
    una = [r for r in rows if not r["answerable"]]

    def m(key, rs, ci=False):
        vals = [r[key] for r in rs if key in r]
        if not vals:
            return None
        return {"mean": round(mean(vals), 3), "ci": _ci(vals, n_boot) if ci else None, "n": len(vals)}

    return {
        "n": len(rows), "n_single": len(single), "n_multi": len(multi), "n_unanswerable": len(una),
        "R@5": m("recall@5", single, True), "MRR": m("mrr", single),
        "multi_All@5": m("all@5", multi, True), "multi_All@10": m("all@10", multi),
        "correct": m("correctness", single, True), "multi_correct": m("correctness", multi, True),
        "faithful": m("faithfulness", rows, True), "cited": m("cited_claim_rate", rows),
        "cites_gold": m("cites_gold", single), "multi_cites_all": m("cites_all_gold", multi),
        "false_refusal": {"mean": round(mean(float(r["insufficient"]) for r in ans), 3)} if ans else None,
        "unans_refusal": {"mean": round(mean(float(r["insufficient"] or r["action"] == "ANSWER_DIRECTLY")
                                              for r in una), 3)} if una else None,
        "latency_s": {"mean": round(mean(r["latency"] for r in rows), 2)},
    }


COLUMNS = ["R@5", "multi_All@5", "correct", "multi_correct", "faithful", "cited", "cites_gold",
           "multi_cites_all", "false_refusal", "unans_refusal", "latency_s"]


def to_markdown(results: dict[str, dict]) -> str:
    def cell(v):
        if not v:
            return "–"
        return f"{v['mean']} ±{v['ci']}" if v.get("ci") is not None else str(v["mean"])
    lines = ["| variant | " + " | ".join(COLUMNS) + " |", "|---" * (len(COLUMNS) + 1) + "|"]
    for name, s in results.items():
        lines.append(f"| {name} | " + " | ".join(cell(s.get(c)) for c in COLUMNS) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------- orchestration
def _variant_configs(base_config: str, spec: dict, base_overrides, base_cfg: Config):
    base = _flatten(apply_overrides({}, base_overrides))
    for v in spec["variants"]:
        overrides = {**base, **(v.get("overrides") or {})}
        if any(k.startswith("ingest.") for k in overrides):
            # ingestion settings change the chunks, so the variant gets its own index
            overrides.setdefault("index_dir", str(Path(base_cfg.index_dir) / v["name"]))
        yield v["name"], load_config(base_config, overrides)


def run_ablations(base_config: str, ablations_file: str, only: list[str] | None = None,
                  limit: Optional[int] = None, use_judge: bool = True, out_dir: str = "eval/results",
                  base_overrides: list[str] | None = None, split: Optional[str] = "test",
                  resume: bool = False, generate: bool = True) -> str:
    """`base_overrides` (CLI --set) apply to every variant; variant overrides win on conflicts."""
    spec = yaml.safe_load(Path(ablations_file).read_text())
    base_cfg = load_config(base_config, base_overrides)
    out = Path(out_dir)
    items = ds.load(base_cfg.eval.dataset)
    if split:
        items = [i for i in items if ds.split_of(i) == split]
    if limit:
        by_type: dict[str, list] = {}
        for i in items:
            by_type.setdefault(i.get("type", "single"), []).append(i)
        items = [i for group in by_type.values() for i in group[:limit]]
    variants = [(n, c) for n, c in _variant_configs(base_config, spec, base_overrides, base_cfg)
                if not only or n in only]

    if generate:
        for name, cfg in variants:
            path = out / f"{name}.jsonl"
            if resume and path.exists():
                print(f"skipping {name} (generated)")
                continue
            missing = [a for a in cfg.llm.adapters.values() if not Path(a).exists()]
            if missing:
                print(f"skipping {name}: adapter not trained yet ({', '.join(missing)})")
                continue
            print(f"\n=== generate: {name} ===")
            release_llms(keep=[cfg.llm])     # keep GPU memory for the models this variant needs
            release_models()
            if not (cfg.index_path / "chunks.jsonl").exists():
                ingest(cfg)
            generate_variant(cfg, items, path)
        release_llms()
        release_models()

    if use_judge:
        judge = get_llm(base_cfg.eval.judge)
        for name, _ in variants:
            path = out / f"{name}.jsonl"
            if path.exists():
                print(f"=== judge: {name} ===")
                judge_file(path, judge)
        release_llms()

    results = {}
    for v in spec["variants"]:
        path = out / f"{v['name']}.jsonl"
        if path.exists():
            results[v["name"]] = summarize([json.loads(line) for line in open(path)], base_cfg.eval.bootstrap)
            (out / f"{v['name']}.summary.json").write_text(json.dumps(results[v["name"]], indent=1))
    table = to_markdown(results)
    first = next(iter(results.values()), {})
    header = (f"Split: {split or 'all'}: {first.get('n_single')} single-hop, {first.get('n_multi')} multi-hop, "
              f"{first.get('n_unanswerable')} unanswerable questions. ± = 95% bootstrap CI.\n\n")
    (out / "summary.md").write_text(header + table + "\n")
    return header + table


def calibrate_relevance(cfg: Config, splits: tuple[str, ...] = ("train", "dev")) -> float:
    """Pick agent.min_relevance: the top-reranker-score threshold that best separates answerable
    from unanswerable questions (Youden's J), using retrieval only, no LLM calls.
    Never uses the test split; train is fine because only one scalar is being fitted."""
    from localrag.retrieval.index import Index
    from localrag.retrieval.retriever import Retriever

    items = [i for i in ds.load(cfg.eval.dataset) if ds.split_of(i) in splits]
    retriever = Retriever(Index(cfg.index_path, cfg.retrieval.embedding_model, cfg.retrieval.query_prompt),
                          cfg.retrieval)
    scored = [(retriever.retrieve(i["question"])[0].score, i["answerable"]) for i in items]
    pos = sorted(s for s, a in scored if a)
    neg = sorted(s for s, a in scored if not a)
    best_t, best_j = 0.0, -1.0
    for t in sorted({0.0} | {s for s, _ in scored}):
        tpr = mean(float(s >= t) for s in pos)   # answerable, not refused
        tnr = mean(float(s < t) for s in neg)    # unanswerable, refused
        if tpr + tnr - 1 > best_j:
            best_t, best_j = t, tpr + tnr - 1
    print(f"{'+'.join(splits)}: {len(pos)} answerable, {len(neg)} unanswerable")
    print(f"answerable top score   min/median: {pos[0]:.3f} / {pos[len(pos) // 2]:.3f}")
    print(f"unanswerable top score median/max: {neg[len(neg) // 2]:.3f} / {neg[-1]:.3f}")
    print(f"best threshold {best_t:.4f} (Youden J = {best_j:.3f})")
    return best_t
