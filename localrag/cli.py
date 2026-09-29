"""Command line interface.

    python -m localrag ingest
    python -m localrag ask "What are the design verification requirements?"
    python -m localrag eval-gen --n 120
    python -m localrag eval --only baseline hybrid
Any config value can be overridden with repeatable --set key=value flags before the command.
"""
from __future__ import annotations

import argparse
import json

from localrag.config import load_config


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="localrag")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="config override, repeatable: --set retrieval.mode=dense --set agent.decompose=false")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("ingest", help="parse, chunk and index the PDFs")

    ask = sub.add_parser("ask", help="answer a question")
    ask.add_argument("question")
    ask.add_argument("--trace", action="store_true", help="print retrieval and claim details")

    gen = sub.add_parser("eval-gen", help="generate the synthetic eval set with the teacher model")
    gen.add_argument("--n-single", type=int, default=300)
    gen.add_argument("--n-multi", type=int, default=150)
    gen.add_argument("--n-unanswerable", type=int, default=60)

    ev = sub.add_parser("eval", help="run ablation variants (generate, then judge)")
    ev.add_argument("--ablations", default="configs/ablations.yaml")
    ev.add_argument("--only", nargs="*")
    ev.add_argument("--limit", type=int, help="use only the first N questions of each type")
    ev.add_argument("--split", default="test", choices=["train", "dev", "test", "all"])
    ev.add_argument("--resume", action="store_true", help="skip variants that already have answers")
    ev.add_argument("--no-judge", action="store_true", help="skip LLM-judge metrics")
    ev.add_argument("--judge-only", action="store_true", help="only grade existing answers")
    ev.add_argument("--out-dir", default="eval/results")

    sub.add_parser("calibrate", help="pick agent.min_relevance on the train+dev splits")

    tr = sub.add_parser("train", help="optional: DPO / GRPO fine-tuning of the generator (LoRA)")
    tr.add_argument("method", choices=["dpo", "dpo-refusal", "grpo", "grpo-decompose"])
    tr.add_argument("--limit", type=int, help="use only the first N train questions")

    args = p.parse_args(argv)
    cfg = load_config(args.config, args.set)

    if args.cmd == "ingest":
        from localrag.pipeline import ingest
        ingest(cfg)

    elif args.cmd == "ask":
        from localrag.pipeline import RAGPipeline
        ans = RAGPipeline(cfg).answer(args.question)
        if args.trace:
            print(f"action: {ans.action}\nsubqueries: {ans.subqueries}\ntimings: {ans.timings}\n")
            for i, h in enumerate(ans.passages, 1):
                print(f"[{i}] {h.score:.3f} {h.chunk.source} p.{h.chunk.page} | {h.chunk.section[:60]}")
            for c in ans.claims:
                print(f"  claim supported={c.supported} score={c.support_score}: {c.text[:80]} {c.citations}")
            print()
        print(ans.text)

    elif args.cmd == "eval-gen":
        from localrag.eval import dataset as ds
        from localrag.llm import get_llm, release_llms
        from localrag.retrieval.index import Index
        from localrag.retrieval.retriever import Retriever
        index = Index(cfg.index_path, cfg.retrieval.embedding_model, cfg.retrieval.query_prompt)
        items = ds.generate_dataset(list(index.chunks.values()), get_llm(cfg.eval.teacher),
                                    args.n_single, args.n_multi, args.n_unanswerable,
                                    embedding_model=cfg.retrieval.embedding_model)
        items = ds.filter_unanswerable(items, Retriever(index, cfg.retrieval), get_llm(cfg.eval.teacher))
        release_llms()
        ds.save(items, cfg.eval.dataset)
        counts = {}
        for i in items:
            key = (i["type"], ds.split_of(i))
            counts[key] = counts.get(key, 0) + 1
        print(f"wrote {len(items)} items to {cfg.eval.dataset}: {counts}")

    elif args.cmd == "eval":
        from localrag.eval.runner import run_ablations
        print(run_ablations(args.config, args.ablations, args.only, args.limit, not args.no_judge,
                            out_dir=args.out_dir, base_overrides=args.set,
                            split=None if args.split == "all" else args.split,
                            resume=args.resume, generate=not args.judge_only))

    elif args.cmd == "calibrate":
        from localrag.eval.runner import calibrate_relevance
        calibrate_relevance(cfg)

    elif args.cmd == "train":
        from localrag.rl.train import TRAINERS
        TRAINERS[args.method](cfg, args.limit)


if __name__ == "__main__":
    main()
