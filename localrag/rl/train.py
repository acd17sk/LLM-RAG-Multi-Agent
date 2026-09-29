"""Optional preference / RL fine-tuning of the small generator with LoRA.

    python -m localrag train dpo              # preference pairs ranked by the verifiable reward
    python -m localrag train dpo-refusal      # + refusal pairs (unanswerable / evidence removed)
    python -m localrag train grpo             # online RL with the same reward (answer step)
    python -m localrag train grpo-decompose   # online RL for query decomposition (retrieval recall reward)

Each run writes a PEFT adapter to adapters/<name>/ and a GGUF copy to adapters/<name>.gguf,
which the pipeline loads with `llm.adapters.<step>: adapters/<name>.gguf`.
Only the TRAIN split is used; evaluation stays on held-out test questions.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from localrag.config import Config
from localrag.rl import data
from localrag.rl.export import lora_to_gguf
from localrag.retrieval.index import release_models
from localrag.rl.rewards import AnswerExample, RewardWeights, answer_reward, decompose_reward

STEP_OF = {"dpo": "answer", "dpo-refusal": "answer", "grpo": "answer", "grpo-decompose": "decompose"}


def _tokenizer(cfg: Config):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.rl.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def _render(tok, messages: list[dict]) -> str:
    """The prompt string exactly as llama-server renders it at inference (thinking disabled)."""
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def _model(cfg: Config):
    import torch
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(cfg.rl.base_model, dtype=torch.bfloat16, device_map="cuda")


def _peft(cfg: Config):
    from peft import LoraConfig
    return LoraConfig(r=cfg.rl.lora_r, lora_alpha=cfg.rl.lora_alpha, lora_dropout=0.0,
                      target_modules=cfg.rl.target_modules, task_type="CAUSAL_LM")


def _nli(cfg: Config):
    from localrag.retrieval.rerank import NLIScorer
    return NLIScorer(cfg.agent.nli_model).score


def _finish(trainer, cfg: Config, name: str) -> Path:
    out = Path(cfg.rl.output_dir) / name
    trainer.save_model(str(out))
    gguf = lora_to_gguf(out, Path(cfg.rl.output_dir) / f"{name}.gguf", cfg.rl.base_model, cfg.rl.llama_cpp_tag)
    print(f"\nAdapter: {out}/  GGUF: {gguf}\nUse it with:  --set llm.adapters.{STEP_OF[name]}={gguf}")
    return gguf


# ------------------------------------------------------------------ DPO
def build_dpo_pairs(cfg: Config, rows: list[dict], scorer, seed: int = 0) -> list[dict]:
    """Sample answers from the current model (grammar-constrained, via llama-server), score them
    with the verifiable reward, and keep (best, worst) pairs whose reward gap is large enough."""
    from localrag.agents import grounded_request
    from localrag.llm import get_llm, release_llms

    llm = get_llm(cfg.llm)
    rng = random.Random(seed)
    pairs = []
    for n, r in enumerate(rows, start=1):
        ex = AnswerExample(r["passages"], r["gold"], r["answerable"], r["reference"])
        _, schema = grounded_request("", [f"[{i}]" for i in range(1, len(r["passages"]) + 1)])
        samples = {llm.chat(r["prompt"], schema=schema, max_tokens=cfg.rl.max_completion_length,
                            temperature=cfg.rl.temperature, seed=rng.randrange(2**31))
                   for _ in range(cfg.rl.dpo_samples)}
        scored = sorted(((answer_reward(s, ex, scorer)[0], s) for s in samples), reverse=True)
        if len(scored) >= 2 and scored[0][0] - scored[-1][0] >= cfg.rl.dpo_min_margin:
            pairs.append({"prompt": r["prompt"], "chosen": scored[0][1], "rejected": scored[-1][1],
                          "margin": scored[0][0] - scored[-1][0]})
        if n % 20 == 0:
            print(f"  sampled {n}/{len(rows)} prompts -> {len(pairs)} pairs")
    release_llms()
    return pairs


def _answer_pairs(cfg: Config, limit: int | None) -> list[dict]:
    pairs_file = Path(cfg.rl.output_dir) / "dpo_pairs.jsonl"
    if pairs_file.exists() and not limit:
        # sampling is the slow part; delete the file to re-sample
        pairs = [json.loads(line) for line in open(pairs_file)]
        print(f"reusing {len(pairs)} preference pairs from {pairs_file}")
        return pairs
    rows = data.answer_examples(cfg, "train", limit)
    pairs = build_dpo_pairs(cfg, rows, _nli(cfg))
    Path(cfg.rl.output_dir).mkdir(exist_ok=True)
    with open(pairs_file, "w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"{len(pairs)} preference pairs from {len(rows)} train prompts")
    return pairs


REFUSAL_ANSWER = json.dumps({"claims": []})


def build_refusal_pairs(cfg: Config, scorer, max_counterfactual: int = 80, seed: int = 0) -> list[dict]:
    """Pairs that teach *when* to refuse: chosen = empty claim list, rejected = the base model's answer.

    Two sources, both from the train split:
      unanswerable     questions the corpus does not answer (already judge-filtered)
      counterfactual   answerable questions with the gold pages removed from the retrieved passages,
                       kept only if NLI confirms the remaining passages do not entail the reference
                       answer, i.e. the evidence genuinely isn't there
    """
    from localrag.agents import format_passages, grounded_request
    from localrag.eval import dataset as ds
    from localrag.llm import get_llm, release_llms
    from localrag.rl.rewards import _SENTENCES, parse_claims

    rng = random.Random(seed)
    retriever = data._retriever(cfg)
    train = [i for i in ds.load(cfg.eval.dataset) if ds.split_of(i) == "train"]
    cases = [("unanswerable", it, retriever.retrieve(it["question"])) for it in train if not it["answerable"]]

    singles = [i for i in train if i["answerable"] and i.get("type") == "single"]
    rng.shuffle(singles)
    n_cf = 0
    for it in singles:
        if n_cf >= max_counterfactual:
            break
        gold = {tuple(g) for g in it["gold"]}
        hits = [h for h in retriever.retrieve(it["question"], k=4 * cfg.retrieval.top_k)
                if (h.chunk.source, h.chunk.page) not in gold][:cfg.retrieval.top_k]
        ref = [x for x in _SENTENCES.split(it["answer"]) if len(x.split()) >= 4]
        if not hits or not ref:
            continue
        entail = scorer([(r, h.chunk.text) for r in ref for h in hits])
        if max(entail) < 0.5:          # no remaining passage supports any part of the reference
            cases.append(("counterfactual", it, hits))
            n_cf += 1

    llm = get_llm(cfg.llm)
    pairs = []
    for kind, it, hits in cases:
        messages, schema = grounded_request(it["question"], format_passages(hits))
        for _ in range(3):             # find a sampled answer that makes claims (the behaviour to unlearn)
            answer = llm.chat(messages, schema=schema, max_tokens=cfg.rl.max_completion_length,
                              temperature=0.8, seed=rng.randrange(2**31))
            if parse_claims(answer, len(hits)):
                pairs.append({"prompt": messages, "chosen": REFUSAL_ANSWER, "rejected": answer, "kind": kind})
                break
    release_llms()
    print(f"refusal pairs: {sum(p['kind'] == 'unanswerable' for p in pairs)} unanswerable, "
          f"{sum(p['kind'] == 'counterfactual' for p in pairs)} counterfactual")
    return pairs


def _train_dpo_on(cfg: Config, pairs: list[dict], name: str, sft_weight: float = 0.0) -> Path:
    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer

    release_models()   # free the NLI / retrieval models before training
    tok = _tokenizer(cfg)
    ds = Dataset.from_list([{"prompt": _render(tok, p["prompt"]), "chosen": p["chosen"], "rejected": p["rejected"]}
                            for p in pairs])
    loss = {"loss_type": ["sigmoid", "sft"], "loss_weights": [1.0, sft_weight]} if sft_weight else {}
    args = DPOConfig(output_dir=str(Path(cfg.rl.output_dir) / "runs" / name), beta=cfg.rl.dpo_beta, **loss,
                     learning_rate=cfg.rl.learning_rate, num_train_epochs=cfg.rl.dpo_epochs, max_steps=cfg.rl.max_steps,
                     per_device_train_batch_size=cfg.rl.batch_size, gradient_accumulation_steps=cfg.rl.grad_accum,
                     max_length=cfg.rl.max_length, bf16=True, gradient_checkpointing=True, logging_steps=5,
                     save_strategy="no", report_to="none")
    trainer = DPOTrainer(model=_model(cfg), args=args, train_dataset=ds, processing_class=tok,
                         peft_config=_peft(cfg))
    trainer.train()
    return _finish(trainer, cfg, name)


def train_dpo(cfg: Config, limit: int | None = None) -> Path:
    return _train_dpo_on(cfg, _answer_pairs(cfg, limit), "dpo")


def _question(prompt: list[dict]) -> str:
    return prompt[-1]["content"].rsplit("Question:", 1)[-1].strip()


def anti_refusal_pairs(answer_pairs: list[dict], refusal_pairs: list[dict], seed: int = 0) -> list[dict]:
    """Mirror image of the refusal pairs: evidence present -> the good answer is chosen and the
    empty refusal is REJECTED. Wherever possible these are minimal pairs with the counterfactuals
    (same question; only the presence of the evidence differs), so the model has to read the
    passages to tell the two cases apart."""
    from localrag.rl.rewards import parse_claims

    usable = [p for p in answer_pairs if parse_claims(p["chosen"], 99)]
    by_question = {_question(p["prompt"]): p for p in usable}
    picked, used = [], set()
    for r in refusal_pairs:
        match = by_question.get(_question(r["prompt"]))
        if match is not None and id(match) not in used:
            picked.append(match)
            used.add(id(match))
    rest = [p for p in usable if id(p) not in used]
    random.Random(seed).shuffle(rest)
    picked += rest[:max(0, len(refusal_pairs) - len(picked))]
    return [{"prompt": p["prompt"], "chosen": p["chosen"], "rejected": REFUSAL_ANSWER, "kind": "anti_refusal"}
            for p in picked]


def train_dpo_refusal(cfg: Config, limit: int | None = None) -> Path:
    """DPO on answer pairs + refusal pairs + mirrored anti-refusal pairs, with an SFT anchor (RPO).

    A first version trained on answer + refusal pairs only and collapsed to refusing everything:
    the refusal string was only ever *chosen*, so raising its likelihood everywhere satisfied every
    preference. The anti-refusal pairs make the same string *rejected* whenever evidence is present.
    """
    answer = _answer_pairs(cfg, limit)
    refusal_file = Path(cfg.rl.output_dir) / "dpo_refusal_pairs.jsonl"
    if refusal_file.exists() and not limit:
        refusal = [json.loads(line) for line in open(refusal_file)]
        print(f"reusing {len(refusal)} refusal pairs from {refusal_file}")
    else:
        refusal = build_refusal_pairs(cfg, _nli(cfg))
        with open(refusal_file, "w") as f:
            for p in refusal:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
    anti = anti_refusal_pairs(answer, refusal)
    minimal = len({_question(p["prompt"]) for p in anti} & {_question(p["prompt"]) for p in refusal})
    print(f"training on {len(answer)} answer + {len(refusal)} refusal + {len(anti)} anti-refusal pairs "
          f"({minimal} minimal pairs), SFT weight {cfg.rl.refusal_sft_weight}")
    pairs = answer + refusal + anti
    random.Random(0).shuffle(pairs)
    return _train_dpo_on(cfg, pairs, "dpo-refusal", sft_weight=cfg.rl.refusal_sft_weight)


# ------------------------------------------------------------------ GRPO
def _grpo_args(cfg: Config, name: str):
    from trl import GRPOConfig
    # one generation round = prompts_per_step x num_generations sequences, scored in micro-batches,
    # and accumulated into exactly one optimizer step
    gen_batch = cfg.rl.grpo_prompts_per_step * cfg.rl.num_generations
    return GRPOConfig(output_dir=str(Path(cfg.rl.output_dir) / "runs" / name),
                      learning_rate=cfg.rl.learning_rate, num_train_epochs=cfg.rl.epochs, max_steps=cfg.rl.max_steps,
                      per_device_train_batch_size=cfg.rl.grpo_micro_batch,
                      gradient_accumulation_steps=gen_batch // cfg.rl.grpo_micro_batch,
                      num_generations=cfg.rl.num_generations, generation_batch_size=gen_batch,
                      max_completion_length=cfg.rl.max_completion_length, temperature=cfg.rl.temperature,
                      beta=cfg.rl.grpo_beta, loss_type="dr_grpo", mask_truncated_completions=True,
                      bf16=True, gradient_checkpointing=True, logging_steps=1, log_completions=False,
                      save_strategy="no", report_to="none")


def train_grpo(cfg: Config, limit: int | None = None) -> Path:
    from datasets import Dataset
    from trl import GRPOTrainer

    tok = _tokenizer(cfg)
    rows = data.answer_examples(cfg, "train", limit)
    ds = Dataset.from_list([{"prompt": _render(tok, r["prompt"]), "passages": r["passages"], "gold": r["gold"],
                             "answerable": r["answerable"], "reference": r["reference"]} for r in rows])
    scorer, weights = _nli(cfg), RewardWeights()

    def reward(prompts, completions, passages, gold, answerable, reference, **_):
        return [answer_reward(c, AnswerExample(p, g, a, ref), scorer, weights)[0]
                for c, p, g, a, ref in zip(completions, passages, gold, answerable, reference)]

    trainer = GRPOTrainer(model=_model(cfg), reward_funcs=[reward], args=_grpo_args(cfg, "grpo"),
                          train_dataset=ds, processing_class=tok, peft_config=_peft(cfg))
    trainer.train()
    return _finish(trainer, cfg, "grpo")


def train_grpo_decompose(cfg: Config, limit: int | None = None) -> Path:
    from datasets import Dataset
    from trl import GRPOTrainer

    tok = _tokenizer(cfg)
    rows = data.decompose_examples(cfg, "train", limit)
    # Arrow columns need one type, so gold pages are encoded "source|page"
    ds = Dataset.from_list([{"prompt": _render(tok, r["prompt"]), "question": r["question"],
                             "gold": [f"{src}|{page}" for src, page in r["gold"]]} for r in rows])
    retriever = data._retriever(cfg)

    def reward(prompts, completions, question, gold, **_):
        return [decompose_reward(c, q, {(x.rsplit("|", 1)[0], int(x.rsplit("|", 1)[1])) for x in g},
                                 lambda q_, s, k: retriever.retrieve(q_, s, k=k),
                                 max_queries=cfg.agent.max_subqueries)[0]
                for c, q, g in zip(completions, question, gold)]

    args = _grpo_args(cfg, "grpo-decompose")
    args.max_completion_length = 200
    trainer = GRPOTrainer(model=_model(cfg), reward_funcs=[reward], args=args,
                          train_dataset=ds, processing_class=tok, peft_config=_peft(cfg))
    trainer.train()
    return _finish(trainer, cfg, "grpo-decompose")


TRAINERS = {"dpo": train_dpo, "dpo-refusal": train_dpo_refusal, "grpo": train_grpo,
            "grpo-decompose": train_grpo_decompose}
