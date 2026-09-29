# localrag: accurate RAG with sub-1B local models

A fully local question-answering system over PDF documents, built to show how far a
**0.6–0.8B parameter model** can go when the retrieval, generation and training around it are
engineered carefully. It runs on a single consumer GPU (developed on an RTX 5070 Ti, 16 GB),
sends no data to external services, and every design choice is measured by a leave-one-out
ablation on a held-out evaluation set.

The demo corpus is public US FDA medical-device regulation (21 CFR 820, software
validation, off-the-shelf software, cybersecurity, and AI/ML SaMD guidance).

## Pipeline

```
PDFs ─► parser ─────────► section-aware chunker ─► [child chunks] ─► BM25 | SPLADE   ─┐
        hand-written       heading paths, tables     parent-child      dense (Chroma) ─┤
        or Docling         kept whole, hash IDs      "small-to-big"    [+ LLM context] │
                                                                                      ▼
question ─► router ─► decomposer ─► per-query sparse + dense ─► weighted RRF ─► rerank
            (grammar-   (grammar-                                              (cross-encoder or
             constrained) constrained, optional RL adapter)                    Qwen3-Reranker)
                                                                                      │
                   ┌── CRAG: weak evidence → rewrite query → retrieve again ◄─────────┤
                   │   relevance gate: still weak → refuse                            ▼
answer ◄── self-correction ◄── claim verification ◄── grounded generation (claims + citation IDs,
           (regenerate once       (NLI entailment or      grammar-constrained; optional
            with feedback)         reranker, per claim)    DPO/GRPO adapter)
```

### Pipeline options

Every stage is selectable. Where one technique *replaces* another, the parameter is a string:

| Stage | Parameter | Options (default first) |
|---|---|---|
| Parser | `ingest.parser` | `pymupdf` (hand-written) · `docling` |
| Sparse retrieval | `retrieval.sparse` | `bm25` · `splade` |
| Dense retrieval | `retrieval.dense_backend` | `bi_encoder` · `late_interaction` (ColBERT MaxSim) |
| Retrieval mix | `retrieval.mode` | `hybrid` · `dense` · `sparse` |
| Reranker | `retrieval.reranker_backend` | `cross_encoder` · `llama` (Qwen3-Reranker) |
| Retrieval control | `agent.retrieval_strategy` | `single` · `iterative` (LLM names what's missing and searches again) |
| What the LLM reads | `agent.context` | `chunk` · `window` (+ neighbours) · `section` (retrieve less, read more) |
| Compression | `agent.compression` | `none` · `sentences` (extractive, reranker-scored) |
| Answer format | `agent.answer_mode` | `grounded` · `grounded_evidence` · `freetext` |
| Claim verifier | `agent.verifier` | `reranker` · `nli` · `none` |
| Adapters | `llm.adapters.{answer,decompose}` | path to a trained `.gguf` LoRA, or unset |

Booleans switch optional stages on or off: `agent.route`, `agent.decompose`, `agent.crag`,
`agent.self_correct`, `ingest.contextualize`, plus `ingest.child_size` (parent-child) and
`agent.min_relevance` (refusal gate).

### What is hand-written (and why)

| Component | What it does | Why it matters for small models |
|---|---|---|
| `ingest/parser.py` | Font-statistics heading hierarchy, column-aware reading order, running header/footer removal, de-hyphenation | Clean, well-scoped chunks; two-column regulations aren't interleaved |
| `ingest/chunker.py` | Chunks never cross section/page boundaries, carry their heading path; tables kept whole (split by rows with repeated header); parent-child children; content-hash IDs | Section context in every embedding; idempotent re-ingestion |
| `retrieval/bm25.py` | Okapi BM25 with an inverted index; tokenizer keeps identifiers like `820.30` | Exact regulatory references are where dense retrieval fails |
| `retrieval/index.py` | SPLADE inverted index over learned term weights; section/window context expansion | Learned sparse retrieval without a search engine |
| `retrieval/colbert.py` | ColBERT token encoder (reproducing the model's PyLate config) and exact MaxSim search | Late interaction without extra dependencies |
| `retrieval/fusion.py` | Weighted Reciprocal Rank Fusion over (query × retriever) rankings | No score calibration needed; the original query outweighs sub-queries |
| `agents.py` | Router, decomposer, CRAG query rewriter, grounded answerer. All structured outputs are JSON-Schema → llama.cpp grammar; citations are an `enum` of the passage numbers shown | A 0.8B model *cannot* emit malformed JSON or cite a non-existent source |
| `agents.verify_claims` | Scores each claim against its cited passages (and, for multi-citation claims, against them combined) with an NLI model | Unsupported claims are dropped before the user sees them |
| `rl/rewards.py` | Verifiable reward: format, NLI-entailed citations, gold-page citation, refusal calibration, length | Rewards that can't be gamed without actually being faithful |
| `eval/` | Synthetic single- and multi-hop QA, page-level retrieval metrics, cross-family LLM judge, bootstrap CIs, dev-calibrated thresholds, two-phase runner | Every technique has to earn its place in the numbers |

Libraries are used for commodity parts only: llama.cpp (`llama-server`) for inference,
sentence-transformers for embedding / reranking / NLI models, ChromaDB for vectors,
PyMuPDF and Docling for reading PDFs, TRL + PEFT for training.

## Optional: RL / preference fine-tuning

The generator can be improved with LoRA adapters trained on the **train split** using a
verifiable reward (`localrag/rl/rewards.py`). The claim-level reward uses the same NLI
entailment check as the verifier, so the model is paid only for claims its cited passages
actually support, plus correct refusals of unanswerable questions.

```bash
python -m localrag train dpo              # sample answers, rank by reward, train on best-vs-worst pairs
python -m localrag train dpo-refusal      # + refusal and mirrored anti-refusal pairs (learns when NOT to answer)
python -m localrag train grpo             # online RL (GRPO, Dr. GRPO loss) with the same reward
python -m localrag train grpo-decompose   # RL for query decomposition; reward = recall of gold pages
```

Each command writes `adapters/<name>/` (PEFT) and `adapters/<name>.gguf`. llama-server loads
all configured adapters once and enables one per request, so different pipeline steps can
use different adapters on the same base model:

```bash
python -m localrag --set llm.adapters.answer=adapters/grpo.gguf ask "..."
```

**Teaching refusal.** `dpo-refusal` adds pairs where refusing (`{"claims": []}`) is the chosen
answer: unanswerable train questions, plus *counterfactuals*, which are answerable questions with the
gold pages removed from the passages, kept only if NLI confirms the rest can't answer them. A first
version trained on these alone **collapsed to refusing everything** (100% false refusals): the
refusal string was only ever *chosen*, so making it likely everywhere satisfied every preference.
The fix is mirrored **anti-refusal pairs**: the same questions *with* their evidence, where the
refusal is *rejected* (70 of 100 are exact minimal pairs), plus a small SFT term on chosen answers
(RPO-style). The model then has to read the passages to decide.

In `demo.ipynb` this is a parameter (`ANSWER_ADAPTER = "grpo"`), and a cell compares base vs.
DPO vs. GRPO answers side by side. Nothing requires training: without adapters everything runs
on the base model.

## Setup

```bash
conda create -n rag -c conda-forge python=3.12 "llama.cpp=*=cuda130*"   # or the cpu build
conda activate rag
pip install torch --index-url https://download.pytorch.org/whl/cu130   # match your CUDA
pip install -r requirements.txt
pip install -r requirements-rl.txt                                       # optional: training
```

Models are downloaded from the Hugging Face Hub on first use.

## Usage

```bash
python -m localrag ingest                         # parse, chunk and index documents/*.pdf
python -m localrag ask "What must design verification confirm?" --trace
python -m localrag eval-gen                       # build eval/dataset.jsonl with the teacher model
python -m localrag calibrate                      # pick the refusal threshold on the dev split
python -m localrag eval                           # all variants in configs/ablations.yaml, test split
python -m localrag eval --only default crag --limit 20
python -m pytest tests
```

All settings live in `configs/default.yaml` and can be overridden per command, e.g.
`python -m localrag --set retrieval.sparse=splade --set agent.crag=true ask "..."`.

## Evaluation

`eval/dataset.jsonl` is written by a 9B **teacher** (Qwen3.5-9B) and graded by a 12B **judge**
from a different model family (Gemma 4 12B), so the judge isn't grading text in its own style.

- **Question types:** single-hop (direct / paraphrased / practical), multi-hop *comparison*
  (two documents) and *bridge* (two sections), and unanswerable questions. Unanswerable
  labels are double-checked against retrieval by the model, and noisy ones dropped.
- **Splits:** train (50%) / dev (15%) / test (35%), assigned by **source page**, so no gold page is
  shared between training and test. Thresholds are calibrated on dev, results reported on test.
- **Metrics:** R@5 (single-hop), All@5 (all gold pages retrieved, multi-hop), judge correctness,
  judge faithfulness of cited claims, citation rate, gold-page citation, false/correct refusal rates,
  latency. `±` is a 95% bootstrap confidence interval.
- **Two-phase runner:** answers for all variants are generated first (only small models on the GPU),
  then the judge grades everything in one pass; grades are cached in the result files.

Full table (32 variants): `eval/results/summary.md`. Superseded runs are kept in
`eval/results_extra/` (the collapsed refusal adapter, iterative retrieval before a loop fix), and
the first iteration on a simpler single-hop set is in `eval/results_v1/`.

## Results

Test split: 104 single-hop, 47 multi-hop, 17 unanswerable questions; ± is the 95% bootstrap CI.
*correct* = judge agreement with the reference answer; *faithful* = share of answer claims the
judge finds supported by the passages they cite.

| variant | correct | multi-hop correct | faithful | cites gold page | refuses unanswerable | false refusals |
|---|---|---|---|---|---|---|
| v0: original design (dense, prompt-only citations, Qwen3-0.6B) | 0.81 ±.07 | 0.59 | 0.12 ±.04 | 0.22 | 0% | 0% |
| free-text answers, current retrieval | 0.86 ±.06 | 0.71 | 0.04 ±.03 | 0.06 | 0% | 0% |
| **default** (hybrid + rerank, grounded claims, reranker verifier) | 0.78 ±.06 | 0.62 | 0.90 ±.03 | 0.92 | 6% | 0% |
| NLI verifier | 0.74 ±.06 | 0.48 | 0.96 ±.02 | 0.89 | 41% | 5% |
| **NLI verifier + DPO adapter** | **0.91 ±.04** | 0.64 | **0.98 ±.01** | **0.94** | 0% | 0% |
| NLI verifier + GRPO adapter | 0.82 ±.06 | 0.43 | 0.98 ±.02 | 0.89 | 12% | 4% |
| **NLI verifier + DPO-refusal adapter** | 0.83 ±.05 | 0.54 | 0.94 ±.02 | 0.90 | **100%** | 8% |
| DPO-refusal v1 (no anti-refusal pairs): collapsed | – | – | – | 0.00 | 100% | **100%** |
| relevance gate (threshold from train+dev) | 0.64 ±.08 | 0.52 | 0.93 ±.03 | 0.78 | 94% | 21% |
| DPO adapter + relevance gate | 0.75 ±.07 | 0.55 | 0.97 ±.02 | 0.79 | 94% | 21% |

### What the ablations show

- **Grounding is the core win.** Prompt-only citations are almost never faithful (4–12% of claims),
  whatever the model. Schema-enforced citations plus verification bring this to 90–98%.
- **Preference tuning removed the faithfulness/correctness trade-off.** Grounded answers were
  less complete than free text (0.78 vs 0.86 correct). The DPO adapter, trained on the train split
  with the verifiable reward (support + gold citation + coverage), reaches **0.91 correct with 0.98
  faithful**: better than free text on correctness, while nearly every claim is backed by its citation.
- **DPO and GRPO optimized the same reward differently.** DPO writes more claims, more carefully
  (5.9 per answer, 95% verifier-accepted). GRPO learned to say *less* (3.1 claims per answer):
  equally faithful, but it lost multi-hop completeness (0.43). With online RL the cheapest path
  through the reward was to be terse, even with a coverage term; a higher coverage weight is the obvious next experiment.
- **Refusal can be learned, and it beats a threshold.** The DPO and GRPO adapters never refused:
  their training samples contained no refusals. The score-threshold gate catches 94% of unanswerable
  questions but wrongly refuses 21% of answerable ones, because it sees one retrieval score, not the
  evidence. The DPO-refusal adapter refuses **all 17** unanswerable test questions at **8%** false
  refusals, and still beats the untuned pipeline on correctness (0.83 vs 0.74–0.78). Stacking the gate
  on top only adds false refusals. Remaining weakness: comparison questions (30% false refusals). When
  one side of a comparison is missing, the model refuses instead of answering the side it has; partial-evidence
  pairs are the next fix. 17/17 has a wide CI (true rate plausibly ≥ ~80%).
- **Retrieval.** The cross-encoder reranker matters most: without it, multi-hop All@5 drops from 0.49 to 0.30.
  BM25 vs SPLADE, contextual retrieval, parent-child, Qwen3-Embedding and Docling are all within
  the confidence intervals on this corpus (contextual / parent-child / Qwen3-Embedding trend up on
  multi-hop: 0.53–0.55 vs 0.49).
- **Newer retrieval and reading strategies didn't beat the default on this corpus:** late interaction
  (ColBERT: 0.76 correct), whole sections ("retrieve less, read more": 0.77), extractive compression
  (0.75), all within the CI of the default (0.78). Adjacent-chunk windows *lowered* faithfulness
  (0.90 → 0.84): with more text per passage, the 0.8B model more often blends details. More context
  is not free for a small reader. Iterative retrieval also showed no gain (0.77 correct,
  multi-hop All@5 0.47, +50% latency): the 0.8B model almost never judges its evidence complete
  (2 hops on 119 of 168 questions), so a trained search policy (Search-R1-style RL) is the next step.
- **No measurable gain:** CRAG (fired on 19% of questions), self-correction, evidence-first
  prompting, query decomposition, and the RL-trained decomposer. These are reported as-is.
- **The generator matters in the grounded pipeline:** Qwen3.5-0.8B scores 0.78 correct vs 0.65 for Qwen3-0.6B.

**Recommended configurations** (both use `--set agent.verifier=nli --set agent.support_threshold=0.5`):

- **Open-ended users → `llm.adapters.answer=adapters/dpo-refusal.gguf`** (`train dpo-refusal`).
  Refuses what the documents don't cover, 0.83 correct, 0.94 faithful.
- **Questions known to be in scope → `llm.adapters.answer=adapters/dpo.gguf`** (`train dpo`).
  0.91 correct, 0.98 faithful, but it will attempt every question.

### Limitations

- The questions are synthetic (written by a 9B teacher from single chunks or chunk pairs). Some
  comparison questions are contrived, and the unanswerable set is small (17 test questions).
- The judge is an LLM (Gemma 4 12B, a different family from the generator and teacher). It has not been
  calibrated against human grades.
- The corpus is 5 documents (~170 pages). Retrieval differences that are within the CIs here may
  matter on larger corpora.
- Variants generated after a robustness fix use a 1,200-token answer budget instead of 700. Only
  evidence-first answers approach that limit.
- The verifier checks that a cited passage supports a claim's *content*, not that the claim names
  the right *source document*. A claim can say "the OTS guidance requires X" while correctly citing
  a cybersecurity-guidance passage that says X.
- Each adapter comes from a single training run (no seeds averaged).
