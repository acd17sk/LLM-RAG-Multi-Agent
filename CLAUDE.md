# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`localrag`: a fully local RAG system over the PDFs in `documents/` (public FDA medical-device regulations), meant as a portfolio piece that shows what sub-1B models (default Qwen3.5-0.8B, compared against Qwen3-0.6B) can do inside a carefully engineered pipeline, plus optional DPO/GRPO fine-tuning. Core algorithms (parsing heuristics, chunking, BM25, SPLADE index, RRF, grounded citations, claim verification, CRAG loop, rewards, metrics) are hand-written on purpose. Use libraries only for commodity parts (llama.cpp, sentence-transformers, Chroma, PyMuPDF/Docling, TRL/PEFT), and don't swap the hand-written parts for LangChain or LlamaIndex equivalents.

## Environment and commands

- Conda env `rag` (`/home/stefos/miniconda3/envs/rag`) on an RTX 5070 Ti (Blackwell, sm_120, 16 GB): CUDA 13 builds throughout (torch `cu130`, conda-forge `llama.cpp` cuda130). The inference backend is the `llama-server` binary, not `llama-cpp-python`.
- When pip-installing into the env, pin the existing versions (`pip install X -c <(pip freeze | grep -E '^(torch|transformers|sentence-transformers)==')`) so torch keeps its CUDA build.
- `python -m localrag ingest | ask "..." --trace | eval-gen | calibrate | eval | train {dpo,dpo-refusal,grpo,grpo-decompose}`
- `python -m localrag eval [--only v1 v2] [--limit N] [--split test] [--resume] [--no-judge | --judge-only]`: generates answers for every variant in `configs/ablations.yaml` first, then grades them all with the judge (grades cached in `eval/results/<variant>.jsonl`).
- `python -m pytest tests`, or a single test: `python -m pytest tests/test_core.py::test_rrf_weights`
- Config overrides: repeatable `--set key=value`, placed before the subcommand. Unknown keys are errors (`extra="forbid"`), so a stale key fails loudly.
- `HF_HUB_DISABLE_XET=1` if a Hugging Face download stalls.
- Use `python -u` (or `PYTHONUNBUFFERED=1`) when redirecting long runs to a log file, or the log stays empty for a long time.
- **Never use `pkill -f <pattern>` or `pkill -x llama-server`.** The first matches the shell's own command line, and the second kills llama-servers started by other running jobs (dataset generation, eval). Kill by specific PID. For the same reason, a `while pgrep -f "<pattern>"` waiter never exits when the pattern appears in the waiter's own command line.

## Architecture

- `config.py`: strict pydantic models are the single source of settings. `configs/default.yaml` holds the defaults, and ablations are dotted-key overrides on top.
- `llm.py`:
  - `get_llm(cfg)` returns a shared llama-server per model config from a registry. `release_llms(keep=...)` frees GPU memory between eval variants, and `retrieval/index.release_models()` does the same for torch models.
  - `LLM.json()` passes a JSON Schema as `response_format`, which llama.cpp turns into a grammar.
  - Servers are pinned to the CUDA device, because the CUDA+Vulkan build also lists the same GPU via Vulkan plus the integrated GPU.
  - LoRA adapters (`llm.adapters: {step: path.gguf}`) are loaded with `--lora-init-without-apply` and enabled per request via `adapter="answer"|"decompose"`.
- **Ingestion** (`ingest/`): a parser (`parser.py` hand-written, or `docling_parser.py`) emits ordered `Block`s (heading with level, paragraph, or table). The chunker keeps a heading stack for the `section` path and never crosses a section or page boundary. `make_children` creates parent-child units. `Chunk.index_text` (section + optional LLM context + text) is what gets indexed, while `Chunk.text` is what the LLM sees.
- **Index** (`retrieval/index.py`): `chunks.jsonl` (plus `children.jsonl` if parent-child) is the source of truth. BM25 is built at ingest; SPLADE and dense collections are built lazily per model. With children present, search runs over children and `Index.parent()` maps hits back to parents.
- **Pipeline options:** where one technique replaces another, the parameter is a string (`retrieval.dense_backend`, `retrieval.sparse`, `agent.retrieval_strategy`, `agent.context`, `agent.compression`, `agent.answer_mode`, `agent.verifier`, `ingest.parser`). Optional stages are booleans. The README has the full table, and `demo.ipynb` exposes all of them in its parameter cell.
- **Late interaction** (`retrieval/colbert.py`): hand-written ColBERT encoder reproducing the model's PyLate config (prefixes, lengths, skiplist, bias-free projection) plus exact MaxSim. PyLate itself is incompatible with sentence-transformers 6.
- **Retrieval** (`retrieval/retriever.py`, `rerank.py`): every (query, retriever) ranking is fused by weighted RRF (the original query gets 2× weight), then the top `rerank_k` are scored by a `PairScorer` (cross-encoder or llama `/v1/rerank`). The same interface serves the claim verifier (reranker or NLI entailment).
- **Agents** (`agents.py`, `pipeline.py`): route → decompose → retrieve → optional CRAG rewrite loop → relevance gate → grounded answer → verify → optional self-correction. `grounded_request` and `decompose_request` build the exact messages and schemas, and are shared with RL training so train and inference prompts match. Prompts state the JSON format explicitly, because training generation is unconstrained. The grounded answer has no up-front "insufficient" flag, since small models set it immediately and refuse everything; an empty claim list is the refusal.
- **Eval** (`eval/`):
  - The dataset has `type` (single, comparison, bridge, unanswerable) and `split` (train, dev, test).
  - Splits are assigned by gold page, so train and test never share a page.
  - Metrics use (source, page) relevance, so they survive chunking changes.
  - Variants overriding `ingest.*` get `index/<variant>/`, and variants whose adapter files are missing are skipped.
  - The teacher (Qwen3.5-9B) and judge (Gemma 4 12B) are deliberately different model families.
- **Refusal training** (`train dpo-refusal`): refusal pairs (unanswerable questions + gold-removed counterfactuals) must be mirrored by anti-refusal pairs (same questions with evidence, refusal rejected) plus an SFT term. Without the mirror, DPO collapses to always refusing; that run is kept in `eval/results_extra/`.
- **RL** (`rl/`): `rewards.py` (verifiable, unit-tested), `data.py` (train-split prompts), `train.py` (TRL DPO/GRPO with LoRA on the HF weights `Qwen/Qwen3.5-0.8B`), `export.py` (PEFT → GGUF via llama.cpp's converter at tag `rl.llama_cpp_tag`, cached in `~/.cache/localrag/`). Prompts are rendered with `enable_thinking=False`, as llama-server does.

## Gotchas

- The PDF file names in `documents/` were renamed to match their real contents, because FDA download IDs pointed to different documents than expected.
- The 21 CFR 820 PDF is two-column with wide margins. Re-check it after any parser change.
- Thresholds depend on the scorer: bge-reranker and Qwen3-Reranker both return 0–1, but `support_threshold` should be about 0.05 for the reranker verifier and 0.5 for NLI.
- Qwen3.5-0.8B on HF is a vision-language checkpoint. `AutoModelForCausalLM` loads its text part (`Qwen3_5ForCausalLM`), which has 18 linear-attention and 6 full-attention layers. LoRA targets include `in_proj_qkv`, `in_proj_z` and `out_proj`.
