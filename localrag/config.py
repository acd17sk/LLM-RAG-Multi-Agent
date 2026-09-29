"""Typed configuration loaded from YAML, with dotted-key overrides for ablations."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    # unknown keys are errors, so a typo'd or stale override can't silently do nothing
    model_config = ConfigDict(extra="forbid")


class LLMConfig(_Strict):
    repo_id: str = "unsloth/Qwen3.5-0.8B-GGUF"
    filename: str = "Qwen3.5-0.8B-Q8_0.gguf"
    n_ctx: int = 8192
    n_gpu_layers: int = -1
    # Base URL of an already-running OpenAI-compatible server. If unset, a
    # llama-server process is launched for this model.
    base_url: Optional[str] = None
    port: Optional[int] = None
    enable_thinking: bool = False
    extra_args: list[str] = []      # extra llama-server flags, e.g. ["--rerank"]
    # Optional LoRA adapters (GGUF), by pipeline step: {"answer": path, "decompose": path}.
    # All are loaded once; each request enables only the adapter for its step.
    adapters: dict[str, str] = {}


class IngestConfig(_Strict):
    pdf_dir: str = "documents"
    parser: Literal["pymupdf", "docling"] = "pymupdf"
    min_block_chars: int = 20
    chunk_size: int = 1200          # characters
    chunk_overlap: int = 200        # characters
    # parent-child ("small-to-big"): index small child chunks, return their parent chunk. 0 = off
    child_size: int = 0
    child_overlap: int = 50
    contextualize: bool = False     # prepend LLM-written situating context to each chunk


class RetrievalConfig(_Strict):
    mode: Literal["dense", "sparse", "hybrid"] = "hybrid"
    sparse: Literal["bm25", "splade"] = "bm25"
    splade_model: str = "prithivida/Splade_PP_en_v1"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    query_prompt: Optional[str] = None   # e.g. "query" for Qwen3-Embedding
    # cross_encoder: sentence-transformers model id; llama: GGUF reranker served by llama-server
    reranker_backend: Literal["cross_encoder", "llama"] = "cross_encoder"
    reranker_model: Optional[str] = "BAAI/bge-reranker-v2-m3"
    llama_reranker: LLMConfig = Field(default_factory=lambda: LLMConfig(
        repo_id="ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF", filename="qwen3-reranker-0.6b-q8_0.gguf",
        # in rerank mode each (query, document) must fit one physical batch; the default 512 is too small
        n_ctx=8192, extra_args=["--rerank", "-b", "4096", "-ub", "4096"]))
    fetch_k: int = 20                    # candidates per query per retriever
    rerank_k: int = 20                   # candidates sent to the cross-encoder
    top_k: int = 5                       # passages given to the generator
    rrf_k: int = 60
    bm25_k1: float = 1.5
    bm25_b: float = 0.75


class AgentConfig(_Strict):
    # Tells the router what the knowledge base covers; source file names are appended automatically.
    corpus_description: str = ""
    route: bool = True            # orchestrator decides SEARCH vs ANSWER_DIRECTLY
    decompose: bool = True        # multi-query decomposition
    max_subqueries: int = 3
    answer_mode: Literal["grounded", "grounded_evidence", "freetext"] = "grounded"
    # Corrective RAG: if the best reranker score is below crag_threshold, rewrite the query and
    # retrieve again (up to crag_retries times) before generating.
    crag: bool = False
    crag_threshold: float = 0.5
    crag_retries: int = 1
    # refuse without generating when the best reranker score (after CRAG) is below this
    min_relevance: float = 0.0
    # claim verification against cited passages: none | reranker (relevance proxy) | nli (entailment)
    verifier: Literal["none", "reranker", "nli"] = "reranker"
    nli_model: str = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
    support_threshold: float = 0.05      # reranker score, or entailment probability for nli
    # Self-correction: if fewer than this share of claims is supported, regenerate once with feedback
    self_correct: bool = False
    self_correct_below: float = 0.5
    max_new_tokens: int = 1200
    temperature: float = 0.1


class EvalConfig(_Strict):
    dataset: str = "eval/dataset.jsonl"
    # teacher writes the questions; judge grades answers. Different model families on purpose,
    # so the judge isn't grading text in its own family's style.
    teacher: LLMConfig = Field(default_factory=lambda: LLMConfig(
        repo_id="unsloth/Qwen3.5-9B-GGUF", filename="Qwen3.5-9B-Q4_K_M.gguf"))
    judge: LLMConfig = Field(default_factory=lambda: LLMConfig(
        repo_id="ggml-org/gemma-4-12B-it-GGUF", filename="gemma-4-12B-it-Q4_0.gguf", n_ctx=16384))
    bootstrap: int = 1000               # resamples for confidence intervals
    ks: list[int] = [1, 3, 5, 10]


class RLConfig(_Strict):
    """Optional preference/RL fine-tuning of the generator (see localrag/rl)."""
    base_model: str = "Qwen/Qwen3.5-0.8B"      # HF weights matching the GGUF used for inference
    output_dir: str = "adapters"
    llama_cpp_tag: str = "b10751"               # llama.cpp source version for LoRA -> GGUF conversion
    lora_r: int = 16
    lora_alpha: int = 32
    target_modules: list[str] = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
                                 "down_proj", "in_proj_qkv", "in_proj_z", "out_proj"]
    learning_rate: float = 5e-5                 # LoRA needs a higher LR than full fine-tuning
    epochs: float = 2.0                         # GRPO epochs over the train prompts (62 steps each)
    dpo_epochs: float = 3.0                     # ~194 pairs / 16 per step is only ~12 steps per epoch
    max_steps: int = -1
    # memory: the 248k-token vocabulary makes logits large, so train on micro-batches
    batch_size: int = 1                         # DPO pairs per forward pass
    grad_accum: int = 16                        # DPO: effective batch = batch_size * grad_accum
    max_length: int = 3072                      # DPO prompt + completion tokens
    max_completion_length: int = 512
    # DPO: sample this many answers per prompt from the base model, pair best vs worst by reward
    dpo_samples: int = 6
    dpo_min_margin: float = 0.5
    dpo_beta: float = 0.1
    # GRPO
    num_generations: int = 6
    grpo_prompts_per_step: int = 4              # one optimizer step = this many prompts x num_generations
    grpo_micro_batch: int = 2                   # sequences per forward pass
    grpo_beta: float = 0.0                      # KL coefficient (0 = no reference model, DAPO-style)
    temperature: float = 1.0


class Config(_Strict):
    index_dir: str = "index"
    llm: LLMConfig = Field(default_factory=LLMConfig)
    ingest: IngestConfig = Field(default_factory=IngestConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    rl: RLConfig = Field(default_factory=RLConfig)

    @property
    def index_path(self) -> Path:
        return Path(self.index_dir)


def _set_dotted(d: dict, key: str, value: Any) -> None:
    parts = key.split(".")
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = value


def apply_overrides(raw: dict, overrides: dict[str, Any] | list[str] | None) -> dict:
    """Apply {"retrieval.mode": "dense"} or ["retrieval.mode=dense"] style overrides."""
    raw = copy.deepcopy(raw)
    if not overrides:
        return raw
    if isinstance(overrides, list):
        overrides = {k: yaml.safe_load(v) for k, v in (o.split("=", 1) for o in overrides)}
    for k, v in overrides.items():
        _set_dotted(raw, k, v)
    return raw


def load_config(path: str | Path | None = "configs/default.yaml",
                overrides: dict[str, Any] | list[str] | None = None) -> Config:
    raw = {}
    if path and Path(path).exists():
        raw = yaml.safe_load(Path(path).read_text()) or {}
    return Config.model_validate(apply_overrides(raw, overrides))
