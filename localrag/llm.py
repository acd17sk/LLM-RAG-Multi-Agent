"""Local LLM access through llama.cpp's OpenAI-compatible `llama-server`.

Running the model in a separate server process (instead of in-process bindings)
gives us the newest llama.cpp architectures (e.g. Qwen3.5), GPU offload, and
grammar-constrained JSON output via `response_format: json_schema`.
"""
from __future__ import annotations

import atexit
import json
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

import requests
from huggingface_hub import hf_hub_download

from localrag.config import LLMConfig


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _preferred_device(binary: str) -> list[str]:
    """Pin to the first CUDA device if present. Builds with CUDA and Vulkan list the same GPU twice
    (plus any integrated GPU), and splitting a model across them is slower than one device."""
    try:
        out = subprocess.run([binary, "--list-devices"], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    cuda = [line.split(":")[0].strip() for line in out.splitlines() if line.strip().startswith("CUDA")]
    return ["--device", cuda[0]] if cuda else []


class LLMOutputError(RuntimeError):
    """The model's structured output could not be parsed (e.g. truncated)."""


class LlamaServer:
    """Starts `llama-server` for a GGUF model and stops it on exit."""

    def __init__(self, cfg: LLMConfig, log_dir: str | Path = "logs"):
        # fall back to the active interpreter's env (conda env used without `conda activate`)
        binary = shutil.which("llama-server") or shutil.which("llama-server", path=str(Path(sys.executable).parent))
        if not binary:
            raise RuntimeError("llama-server not found on PATH (conda install -c conda-forge llama.cpp)")
        model_path = hf_hub_download(cfg.repo_id, cfg.filename)
        self.port = cfg.port or _free_port()
        self.base_url = f"http://127.0.0.1:{self.port}"
        Path(log_dir).mkdir(exist_ok=True)
        self._log = open(Path(log_dir) / f"llama-server-{Path(cfg.filename).stem}.log", "w")
        adapters = [a for a in cfg.adapters.values()]
        lora_args = ["--lora", ",".join(adapters), "--lora-init-without-apply"] if adapters else []
        self.proc = subprocess.Popen(
            [binary, "-m", model_path, "--port", str(self.port), "-c", str(cfg.n_ctx),
             "-ngl", str(cfg.n_gpu_layers if cfg.n_gpu_layers >= 0 else 999),
             "--jinja", "--parallel", "1", "--no-webui", *_preferred_device(binary), *lora_args, *cfg.extra_args],
            stdout=self._log, stderr=subprocess.STDOUT)
        atexit.register(self.stop)
        self._wait_ready()

    def _wait_ready(self, timeout: float = 300):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited early; see {self._log.name}")
            try:
                if requests.get(f"{self.base_url}/health", timeout=2).status_code == 200:
                    return
            except requests.ConnectionError:
                pass
            time.sleep(0.5)
        raise TimeoutError("llama-server did not become ready")

    def stop(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()


class LLM:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.server: Optional[LlamaServer] = None
        if cfg.base_url:
            self.base_url = cfg.base_url.rstrip("/")
        else:
            self.server = LlamaServer(cfg)
            self.base_url = self.server.base_url
        self.name = Path(cfg.filename).stem

    def _lora(self, adapter: Optional[str]) -> list[dict]:
        """Per-request adapter scales: only `adapter` is active (none -> base model)."""
        return [{"id": i, "scale": 1.0 if name == adapter else 0.0}
                for i, name in enumerate(self.cfg.adapters)]

    def chat(self, messages: list[dict], *, schema: Optional[dict] = None, max_tokens: int = 512,
             temperature: float = 0.1, adapter: Optional[str] = None, **extra: Any) -> str:
        body: dict[str, Any] = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "chat_template_kwargs": {"enable_thinking": self.cfg.enable_thinking},
            **extra,
        }
        if self.cfg.adapters:
            body["lora"] = self._lora(adapter)
        if schema is not None:
            body["response_format"] = {"type": "json_schema",
                                       "json_schema": {"name": "output", "schema": schema}}
        r = requests.post(f"{self.base_url}/v1/chat/completions", json=body, timeout=600)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"] or ""

    def json(self, messages: list[dict], schema: dict, **kw: Any) -> dict:
        """Grammar-constrained generation: the output follows `schema`, unless it is cut off by
        max_tokens. In that case retry once with double the budget before giving up."""
        text = self.chat(messages, schema=schema, **kw)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            kw["max_tokens"] = 2 * kw.get("max_tokens", 512)
            text = self.chat(messages, schema=schema, **kw)
            try:
                return json.loads(text)
            except json.JSONDecodeError as e:
                raise LLMOutputError(f"unparseable output after retry ({len(text)} chars)") from e

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        """Relevance scores from a reranker model served with `llama-server --rerank`."""
        r = requests.post(f"{self.base_url}/v1/rerank",
                          json={"query": query, "documents": documents, "top_n": len(documents)}, timeout=600)
        r.raise_for_status()
        scores = [0.0] * len(documents)
        for item in r.json()["results"]:
            scores[item["index"]] = float(item["relevance_score"])
        return scores

    def close(self):
        if self.server:
            self.server.stop()


# One server per distinct model config, shared by the pipeline, reranker, eval and judge.
_REGISTRY: dict[tuple, LLM] = {}


def _key(cfg: LLMConfig) -> tuple:
    return (cfg.base_url, cfg.repo_id, cfg.filename, cfg.n_ctx, tuple(cfg.extra_args),
            tuple(sorted(cfg.adapters.items())))


def get_llm(cfg: LLMConfig) -> LLM:
    key = _key(cfg)
    if key not in _REGISTRY:
        _REGISTRY[key] = LLM(cfg)
    return _REGISTRY[key]


def release_llms(keep: list[LLMConfig] = ()) -> None:
    """Stop every server except those in `keep`, to free GPU memory."""
    keys = {_key(c) for c in keep}
    for key in [k for k in _REGISTRY if k not in keys]:
        _REGISTRY.pop(key).close()
