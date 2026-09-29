"""Convert a PEFT LoRA adapter to GGUF so llama-server can load it next to the base model."""
from __future__ import annotations

import io
import os
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

from huggingface_hub import snapshot_download

CACHE = Path.home() / ".cache" / "localrag"


def llama_cpp_sources(tag: str) -> Path:
    """llama.cpp's Python converters at the same version as the installed llama-server (cached)."""
    root = CACHE / f"llama.cpp-{tag}"
    if not (root / "convert_lora_to_gguf.py").exists():
        print(f"Fetching llama.cpp {tag} converter sources ...")
        url = f"https://codeload.github.com/ggml-org/llama.cpp/tar.gz/refs/tags/{tag}"
        data = urllib.request.urlopen(url, timeout=300).read()
        wanted = ("convert_hf_to_gguf.py", "convert_lora_to_gguf.py", "conversion/", "gguf-py/")
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
            for m in tar.getmembers():
                rel = m.name.split("/", 1)[-1]
                if rel.startswith(wanted):
                    m.name = rel
                    tar.extract(m, root, filter="data")
    return root


def lora_to_gguf(adapter_dir: str | Path, out_file: str | Path, base_model: str, tag: str) -> Path:
    src = llama_cpp_sources(tag)
    env = {**os.environ, "PYTHONPATH": f"{src / 'gguf-py'}{os.pathsep}{src}"}
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(src / "convert_lora_to_gguf.py"), str(adapter_dir),
                    "--base", snapshot_download(base_model), "--outfile", str(out_file), "--outtype", "f16"],
                   check=True, env=env)
    return out_file
