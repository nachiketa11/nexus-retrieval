"""Assemble a self-contained Vercel deployment of the NEXUS web app.

    python scripts/build_vercel.py            # -> .vercel-build/
    cd .vercel-build && vercel deploy --prod

Steps:
1. download the quantized ONNX models (E5-base-v2, MS MARCO MiniLM cross-encoder) into models/;
2. precompute E5 embeddings of the Samsung/demo corpus (models/demo_corpus_e5.npz);
3. copy web/app.py, web/public/, the torch-free parts of src/ and the models into the
   build directory together with a lightweight requirements.txt and vercel.json.

The full repository requirements (torch, sentence-transformers, FAISS) exceed the
serverless bundle limit, which is why the deployment uses ONNX Runtime instead.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

MODELS = {
    "e5-base-v2": "Xenova/e5-base-v2",
    "ms-marco-MiniLM-L-6-v2": "Xenova/ms-marco-MiniLM-L-6-v2",
}
MODEL_FILES = ("onnx/model_quantized.onnx", "tokenizer.json", "config.json")
MAX_FILE_BYTES = 90 * 1024 * 1024
SRC_FILES = [
    "src/__init__.py",
    "src/agent/__init__.py",
    "src/agent/nexus_agent.py",
    "src/data/__init__.py",
    "src/data/samsung_demo.py",
    "src/retrieval/__init__.py",
    "src/retrieval/bm25.py",
    "src/retrieval/hybrid.py",
    "src/retrieval/version_aware.py",
    "src/retrieval/onnx_models.py",
    "src/serving/__init__.py",
    "src/serving/engine.py",
]
REQUIREMENTS = """fastapi>=0.115
numpy>=1.26
rank-bm25>=0.2.2
onnxruntime>=1.20
tokenizers>=0.20
"""
VERCEL_JSON = {
    "$schema": "https://openapi.vercel.sh/vercel.json",
    "functions": {"app.py": {"maxDuration": 60}},
}
PYPROJECT = """[project]
name = "nexus-web"
version = "2.0.0"
requires-python = ">=3.12,<3.13"
dependencies = [
    "fastapi>=0.115",
    "numpy>=1.26",
    "rank-bm25>=0.2.2",
    "onnxruntime>=1.20",
    "tokenizers>=0.20",
]
"""


def fetch_models(model_root: Path) -> None:
    from huggingface_hub import hf_hub_download

    for name, repo in MODELS.items():
        for filename in MODEL_FILES:
            target = model_root / name / filename
            if target.exists():
                continue
            print(f"downloading {repo}/{filename}")
            hf_hub_download(repo, filename, local_dir=str(model_root / name))


def precompute_embeddings(model_root: Path) -> None:
    import numpy as np
    from src.data.samsung_demo import load_samsung_demo_data
    from src.retrieval.onnx_models import OnnxE5Encoder
    from src.serving.engine import EMBEDDINGS_FILE

    _, corpus = load_samsung_demo_data()
    ids = sorted(corpus)
    texts = [corpus[i]["text"] for i in ids]
    vectors = OnnxE5Encoder(model_root / "e5-base-v2").encode(texts, prefix="passage:")
    np.savez_compressed(model_root / EMBEDDINGS_FILE, ids=np.array(ids), texts=np.array(texts), vectors=vectors)
    print(f"precomputed {len(ids)} passage embeddings -> {EMBEDDINGS_FILE}")


def assemble(out: Path, model_root: Path) -> None:
    # Rebuild everything except `.vercel/`, which links the directory to its Vercel project.
    out.mkdir(parents=True, exist_ok=True)
    for child in out.iterdir():
        if child.name != ".vercel":
            shutil.rmtree(child) if child.is_dir() else child.unlink()
    shutil.copy2(BASE_DIR / "web" / "app.py", out / "app.py")
    shutil.copytree(BASE_DIR / "web" / "public", out / "public")
    for rel in SRC_FILES:
        dest = out / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(BASE_DIR / rel, dest)
    for name in MODELS:
        dest_dir = out / "models" / name
        (dest_dir / "onnx").mkdir(parents=True, exist_ok=True)
        shutil.copy2(model_root / name / "tokenizer.json", dest_dir / "tokenizer.json")
        model_bytes = (model_root / name / "onnx" / "model_quantized.onnx").read_bytes()
        if len(model_bytes) <= MAX_FILE_BYTES:
            (dest_dir / "onnx" / "model_quantized.onnx").write_bytes(model_bytes)
        else:  # Vercel rejects single files over 100 MB; the loader concatenates the parts.
            for index, start in enumerate(range(0, len(model_bytes), MAX_FILE_BYTES)):
                (dest_dir / "onnx" / f"model_quantized.onnx.part{index}").write_bytes(
                    model_bytes[start:start + MAX_FILE_BYTES])
    from src.serving.engine import EMBEDDINGS_FILE
    shutil.copy2(model_root / EMBEDDINGS_FILE, out / "models" / EMBEDDINGS_FILE)
    (out / "requirements.txt").write_text(REQUIREMENTS, encoding="utf-8")
    (out / "pyproject.toml").write_text(PYPROJECT, encoding="utf-8")
    (out / ".python-version").write_text("3.12\n", encoding="utf-8")
    (out / "vercel.json").write_text(json.dumps(VERCEL_JSON, indent=2) + "\n", encoding="utf-8")
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"assembled {out} ({size / 1e6:.1f} MB before dependencies)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(BASE_DIR / ".vercel-build"))
    parser.add_argument("--models", default=str(BASE_DIR / "models"))
    args = parser.parse_args()
    model_root = Path(args.models)
    fetch_models(model_root)
    precompute_embeddings(model_root)
    assemble(Path(args.out), model_root)


if __name__ == "__main__":
    main()
