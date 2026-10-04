"""Torch-free model backends built on ONNX Runtime.

These run the same model families as the PyTorch path (E5-base-v2 bi-encoder
and the MS MARCO MiniLM cross-encoder) from quantized ONNX exports, so the
pipeline fits in a serverless bundle. Expected layout per model directory::

    <model_dir>/onnx/model_quantized.onnx          (or model_quantized.onnx.part0, .part1, ...)
    <model_dir>/tokenizer.json

Split parts exist because serverless uploads cap single files at 100 MB; they are
concatenated in memory. Download with ``python scripts/build_vercel.py``.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_MODEL_ROOT = Path(os.getenv("NEXUS_ONNX_DIR", Path(__file__).resolve().parents[2] / "models"))
E5_DIR_NAME = "e5-base-v2"
CROSS_ENCODER_DIR_NAME = "ms-marco-MiniLM-L-6-v2"


def _model_parts(model_dir: Path) -> List[Path]:
    single = model_dir / "onnx" / "model_quantized.onnx"
    if single.exists():
        return [single]
    return sorted((model_dir / "onnx").glob("model_quantized.onnx.part*"),
                  key=lambda p: int(p.suffix.replace(".part", "")))


def onnx_models_available(root: Optional[Path] = None) -> bool:
    root = Path(root or DEFAULT_MODEL_ROOT)
    return all(
        _model_parts(root / name) and (root / name / "tokenizer.json").exists()
        for name in (E5_DIR_NAME, CROSS_ENCODER_DIR_NAME)
    )


class _OnnxBert:
    def __init__(self, model_dir: Path, max_length: int):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        options = ort.SessionOptions()
        options.intra_op_num_threads = int(os.getenv("NEXUS_ONNX_THREADS", "2"))
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        parts = _model_parts(model_dir)
        if not parts:
            raise FileNotFoundError(f"No ONNX model found under {model_dir / 'onnx'}")
        model = str(parts[0]) if len(parts) == 1 else b"".join(part.read_bytes() for part in parts)
        self.session = ort.InferenceSession(
            model,
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.input_names = {i.name for i in self.session.get_inputs()}
        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=max_length)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")

    def _feed(self, encodings) -> Dict[str, np.ndarray]:
        feed = {
            "input_ids": np.array([e.ids for e in encodings], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in encodings], dtype=np.int64),
        }
        if "token_type_ids" in self.input_names:
            feed["token_type_ids"] = np.array([e.type_ids for e in encodings], dtype=np.int64)
        return feed


class OnnxE5Encoder(_OnnxBert):
    """E5 bi-encoder: mean pooling + L2 normalisation, with E5 prefixes."""

    def __init__(self, model_dir: Optional[Path] = None, max_length: int = 512):
        super().__init__(Path(model_dir or DEFAULT_MODEL_ROOT / E5_DIR_NAME), max_length)

    def encode(self, texts: Sequence[str], prefix: str, batch_size: int = 16) -> np.ndarray:
        chunks: List[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = [f"{prefix} {t}" for t in texts[start:start + batch_size]]
            feed = self._feed(self.tokenizer.encode_batch(batch))
            hidden = self.session.run(None, feed)[0]
            mask = feed["attention_mask"][..., None].astype(np.float32)
            pooled = (hidden * mask).sum(axis=1) / np.clip(mask.sum(axis=1), 1e-9, None)
            pooled /= np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12, None)
            chunks.append(pooled.astype(np.float32))
        return np.vstack(chunks) if chunks else np.zeros((0, 768), dtype=np.float32)


class OnnxCrossEncoder(_OnnxBert):
    """MS MARCO MiniLM cross-encoder returning one relevance logit per pair."""

    def __init__(self, model_dir: Optional[Path] = None, max_length: int = 512):
        super().__init__(Path(model_dir or DEFAULT_MODEL_ROOT / CROSS_ENCODER_DIR_NAME), max_length)

    def predict(self, pairs: Sequence[Tuple[str, str]], batch_size: int = 16) -> np.ndarray:
        scores: List[np.ndarray] = []
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start:start + batch_size]
            feed = self._feed(self.tokenizer.encode_batch([(q, d) for q, d in batch]))
            logits = self.session.run(None, feed)[0]
            scores.append(logits.reshape(len(batch), -1)[:, 0])
        return np.concatenate(scores) if scores else np.zeros((0,), dtype=np.float32)


class OnnxDenseRetriever:
    """Exact inner-product search over E5 embeddings (NumPy; no FAISS needed for small corpora)."""

    def __init__(self, encoder: OnnxE5Encoder):
        self.encoder = encoder
        self.corpus_ids: List[str] = []
        self.matrix: Optional[np.ndarray] = None

    def fit(self, corpus: Dict[str, Dict], embeddings: Optional[Dict[str, np.ndarray]] = None
            ) -> "OnnxDenseRetriever":
        """Index ``corpus``; ``embeddings`` (doc_id -> vector) skips encoding for precomputed docs."""
        self.corpus_ids = sorted(corpus.keys())
        embeddings = embeddings or {}
        missing = [cid for cid in self.corpus_ids if cid not in embeddings]
        if missing:
            encoded = self.encoder.encode([corpus[cid].get("text", "") for cid in missing], prefix="passage:")
            embeddings = {**embeddings, **dict(zip(missing, encoded))}
        self.matrix = np.vstack([embeddings[cid] for cid in self.corpus_ids]).astype(np.float32)
        return self

    def retrieve(self, query: str, top_k: int = 100) -> List[Tuple[str, float]]:
        if self.matrix is None:
            raise RuntimeError("OnnxDenseRetriever must be fit before calling retrieve.")
        q = self.encoder.encode([query], prefix="query:")[0]
        sims = self.matrix @ q
        order = sorted(range(len(sims)), key=lambda i: (-float(sims[i]), self.corpus_ids[i]))
        return [(self.corpus_ids[i], float(sims[i])) for i in order[:top_k]]
