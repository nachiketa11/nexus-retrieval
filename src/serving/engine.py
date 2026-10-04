"""Lightweight NEXUS engine for serverless deployment (no torch / FAISS).

Runs BM25, E5 dense retrieval and the MS MARCO cross-encoder through ONNX Runtime
over the Samsung/demo corpus, and exposes both the classic pipeline and the agent.
If the ONNX models are missing, the engine degrades to BM25 + metadata policy and
reports that in ``info()``.
"""

import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..agent import NexusAgent
from ..data.samsung_demo import load_samsung_demo_data
from ..retrieval.bm25 import BM25Retriever
from ..retrieval.hybrid import reciprocal_rank_fusion
from ..retrieval.onnx_models import DEFAULT_MODEL_ROOT, onnx_models_available
from ..retrieval.version_aware import VersionAwareFilter, parse_version_intent

EMBEDDINGS_FILE = "demo_corpus_e5.npz"
METHODS = ("bm25", "dense", "hybrid")


class NexusEngine:
    def __init__(self, model_root: Optional[Path] = None, enable_models: bool = True):
        self.model_root = Path(model_root or DEFAULT_MODEL_ROOT)
        self.queries, self.corpus = load_samsung_demo_data()
        self.bm25 = BM25Retriever().fit(self.corpus)
        self.version_filter = VersionAwareFilter()
        self.models_enabled = enable_models and onnx_models_available(self.model_root)
        self._dense = None
        self._cross = None
        self._lock = threading.Lock()
        self.model_load_ms: Optional[float] = None

    # ------------------------------------------------------------------ models
    def _ensure_models(self) -> None:
        if not self.models_enabled or self._dense is not None:
            return
        with self._lock:
            if self._dense is not None:
                return
            from ..retrieval.onnx_models import OnnxCrossEncoder, OnnxDenseRetriever, OnnxE5Encoder

            started = time.perf_counter()
            encoder = OnnxE5Encoder(self.model_root / "e5-base-v2")
            precomputed = self._load_embeddings()
            dense = OnnxDenseRetriever(encoder).fit(self.corpus, embeddings=precomputed)
            self._cross = OnnxCrossEncoder(self.model_root / "ms-marco-MiniLM-L-6-v2")
            self._dense = dense
            self.model_load_ms = round((time.perf_counter() - started) * 1000, 1)

    def _load_embeddings(self) -> Dict[str, np.ndarray]:
        path = self.model_root / EMBEDDINGS_FILE
        if not path.exists():
            return {}
        data = np.load(path, allow_pickle=False)
        ids, vectors = data["ids"], data["vectors"]
        stored = {str(i): v for i, v in zip(ids, vectors)}
        # Only reuse vectors whose passage text is unchanged.
        texts = {str(i): str(t) for i, t in zip(ids, data["texts"])}
        return {cid: v for cid, v in stored.items() if self.corpus.get(cid, {}).get("text") == texts.get(cid)}

    def dense_search(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        self._ensure_models()
        return self._dense.retrieve(query, top_k=top_k)

    def rerank(self, query: str, ids) -> List[Tuple[str, float]]:
        self._ensure_models()
        ids = list(ids)
        scores = self._cross.predict([(query, self.corpus[c].get("text", "")) for c in ids])
        return sorted(zip(ids, map(float, scores)), key=lambda x: (-x[1], x[0]))

    # ------------------------------------------------------------------ API
    def info(self) -> Dict[str, Any]:
        return {
            "system": "NEXUS — Agentic Code Intelligence",
            "dataset": "samsung_demo",
            "corpus_size": len(self.corpus),
            "dense_model": "intfloat/e5-base-v2 (ONNX int8)" if self.models_enabled else None,
            "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2 (ONNX int8)" if self.models_enabled else None,
            "models_enabled": self.models_enabled,
            "models_loaded": self._dense is not None,
            "model_load_ms": self.model_load_ms,
            "methods": list(METHODS) if self.models_enabled else ["bm25"],
        }

    def examples(self) -> List[Dict[str, str]]:
        return [{"id": qid, "text": q["text"], "expected_doc": q["expected_doc"]} for qid, q in self.queries.items()]

    def search(self, query: str, method: str = "hybrid", rerank: bool = False, top_k: int = 5,
               version: Optional[str] = None) -> Dict[str, Any]:
        method = method.lower()
        if method not in METHODS:
            raise ValueError(f"method must be one of {', '.join(METHODS)}")
        if method != "bm25" or rerank:
            if not self.models_enabled:
                raise RuntimeError("Dense retrieval and reranking need the ONNX models; use method=bm25.")
            self._ensure_models()

        started = time.perf_counter()
        scores: Dict[str, Dict[str, float]] = {}
        lexical = self.bm25.retrieve(query, top_k=50)
        for cid, s in lexical:
            scores.setdefault(cid, {})["bm25"] = round(s, 4)
        ranking = [c for c, _ in lexical]
        primary: Dict[str, float] = dict(lexical)

        if method in ("dense", "hybrid"):
            semantic = self.dense_search(query, 50)
            for cid, s in semantic:
                scores.setdefault(cid, {})["dense"] = round(s, 4)
            if method == "dense":
                ranking, primary = [c for c, _ in semantic], dict(semantic)
            else:
                fused = reciprocal_rank_fusion([[c for c, _ in semantic], ranking], top_k=50)
                for cid, s in fused:
                    scores[cid]["rrf"] = round(s, 5)
                ranking, primary = [c for c, _ in fused], dict(fused)

        intent = parse_version_intent(query)
        if version:
            ranking = self.version_filter.filter_candidates(ranking, self.corpus, version=version)
        elif intent:
            ranking = [c for c, _ in self.version_filter.rerank_by_metadata_match(ranking, self.corpus, query)]

        if rerank and ranking:
            head = self.rerank(query, ranking[: max(top_k * 3, 15)])
            for cid, s in head:
                scores[cid]["rerank"] = round(s, 4)
            ranking = [c for c, _ in head] + ranking[len(head):]
            primary = dict(head)

        results = []
        for rank, cid in enumerate(ranking[:top_k], 1):
            doc = self.corpus[cid]
            results.append({
                "doc_id": cid,
                "rank": rank,
                "score": primary.get(cid),
                "text": doc.get("text", ""),
                "metadata": doc.get("metadata") or {},
                "signals": scores.get(cid, {}),
            })
        return {
            "query": query,
            "method": method + ("+rerank" if rerank else ""),
            "version": version,
            "intent": intent,
            "total": len(results),
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "results": results,
        }

    def agent(self, query: str, top_k: int = 5, version: Optional[str] = None,
              use_models: bool = True, rerank: Optional[bool] = None) -> Dict[str, Any]:
        use_models = use_models and self.models_enabled
        if use_models:
            self._ensure_models()
        agent = NexusAgent(
            self.corpus,
            bm25=self.bm25,
            dense=self.dense_search if use_models else None,
            reranker=self.rerank if use_models else None,
        )
        return agent.run(query, top_k=top_k, version=version, use_rerank=rerank)
