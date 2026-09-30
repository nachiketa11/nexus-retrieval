from typing import Dict, List, Tuple, Optional
import time
from sentence_transformers import CrossEncoder

_CROSS_ENCODER_CACHE: Dict[str, CrossEncoder] = {}


def _get_cross_encoder(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> CrossEncoder:
    """Cached loader for CrossEncoder."""
    if model_name in _CROSS_ENCODER_CACHE:
        return _CROSS_ENCODER_CACHE[model_name]
    model = CrossEncoder(model_name)
    _CROSS_ENCODER_CACHE[model_name] = model
    return model


class CodeReranker:
    """Cross-encoder reranker for top candidate passages."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model: Optional[CrossEncoder] = None

    def _load(self):
        if self._model is None:
            self._model = _get_cross_encoder(self.model_name)

    def rerank_query(
        self,
        query: str,
        candidates: List[str],
        corpus: Dict[str, Dict],
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """Rerank a list of candidate document IDs for a single query.

        Returns:
            List of (doc_id, score) pairs sorted by score descending.
        """
        if not candidates:
            return []

        # Filter candidates present in corpus
        valid_candidates = [cid for cid in candidates if cid in corpus]
        if not valid_candidates:
            return []

        self._load()
        pairs = [(query, corpus[cid].get("text", "")) for cid in valid_candidates]

        if hasattr(self._model, "predict"):
            scores = self._model.predict(pairs)
        else:
            # Fallback for dummy mock models in tests
            scores = [1.0 / (i + 1) for i in range(len(pairs))]

        doc_scores = list(zip(valid_candidates, [float(s) for s in scores]))
        # Sort deterministically: score descending, doc_id ascending
        sorted_pairs = sorted(doc_scores, key=lambda x: (-x[1], x[0]))

        return sorted_pairs[:top_k]

    def rerank_batch(
        self,
        queries: Dict[str, Dict],
        candidates_map: Dict[str, List[str]],
        corpus: Dict[str, Dict],
        top_k: int = 10,
    ) -> Dict[str, List[str]]:
        """Rerank candidates for a dictionary of queries."""
        reranked_results: Dict[str, List[str]] = {}

        for qid, q_data in queries.items():
            query_text = q_data.get("text", "")
            cands = candidates_map.get(qid, [])
            reranked_pairs = self.rerank_query(query_text, cands, corpus, top_k=top_k)
            reranked_results[qid] = [doc_id for doc_id, _ in reranked_pairs]

        return reranked_results
