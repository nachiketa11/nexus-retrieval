import os
import numpy as np
import faiss
from typing import Dict, List, Tuple

# Cached‑only evaluation for dense E0

def evaluate_cached_e0(
    top_k: int = 100,
    split: str = "test",
    model_name: str = "intfloat/e5-base-v2",
) -> Tuple[Dict[str, List[str]], Dict[str, float]]:
    """Evaluate the dense‑E0 baseline using previously cached embeddings.

    Returns
    -------
    retrieved: dict mapping query_id → list of retrieved doc_ids (up to ``top_k``)
    metrics:   dict with ``ndcg@10``, ``mrr``, ``recall@10``, ``recall@100``
    """
    # Load dataset (queries, corpus, qrels)
    from ..data.coir import load_coir
    from ..evaluation.metrics import evaluate as eval_metrics

    queries, corpus, qrels = load_coir(split=split)

    # Deterministic ordering – must match the ordering used when the embeddings were cached
    sorted_cids = sorted(corpus.keys())
    id_map = {i: cid for i, cid in enumerate(sorted_cids)}
    sorted_qids = sorted(queries.keys())

    # Locate cache files (project ``cache`` directory)
    cache_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "cache"))
    corpus_cache = os.path.join(cache_dir, f"corpus_{model_name.replace('/', '_')}_{split}.npy")
    query_cache = os.path.join(cache_dir, f"queries_{model_name.replace('/', '_')}_{split}.npy")
    if not os.path.exists(corpus_cache) or not os.path.exists(query_cache):
        raise FileNotFoundError("Cached embeddings not found – run the full E0 evaluation once to generate them.")

    corpus_emb = np.load(corpus_cache)
    query_emb = np.load(query_cache)

    # Verify that the cache sizes match the number of IDs
    if corpus_emb.shape[0] != len(sorted_cids):
        raise ValueError(
            f"Corpus cache size {corpus_emb.shape[0]} does not match number of corpus IDs {len(sorted_cids)}"
        )
    if query_emb.shape[0] != len(sorted_qids):
        raise ValueError(
            f"Query cache size {query_emb.shape[0]} does not match number of query IDs {len(sorted_qids)}"
        )

    # Build FAISS index from cached corpus embeddings
    dim = corpus_emb.shape[1]
    base = faiss.IndexFlatIP(dim)
    index = faiss.IndexIDMap(base)
    ids = np.array(list(id_map.keys()), dtype=np.int64)
    index.add_with_ids(corpus_emb.astype(np.float32), ids)

    # Search using cached query embeddings
    distances, faiss_ids = index.search(query_emb.astype(np.float32), top_k)
    retrieved: Dict[str, List[str]] = {}
    for q_idx, qid in enumerate(sorted_qids):
        retrieved[qid] = [id_map[int(idx)] for idx in faiss_ids[q_idx] if idx != -1]

    # Compute metrics
    metrics = eval_metrics(retrieved, qrels)
    return retrieved, metrics
