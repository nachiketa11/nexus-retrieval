from typing import Dict, List, Tuple, Set


def reciprocal_rank_fusion(
    rankings_list: List[List[str]],
    k: int = 60,
    top_k: int = 100,
) -> List[Tuple[str, float]]:
    """Compute Reciprocal Rank Fusion (RRF) scores across multiple ranked document ID lists.

    Formula:
        RRF_score(d) = sum_{r in rankings} 1 / (k + rank(d, r))
    where rank is 1-indexed.

    Tie-breaking is deterministic: sort by (RRF_score DESC, doc_id ASC).
    """
    scores: Dict[str, float] = {}

    for ranking in rankings_list:
        for rank_idx, doc_id in enumerate(ranking, start=1):
            rrf_val = 1.0 / (k + rank_idx)
            scores[doc_id] = scores.get(doc_id, 0.0) + rrf_val

    # Sort deterministically: score descending, then doc_id ascending
    sorted_docs = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return sorted_docs[:top_k]


def hybrid_retrieve(
    dense_results: Dict[str, List[str]],
    bm25_results: Dict[str, List[str]],
    k: int = 60,
    top_k: int = 100,
) -> Dict[str, List[str]]:
    """Combine Dense and BM25 rankings for a set of query IDs via RRF.

    Returns:
        Dict[query_id, List[doc_id]]
    """
    hybrid_rankings: Dict[str, List[str]] = {}
    all_qids = set(dense_results.keys()) | set(bm25_results.keys())

    for qid in all_qids:
        dense_list = dense_results.get(qid, [])
        bm25_list = bm25_results.get(qid, [])

        fused = reciprocal_rank_fusion([dense_list, bm25_list], k=k, top_k=top_k)
        hybrid_rankings[qid] = [doc_id for doc_id, _ in fused]

    return hybrid_rankings
