import math
from typing import List, Set, Dict, Tuple


def ndcg_at_k(ranked: List[str], relevant: Set[str], k: int = 10) -> float:
    """Normalized Discounted Cumulative Gain @k.

    * ``ranked`` – list of document ids ordered by predicted relevance.
    * ``relevant`` – set of ground‑truth relevant document ids for the query.
    * ``k`` – cut‑off (default 10).

    The gain for a relevant document is 1, otherwise 0.
    The ideal DCG is the sum of the top‑k gains when the list is perfectly sorted.
    """
    if k <= 0:
        return 0.0
    dcg = 0.0
    for i, doc_id in enumerate(ranked[:k]):
        gain = 1.0 if doc_id in relevant else 0.0
        if gain > 0:
            dcg += gain / math.log2(i + 2)  # i starts at 0 → rank = i+1, denominator log2(rank+1)
    # Ideal DCG: all relevant docs at top positions (up to k)
    ideal_rels = min(len(relevant), k)
    if ideal_rels == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_rels))
    return dcg / idcg


def mrr(ranked_lists: List[List[str]], relevance_sets: List[Set[str]]) -> float:
    """Mean Reciprocal Rank.

    ``ranked_lists`` – list of rankings (one per query).
    ``relevance_sets`` – list of sets of relevant doc ids (same order).
    """
    if not ranked_lists:
        return 0.0
    reciprocal_ranks = []
    for ranked, rel in zip(ranked_lists, relevance_sets):
        rr = 0.0
        for i, doc_id in enumerate(ranked, start=1):
            if doc_id in rel:
                rr = 1.0 / i
                break
        reciprocal_ranks.append(rr)
    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def recall_at_k(ranked: List[str], relevant: Set[str], k: int) -> float:
    """Recall@k.

    ``ranked`` – list of document ids ordered by prediction.
    ``relevant`` – set of ground‑truth relevant doc ids.
    ``k`` – cut‑off.
    """
    if k <= 0:
        return 0.0
    retrieved = set(ranked[:k])
    if not relevant:
        return 0.0
    return len(retrieved & relevant) / len(relevant)


def evaluate(ranked_results: Dict[str, List[str]], qrels: Dict[str, List[str]]) -> Dict[str, float]:
    """Compute the four required metrics over a ranking dict.

    ``ranked_results`` – mapping query_id → list of doc_ids ordered by score.
    ``qrels`` – mapping query_id → list of relevant doc_ids.
    Returns a dict with keys ``ndcg@10``, ``mrr``, ``recall@10``, ``recall@100``.
    """
    ndcg_vals = []
    mrr_vals = []  # will be aggregated later via mrr()
    recall10_vals = []
    recall100_vals = []
    ranked_lists = []
    relevance_sets = []

    for qid, ranking in ranked_results.items():
        rel = set(qrels.get(qid, []))
        if not rel:
            # skip queries with no relevance judgments – they do not affect the averages
            continue
        ndcg_vals.append(ndcg_at_k(ranking, rel, k=10))
        recall10_vals.append(recall_at_k(ranking, rel, k=10))
        recall100_vals.append(recall_at_k(ranking, rel, k=100))
        ranked_lists.append(ranking)
        relevance_sets.append(rel)

    overall = {
        "ndcg@10": sum(ndcg_vals) / len(ndcg_vals) if ndcg_vals else 0.0,
        "mrr": mrr(ranked_lists, relevance_sets),
        "recall@10": sum(recall10_vals) / len(recall10_vals) if recall10_vals else 0.0,
        "recall@100": sum(recall100_vals) / len(recall100_vals) if recall100_vals else 0.0,
    }
    return overall
