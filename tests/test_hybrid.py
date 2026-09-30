from src.retrieval.hybrid import reciprocal_rank_fusion, hybrid_retrieve
from src.evaluation.metrics import evaluate


def test_reciprocal_rank_fusion_scoring():
    rankings_dense = ["docA", "docB", "docC"]
    rankings_bm25 = ["docB", "docA", "docD"]

    # docA: 1/(60+1) + 1/(60+2) = 1/61 + 1/62
    # docB: 1/(60+2) + 1/(60+1) = 1/62 + 1/61
    # docA and docB tie on score, tie breaking resolves lexicographically
    fused = reciprocal_rank_fusion([rankings_dense, rankings_bm25], k=60, top_k=4)

    fused_dict = dict(fused)
    assert "docA" in fused_dict
    assert "docB" in fused_dict
    assert "docC" in fused_dict
    assert "docD" in fused_dict


def test_hybrid_retrieve_batch():
    dense_res = {"q1": ["doc1", "doc2"]}
    bm25_res = {"q1": ["doc2", "doc3"]}

    hybrid_res = hybrid_retrieve(dense_res, bm25_res, top_k=3)
    assert "q1" in hybrid_res
    assert len(hybrid_res["q1"]) == 3
    assert set(hybrid_res["q1"]) == {"doc1", "doc2", "doc3"}


def test_recall_at_100_uses_results_beyond_top_10():
    ranking = [f"doc{i}" for i in range(1, 12)]
    metrics = evaluate({"q1": ranking}, {"q1": ["doc11"]})
    assert metrics["recall@10"] == 0.0
    assert metrics["recall@100"] == 1.0
