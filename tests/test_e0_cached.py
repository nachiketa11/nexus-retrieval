import os
import pytest
import numpy as np

from src.retrieval.e0_cached import evaluate_cached_e0

def test_evaluate_cached_runs_without_model(monkeypatch):
    # Ensure that SentenceTransformer is not imported/used
    def fake_import(name, *args, **kwargs):
        if name == "sentence_transformers":
            raise AssertionError("SentenceTransformer should not be imported in cached evaluation")
        return __import__(name, *args, **kwargs)
    monkeypatch.setitem(__builtins__, "__import__", fake_import)

    # Verify cache files exist
    cache_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src", "retrieval", "..", "cache"))
    corpus_file = os.path.join(cache_dir, "corpus_intfloat_e5-base-v2_test.npy")
    query_file = os.path.join(cache_dir, "queries_intfloat_e5-base-v2_test.npy")
    assert os.path.exists(corpus_file), f"Corpus cache missing: {corpus_file}"
    assert os.path.exists(query_file), f"Query cache missing: {query_file}"

    # Run cached evaluation (limit top_k for speed)
    retrieved, metrics = evaluate_cached_e0(top_k=10)

    # Basic sanity checks
    assert isinstance(retrieved, dict)
    assert isinstance(metrics, dict)
    expected_keys = {"ndcg@10", "mrr", "recall@10", "recall@100"}
    assert expected_keys.issubset(metrics.keys())
    # Ensure some queries retrieved
    assert len(retrieved) > 0
    # Ensure metric values are floats
    for v in metrics.values():
        assert isinstance(v, float)
