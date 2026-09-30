import os
import builtins
import pytest
import numpy as np

from src.retrieval.e0_cached import evaluate_cached_e0


def test_evaluate_cached_runs_without_model(monkeypatch):
    """Verify that evaluate_cached_e0 runs using cached files without importing sentence_transformers."""
    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "sentence_transformers":
            raise AssertionError("SentenceTransformer should not be imported in cached evaluation")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    cache_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "cache")
    )
    corpus_file = os.path.join(cache_dir, "corpus_intfloat_e5-base-v2_test.npy")
    query_file = os.path.join(cache_dir, "queries_intfloat_e5-base-v2_test.npy")

    if not os.path.exists(corpus_file) or not os.path.exists(query_file):
        pytest.skip("Cache files not generated yet on disk")

    retrieved, metrics = evaluate_cached_e0(top_k=10)

    assert isinstance(retrieved, dict)
    assert isinstance(metrics, dict)
    expected_keys = {"ndcg@10", "mrr", "recall@10", "recall@100"}
    assert expected_keys.issubset(metrics.keys())
    assert len(retrieved) > 0
    for v in metrics.values():
        assert isinstance(v, float)
