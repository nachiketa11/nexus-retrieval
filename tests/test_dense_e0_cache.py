import os
import builtins
import re
from typing import Dict

import numpy as np
import pytest

from src.retrieval import dense_e0


class DummySentenceTransformer:
    """Lightweight stand-in to exercise disk caching without model inference."""

    def __init__(self, model_name):
        self.model_name = model_name

    def encode(self, texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True):
        return np.ones((len(texts), 4), dtype=np.float32)

# Helper to compute cache file paths for the test split
def _cache_file(name: str) -> str:
    safe_model = "intfloat_e5-base-v2"
    cache_dir = os.path.abspath(os.path.join(os.path.dirname(dense_e0.__file__), "..", "..", "cache"))
    return os.path.join(cache_dir, f"{name}_{safe_model}_test.npy")

@pytest.fixture(autouse=True)
def clean_cache():
    """Remove any existing cache files before each test.
    This ensures a deterministic environment for cache‑creation tests.
    """
    for fname in ["corpus", "queries"]:
        path = _cache_file(fname)
        if os.path.exists(path):
            os.remove(path)
    yield
    # Cleanup after test as well
    for fname in ["corpus", "queries"]:
        path = _cache_file(fname)
        if os.path.exists(path):
            os.remove(path)

def test_cache_creation_and_load(capsys, monkeypatch):
    monkeypatch.setattr(dense_e0, "SentenceTransformer", DummySentenceTransformer)
    monkeypatch.setattr(dense_e0, "_load_model", lambda model_name="intfloat/e5-base-v2": DummySentenceTransformer(model_name))
    # Minimal corpus and queries – enough to trigger encoding
    corpus = {"doc1": {"text": "def foo(): pass"}}
    queries = {"q1": {"text": "how to write a function?"}}

    # First run – should encode and create cache files
    index, id_map = dense_e0.build_corpus_index(corpus, split="test")
    corpus_cache = _cache_file("corpus")
    assert os.path.exists(corpus_cache), "Corpus cache file was not created"
    # Capture output from second run – should load from cache
    index2, id_map2 = dense_e0.build_corpus_index(corpus, split="test")
    captured = capsys.readouterr().out
    assert re.search(r"Loaded .*corpus_intfloat_e5-base-v2_test\.npy from cache", captured), "Cache load message not found"
    # Ensure deterministic mapping remains the same
    assert id_map == id_map2

    # Retrieve top‑k – first call creates query cache
    results = dense_e0.retrieve_top_k(queries, index, {0: "doc1"}, split="test")
    query_cache = _cache_file("queries")
    assert os.path.exists(query_cache), "Query cache file was not created"
    # Second retrieval should load query embeddings from cache
    results2 = dense_e0.retrieve_top_k(queries, index, {0: "doc1"}, split="test")
    captured2 = capsys.readouterr().out
    assert re.search(r"Loaded .*queries_intfloat_e5-base-v2_test\.npy from cache", captured2), "Query cache load message not found"
    # Retrieval results must be identical
    assert results == results2
