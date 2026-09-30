import numpy as np
import builtins
from typing import List, Dict

import pytest

# Import the module under test
from src.retrieval import dense_e0


class DummyModel:
    """A minimal mock of SentenceTransformer with deterministic encode.

    encode returns an identity matrix of shape (len(texts), dim) where dim = len(texts).
    This ensures orthogonal vectors and already L2‑normalized (unit vectors).
    """

    def encode(self, texts: List[str], batch_size: int = 32, show_progress_bar: bool = False, normalize_embeddings: bool = True):
        # Return an identity matrix: each input gets a distinct basis vector.
        dim = len(texts)
        return np.eye(dim, dtype=np.float32)


# Monkeypatch the internal model loader to return the dummy model
@pytest.fixture(autouse=True)
def patch_load_model(monkeypatch):
    monkeypatch.setattr(dense_e0, "_load_model", lambda model_name="intfloat/e5-base-v2": DummyModel())
    yield


def test_build_corpus_index_deterministic_mapping():
    corpus = {
        "docB": {"text": "text B"},
        "docA": {"text": "text A"},
    }
    index, id_map = dense_e0.build_corpus_index(corpus)
    # IDs should be assigned in lexicographic order: docA -> 0, docB -> 1
    assert id_map[0] == "docA"
    assert id_map[1] == "docB"
    # Index should contain exactly 2 vectors of dimension 2
    assert index.ntotal == 2
    assert index.d == 2


def test_retrieve_top_k_returns_correct_ids():
    # Corpus with two documents
    corpus = {
        "doc1": {"text": "alpha"},
        "doc2": {"text": "beta"},
    }
    index, id_map = dense_e0.build_corpus_index(corpus)
    # Query that matches first document (identity vector at position 0)
    queries = {"q1": {"text": "alpha"}}
    results = dense_e0.retrieve_top_k(queries, index, id_map, top_k=2)
    # The top result should be doc1 (lexicographically first)
    assert results["q1"][0] == "doc1"
    # Should return two results (second is the other doc)
    assert len(results["q1"]) == 2


def test_prefixes_are_applied():
    # Capture the texts passed to DummyModel.encode by monkeypatching its method
    captured_texts = []

    class CapturingDummyModel(DummyModel):
        def encode(self, texts, **kwargs):
            captured_texts.extend(texts)
            return super().encode(texts, **kwargs)

    # Patch the loader to return the capturing model
    def load_capturing(model_name="intfloat/e5-base-v2"):
        return CapturingDummyModel()

    # Apply the patch
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(dense_e0, "_load_model", load_capturing)

    corpus = {"doc": {"text": "code snippet"}}
    index, id_map = dense_e0.build_corpus_index(corpus)
    queries = {"q": {"text": "how to do X?"}}
    dense_e0.retrieve_top_k(queries, index, id_map, top_k=1)

    # Verify that prefixes were added
    assert any(t.startswith("passage: ") for t in captured_texts)
    assert any(t.startswith("query: ") for t in captured_texts)
    monkeypatch.undo()
