import pytest
from src.data.coir import load_coir

def test_load_coir_test_split():
    queries, corpus, qrels = load_coir(split="test")
    # Basic sanity checks
    assert isinstance(queries, dict) and len(queries) > 0
    assert isinstance(corpus, dict) and len(corpus) > 0
    assert isinstance(qrels, dict) and len(qrels) > 0
    # Verify that each query id exists in queries dict
    for qid in qrels:
        assert qid in queries
        # each relevance list should refer to existing corpus ids
        for doc_id in qrels[qid]:
            assert doc_id in corpus
