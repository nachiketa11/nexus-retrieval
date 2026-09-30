from src.retrieval.bm25 import BM25Retriever, tokenize_code


def test_tokenize_code():
    tokens = tokenize_code("def getUserInfo_v2(): pass")
    assert "getuserinfo_v2" in tokens
    assert "getuserinfo" in tokens or "user" in tokens
    assert "pass" in tokens


def test_bm25_retriever_ranking():
    corpus = {
        "doc1": {"text": "def authenticate_user_with_token(token): pass"},
        "doc2": {"text": "def parse_json_response(data): pass"},
        "doc3": {"text": "def connect_database_socket(port): pass"},
    }
    retriever = BM25Retriever()
    retriever.fit(corpus)

    results = retriever.retrieve("parse JSON response", top_k=3)
    assert len(results) == 3
    top_doc_id, top_score = results[0]
    assert top_doc_id == "doc2"
    assert top_score > 0.0
