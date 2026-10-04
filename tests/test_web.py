"""Web (serverless) API tests. Neural paths run only when the ONNX models are present."""

import pytest
from fastapi.testclient import TestClient

import web.app as web_app
from src.retrieval.onnx_models import onnx_models_available
from src.serving import NexusEngine

needs_models = pytest.mark.skipif(not onnx_models_available(),
                                  reason="ONNX models not downloaded (python scripts/build_vercel.py)")


@pytest.fixture()
def lexical_client(monkeypatch):
    monkeypatch.setattr(web_app, "_engine", NexusEngine(enable_models=False))
    return TestClient(web_app.app)


def test_health_and_info(lexical_client):
    assert lexical_client.get("/api/health").json()["status"] == "ok"
    info = lexical_client.get("/api/info").json()
    assert info["corpus_size"] >= 30
    assert info["methods"] == ["bm25"]


def test_examples(lexical_client):
    examples = lexical_client.get("/api/examples").json()
    assert examples and {"id", "text", "expected_doc"} <= set(examples[0])


def test_bm25_search_and_version_filter(lexical_client):
    res = lexical_client.post("/api/search", json={"query": "authentication", "method": "bm25", "version": "v3"})
    assert res.status_code == 200
    data = res.json()
    assert data["version"] == "3"
    assert all(r["metadata"]["version"].startswith("3") for r in data["results"])


def test_neural_method_without_models_is_rejected(lexical_client):
    res = lexical_client.post("/api/search", json={"query": "parse JSON", "method": "dense"})
    assert res.status_code == 422


def test_agent_endpoint_lexical(lexical_client):
    res = lexical_client.post("/api/agent", json={"query": "Find deprecated authentication API and its replacement"})
    assert res.status_code == 200
    data = res.json()
    assert data["results"][0]["doc_id"] == "samsung_auth_v3"
    assert data["trace"] and data["answer"]["summary"]


def test_rejects_empty_query(lexical_client):
    assert lexical_client.post("/api/agent", json={"query": ""}).status_code == 422
    assert lexical_client.post("/api/search", json={"query": "   ", "method": "bm25"}).status_code == 422


def test_bm25_returns_nothing_for_unmatched_terms():
    out = NexusEngine(enable_models=False).search("zzzz qqqq", method="bm25")
    assert out["results"] == []


@needs_models
def test_hybrid_rerank_with_onnx_models():
    engine = NexusEngine()
    out = engine.search("stream heart rate from Galaxy Watch", method="hybrid", rerank=True, top_k=3)
    assert out["results"][0]["doc_id"] == "health_heart_rate_v2"
    assert {"bm25", "dense", "rrf", "rerank"} <= set(out["results"][0]["signals"])
    agent = engine.agent("upgrade a Tizen Galaxy Watch app to the current platform")
    assert agent["results"][0]["doc_id"] == "wearos_tile_v1"
