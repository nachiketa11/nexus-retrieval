import pytest
import numpy as np
from fastapi.testclient import TestClient

from src.api.app import app, STATE, load_system_indexes
from src.retrieval import dense


class DummyModel:
    def encode(self, texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True):
        # Return synthetic 768-dim normalized embeddings
        arr = np.ones((len(texts), 768), dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / norms


@pytest.fixture(autouse=True)
def patch_and_init_api(monkeypatch):
    """Patch model loading for ultra-fast API unit tests."""
    monkeypatch.setattr(dense, "_load_model", lambda model_name="intfloat/e5-base-v2": DummyModel())
    load_system_indexes(dataset="samsung_demo", force=True)


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "uptime_seconds" in data


def test_info_endpoint():
    with TestClient(app) as client:
        response = client.get("/info")
        assert response.status_code == 200
        data = response.json()
        assert "dense_model" in data
        assert "reranker_model" in data


def test_retrieve_endpoint():
    with TestClient(app) as client:
        payload = {
            "query": "authentication using SDK version 3",
            "top_k": 5,
            "method": "hybrid",
            "rerank": False,
            "dataset": "samsung_demo",
        }
        response = client.post("/retrieve", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["query"] == payload["query"]
        assert "results" in data
        assert len(data["results"]) <= 5
