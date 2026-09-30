from src.config.settings import get_settings, NexusSettings


def test_nexus_settings_defaults():
    settings = get_settings()
    assert settings.dense_model_name == "intfloat/e5-base-v2"
    assert settings.reranker_model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert settings.rrf_k == 60
    assert settings.bm25_k1 == 1.5
    assert settings.bm25_b == 0.75
    assert settings.cache_dir.exists()
