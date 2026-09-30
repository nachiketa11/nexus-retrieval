import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory

from src.retrieval import dense


class CountingModel:
    def __init__(self):
        self.calls = 0

    def encode(self, texts, **kwargs):
        self.calls += 1
        return np.ones((len(texts), 4), dtype=np.float32)


def test_corpus_cache_tracks_order_text_and_model_configuration(monkeypatch):
    model = CountingModel()
    monkeypatch.setattr(dense, "_load_model", lambda name: model)
    with TemporaryDirectory(dir=Path(__file__).parent) as cache_dir:
        cache_path = Path(cache_dir)
        monkeypatch.setattr(dense, "get_settings", lambda: type("Settings", (), {"cache_dir": cache_path})())

        corpus = {"doc-a": {"text": "alpha"}, "doc-b": {"text": "beta"}}
        Dense = dense.DenseE5Retriever
        Dense(model_name="fake/model", batch_size=8).build_index(corpus)
        assert model.calls == 1

        Dense(model_name="fake/model", batch_size=8).build_index(corpus)
        assert model.calls == 1  # identical input/configuration reuses vectors

        reordered = {"doc-b": {"text": "beta"}, "doc-a": {"text": "alpha"}}
        Dense(model_name="fake/model", batch_size=8).build_index(reordered)
        assert model.calls == 2

        changed_text = {"doc-b": {"text": "changed"}, "doc-a": {"text": "alpha"}}
        Dense(model_name="fake/model", batch_size=8).build_index(changed_text)
        assert model.calls == 3

        Dense(model_name="fake/model", batch_size=16).build_index(changed_text)
        assert model.calls == 4

        assert len(list(cache_path.glob("corpus_*.npy"))) == 4
