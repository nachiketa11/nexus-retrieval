import pytest

from src.agent import NexusAgent
from src.data.samsung_demo import load_samsung_demo_data
from src.retrieval.version_aware import VersionAwareFilter, parse_version_intent, replacement_chain


@pytest.fixture(scope="module")
def corpus():
    return load_samsung_demo_data()[1]


@pytest.fixture(scope="module")
def agent(corpus):
    return NexusAgent(corpus)  # BM25 + metadata tools only: no model downloads


def test_demo_corpus_links_are_consistent():
    queries, corpus = load_samsung_demo_data()
    assert len(corpus) >= 30
    for qid, q in queries.items():
        assert q["expected_doc"] in corpus, qid
    for cid, doc in corpus.items():
        replacement = doc["metadata"].get("replacement")
        assert replacement is None or replacement in corpus, cid
        if replacement:
            assert doc["metadata"]["deprecated"] is True, cid


def test_migration_version_is_source_not_target():
    intent = parse_version_intent("Migrate legacy v1 authentication to the current version")
    assert intent.get("source_version") == "1"
    assert "version" not in intent
    assert intent["migration"] and intent["replacement"] and intent["current"]


def test_replacement_intent_promotes_successor_not_deprecated(corpus):
    ranked = VersionAwareFilter().rerank_by_metadata_match(
        ["samsung_auth_v1", "samsung_auth_v2", "samsung_auth_v3"], corpus,
        "replacement for deprecated authentication API")
    assert ranked[0][0] == "samsung_auth_v3"


def test_replacement_chain(corpus):
    assert replacement_chain("samsung_auth_v1", corpus) == ["samsung_auth_v3"]
    assert replacement_chain("samsung_auth_v3", corpus) == []


def test_lexical_agent_beats_bm25_on_labelled_queries(agent, corpus):
    from src.retrieval.bm25 import BM25Retriever

    queries = load_samsung_demo_data()[0]
    bm25 = BM25Retriever().fit(corpus)
    agent_hits = sum(agent.run(q["text"])["results"][0]["doc_id"] == q["expected_doc"] for q in queries.values())
    bm25_hits = sum(bm25.retrieve(q["text"], 1)[0][0] == q["expected_doc"] for q in queries.values())
    assert agent_hits >= len(queries) - 1
    assert agent_hits > bm25_hits


@pytest.mark.skipif(not __import__("src.retrieval.onnx_models", fromlist=["x"]).onnx_models_available(),
                    reason="ONNX models not downloaded")
def test_neural_agent_top1_on_all_labelled_queries():
    from src.serving import NexusEngine

    engine = NexusEngine()
    queries = load_samsung_demo_data()[0]
    misses = [qid for qid, q in queries.items()
              if engine.agent(q["text"])["results"][0]["doc_id"] != q["expected_doc"]]
    assert misses == []


def test_agent_trace_covers_every_phase(agent):
    out = agent.run("migrate from IAP v4 startPayment")
    phases = [s["phase"] for s in out["trace"]]
    for phase in ("analyse", "plan", "act", "reflect", "respond"):
        assert phase in phases
    assert out["answer"]["migration"]["path"] == ["iap_purchase_v4", "iap_purchase_v6"]
    assert 0.0 <= out["confidence"] <= 1.0


def test_agent_respects_explicit_old_version(agent):
    out = agent.run("Knox camera v2 java")
    assert out["results"][0]["doc_id"] == "knox_camera_policy_v2"
    assert "superseded" in out["results"][0]["note"]


def test_agent_relaxes_impossible_version_filter(agent):
    out = agent.run("authentication", version="9")
    assert out["results"], "agent should relax a filter that removes every candidate"
    assert any(s["phase"] == "refine" for s in out["trace"])


def test_agent_flags_low_confidence(agent):
    out = agent.run("make it faster")
    assert out["confidence"] < agent.confidence_threshold
    assert out["answer"]["summary"].startswith("Low confidence")


def test_agent_uses_injected_neural_tools(corpus):
    calls = {"dense": 0, "rerank": 0}

    def dense(query, k):
        calls["dense"] += 1
        return [(cid, 1.0 / (i + 1)) for i, cid in enumerate(sorted(corpus)[:k])]

    def rerank(query, ids):
        calls["rerank"] += 1
        return [(cid, float(len(ids) - i)) for i, cid in enumerate(ids)]

    out = NexusAgent(corpus, dense=dense, reranker=rerank).run("stream heart rate from Galaxy Watch")
    assert calls["dense"] >= 1 and calls["rerank"] >= 1
    assert {"dense", "rrf_fusion", "cross_encoder_rerank"} <= set(out["plan"])
