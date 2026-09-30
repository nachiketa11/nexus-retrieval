from src.retrieval.version_aware import parse_version_intent, VersionAwareFilter


def test_parse_version_intent():
    intent1 = parse_version_intent("authentication using SDK version 3")
    assert intent1.get("version") == "3"

    intent2 = parse_version_intent("replacement for deprecated authentication API")
    assert intent2.get("deprecated") is True
    assert intent2.get("replacement") is True


def test_version_filter_candidates():
    v_filter = VersionAwareFilter()
    corpus = {
        "doc1": {"metadata": {"version": "1.0", "deprecated": True}},
        "doc2": {"metadata": {"version": "3.0", "deprecated": False}},
    }

    filtered = v_filter.filter_candidates(["doc1", "doc2"], corpus, version="3.0")
    assert filtered == ["doc2"]
