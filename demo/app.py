import time

import streamlit as st

from src.data.samsung_demo import load_samsung_demo_data
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.hybrid import reciprocal_rank_fusion
from src.retrieval.version_aware import VersionAwareFilter, parse_version_intent


st.set_page_config(page_title="NEXUS — Agentic Code Intelligence", page_icon="⌕", layout="wide")
st.title("NEXUS — Agentic Code Intelligence Retrieval")
st.caption("Search code by intent, identifiers, and version context. NEXUS retrieves and ranks code examples; it does not generate code.")

with st.sidebar:
    st.header("Search configuration")
    dataset_label = st.selectbox("Dataset", ["Samsung/demo examples", "CoIR AppsRetrieval · test split"])
    method = st.selectbox("Retrieval method", ["BM25", "Dense E5", "Hybrid RRF"])
    use_reranker = st.checkbox("Rerank candidates with cross-encoder", value=False)
    explicit_version = st.text_input("Version filter (optional)", placeholder="3.0")
    top_k = st.slider("Results", min_value=1, max_value=10, value=5)
    st.caption("The Samsung/demo corpus is synthetic, small, and separate from the official CoIR benchmark.")


@st.cache_resource
def load_dataset(dataset_name: str):
    if dataset_name == "samsung_demo":
        queries, corpus = load_samsung_demo_data()
        return queries, corpus
    from src.data.coir import load_coir
    queries, corpus, _ = load_coir(split="test")
    return queries, corpus


@st.cache_resource
def build_bm25(_corpus, dataset_name: str):
    return BM25Retriever().fit(_corpus)


@st.cache_resource
def build_dense(_corpus, dataset_name: str):
    from src.config.settings import get_settings
    from src.retrieval.dense import DenseE5Retriever
    settings = get_settings()
    retriever = DenseE5Retriever(model_name=settings.dense_model_name, batch_size=settings.dense_batch_size)
    retriever.build_index(_corpus)
    return retriever


@st.cache_resource
def build_reranker():
    from src.ranking.reranker import CodeReranker
    return CodeReranker()


dataset_name = "samsung_demo" if dataset_label.startswith("Samsung") else "coir"
try:
    queries, corpus = load_dataset(dataset_name)
    bm25 = build_bm25(corpus, dataset_name)
except Exception as exc:
    st.error(f"Could not load the selected dataset: {exc}")
    st.stop()

st.markdown(f"**Corpus:** {len(corpus):,} code records · **Dataset:** {dataset_label}")
st.divider()

examples = [
    "Find current SDK v3 authentication API",
    "Find deprecated authentication API and its replacement",
    "Find code similar to parsing a JSON response",
    "Migrate legacy v1 authentication to the current version",
]
query_source = st.radio("Query input", ["Write a query", "Use an example"], horizontal=True)
if query_source == "Use an example":
    query = st.selectbox("Example query", examples)
else:
    query = st.text_input("Natural-language coding query", placeholder="e.g. Find deprecated authentication API and its replacement")

if st.button("Search code", type="primary", disabled=not query.strip()):
    started = time.perf_counter()
    try:
        if method == "BM25":
            scored = bm25.retrieve(query, top_k=100)
        else:
            dense = build_dense(corpus, dataset_name)
            dense_ranked = dense.retrieve({"q": {"text": query}}, top_k=100).get("q", [])
            if method == "Dense E5":
                scored = [(doc_id, None) for doc_id in dense_ranked]
            else:
                lexical = bm25.retrieve_batch({"q": {"text": query}}, top_k=100).get("q", [])
                scored = reciprocal_rank_fusion([dense_ranked, lexical], top_k=100)

        candidates = [doc_id for doc_id, _ in scored]
        intent = parse_version_intent(query)
        filter_version = explicit_version.strip() or intent.get("version")
        version_filter = VersionAwareFilter()
        if filter_version:
            candidates = version_filter.filter_candidates(candidates, corpus, version=filter_version)
        elif intent:
            candidates = [doc_id for doc_id, _ in version_filter.rerank_by_metadata_match(candidates, corpus, query)]

        base_scores = dict(scored)
        reranked = False
        if use_reranker and candidates:
            reranker = build_reranker()
            ranked = reranker.rerank_query(query, candidates, corpus, top_k=top_k)
            reranked = True
        else:
            ranked = [(doc_id, base_scores.get(doc_id)) for doc_id in candidates[:top_k]]

        elapsed_ms = (time.perf_counter() - started) * 1000
        st.subheader("Retrieved results")
        a, b, c = st.columns(3)
        a.metric("Method", method + (" + reranker" if reranked else ""))
        b.metric("Returned", len(ranked))
        c.metric("Retrieval latency", f"{elapsed_ms:.1f} ms")
        st.write(f"**Query:** {query}")
        if filter_version:
            st.info(f"Version constraint applied: {filter_version}")
        elif intent:
            st.caption(f"Detected intent: {', '.join(intent.keys())}")

        if not ranked:
            st.info("No matching code records were found. Try a broader query or remove the version filter.")
        for rank, (doc_id, score) in enumerate(ranked, start=1):
            doc = corpus.get(doc_id, {})
            metadata = doc.get("metadata") or doc.get("meta_information") or {}
            title = metadata.get("file") or doc.get("title") or doc_id
            score_label = f"Score {score:.4f}" if score is not None else "Ranked by retrieval order"
            with st.container(border=True):
                st.markdown(f"**{rank}. {title}** · `{doc_id}` · {score_label}")
                details = [metadata.get("language") or doc.get("language"), metadata.get("version"), metadata.get("repository")]
                details = [str(value) for value in details if value]
                if details:
                    st.caption(" · ".join(details))
                if metadata.get("deprecated"):
                    replacement = metadata.get("replacement")
                    st.warning(f"Deprecated API{f' · replacement: {replacement}' if replacement else ''}")
                st.code(doc.get("text", ""), language=(metadata.get("language") or doc.get("language") or "text").lower())
    except Exception as exc:
        st.error(f"Search failed: {exc}")
else:
    st.info("Enter a coding question or choose an example, then select Search code.")
