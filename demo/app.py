import time
import streamlit as st
import pandas as pd

from src.data.samsung_demo import load_samsung_demo_data
from src.retrieval.dense import DenseE5Retriever
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.hybrid import hybrid_retrieve
from src.retrieval.version_aware import VersionAwareFilter, parse_version_intent
from src.ranking.reranker import CodeReranker

st.set_page_config(
    page_title="Nexus Code Retrieval Engine | Samsung PRISM",
    page_icon="🔍",
    layout="wide",
)

st.title("⚡ Nexus Code Retrieval Engine")
st.caption("Samsung PRISM - Hybrid Dense/Lexical Retrieval & Version-Aware Code Search")

# Sidebar Configuration
st.sidebar.header("⚙️ Retrieval Parameters")

dataset_choice = st.sidebar.selectbox(
    "Corpus Dataset",
    options=["Samsung PRISM Demo (Version-Aware)", "CoIR AppsRetrieval Test Split"],
)

method = st.sidebar.selectbox(
    "Retrieval Pipeline Method",
    options=["hybrid-rerank", "hybrid", "dense", "bm25"],
    index=0,
)

enable_reranker = st.sidebar.checkbox("Enable Cross-Encoder Reranking", value=(method == "hybrid-rerank"))
version_filter_input = st.sidebar.text_input("Filter by SDK/Library Version (e.g. 3.0)", value="")
top_k = st.sidebar.slider("Top-K Results", min_value=1, max_value=20, value=5)


@st.cache_resource
def get_demo_data(dataset_key: str):
    if dataset_key.startswith("Samsung"):
        return load_samsung_demo_data()
    else:
        from src.data.coir import load_coir
        queries, corpus, _ = load_coir(split="test")
        return queries, corpus


queries_dict, corpus = get_demo_data(dataset_choice)

# Pre-indexed retrievers
@st.cache_resource
def get_retrievers(_corpus):
    dense = DenseE5Retriever()
    dense.build_index(_corpus)

    bm25 = BM25Retriever()
    bm25.fit(_corpus)

    reranker = CodeReranker()
    v_filter = VersionAwareFilter()
    return dense, bm25, reranker, v_filter


with st.spinner("Loading models and building vector/BM25 indices..."):
    dense_retriever, bm25_retriever, reranker_model, v_filter_model = get_retrievers(corpus)

st.markdown("### 🔎 Query Interface")

preset_queries = [
    "authentication using SDK version 3",
    "replacement for deprecated authentication API",
    "parse JSON response",
    "sensor streaming in Knox SDK version 2",
]

selected_preset = st.selectbox("Sample Preset Queries:", ["-- Select or type custom query below --"] + preset_queries)

if selected_preset and not selected_preset.startswith("--"):
    user_query = selected_preset
else:
    user_query = st.text_input("Natural Language Search Query:", value="authentication using SDK version 3")

if st.button("🚀 Execute Retrieval", type="primary"):
    start_t = time.time()
    query_obj = {"q1": {"text": user_query}}

    # Step 1: Candidate retrieval
    if method == "dense":
        res = dense_retriever.retrieve(query_obj, top_k=50)
        cands = res.get("q1", [])
    elif method == "bm25":
        res = bm25_retriever.retrieve_batch(query_obj, top_k=50)
        cands = res.get("q1", [])
    else:
        d_res = dense_retriever.retrieve(query_obj, top_k=50)
        b_res = bm25_retriever.retrieve_batch(query_obj, top_k=50)
        f_res = hybrid_retrieve(d_res, b_res, top_k=50)
        cands = f_res.get("q1", [])

    # Step 2: Version intent & filtering
    if version_filter_input.strip():
        cands = v_filter_model.filter_candidates(cands, corpus, version=version_filter_input.strip())
    elif parse_version_intent(user_query):
        boosted = v_filter_model.rerank_by_metadata_match(cands, corpus, user_query)
        cands = [cid for cid, _ in boosted]

    # Step 3: Reranking
    if enable_reranker:
        reranked_pairs = reranker_model.rerank_query(user_query, cands, corpus, top_k=top_k)
    else:
        reranked_pairs = [(cid, 1.0 / (i + 1)) for i, cid in enumerate(cands[:top_k])]

    elapsed_ms = (time.time() - start_t) * 1000

    st.success(f"Retrieved top {len(reranked_pairs)} results in **{elapsed_ms:.1f} ms**")

    # Display Metrics & Intent Analysis
    col1, col2, col3 = st.columns(3)
    col1.metric("Method", method)
    col2.metric("Reranking", "Enabled" if enable_reranker else "Disabled")
    intent = parse_version_intent(user_query)
    col3.metric("Detected Version Intent", str(intent.get("version") or "None"))

    st.markdown("---")
    st.markdown("### 🏆 Top Ranked Code Snippets")

    for rank, (cid, score) in enumerate(reranked_pairs, start=1):
        doc = corpus.get(cid, {})
        text = doc.get("text", "")
        meta = doc.get("meta_information") or doc.get("metadata") or {}

        with st.expander(f"Rank {rank} | ID: {cid} | Score: {score:.4f}", expanded=(rank == 1)):
            col_a, col_b, col_c = st.columns(3)
            col_a.write(f"**Language:** `{meta.get('language') or 'code'}`")
            col_b.write(f"**Version:** `{meta.get('version') or 'N/A'}`")
            col_c.write(f"**Deprecated:** `{meta.get('deprecated', False)}`")

            if meta.get("replacement"):
                st.info(f"💡 Recommended Replacement API: `{meta.get('replacement')}`")

            st.code(text, language=str(meta.get("language", "python")).lower())
