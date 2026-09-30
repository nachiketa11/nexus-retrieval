import html
import time

import streamlit as st

from src.data.samsung_demo import load_samsung_demo_data
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.hybrid import reciprocal_rank_fusion
from src.retrieval.version_aware import VersionAwareFilter, parse_version_intent


st.set_page_config(page_title="NEXUS — Code Intelligence", page_icon="N", layout="wide")

st.markdown(
    """
    <style>
    :root {
      --nx-bg: #000000;
      --nx-surface: #0d0d0d;
      --nx-surface-alt: #151515;
      --nx-border: rgba(255,255,255,.14);
      --nx-border-solid: rgba(255,255,255,.24);
      --nx-text: #fafafa;
      --nx-dim: #858585;
      --nx-faint: #4c4c4c;
      --nx-red: #e5342b;
      --nx-red-dim: rgba(229,52,43,.14);
      --nx-mono: "JetBrains Mono", "Cascadia Code", "SFMono-Regular", Consolas, monospace;
      --nx-sans: "Space Grotesk", "Avenir Next", "Segoe UI", sans-serif;
    }
    .stApp {
      color: var(--nx-text);
      background-color: var(--nx-bg);
      background-image: radial-gradient(rgba(255,255,255,.10) .55px, transparent .55px);
      background-size: 20px 20px;
      font-family: var(--nx-sans);
    }
    .stApp::before {
      content: ""; position: fixed; inset: 0; z-index: 0; pointer-events: none;
      opacity: .12; background-image: repeating-linear-gradient(0deg, transparent 0 3px, rgba(255,255,255,.018) 4px);
      animation: nx-grain 8s steps(5) infinite;
    }
    @keyframes nx-grain { 0%,100% { transform: translateY(0); } 50% { transform: translateY(-3px); } }
    @keyframes nx-in { from { opacity: 0; transform: translateY(12px); filter: blur(3px); } to { opacity: 1; transform: translateY(0); filter: blur(0); } }
    @keyframes nx-pulse { 0%,100% { opacity: 1; box-shadow: 0 0 0 0 rgba(229,52,43,.22); } 50% { opacity: .7; box-shadow: 0 0 0 5px rgba(229,52,43,0); } }
    @keyframes nx-caret { 50% { opacity: 0; } }
    @keyframes nx-grow { from { transform: scaleX(0); } to { transform: scaleX(1); } }

    .stApp > header { background: rgba(0,0,0,.92); }
    .stApp [data-testid="stAppViewContainer"] { background: transparent; }
    .stApp [data-testid="stMainBlockContainer"] { max-width: 1360px; padding: 1.4rem clamp(1rem, 4vw, 3.8rem) 4rem; position: relative; z-index: 1; }
    .stApp [data-testid="stSidebar"] { background: #080808; border-right: 1px dashed var(--nx-border); }
    .stApp [data-testid="stSidebar"] > div:first-child { padding-top: 1.5rem; }
    .stApp [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: var(--nx-dim); }
    .stApp #MainMenu, .stApp footer { visibility: hidden; }
    .stApp h1, .stApp h2, .stApp h3, .stApp p, .stApp label { font-family: var(--nx-sans); }
    .stApp code, .stApp pre, .stApp [data-testid="stMetricValue"] { font-family: var(--nx-mono) !important; }
    .stApp [data-testid="stWidgetLabel"] p, .stApp label p { color: var(--nx-dim); font-size: .68rem; letter-spacing: .12em; text-transform: uppercase; }
    .stApp [data-testid="stTextInputRootElement"] { background: var(--nx-surface); border: 1px solid var(--nx-border-solid); border-radius: 2px; transition: border-color .22s ease, box-shadow .22s ease; }
    .stApp [data-testid="stTextInputRootElement"]:focus-within { border-color: var(--nx-red); box-shadow: 0 0 0 1px var(--nx-red-dim), inset 0 -2px var(--nx-red); }
    .stApp [data-testid="stTextInputRootElement"] input { background: transparent; color: var(--nx-text); font-family: var(--nx-mono); font-size: .94rem; }
    .stApp [data-testid="stTextInputRootElement"] input::placeholder { color: #5a5a5a; }
    .stApp [data-testid="stSelectbox"] [data-baseweb="select"] > div { background: var(--nx-surface); border: 1px solid var(--nx-border); border-radius: 2px; }
    .stApp [data-testid="stRadio"] > div { gap: .5rem; }
    .stApp [data-testid="stRadio"] label[data-baseweb="radio"] { border: 1px dashed var(--nx-border); border-radius: 2px; padding: .48rem .82rem; margin: 0; transition: border-color .18s ease, background .18s ease, color .18s ease; }
    .stApp [data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) { border-color: var(--nx-red); background: var(--nx-red-dim); color: var(--nx-text); }
    .stApp [data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child { display: none; }
    .stApp [data-testid="stRadio"] label[data-baseweb="radio"] p { font-family: var(--nx-mono); font-size: .72rem; letter-spacing: .04em; }
    .stApp [data-testid="stButton"] button, .stApp [data-testid="stFormSubmitButton"] button { min-height: 2.8rem; border-radius: 2px; border: 1px solid var(--nx-red); background: var(--nx-red); color: #fff; font-family: var(--nx-mono); font-size: .76rem; letter-spacing: .09em; transition: transform .18s ease, background .18s ease, box-shadow .18s ease; }
    .stApp [data-testid="stButton"] button:hover, .stApp [data-testid="stFormSubmitButton"] button:hover { transform: translateY(-1px); background: #f14339; box-shadow: 0 5px 22px rgba(229,52,43,.16); }
    .stApp [data-testid="stButton"] button:focus-visible, .stApp [data-testid="stFormSubmitButton"] button:focus-visible { outline: 2px solid #fff; outline-offset: 3px; }
    .stApp [data-testid="stCheckbox"] label p { color: #ccc; font-size: .82rem; letter-spacing: 0; text-transform: none; }
    .stApp [data-testid="stExpander"] { border: 1px dashed var(--nx-border); border-radius: 2px; }

    .nexus-nav { display:flex; align-items:center; justify-content:space-between; gap:1rem; border-bottom:1px dashed var(--nx-border); padding:.4rem 0 1rem; margin-bottom:3.2rem; animation:nx-in .55s ease both; }
    .nexus-brand { display:flex; align-items:center; gap:.8rem; }
    .nexus-mark { display:grid; place-items:center; width:2rem; height:2rem; border:1px solid var(--nx-border-solid); color:white; font:700 .95rem var(--nx-mono); }
    .nexus-wordmark { font:600 1.05rem var(--nx-sans); letter-spacing:.16em; }
    .nexus-system { color:var(--nx-dim); font: .63rem var(--nx-mono); letter-spacing:.13em; }
    .nexus-live { display:flex; align-items:center; gap:.55rem; color:#c8c8c8; font:.63rem var(--nx-mono); letter-spacing:.09em; }
    .nexus-dot { width:7px; height:7px; border-radius:50%; background:var(--nx-red); animation:nx-pulse 2.4s ease-in-out infinite; }
    .nexus-hero { position:relative; padding:0 0 2.5rem; margin-bottom:1.2rem; animation:nx-in .72s .05s ease both; }
    .nexus-kicker, .nexus-eyebrow { color:var(--nx-red); font:.64rem var(--nx-mono); letter-spacing:.16em; text-transform:uppercase; }
    .nexus-headline { max-width:900px; margin:.9rem 0 1rem; color:var(--nx-text); font:500 clamp(2.5rem, 6.6vw, 5.9rem)/.98 var(--nx-sans); letter-spacing:-.055em; }
    .nexus-headline span { color:#717171; }
    .nexus-description { max-width:600px; color:#aaa; font:400 clamp(.94rem, 1.6vw, 1.1rem)/1.7 var(--nx-sans); }
    .nexus-hero-meta { display:flex; flex-wrap:wrap; gap:.55rem 1.4rem; border-top:1px dashed var(--nx-border); margin-top:2rem; padding-top:.8rem; color:#666; font:.61rem var(--nx-mono); letter-spacing:.08em; }
    .nexus-hero-meta span::before { content:"/ "; color:var(--nx-red); }
    .nexus-section-label { display:flex; align-items:center; gap:.8rem; color:#999; font:.66rem var(--nx-mono); letter-spacing:.13em; text-transform:uppercase; margin:1rem 0 .8rem; }
    .nexus-section-label::after { content:""; flex:1; border-top:1px dashed var(--nx-border); }
    .nexus-search-frame { border:1px dashed var(--nx-border-solid); border-left:2px solid var(--nx-red); background:linear-gradient(110deg, #0d0d0d, #080808); padding:1.15rem clamp(.8rem, 2.5vw, 1.6rem) .25rem; position:relative; }
    .nexus-search-frame:focus-within { border-color:rgba(229,52,43,.7); box-shadow:0 0 28px rgba(229,52,43,.06); }
    .nexus-prompt { color:var(--nx-red); font: .75rem var(--nx-mono); }
    .nexus-enter { color:#555; font:.62rem var(--nx-mono); text-align:right; padding:.2rem 0 .5rem; }
    .nexus-enter b { color:#999; font-weight:400; }
    .nexus-control-row { display:flex; align-items:center; justify-content:space-between; gap:1rem; flex-wrap:wrap; margin:.9rem 0 1.35rem; }
    .nexus-method-title { color:#676767; font:.6rem var(--nx-mono); letter-spacing:.12em; }
    .nexus-stats { display:grid; grid-template-columns:repeat(5, minmax(0,1fr)); margin:1.8rem 0 2.4rem; border:1px dashed var(--nx-border); background:rgba(8,8,8,.82); animation:nx-in .5s ease both; }
    .nexus-stat { padding:.85rem 1rem; min-width:0; border-right:1px dashed var(--nx-border); }
    .nexus-stat:last-child { border-right:0; }
    .nexus-stat-label { color:#666; font:.58rem var(--nx-mono); letter-spacing:.1em; }
    .nexus-stat-value { color:#f4f4f4; margin-top:.4rem; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; font:.83rem var(--nx-mono); animation:nx-in .42s ease both; }
    .nexus-pipeline { display:flex; align-items:center; justify-content:center; gap:.7rem; flex-wrap:wrap; background:#080808; border-block:1px dashed var(--nx-border); padding:1rem; margin:1.4rem 0 2rem; animation:nx-in .48s ease both; }
    .nexus-node { color:#bbb; border:1px solid #292929; padding:.45rem .7rem; font:.62rem var(--nx-mono); letter-spacing:.06em; animation:nx-in .45s ease both; }
    .nexus-node.active { color:#fff; border-color:var(--nx-red); background:var(--nx-red-dim); }
    .nexus-arrow { color:#555; font:.8rem var(--nx-mono); }
    .nexus-result { border:1px solid var(--nx-border); background:linear-gradient(115deg,rgba(15,15,15,.97),rgba(7,7,7,.97)); margin:.8rem 0 1rem; padding:1.05rem clamp(.75rem,2vw,1.35rem); animation:nx-in .52s calc(var(--nx-order, 0) * 90ms) ease both; transition:border-color .2s ease, transform .2s ease, background .2s ease; }
    .nexus-result:hover { border-color:rgba(229,52,43,.62); transform:translateY(-2px); background:#101010; }
    .nexus-result-head { display:grid; grid-template-columns:3.3rem minmax(0,1fr) auto; align-items:start; gap:.7rem; border-bottom:1px dashed var(--nx-border); padding-bottom:.9rem; }
    .nexus-rank { color:#5b5b5b; font:400 1.75rem/1 var(--nx-mono); }
    .nexus-path { color:#f0f0f0; overflow-wrap:anywhere; font:.9rem/1.45 var(--nx-mono); }
    .nexus-docid { margin-top:.3rem; color:#777; font:.62rem var(--nx-mono); letter-spacing:.035em; overflow-wrap:anywhere; }
    .nexus-score { color:#ddd; text-align:right; white-space:nowrap; font:.66rem var(--nx-mono); }
    .nexus-score small { display:block; margin-top:.28rem; color:#686868; font-size:.54rem; }
    .nexus-result-meta { display:flex; flex-wrap:wrap; gap:.4rem .55rem; padding:.75rem 0; }
    .nexus-tag { color:#999; border:1px dashed #303030; padding:.22rem .42rem; font:.57rem var(--nx-mono); letter-spacing:.045em; }
    .nexus-tag.warning { color:#f18b84; border-color:rgba(229,52,43,.42); background:var(--nx-red-dim); }
    .nexus-scorebar { height:2px; background:#252525; margin:0 0 .9rem; overflow:hidden; }
    .nexus-scorebar span { display:block; height:100%; background:var(--nx-red); transform-origin:left; animation:nx-grow .6s .12s ease both; }
    .nexus-score-note { color:#505050; font:.55rem var(--nx-mono); margin:-.62rem 0 .9rem; text-align:right; }
    .nexus-code-label { color:#626262; font:.56rem var(--nx-mono); letter-spacing:.12em; margin-bottom:.35rem; }
    .nexus-code { overflow:auto; max-height:390px; padding:.9rem 1rem; border:1px dashed #292929; background:#050505; color:#d3d3d3; font:.72rem/1.65 var(--nx-mono); tab-size:4; white-space:pre; }
    .nexus-footnote { color:#626262; font:.63rem/1.6 var(--nx-mono); border-left:1px solid var(--nx-red); padding:.35rem .75rem; margin-top:1rem; }
    .nexus-empty { padding:1.5rem; border:1px dashed var(--nx-border); color:#888; background:#080808; font:.77rem/1.7 var(--nx-mono); }
    .nexus-load { height:3px; overflow:hidden; background:#202020; margin:.9rem 0; }
    .nexus-load::after { content:""; display:block; height:100%; width:35%; background:var(--nx-red); animation:nx-load 1s ease-in-out infinite alternate; }
    @keyframes nx-load { from { transform:translateX(-5%); } to { transform:translateX(195%); } }
    .nexus-footer { display:flex; justify-content:space-between; gap:1rem; flex-wrap:wrap; color:#4d4d4d; border-top:1px dashed var(--nx-border); margin-top:3rem; padding-top:.8rem; font:.58rem var(--nx-mono); letter-spacing:.08em; }

    @media (max-width: 760px) {
      .stApp [data-testid="stMainBlockContainer"] { padding:1rem 1rem 2.5rem; }
      .nexus-nav { margin-bottom:2.1rem; }
      .nexus-system { display:none; }
      .nexus-headline { font-size:clamp(2.55rem, 13vw, 4.2rem); }
      .nexus-stats { grid-template-columns:repeat(2,minmax(0,1fr)); }
      .nexus-stat { border-bottom:1px dashed var(--nx-border); }
      .nexus-stat:nth-child(2n) { border-right:0; }
      .nexus-stat:last-child { grid-column:1/-1; border-bottom:0; }
      .nexus-result-head { grid-template-columns:2.2rem minmax(0,1fr); }
      .nexus-score { grid-column:2; text-align:left; }
      .nexus-score small { display:inline; margin-left:.4rem; }
      .nexus-rank { font-size:1.35rem; }
      .nexus-code { font-size:.66rem; }
    }
    @media (prefers-reduced-motion: reduce) {
      *, *::before, *::after { animation-duration:.01ms !important; animation-iteration-count:1 !important; scroll-behavior:auto !important; transition-duration:.01ms !important; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _safe(value) -> str:
    return html.escape(str(value or ""), quote=True)


def _pipeline(method: str) -> str:
    if method == "BM25":
        nodes = ["QUERY", "BM25", "RANKED CANDIDATES"]
    elif method == "Dense E5":
        nodes = ["QUERY", "DENSE E5 · FAISS", "RANKED CANDIDATES"]
    else:
        nodes = ["QUERY", "DENSE E5", "+", "BM25", "RRF FUSION", "RANKED CANDIDATES"]
    bits = []
    for index, node in enumerate(nodes):
        active = " active" if node in {"QUERY", "RRF FUSION", "BM25", "DENSE E5", "DENSE E5 · FAISS", "RANKED CANDIDATES"} else ""
        bits.append(f'<span class="nexus-node{active}">{_safe(node)}</span>')
        if index < len(nodes) - 1:
            bits.append('<span class="nexus-arrow" aria-hidden="true">→</span>')
    return '<div class="nexus-pipeline" aria-label="Retrieval pipeline">' + "".join(bits) + "</div>"


st.markdown(
    """
    <div class="nexus-nav">
      <div class="nexus-brand"><div class="nexus-mark">N/</div><div class="nexus-wordmark">NEXUS</div><div class="nexus-system">AGENTIC CODE INTELLIGENCE</div></div>
      <div class="nexus-live"><span class="nexus-dot"></span><span>CORPUS SEARCH READY</span></div>
    </div>
    <section class="nexus-hero">
      <div class="nexus-kicker">RETRIEVAL SYSTEM / 001</div>
      <h1 class="nexus-headline">Find the code<br><span>behind the intent.</span></h1>
      <div class="nexus-description">Search implementation patterns across code and version context. NEXUS ranks existing snippets for inspection; it does not generate code.</div>
      <div class="nexus-hero-meta"><span>SEMANTIC + LEXICAL SIGNALS</span><span>VERSION-AWARE SEARCH</span><span>RANKED CODE CONTEXT</span></div>
    </section>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown('<div class="nexus-eyebrow">SYSTEM / CONFIGURATION</div>', unsafe_allow_html=True)
    dataset_label = st.selectbox("CORPUS", ["Samsung/demo examples", "CoIR AppsRetrieval · test split"])
    use_reranker = st.checkbox("Cross-encoder rerank", value=False)
    explicit_version = st.text_input("Version filter", placeholder="e.g. 3.0")
    top_k = st.slider("Result count", min_value=1, max_value=10, value=5)
    st.markdown("---")
    st.caption("Samsung/demo examples are synthetic and separate from the official CoIR benchmark.")

dataset_name = "samsung_demo" if dataset_label.startswith("Samsung") else "coir"

with st.spinner("Loading corpus and lexical index…"):
    try:
        if dataset_name == "samsung_demo":
            queries, corpus = load_samsung_demo_data()
        else:
            from src.data.coir import load_coir
            queries, corpus, _ = load_coir(split="test")
        bm25_retriever = BM25Retriever().fit(corpus)
    except Exception as exc:
        st.error(f"Corpus initialization failed: {exc}")
        st.stop()

st.markdown('<div class="nexus-section-label">01 / Retrieval signal</div>', unsafe_allow_html=True)
method = st.radio("METHOD", ["BM25", "Dense E5", "RRF"], horizontal=True, label_visibility="collapsed")
st.markdown('<div class="nexus-section-label">02 / Query terminal</div>', unsafe_allow_html=True)

examples = [
    "Find current SDK v3 authentication API",
    "Find deprecated authentication API and its replacement",
    "Find code similar to parsing a JSON response",
    "Migrate legacy v1 authentication to the current version",
]

with st.form("nexus_search", clear_on_submit=False):
    query_source = st.radio("QUERY SOURCE", ["Write a query", "Use an example"], horizontal=True)
    if query_source == "Use an example":
        query = st.selectbox("EXAMPLE QUERY", examples)
    else:
        query = st.text_input("QUERY / NATURAL LANGUAGE", placeholder="Describe the code behavior, symbol, or version you need…")
    prompt_col, action_col = st.columns([5, 1.2], vertical_alignment="bottom")
    with prompt_col:
        st.markdown('<div class="nexus-enter"><b>↳</b> Press run to search</div>', unsafe_allow_html=True)
    with action_col:
        submitted = st.form_submit_button("RUN SEARCH  ↗", type="primary", use_container_width=True)

if submitted:
    started = time.perf_counter()
    loading_slot = st.empty()
    loading_slot.markdown('<div class="nexus-load" aria-label="Retrieval in progress"></div>', unsafe_allow_html=True)
    try:
        if method == "BM25":
            scored = bm25_retriever.retrieve(query, top_k=100)
        else:
            from src.config.settings import get_settings
            from src.retrieval.dense import DenseE5Retriever
            settings = get_settings()
            dense_retriever = DenseE5Retriever(
                model_name=settings.dense_model_name,
                batch_size=settings.dense_batch_size,
            )
            dense_retriever.build_index(corpus)
            dense_ranked = dense_retriever.retrieve({"q": {"text": query}}, top_k=100).get("q", [])
            if method == "Dense E5":
                scored = [(doc_id, None) for doc_id in dense_ranked]
            else:
                lexical_ranked = bm25_retriever.retrieve_batch({"q": {"text": query}}, top_k=100).get("q", [])
                scored = reciprocal_rank_fusion([dense_ranked, lexical_ranked], top_k=100)

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
            from src.ranking.reranker import CodeReranker
            reranker = CodeReranker()
            ranked = reranker.rerank_query(query, candidates, corpus, top_k=top_k)
            reranked = True
        else:
            ranked = [(doc_id, base_scores.get(doc_id)) for doc_id in candidates[:top_k]]
        elapsed_ms = (time.perf_counter() - started) * 1000
        loading_slot.empty()

        method_label = method + (" + CROSS-ENCODER" if reranked else "")
        dataset_short = "SAMSUNG / DEMO" if dataset_name == "samsung_demo" else "COIR / APPSRETRIEVAL TEST"
        stats = [
            ("METHOD", method_label),
            ("RESULTS", str(len(ranked))),
            ("LATENCY", f"{elapsed_ms:.1f} ms"),
            ("CORPUS", f"{len(corpus):,} RECORDS"),
            ("DATASET", dataset_short),
        ]
        stat_html = ''.join(
            f'<div class="nexus-stat"><div class="nexus-stat-label">{_safe(label)}</div><div class="nexus-stat-value">{_safe(value)}</div></div>'
            for label, value in stats
        )
        st.markdown(f'<div class="nexus-stats">{stat_html}</div>', unsafe_allow_html=True)
        st.markdown(_pipeline(method), unsafe_allow_html=True)
        st.markdown('<div class="nexus-section-label">03 / Ranked code candidates</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="nexus-footnote">QUERY&nbsp;&nbsp; {_safe(query)}<br>Candidate order is a retrieval ranking, not a relevance guarantee. Scores are method-specific and are not calibrated probabilities.</div>', unsafe_allow_html=True)
        if filter_version:
            st.caption(f"VERSION CONSTRAINT / {filter_version}")
        elif intent:
            st.caption("QUERY INTENT / " + " · ".join(intent.keys()))

        if not ranked:
            st.markdown('<div class="nexus-empty">NO CANDIDATES / Try broadening the query or removing the version constraint.</div>', unsafe_allow_html=True)
        else:
            score_magnitude = max((abs(float(score)) for _, score in ranked if score is not None), default=0.0)
            for rank, (doc_id, score) in enumerate(ranked, start=1):
                doc = corpus.get(doc_id, {})
                metadata = doc.get("metadata") or doc.get("meta_information") or {}
                path = metadata.get("file") or doc.get("title") or doc_id
                language = metadata.get("language") or doc.get("language") or "CODE"
                version = metadata.get("version")
                repository = metadata.get("repository")
                deprecated = bool(metadata.get("deprecated"))
                score_text = f"{float(score):.4f}" if score is not None else "RANK ORDER"
                bar_percent = min(100, abs(float(score)) / score_magnitude * 100) if score is not None and score_magnitude else 0
                tags = [language.upper()]
                if version:
                    tags.append(f"VERSION {version}")
                if repository:
                    tags.append(str(repository))
                if deprecated:
                    tags.append("DEPRECATED")
                tag_html = ''.join(f'<span class="nexus-tag{" warning" if tag == "DEPRECATED" else ""}">{_safe(tag)}</span>' for tag in tags)
                replacement = metadata.get("replacement")
                replacement_html = f'<span class="nexus-tag warning">REPLACEMENT / {_safe(replacement)}</span>' if replacement else ""
                score_note = "RELATIVE MAGNITUDE" if score is not None else "NO SCORE EXPOSED BY RETRIEVER"
                code = _safe(doc.get("text", ""))
                card_html = f'''
                <article class="nexus-result" style="--nx-order:{rank - 1}">
                  <div class="nexus-result-head">
                    <div class="nexus-rank">{rank:02d}</div>
                    <div><div class="nexus-path">{_safe(path)}</div><div class="nexus-docid">{_safe(doc_id)}</div></div>
                    <div class="nexus-score">SCORE&nbsp; {score_text}<small>{score_note}</small></div>
                  </div>
                  <div class="nexus-result-meta">{tag_html}{replacement_html}</div>
                  <div class="nexus-scorebar" aria-label="relative score magnitude"><span style="width:{bar_percent:.1f}%"></span></div>
                  <div class="nexus-score-note">{score_note}</div>
                  <div class="nexus-code-label">CODE / SOURCE PREVIEW</div>
                  <pre class="nexus-code"><code>{code}</code></pre>
                </article>
                '''
                st.markdown(card_html, unsafe_allow_html=True)
    except Exception as exc:
        loading_slot.empty()
        st.error(f"Search failed: {exc}")
else:
    st.markdown('<div class="nexus-empty">AWAITING QUERY / Enter a coding question or choose a preset, then run search.</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="nexus-footer"><span>NEXUS / RETRIEVAL INTELLIGENCE</span><span>DEMO CORPUS IS SYNTHETIC · NOT AN OFFICIAL SAMSUNG DATASET</span></div>',
    unsafe_allow_html=True,
)
