# NEXUS architecture

NEXUS retrieves code records from a selected corpus. It ranks and presents existing snippets; it does not generate code.

## Online query path

```text
Natural-language query
          |
          v
  Dense E5 retrieval -------------------+
  BM25 lexical retrieval ---------------+--> RRF fusion (hybrid methods)
                                             |
                                  optional cross-encoder
                                             |
                                  version metadata filter/boost
                                             |
                                       ranked results
```

Dense-only and BM25-only methods omit fusion. The cross-encoder runs when selected by the caller. Version-aware filtering (explicit version) or boosting (parsed intent) uses corpus metadata. In the API, one process is configured for one dataset; a request for another dataset is rejected instead of triggering an on-demand index build.

The pipeline above is what `/retrieve` and the PIPELINE mode of the UI run. Cross-encoder reranking runs before the version policy so metadata boosts are not discarded.

## Agent path (`src/agent/nexus_agent.py`)

```text
query
  └─ ANALYSE   intent (target vs. source version, deprecated/replacement/current), language, SDK, identifiers
  └─ PLAN      select tools from the analysis and the tools injected (BM25 always; dense/RRF/cross-encoder if available)
  └─ ACT       bm25 → dense_e5 → rrf_fusion → cross_encoder → version_policy | version_filter → context_boost
               → deprecation_resolver (follow metadata.replacement links to the current successor)
  └─ REFLECT   confidence = mean(term coverage, retriever agreement, margin over best competitor, metadata consistency)
  └─ REFINE    if confidence < 0.55: pseudo-relevance-feedback expansion, or relax a filter that removed everything
  └─ RESPOND   best match, migration path, warnings, ranked results with per-signal scores, full trace
```

The agent receives retrieval tools as callables, so the same logic runs on three stacks:

| Stack | Entry point | Dense / reranker |
| :--- | :--- | :--- |
| PyTorch + FAISS | `src/api/app.py` (`/agent`) | sentence-transformers E5-base-v2, FAISS; CrossEncoder |
| Serverless (deployed) | `web/app.py` → `src/serving/engine.py` | ONNX Runtime int8 E5-base-v2 (exact NumPy search); ONNX int8 cross-encoder |
| Lexical only | `python -m src.cli agent --lexical` | none — BM25 + metadata policies |

## Corpus and index setup

```text
CoIR loader or Samsung/demo records
                 |
       code text and metadata
          /              \
 Dense E5 embeddings     BM25 token index
          |              |
       FAISS index     lexical rankings
```

`CodeChunker` is available as a utility, but it is not currently wired into corpus ingestion or the retrieval index path. Dense vectors and BM25 indexes are built in memory; dense corpus embeddings may be cached locally. There is no persisted, reusable FAISS or BM25 index artifact in the current implementation.

## Datasets

- **CoIR AppsRetrieval** is the external benchmark corpus and relevance set loaded by `src/data/coir.py`.
- **Samsung/demo** is a small synthetic example set in `src/data/samsung_demo.py`. It is separate from the official benchmark and does not imply Samsung endorsement.
