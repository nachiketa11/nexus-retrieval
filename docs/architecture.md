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
                                  version metadata filter/boost
                                             |
                                  optional cross-encoder
                                             |
                                       ranked results
```

Dense-only and BM25-only methods omit fusion. The cross-encoder runs when selected by the caller. Version-aware filtering or boosting uses corpus metadata and parsed query intent. In the API, one process is configured for one dataset; a request for another dataset is rejected instead of triggering an on-demand index build.

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
