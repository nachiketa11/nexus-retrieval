import os
import time
from typing import Dict, List, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

# Simple in‑process model cache to avoid re‑loading the same model multiple times
_MODEL_CACHE: Dict[str, SentenceTransformer] = {}

def _load_model(model_name: str = "intfloat/e5-base-v2") -> SentenceTransformer:
    """Load a SentenceTransformer model, re‑using a cached instance if available.

    The function is isolated so that it can be monkey‑patched in unit tests.
    """
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]
    model = SentenceTransformer(model_name)
    _MODEL_CACHE[model_name] = model
    return model


def _prepare_texts(texts: List[str], prefix: str) -> List[str]:
    """Add the required E5 prefix ('query: ' or 'passage: ') to each text."""
    return [f"{prefix} {t}" for t in texts]


def _encode_texts(
    model: SentenceTransformer,
    texts: List[str],
    batch_size: int = 32,
    show_progress: bool = True,
) -> np.ndarray:
    """Encode a list of texts with the given model and return a (N, D) ndarray.

    Returns
    -------
    np.ndarray
        Normalized embeddings as ``float32`` of shape (N, D).
    """
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,
    )
    return np.asarray(embeddings, dtype="float32")

# ---------------------------------------------------------------------------
# Caching utilities
# ---------------------------------------------------------------------------

def _cache_path(name: str, model_name: str, split: str) -> str:
    """Generate a deterministic cache filename for embeddings.

    Parameters
    ----------
    name: str
        Either "corpus" or "queries".
    model_name: str
        Model identifier (slashes are replaced with underscores).
    split: str
        Dataset split name (e.g., "test").
    """
    safe_model = model_name.replace('/', '_')
    cache_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "cache"))
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{name}_{safe_model}_{split}.npy")


def _load_or_encode_embeddings(
    model: SentenceTransformer,
    texts: List[str],
    cache_file: str,
    batch_size: int = 32,
) -> np.ndarray:
    """Load embeddings from cache if present and compatible, otherwise encode and cache.

    For non‑SentenceTransformer models (e.g., dummy models used in tests) caching is skipped.
    """
    # Skip caching for dummy models that are not SentenceTransformer instances
    if not isinstance(model, SentenceTransformer):
        return _encode_texts(model, texts, batch_size=batch_size, show_progress=True)

    if os.path.exists(cache_file):
        cached = np.load(cache_file)
        if cached.shape[0] == len(texts):
            start = time.time()
            elapsed = time.time() - start
            print(f"Loaded {os.path.basename(cache_file)} from cache in {elapsed:.2f}s")
            return cached
        else:
            print(
                f"Cache {os.path.basename(cache_file)} size mismatch (cached {cached.shape[0]} vs expected {len(texts)}); recomputing."
            )
    # Encode and cache
    start = time.time()
    embeddings = _encode_texts(model, texts, batch_size=batch_size, show_progress=True)
    elapsed = time.time() - start
    np.save(cache_file, embeddings)
    print(f"Encoded and cached {os.path.basename(cache_file)} in {elapsed:.2f}s")
    return embeddings

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_corpus_index(
    corpus: Dict[str, Dict],
    model_name: str = "intfloat/e5-base-v2",
    batch_size: int = 32,
    split: str = "test",
) -> Tuple[faiss.IndexIDMap, Dict[int, str]]:
    """Build a FAISS IndexIDMap for the provided corpus.

    Returns
    -------
    index: faiss.IndexIDMap
        FAISS index with inner‑product (cosine) similarity.
    id_map: Dict[int, str]
        Mapping from FAISS integer ID to the original corpus ID (deterministic).
    """
    # Load model – timing only for informational purposes
    model_load_start = time.time()
    model = _load_model(model_name)
    model_load_time = time.time() - model_load_start

    # Deterministic mapping: sort corpus IDs lexicographically
    sorted_ids = sorted(corpus.keys())
    id_map = {i: cid for i, cid in enumerate(sorted_ids)}
    texts = [_prepare_texts([corpus[cid]["text"]], "passage:")[0] for cid in sorted_ids]

    # Cache handling for corpus embeddings
    cache_file = _cache_path("corpus", model_name, split)
    embeddings = _load_or_encode_embeddings(model, texts, cache_file, batch_size=batch_size)

    # Build FAISS index
    dim = embeddings.shape[1]
    index_start = time.time()
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIDMap(quantizer)
    index.add_with_ids(embeddings, np.array(list(id_map.keys()), dtype="int64"))
    index_build_time = time.time() - index_start

    total_time = model_load_time + index_build_time
    print(f"Model load: {model_load_time:.2f}s, index build: {index_build_time:.2f}s, total: {total_time:.2f}s")
    return index, id_map


def retrieve_top_k(
    queries: Dict[str, Dict],
    index: faiss.IndexIDMap,
    id_map: Dict[int, str],
    model_name: str = "intfloat/e5-base-v2",
    batch_size: int = 32,
    top_k: int = 100,
    split: str = "test",
) -> Dict[str, List[str]]:
    """Retrieve the top‑k corpus IDs for each query."""
    # Load model (cached)
    model_load_start = time.time()
    model = _load_model(model_name)
    model_load_time = time.time() - model_load_start

    query_ids = list(queries.keys())
    raw_texts = [queries[qid]["text"] for qid in query_ids]
    prefixed = _prepare_texts(raw_texts, "query:")

    # Cache handling for query embeddings
    cache_file = _cache_path("queries", model_name, split)
    query_emb = _load_or_encode_embeddings(model, prefixed, cache_file, batch_size=batch_size)

    # Ensure dimensions match index
    if query_emb.shape[1] != index.d:
        if query_emb.shape[1] < index.d:
            pad_width = index.d - query_emb.shape[1]
            query_emb = np.pad(query_emb, ((0, 0), (0, pad_width)), mode='constant')
        else:
            query_emb = query_emb[:, : index.d]

    # Search
    search_start = time.time()
    distances, faiss_ids = index.search(query_emb, top_k)
    search_time = time.time() - search_start

    results: Dict[str, List[str]] = {}
    for q_idx, qid in enumerate(query_ids):
        retrieved_ids = [id_map[int_id] for int_id in faiss_ids[q_idx] if int_id != -1]
        results[qid] = retrieved_ids

    print(f"Model load (query side): {model_load_time:.2f}s, search: {search_time:.2f}s")
    return results


def evaluate_dense_e0(
    load_coir_func,
    model_name: str = "intfloat/e5-base-v2",
    batch_size: int = 32,
    top_k: int = 100,
    limit: int | None = None,
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], Dict[str, List[str]]]:
    """Run the dense retrieval baseline on the test split and return the results.

    Parameters
    ----------
    load_coir_func : callable
        Function to load the CoIR dataset.
    limit : int or None, optional
        If set, run a quick smoke‑test using only the first ``limit`` queries
        and a matching subset of the corpus.

    Returns
    -------
    tuple
        ``(queries, corpus, retrieved)`` where ``retrieved`` maps query IDs to
        lists of retrieved corpus IDs.
    """
    queries, corpus, qrels = load_coir_func(split="test")

    if limit is not None:
        selected_qids = sorted(queries.keys())[:limit]
        queries = {qid: queries[qid] for qid in selected_qids}
        selected_cids = sorted(corpus.keys())[:limit]
        corpus = {cid: corpus[cid] for cid in selected_cids}

    # Build index (model will be cached inside)
    index, id_map = build_corpus_index(corpus, model_name=model_name, batch_size=batch_size, split="test")
    retrieved = retrieve_top_k(
        queries,
        index,
        id_map,
        model_name=model_name,
        batch_size=batch_size,
        top_k=top_k,
        split="test",
    )
    return queries, corpus, retrieved
