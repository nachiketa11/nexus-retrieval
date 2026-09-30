import os
import time
import hashlib
from typing import Dict, List, Tuple, Optional
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from ..config.settings import get_settings
from ..utils.logging import get_logger

logger = get_logger("nexus.dense")
_MODEL_CACHE: Dict[str, SentenceTransformer] = {}


def _load_model(model_name: str = "intfloat/e5-base-v2") -> SentenceTransformer:
    """Load SentenceTransformer model, using in-memory cache if available."""
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]
    logger.info(f"Loading SentenceTransformer model: {model_name}")
    model = SentenceTransformer(model_name)
    _MODEL_CACHE[model_name] = model
    return model


def _prepare_texts(texts: List[str], prefix: str) -> List[str]:
    """Prepend E5 prefix to texts ('query: ' or 'passage: ')."""
    return [f"{prefix} {t}" for t in texts]


def _cache_path(name: str, model_name: str, split: str, fingerprint: str = "") -> str:
    """Generate a deterministic cache path including source/config identity."""
    settings = get_settings()
    safe_model = model_name.replace("/", "_").replace("\\", "_")
    suffix = f"_{fingerprint}" if fingerprint else ""
    return str(settings.cache_dir / f"{name}_{safe_model}_{split}{suffix}.npy")


def _corpus_fingerprint(corpus: Dict[str, Dict], model_name: str, batch_size: int) -> str:
    """Hash ordered IDs, exact passage text, and embedding configuration."""
    digest = hashlib.sha256()
    digest.update(f"model={model_name}\0batch_size={batch_size}\0prefix=passage:\0".encode())
    for corpus_id in corpus:
        digest.update(corpus_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(corpus[corpus_id].get("text", "")).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()[:20]


class DenseE5Retriever:
    """Dense retriever using E5-base-v2 and FAISS IndexIDMap."""

    def __init__(
        self,
        model_name: str = "intfloat/e5-base-v2",
        batch_size: int = 32,
        split: str = "test",
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.split = split
        self.index: Optional[faiss.IndexIDMap] = None
        self.id_map: Dict[int, str] = {}
        self.corpus_ids: List[str] = []

    def build_index(
        self,
        corpus: Dict[str, Dict],
        force_rebuild: bool = False,
    ) -> Tuple[faiss.IndexIDMap, Dict[int, str]]:
        """Encode corpus and build FAISS index."""
        model = _load_model(self.model_name)

        # Sort corpus IDs deterministically
        self.corpus_ids = sorted(corpus.keys())
        self.id_map = {i: cid for i, cid in enumerate(self.corpus_ids)}

        texts = [
            _prepare_texts([corpus[cid].get("text", "")], "passage:")[0]
            for cid in self.corpus_ids
        ]

        fingerprint = _corpus_fingerprint(corpus, self.model_name, self.batch_size)
        cache_file = _cache_path("corpus", self.model_name, self.split, fingerprint)

        embeddings = None
        if not force_rebuild and os.path.exists(cache_file):
            cached = np.load(cache_file, allow_pickle=False)
            if cached.ndim == 2 and cached.shape[0] == len(texts):
                logger.info(f"Loaded corpus embeddings from cache: {os.path.basename(cache_file)}")
                embeddings = cached.astype(np.float32)

        if embeddings is None:
            logger.info(f"Encoding {len(texts)} corpus passages with {self.model_name}...")
            start = time.time()
            raw_emb = model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            embeddings = np.asarray(raw_emb, dtype=np.float32)
            logger.info(f"Encoding complete in {time.time() - start:.2f}s. Saving cache...")
            np.save(cache_file, embeddings)

        dim = embeddings.shape[1]
        quantizer = faiss.IndexFlatIP(dim)
        self.index = faiss.IndexIDMap(quantizer)
        ids_array = np.array(list(self.id_map.keys()), dtype=np.int64)
        self.index.add_with_ids(embeddings, ids_array)

        return self.index, self.id_map

    def retrieve(
        self,
        queries: Dict[str, Dict],
        top_k: int = 100,
    ) -> Dict[str, List[str]]:
        """Retrieve top-k document IDs for queries dict."""
        if self.index is None:
            raise RuntimeError("FAISS index must be built before retrieval.")

        model = _load_model(self.model_name)
        qids = list(queries.keys())
        raw_query_texts = [queries[qid].get("text", "") for qid in qids]
        prefixed_queries = _prepare_texts(raw_query_texts, "query:")

        logger.info(f"Encoding {len(qids)} queries for dense retrieval...")
        query_emb = model.encode(
            prefixed_queries,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        query_emb = np.asarray(query_emb, dtype=np.float32)

        # Handle dimension mismatch if any
        if query_emb.shape[1] != self.index.d:
            if query_emb.shape[1] < self.index.d:
                pad = self.index.d - query_emb.shape[1]
                query_emb = np.pad(query_emb, ((0, 0), (0, pad)), mode="constant")
            else:
                query_emb = query_emb[:, : self.index.d]

        distances, faiss_ids = self.index.search(query_emb, top_k)

        results: Dict[str, List[str]] = {}
        for q_idx, qid in enumerate(qids):
            retrieved_cids = [
                self.id_map[int(f_id)] for f_id in faiss_ids[q_idx] if f_id != -1
            ]
            results[qid] = retrieved_cids

        return results
