from .bm25 import BM25Retriever, tokenize_code
from .hybrid import reciprocal_rank_fusion, hybrid_retrieve
from .version_aware import VersionAwareFilter, parse_version_intent


def __getattr__(name):
    # DenseE5Retriever pulls in torch, sentence-transformers and FAISS; import it
    # only when requested so lightweight (ONNX/serverless) entry points stay small.
    if name == "DenseE5Retriever":
        from .dense import DenseE5Retriever
        return DenseE5Retriever
    raise AttributeError(name)


__all__ = [
    "DenseE5Retriever",
    "BM25Retriever",
    "tokenize_code",
    "reciprocal_rank_fusion",
    "hybrid_retrieve",
    "VersionAwareFilter",
    "parse_version_intent",
]
