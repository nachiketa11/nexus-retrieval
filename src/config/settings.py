import os
from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = Path(os.getenv("NEXUS_CACHE_DIR", BASE_DIR / "cache"))
INDEX_DIR = BASE_DIR / "indexes"
RESULTS_DIR = BASE_DIR / "results"

for _directory in (CACHE_DIR, INDEX_DIR, RESULTS_DIR):
    try:
        _directory.mkdir(parents=True, exist_ok=True)
    except OSError:
        # Read-only filesystems (e.g. serverless bundles) only need these for offline runs.
        pass


@dataclass
class NexusSettings:
    """Centralized configuration for Nexus Retrieval System."""

    # Dense retrieval
    dense_model_name: str = os.getenv("DENSE_MODEL_NAME", "intfloat/e5-base-v2")
    dense_batch_size: int = int(os.getenv("DENSE_BATCH_SIZE", "32"))
    device: str = os.getenv("DEVICE", "cpu")

    # BM25 settings
    bm25_k1: float = float(os.getenv("BM25_K1", "1.5"))
    bm25_b: float = float(os.getenv("BM25_B", "0.75"))

    # RRF settings
    rrf_k: int = int(os.getenv("RRF_K", "60"))
    candidate_count: int = int(os.getenv("CANDIDATE_COUNT", "100"))

    # Reranker settings
    reranker_model_name: str = os.getenv("RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    use_reranker: bool = os.getenv("USE_RERANKER", "false").lower() in ("true", "1", "yes")

    # General retrieval
    top_k: int = int(os.getenv("TOP_K", "10"))

    # Paths
    base_dir: Path = BASE_DIR
    cache_dir: Path = CACHE_DIR
    index_dir: Path = INDEX_DIR
    results_dir: Path = RESULTS_DIR

    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


def get_settings() -> NexusSettings:
    """Return default settings instance."""
    return NexusSettings()
