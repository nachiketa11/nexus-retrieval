import os
import time
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .schemas import (
    AgentRequest,
    RetrieveRequest,
    RetrieveResponse,
    CodeResult,
    HealthResponse,
    InfoResponse,
)
from ..data.coir import load_coir
from ..data.samsung_demo import load_samsung_demo_data
from ..agent import NexusAgent
from ..retrieval.dense import DenseE5Retriever
from ..retrieval.bm25 import BM25Retriever
from ..retrieval.hybrid import hybrid_retrieve
from ..retrieval.version_aware import VersionAwareFilter, parse_version_intent
from ..ranking.reranker import CodeReranker
from ..config.settings import get_settings
from ..utils.logging import get_logger

logger = get_logger("nexus.api")
START_TIME = time.time()

# Global state for loaded models and indexed corpora
STATE: Dict[str, Any] = {
    "corpus": {},
    "dense_retriever": None,
    "bm25_retriever": None,
    "reranker": None,
    "version_filter": None,
    "loaded": False,
    "dataset": None,
}

SUPPORTED_DATASETS = {"coir", "samsung_demo"}


def _configured_dataset() -> str:
    dataset = os.getenv("NEXUS_DATASET", "coir").strip().lower()
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"NEXUS_DATASET must be one of: {', '.join(sorted(SUPPORTED_DATASETS))}")
    return dataset


def load_system_indexes(dataset: str = "coir", force: bool = False):
    """Load models and build indices once at startup."""
    logger.info(f"Loading system indices (Dataset: {dataset})...")
    settings = get_settings()

    dataset = dataset.strip().lower()
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset '{dataset}'. Choose coir or samsung_demo.")
    STATE["loaded"] = False
    if dataset == "samsung_demo":
        _, corpus = load_samsung_demo_data()
    else:
        _, corpus, _ = load_coir(split="test")

    # Build Dense Index
    dense_retriever = DenseE5Retriever(
        model_name=settings.dense_model_name,
        batch_size=settings.dense_batch_size,
    )
    dense_retriever.build_index(corpus, force_rebuild=force)

    # Build BM25 Index
    bm25 = BM25Retriever(k1=settings.bm25_k1, b=settings.bm25_b)
    bm25.fit(corpus)
    # Publish the complete state together so requests never see partial indexes.
    STATE.update({
        "corpus": corpus,
        "dense_retriever": dense_retriever,
        "bm25_retriever": bm25,
        "reranker": CodeReranker(model_name=settings.reranker_model_name),
        "version_filter": VersionAwareFilter(),
        "dataset": dataset,
    })
    STATE["loaded"] = True
    logger.info("System models and indices loaded successfully!")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI Lifespan handler: initialize models once at server startup."""
    if not STATE["loaded"]:
        load_system_indexes(dataset=_configured_dataset(), force=False)
    yield
    logger.info("Shutting down Nexus Retrieval API server...")


app = FastAPI(
    title="Nexus Code Retrieval API",
    description="Production-grade Code Retrieval & Reranking Service for Samsung PRISM",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health():
    """Health check endpoint."""
    uptime = time.time() - START_TIME
    status = "ok" if STATE["loaded"] else "not_ready"
    payload = HealthResponse(
        status=status,
        version="1.0.0",
        uptime_seconds=round(uptime, 2),
    )
    if not STATE["loaded"]:
        return JSONResponse(status_code=503, content=payload.model_dump())
    return payload


@app.get("/info", response_model=InfoResponse)
def info():
    """System information endpoint."""
    settings = get_settings()
    return InfoResponse(
        system="Nexus Code Retrieval Engine",
        dense_model=settings.dense_model_name,
        reranker_model=settings.reranker_model_name,
        indexed_corpus_count=len(STATE.get("corpus", {})),
        device=settings.device,
    )


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(req: RetrieveRequest):
    """Retrieve top-K ranked code snippets for a query."""
    if not STATE["loaded"]:
        raise HTTPException(status_code=503, detail="Index not loaded yet.")

    requested_dataset = req.dataset.strip().lower()
    if requested_dataset not in SUPPORTED_DATASETS:
        raise HTTPException(status_code=422, detail="dataset must be 'coir' or 'samsung_demo'.")
    if requested_dataset != STATE["dataset"]:
        raise HTTPException(
            status_code=409,
            detail=(f"API is initialized for '{STATE['dataset']}'. Set NEXUS_DATASET={requested_dataset} "
                    "and restart, or rebuild indexes for that dataset before querying."),
        )

    start_time = time.time()
    corpus = STATE["corpus"]
    query_text = req.query
    query_obj = {"q1": {"text": query_text}}
    method = req.method.lower()
    rerank_enabled = req.rerank or method == "hybrid-rerank"

    # 1. Base Retrieval
    candidate_cids: List[str] = []

    if method not in {"dense", "bm25", "hybrid", "hybrid-rerank"}:
        raise HTTPException(status_code=422, detail="method must be dense, bm25, hybrid, or hybrid-rerank.")
    if method == "dense":
        res = STATE["dense_retriever"].retrieve(query_obj, top_k=50)
        candidate_cids = res.get("q1", [])

    elif method == "bm25":
        res = STATE["bm25_retriever"].retrieve_batch(query_obj, top_k=50)
        candidate_cids = res.get("q1", [])

    else:  # hybrid or hybrid-rerank
        dense_res = STATE["dense_retriever"].retrieve(query_obj, top_k=50)
        bm25_res = STATE["bm25_retriever"].retrieve_batch(query_obj, top_k=50)
        fused = hybrid_retrieve(dense_res, bm25_res, top_k=50)
        candidate_cids = fused.get("q1", [])

    # 2. Reranking (before the metadata policy so version/deprecation boosts are not discarded)
    scores: Dict[str, float] = {}
    if rerank_enabled and candidate_cids:
        head = candidate_cids[: max(req.top_k * 3, 30)]
        reranked_pairs = STATE["reranker"].rerank_query(query_text, head, corpus, top_k=len(head))
        scores = {cid: float(score) for cid, score in reranked_pairs}
        candidate_cids = [cid for cid, _ in reranked_pairs] + candidate_cids[len(head):]

    # 3. Version Filtering / Boosting
    v_filter: VersionAwareFilter = STATE["version_filter"]
    if req.version:
        candidate_cids = v_filter.filter_candidates(candidate_cids, corpus, version=req.version)
    elif parse_version_intent(query_text):
        boosted = v_filter.rerank_by_metadata_match(candidate_cids, corpus, query_text)
        candidate_cids = [cid for cid, _ in boosted]

    results: List[CodeResult] = []
    for rank_idx, cid in enumerate(candidate_cids[: req.top_k], start=1):
        doc = corpus.get(cid, {})
        meta = doc.get("meta_information") or doc.get("metadata") or {}
        results.append(
            CodeResult(
                doc_id=cid,
                rank=rank_idx,
                score=scores.get(cid, 1.0 / rank_idx),
                text=doc.get("text", ""),
                metadata=meta,
            )
        )

    elapsed_ms = (time.time() - start_time) * 1000

    return RetrieveResponse(
        query=query_text,
        method=method,
        rerank=rerank_enabled,
        version=req.version,
        total=len(results),
        latency_ms=round(elapsed_ms, 2),
        results=results,
    )


@app.post("/agent")
def agent(req: AgentRequest):
    """Agentic retrieval: analyse, plan, retrieve, resolve versions/deprecations, self-check, refine."""
    if not STATE["loaded"]:
        raise HTTPException(status_code=503, detail="Index not loaded yet.")
    corpus = STATE["corpus"]
    dense_retriever = STATE["dense_retriever"]
    reranker = STATE["reranker"]

    def dense_fn(query: str, k: int):
        ranked = dense_retriever.retrieve({"q": {"text": query}}, top_k=k).get("q", [])
        return [(cid, 1.0 / rank) for rank, cid in enumerate(ranked, start=1)]

    def rerank_fn(query: str, ids):
        return reranker.rerank_query(query, list(ids), corpus, top_k=len(ids))

    nexus_agent = NexusAgent(
        corpus,
        bm25=STATE["bm25_retriever"],
        dense=dense_fn if req.use_dense else None,
        reranker=rerank_fn if req.rerank is not False else None,
    )
    return nexus_agent.run(req.query, top_k=req.top_k, version=req.version, use_rerank=req.rerank)


@app.post("/index/rebuild")
def rebuild_index(background_tasks: BackgroundTasks, dataset: str = "coir"):
    """Trigger asynchronous index rebuilding."""
    dataset = dataset.strip().lower()
    if dataset not in SUPPORTED_DATASETS:
        raise HTTPException(status_code=422, detail="dataset must be 'coir' or 'samsung_demo'.")
    STATE["loaded"] = False
    background_tasks.add_task(load_system_indexes, dataset=dataset, force=True)
    return {"status": "accepted", "message": f"Index rebuild triggered for dataset '{dataset}'"}
