from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field


class RetrieveRequest(BaseModel):
    query: str = Field(..., description="Natural language code search query", json_schema_extra={"example": "parse JSON response"})
    top_k: int = Field(default=10, ge=1, le=100, description="Number of results to return")
    method: Literal["dense", "bm25", "hybrid", "hybrid-rerank"] = Field(
        default="hybrid-rerank",
        description="Retrieval method: dense, bm25, hybrid, hybrid-rerank",
    )
    rerank: bool = Field(default=True, description="Enable cross-encoder reranking")
    version: Optional[str] = Field(default=None, description="Filter by library/SDK version")
    dataset: Literal["coir", "samsung_demo"] = Field(default="coir", description="Target corpus: coir or samsung_demo")


class CodeResult(BaseModel):
    doc_id: str
    rank: int
    score: float
    text: str
    metadata: Dict[str, Any]


class RetrieveResponse(BaseModel):
    query: str
    method: str
    rerank: bool
    version: Optional[str]
    total: int
    latency_ms: float
    results: List[CodeResult]


class HealthResponse(BaseModel):
    status: str
    version: str
    uptime_seconds: float


class InfoResponse(BaseModel):
    system: str
    dense_model: str
    reranker_model: str
    indexed_corpus_count: int
    device: str
