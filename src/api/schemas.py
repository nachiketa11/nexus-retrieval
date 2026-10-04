from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field


class RetrieveRequest(BaseModel):
    query: str = Field(..., description="Natural language code search query", json_schema_extra={"example": "parse JSON response"})
    top_k: int = Field(default=10, ge=1, le=100, description="Number of results to return")
    method: Literal["dense", "bm25", "hybrid", "hybrid-rerank"] = Field(
        default="hybrid-rerank",
        description="Retrieval method: dense, bm25, hybrid, hybrid-rerank",
    )
    rerank: bool = Field(default=False, description="Enable cross-encoder reranking (implied by hybrid-rerank)")
    version: Optional[str] = Field(default=None, description="Filter by library/SDK version")
    dataset: Literal["coir", "samsung_demo"] = Field(default="coir", description="Target corpus: coir or samsung_demo")


class AgentRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural language code search query")
    top_k: int = Field(default=5, ge=1, le=50)
    version: Optional[str] = Field(default=None, description="Hard version constraint")
    use_dense: bool = Field(default=True, description="Allow the agent to use dense retrieval")
    rerank: Optional[bool] = Field(default=None, description="Force the cross-encoder on/off; None lets the agent decide")


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
