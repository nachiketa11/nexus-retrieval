"""NEXUS web API + UI (serverless entrypoint, deployed to Vercel).

Serves the static UI from ``public/`` and a JSON API under ``/api``:

    GET  /api/health     liveness
    GET  /api/info       models, corpus size, available methods
    GET  /api/examples   labelled example queries
    POST /api/search     classic pipeline (bm25 | dense | hybrid, optional rerank, version filter)
    POST /api/agent      agentic retrieval with a full reasoning trace

Run locally from the repository root:

    uvicorn web.app:app --reload
"""

import sys
import time
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
# Inside the Vercel bundle `src/` sits next to this file; in the repository it is one level up.
for candidate in (ROOT, ROOT.parent):
    if (candidate / "src" / "__init__.py").exists():
        sys.path.insert(0, str(candidate))
        break

from src.serving import NexusEngine  # noqa: E402

START = time.time()
app = FastAPI(title="NEXUS — Agentic Code Intelligence", version="2.0.0",
              docs_url="/api/docs", openapi_url="/api/openapi.json")
_engine: Optional[NexusEngine] = None


def engine() -> NexusEngine:
    global _engine
    if _engine is None:
        _engine = NexusEngine()
    return _engine


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    method: Literal["bm25", "dense", "hybrid"] = "hybrid"
    rerank: bool = False
    top_k: int = Field(default=5, ge=1, le=20)
    version: Optional[str] = Field(default=None, max_length=20)


class AgentRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(default=5, ge=1, le=20)
    version: Optional[str] = Field(default=None, max_length=20)
    use_models: bool = True
    rerank: Optional[bool] = None


def _clean_version(version: Optional[str]) -> Optional[str]:
    version = (version or "").strip().lstrip("vV")
    return version or None


@app.get("/api/health")
def health():
    return {"status": "ok", "uptime_seconds": round(time.time() - START, 2)}


@app.get("/api/info")
def info():
    return engine().info()


@app.get("/api/examples")
def examples():
    return engine().examples()


@app.post("/api/search")
def search(req: SearchRequest):
    try:
        return engine().search(req.query.strip(), method=req.method, rerank=req.rerank,
                               top_k=req.top_k, version=_clean_version(req.version))
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.post("/api/agent")
def agent(req: AgentRequest):
    return engine().agent(req.query.strip(), top_k=req.top_k, version=_clean_version(req.version),
                          use_models=req.use_models, rerank=req.rerank)


@app.get("/", include_in_schema=False)
def index():
    # On Vercel `public/index.html` is served from the CDN; this covers local runs.
    return FileResponse(ROOT / "public" / "index.html")
