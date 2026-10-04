# NEXUS — Agentic Code Intelligence

NEXUS is an **agentic code retrieval system**. Given a developer's question, the NEXUS agent analyses the intent, plans which retrieval tools to use, searches with lexical and semantic retrievers, applies SDK version and deprecation policy, follows migration links from legacy APIs to their current replacements, checks its own confidence, refines the query when that confidence is low, and returns ranked code together with a step-by-step reasoning trace.

**Live demo:** https://nexus-retrieval-two.vercel.app

## PRISM GenAI Hackathon Submission

**Hackathon Tag:** `PRISM_GENAI_HACKATHON_Y2026` · **Theme:** Agentic AI

- **Live demo:** https://nexus-retrieval-two.vercel.app (API docs at `/api/docs`)
- **Demo Video:** [Watch the demo on YouTube](https://youtu.be/zXi2QaELKTA)
- **Presentation:** [View the PRISM submission PPT](docs/Nexus_Retrieval_PRISM_Submission.pptx)
- **AI Disclosure:** [View the AI Usage Disclosure Form](<docs/AI Usage DISCLOSURE FORM.docx>)
- **Source Code:** this repository (`requirements.txt` for the full stack; `scripts/build_vercel.py` for the serverless build).

## Problem

Finding a useful code example means matching intent expressed in natural language, preserving exact identifiers, and respecting the library or SDK version the developer is on. Real SDKs (Samsung Health, Knox, SmartThings, Galaxy Store IAP, Galaxy Watch) evolve: APIs get deprecated and replaced. A plain search engine happily returns the deprecated v1 snippet because it matches the words best. A single retrieval method also misses one of the signals — semantic intent or exact tokens.

## Solution: a retrieval agent, not just a pipeline

```text
                 ┌──────────── NEXUS agent loop ─────────────────────────────────────────┐
 query ──► ANALYSE ──► PLAN ──► ACT ─────────────────────────────► REFLECT ──► RESPOND ──┼─► answer + ranked code
           intent,     choose   bm25 · dense E5 · RRF fusion ·     confidence    migration │   + reasoning trace
           version,    tools    cross-encoder · version policy ·   from signals  path,     │
           language,            context boost · deprecation          │           warnings │
           SDK, ids             resolver                             ▼                    │
                                  ▲                      low confidence? REFINE ───────────┤
                                  └────────── query expansion (PRF) / relax version filter ┘
```

| Phase | What the agent does |
| :--- | :--- |
| **Analyse** | Parses target version vs. *source* version (in "migrate from v1" the v1 is what you are leaving), deprecation / replacement / "current" intent, programming language, SDK family and code identifiers. |
| **Plan** | Picks tools from the analysis and what is available: BM25 always; E5 dense + RRF when models are loaded; the cross-encoder for natural-language queries; the version policy only for version-sensitive queries; a context boost when a language or SDK is named. |
| **Act** | Runs each tool and logs inputs, top candidates and latency. |
| **Resolve** | The deprecation resolver follows `metadata.replacement` links, so a deprecated hit surfaces its current successor (and the full migration path) — unless the user explicitly asked for the old version or an exact deprecated symbol. |
| **Reflect** | Self-assesses confidence from query-term coverage, dense/lexical agreement, cross-encoder (or BM25) margin over the best competitor, and metadata consistency. |
| **Refine** | Below the threshold it rewrites the query with pseudo-relevance feedback, or relaxes a version filter that eliminated every candidate, and runs another iteration. Low-confidence answers are labelled as such. |

The tools are injected callables, so the same agent runs on the PyTorch/FAISS stack (`src/api`), on the ONNX serverless stack (`web/`, deployed), or BM25-only with no models at all.

### Retrieval tools

- **BM25** (`rank-bm25`) over a code-aware tokenizer that keeps full identifiers and splits snake_case / camelCase.
- **Dense E5-base-v2** with E5 `query:` / `passage:` prefixes — FAISS inner product (PyTorch stack) or exact NumPy search (ONNX stack).
- **Reciprocal-rank fusion** (k = 60).
- **Cross-encoder** `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- **Version-aware policy**: hard filter for an explicit version (`"3"` matches `3.0`, `3.1`); soft boosts for parsed intent.

## Evaluation

### Agent vs. pipelines (Samsung/demo)

24 labelled queries over the 35-snippet Samsung/demo corpus — including version-targeted, deprecated-API, migration and plain functional queries. Reproduce with `python scripts/eval_agent.py --onnx`.

| System | Hit@1 | MRR | NDCG@10 | Recall@5 |
| :--- | :---: | :---: | :---: | :---: |
| BM25 | 0.792 | 0.882 | 0.912 | 1.000 |
| Dense E5 | 0.750 | 0.868 | 0.902 | 1.000 |
| Hybrid RRF | 0.708 | 0.847 | 0.887 | 1.000 |
| Hybrid RRF + cross-encoder | 0.792 | 0.896 | 0.923 | 1.000 |
| **NEXUS agent** | **1.000** | **1.000** | **1.000** | **1.000** |

All plain methods find the right snippet in the top 5; the misses at rank 1 are version and migration queries where they rank deprecated code first. The agent's version policy and deprecation resolver fix exactly those. This set is small and hand-authored by the team, so treat it as a behavioural test of version/deprecation reasoning, not as a public benchmark. With BM25 as its only retriever (no neural models) the agent scores 0.958 Hit@1 (23/24); the miss is a migration query whose words match an unrelated legacy snippet, which the semantic tools resolve.

### CoIR AppsRetrieval (official benchmark)

CoIR AppsRetrieval test split: 3,765 queries over 8,765 code documents. Full-split metrics have **not yet been measured** for this submission, because encoding the corpus on CPU did not finish in the available time. Two runners are provided, and each writes a JSON and Markdown report under `results/` that records dataset, split, query count, settings, environment and timings:

```bash
python scripts/run_benchmark_onnx.py --rerank-limit 500    # ONNX stack (same models as the live demo)
python scripts/run_benchmark.py --dataset coir --split test # PyTorch / FAISS stack
```

Only numbers produced by an actual run with its recorded configuration should be reported here. Reduced-query runs (`--limit`) must not be compared to full-split runs.

## Repository structure

```text
src/
  agent/           NexusAgent: analyse → plan → act → reflect → refine loop with reasoning trace
  api/             FastAPI application (PyTorch/FAISS stack): /retrieve, /agent, /health, /info
  cli/             `python -m src.cli` commands: index, retrieve, agent, evaluate
  config/          Environment-backed settings
  data/            CoIR loader and the synthetic Samsung/demo corpus with version metadata
  evaluation/      Metrics and benchmark runner
  indexing/        Code chunker (utility)
  ranking/         Cross-encoder reranker (sentence-transformers)
  retrieval/       BM25, dense E5 (FAISS), ONNX E5 / cross-encoder backends, RRF, version-aware policy
  serving/         Torch-free engine used by the deployed web app
web/               Serverless FastAPI app (`web/app.py`) and UI (`web/public/index.html`)
demo/              Streamlit application
scripts/           Benchmarks, agent evaluation, Vercel build
tests/             Pytest suite
```

## Run it

### Web app (agent UI + API) — what is deployed

```bash
python -m pip install fastapi uvicorn numpy rank-bm25 onnxruntime tokenizers huggingface_hub
python scripts/build_vercel.py          # downloads ONNX models into models/, precomputes embeddings
uvicorn web.app:app --reload            # http://localhost:8000
```

Without the ONNX models the web app still runs in lexical mode (BM25 + agent policies). API routes: `GET /api/health`, `GET /api/info`, `GET /api/examples`, `POST /api/search`, `POST /api/agent`; interactive docs at `/api/docs`.

```bash
curl -X POST http://localhost:8000/api/agent -H "Content-Type: application/json" \
  -d '{"query": "migrate legacy v1 step count reader to the new API", "top_k": 5}'
```

### Deploy to Vercel

```bash
python scripts/build_vercel.py
cd .vercel-build
vercel deploy --prod
```

The full `requirements.txt` (PyTorch, sentence-transformers, FAISS) exceeds the serverless bundle limit, so the deployment runs the same E5-base-v2 and MS MARCO cross-encoder weights as int8 ONNX models via ONNX Runtime (~135 MB bundle before dependencies). Corpus embeddings are precomputed at build time; the first neural request on a cold instance loads the models (a few seconds).

### Full stack (PyTorch / FAISS)

```bash
python -m venv .venv
python -m pip install -r requirements.txt
uvicorn src.api.app:app --host 0.0.0.0 --port 8000           # NEXUS_DATASET=samsung_demo for the demo corpus
streamlit run demo/app.py
```

The API initializes one dataset at startup (`NEXUS_DATASET`, default `coir`); requests for another dataset return HTTP 409. Routes: `GET /health`, `GET /info`, `POST /retrieve`, `POST /agent`, `POST /index/rebuild`.

### CLI

```bash
python -m src.cli agent --query "Find deprecated authentication API and its replacement"
python -m src.cli agent --query "Knox camera v2 java" --lexical
python -m src.cli retrieve --query "parse JSON response" --method bm25 --dataset samsung_demo
python -m src.cli evaluate --dataset samsung_demo
```

`agent` and BM25 `retrieve` on the demo corpus do not import PyTorch; dense and CoIR commands load the full stack on demand.

### Benchmarks and tests

```bash
python scripts/eval_agent.py --onnx                 # agent vs pipelines on Samsung/demo
python scripts/run_benchmark_onnx.py                # CoIR AppsRetrieval on the ONNX stack
python scripts/run_benchmark.py --dataset coir      # CoIR AppsRetrieval on the PyTorch/FAISS stack
pytest
```

`test_api.py` and the dense tests need the PyTorch stack; `test_web.py` runs its neural test only when the ONNX models are present; `test_coir_loader.py` needs network access to the dataset.

## What changed in this submission round

- **Agentic layer** (`src/agent`): planning, tool use, deprecation-link resolution, self-assessed confidence and query refinement, with a full reasoning trace in the API, CLI and UI.
- **Version-aware fixes**: replacement intent previously boosted the *deprecated* documents (the ones carrying a `replacement` field) instead of their successors; "migrate from v1" was parsed as a request *for* v1, and the Streamlit demo hard-filtered on it, returning only legacy code. Version matching now treats `3` / `3.0` / `3.0.0` as equal and `3` as matching `3.1`.
- **API fixes**: cross-encoder reranking ran after, and discarded, the version policy; `rerank` defaulted to `true`, so even `method=bm25` loaded the cross-encoder; reranking now keeps the candidates below the reranked head; new `/agent` route.
- **Serverless deployment**: ONNX Runtime backends (`src/retrieval/onnx_models.py`), a torch-free engine (`src/serving`), web UI and Vercel build script. Settings no longer crash on read-only filesystems.
- **Demo corpus** expanded from 5 to 35 snippets across Samsung Health, Knox, SmartThings, Galaxy Store IAP, Tizen → Wear OS and common utilities, with 7 deprecation → replacement links and 24 labelled queries.
- The CLI imports PyTorch/FAISS/MTEB lazily; the CoIR loader works without MTEB installed.

## Dataset distinction

- **CoIR AppsRetrieval** is the external benchmark corpus and relevance set loaded by `src/data/coir.py`.
- **Samsung/demo** is a small hand-authored synthetic corpus in `src/data/samsung_demo.py`. The SDK calls are illustrative, it is not an official Samsung dataset, and it does not imply Samsung endorsement.

## Cache behaviour

Dense corpus embeddings are cached under `cache/` as NumPy files keyed by a SHA-256 fingerprint of model, configuration, corpus IDs, text and order. BM25 and FAISS indexes are built in memory. Models, caches, results and the Vercel build output are excluded from Git.

## Limitations and future work

- The agent's planner and policies are deterministic and auditable; an LLM planner (e.g. to decompose multi-part questions or generate a migration diff) is a natural next step.
- The deployed demo serves the Samsung/demo corpus; serving CoIR-scale corpora needs a persisted vector index.
- Indexes are rebuilt in memory per process; persisting FAISS/BM25 artifacts is future work.
- `CodeChunker` is not yet wired into ingestion.
