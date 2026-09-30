# NEXUS — Agentic Code Intelligence

NEXUS is a code intelligence retrieval system that combines semantic search, code-aware lexical search, rank fusion, optional cross-encoder reranking, and metadata-based version matching. It provides Python API and CLI entry points, a Streamlit demonstration, and an evaluation runner.

## Problem

Finding a useful code example often requires matching intent expressed in natural language while preserving exact code identifiers and respecting library or SDK version context. A single retrieval method may miss one of these signals.

## Solution

NEXUS exposes complementary retrieval methods through one pipeline. Dense retrieval handles semantic similarity, BM25 matches identifiers and tokens, reciprocal-rank fusion combines their candidate rankings, and an optional cross-encoder reorders candidates. A metadata-aware stage can filter or boost candidates for version and deprecation intent.

## PRISM GenAI Hackathon Submission

**Hackathon Tag:** `PRISM_GENAI_HACKATHON_Y2026`

### Submission Resources

- **Demo Video:** [Watch the demo on YouTube](https://youtu.be/zN9a6G7nTWU)
- **Presentation:** [View the PRISM submission PPT](docs/Nexus_Retrieval_PRISM_Submission.pptx)
- **AI Disclosure:** [View the AI Usage Disclosure Form](<docs/AI Usage DISCLOSURE FORM.docx>)
- **Source Code:** This GitHub repository contains the project code and `requirements.txt`.

## Architecture

```text
Query
  ├─ Dense E5 embeddings ─ FAISS inner-product search ─┐
  └─ Code-aware tokens ─ BM25 search ─────────────────┤
                                                       └─ RRF fusion
                                                            └─ optional cross-encoder reranking
                                                                 └─ version metadata filtering/boosting
```

The API, CLI, demo, and benchmark use the implementation under `src/`. The dense retriever encodes passages and queries with E5 prefixes and stores vectors in a FAISS index. BM25 and dense retrieval produce candidates; the hybrid path fuses their ranked lists.

See [docs/architecture.md](docs/architecture.md) for the online query path and current indexing boundaries.

### Retrieval pipeline

- **Dense E5 + FAISS:** normalized SentenceTransformer embeddings are searched with a FAISS inner-product index.
- **BM25:** `rank-bm25` ranks tokenized code and identifiers.
- **Hybrid RRF:** reciprocal-rank fusion merges dense and BM25 rankings.
- **Reranking:** a Sentence Transformers cross-encoder scores query and candidate-text pairs.
- **Version-aware retrieval:** metadata can filter by explicit version or adjust ranking for parsed version, deprecated, and replacement intent.

## Repository structure

```text
src/
  api/             FastAPI application and schemas
  cli/             `python -m src.cli` commands
  config/          Environment-backed settings
  data/            CoIR loader and standalone demonstration corpus
  evaluation/      Metrics and benchmark runner
  indexing/        Code chunker
  ranking/         Cross-encoder reranker
  retrieval/       Dense, BM25, hybrid, and version-aware retrieval
demo/              Streamlit application
scripts/            Benchmark and dataset utilities
tests/              Pytest suite
Dockerfile          API container definition
requirements.txt    Runtime and test dependencies
```

## Requirements

Python 3.11 is the container runtime. Install the packages in `requirements.txt` for the API, CLI, demo, benchmark, and tests. Dense retrieval and reranking use PyTorch and Sentence Transformers; the first use may download models. CoIR loading uses Hugging Face `datasets` and MTEB and may download dataset files.

## Installation

From the repository root:

```bash
python -m venv .venv
# Activate the environment using the command for your operating system.
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The default dense model is `intfloat/e5-base-v2`; the reranker is `cross-encoder/ms-marco-MiniLM-L-6-v2`. Settings can be changed with the environment variables defined in `src/config/settings.py`, including `DENSE_MODEL_NAME`, `RERANKER_MODEL_NAME`, `DEVICE`, `TOP_K`, and `CANDIDATE_COUNT`.

## Run the API

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

The API initializes for one dataset at startup. By default it loads CoIR; set `NEXUS_DATASET=samsung_demo` before launch to select the small demo corpus. A retrieval request must name the active dataset; requests for another supported dataset return HTTP 409 and do not trigger indexing. Readiness returns HTTP 503 until initialization completes. The principal routes are `GET /health`, `GET /info`, `POST /retrieve`, and `POST /index/rebuild`. Example request:

```bash
curl -X POST http://localhost:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query":"parse JSON response", "method":"hybrid", "rerank":false, "top_k":5, "dataset":"coir"}'
```

For a lightweight demo-corpus API process, launch with `NEXUS_DATASET=samsung_demo` (PowerShell: `$env:NEXUS_DATASET="samsung_demo"`). Model-backed retrieval still loads E5 on startup in the API implementation.

## Run the CLI

```bash
python -m src.cli --help
python -m src.cli retrieve --query "parse JSON response" --method bm25 --dataset samsung_demo
python -m src.cli index --dataset samsung_demo
```

Available commands are `index`, `retrieve`, and `evaluate`. Retrieval defaults to CoIR and `hybrid-rerank`, which may download the dense and reranker models. Use `--method bm25` with `--dataset samsung_demo` for a lexical-only command on the small demonstration corpus. The `index` command builds indexes in memory for that invocation; current indexes are not persisted for later CLI invocations.

## Run the demo

```bash
streamlit run demo/app.py
```

The demo offers the Samsung/demo corpus and CoIR test split, plus dense, BM25, and hybrid RRF methods with optional cross-encoder reranking. Samsung/demo mode starts with BM25 only and does not load E5 or the reranker unless those options are selected. Displayed scores come from BM25, RRF, or the cross-encoder when available; dense-only ordering has no exposed similarity score and is labeled by rank order. The demo includes current/deprecated API, version, similarity, and migration-style query examples.

## Run the benchmark

```bash
python scripts/run_benchmark.py --dataset coir --split test
```

Use `--limit N` for a reduced-query run or `--force` to re-encode dense vectors. This command runs dense, BM25, hybrid, and hybrid-rerank evaluations and writes `results/benchmark.json` and `results/benchmark.md`. It can download the CoIR dataset and both models and may require substantial time and memory.

### Official benchmark results

Dataset: **CoIR AppsRetrieval**, split: **test**. The loader returned 3,765 queries and 8,765 corpus documents. The final benchmark attempt was stopped during query embedding before scoring or report generation. These metrics are **not yet measured**:

| Method | NDCG@10 | MRR | Recall@10 | Recall@100 | Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Dense E5 | Not yet measured | Not yet measured | Not yet measured | Not yet measured | Not yet measured |
| BM25 | Not yet measured | Not yet measured | Not yet measured | Not yet measured | Not yet measured |
| Hybrid RRF | Not yet measured | Not yet measured | Not yet measured | Not yet measured | Not yet measured |
| Hybrid RRF + reranker | Not yet measured | Not yet measured | Not yet measured | Not yet measured | Not yet measured |

## Tests

```bash
pytest
```

The CoIR loader test requires cached or network-accessible dataset files. Dense cache unit tests use a small fake encoder and do not download or run a model. `test_e0_cached.py` skips when its legacy E0 embedding cache is absent.

## Docker

Build and run the API container:

```bash
docker build -t nexus-retrieval .
docker run --rm -p 8000:8000 nexus-retrieval
```

The image installs dependencies and starts Uvicorn. API startup loads the CoIR test data and the dense model, so first startup may download data/model artifacts and take time. The health check calls `/health`.

## Cache behavior

Dense corpus embeddings are cached under the repository's `cache/` directory as NumPy files. The deterministic filename includes model, split, batch size, passage prefix, corpus IDs, text, and input order through a SHA-256 fingerprint. Identical corpus/configuration reuses the cache; changed or reordered input gets a different cache file. Cache files, BM25 indexes, and FAISS indexes are excluded from Git; BM25 and FAISS indexes are built in memory and are not persisted as reusable index artifacts.

## Dataset distinction

- **Official CoIR AppsRetrieval benchmark:** loaded from the AppsRetrieval task metadata through MTEB and Hugging Face Datasets. Benchmark metrics should be reported only from an actual run with the dataset split and configuration recorded.
- **Samsung/demo dataset:** a small hand-authored demonstration corpus in `src/data/samsung_demo.py`, separate from CoIR. It illustrates version metadata behavior and is not an official Samsung dataset or an evaluation result.

## Reproducibility notes

The benchmark JSON and Markdown record the selected dataset and split, query limit, active dense/BM25/RRF/reranker settings, Python/platform, relevant package versions, counts, metrics, and timings. For published comparisons, also record hardware, operating system power/compute settings, cache warm/cold state, and whether external downloads occurred. Do not compare reduced-query runs with full benchmark runs as if they were equivalent.

## Current limitations

- Indexes are not persisted for reuse; entry points rebuild them in memory.
- The API is configured for one dataset per process and builds dense and BM25 indexes at startup; a request for a different dataset is rejected with HTTP 409.
- Dense-only API and demo methods do not currently expose similarity scores through the public retrieval interface.
- Chunking exists as a utility but is not connected to the retrieval indexing path.
- The API builds indexes at startup, so the CoIR-configured container may have a long startup and download models/data.
- The full official CoIR benchmark has not produced metrics in this submission state.

## Future work

- Persist and version FAISS and lexical indexes for repeatable deployments.
- Allow an API process to serve multiple prebuilt datasets without implicit on-request indexing.
- Expose dense similarity scores consistently through retrieval interfaces.
- Integrate chunking into repository ingestion and evaluate chunk-level retrieval.
- Measure the official CoIR AppsRetrieval test split and publish its generated report.
