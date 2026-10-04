"""CoIR AppsRetrieval benchmark on the torch-free ONNX stack.

Runs BM25, dense E5-base-v2 (ONNX int8), hybrid RRF and hybrid RRF + cross-encoder
(ONNX int8) on the CoIR AppsRetrieval test split and writes results/benchmark_onnx.{json,md}.

    python scripts/run_benchmark_onnx.py                       # all test queries; rerank on 500
    python scripts/run_benchmark_onnx.py --limit 200 --rerank-limit 200

Cross-encoder scoring is the slow step on CPU (one 512-token pair per candidate), so it runs
on the first ``--rerank-limit`` queries (sorted by id); the other methods are also reported on
that same subset so the comparison is like-for-like.
"""

import argparse
import hashlib
import json
import platform
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.config.settings import get_settings  # noqa: E402
from src.data.coir import load_coir  # noqa: E402
from src.evaluation.metrics import evaluate  # noqa: E402
from src.retrieval.bm25 import BM25Retriever  # noqa: E402
from src.retrieval.hybrid import reciprocal_rank_fusion  # noqa: E402
from src.retrieval.onnx_models import OnnxCrossEncoder, OnnxE5Encoder  # noqa: E402


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def encode_cached(encoder, texts, prefix, name, batch_size=16):
    digest = hashlib.sha256(("\0".join(texts) + prefix).encode("utf-8")).hexdigest()[:16]
    path = get_settings().cache_dir / f"onnx_e5_{name}_{digest}.npy"
    if path.exists():
        log(f"loaded cached {name} embeddings ({path.name})")
        return np.load(path)
    chunks, started = [], time.time()
    for start in range(0, len(texts), 256):
        chunks.append(encoder.encode(texts[start:start + 256], prefix=prefix, batch_size=batch_size))
        done = min(start + 256, len(texts))
        rate = done / (time.time() - started)
        log(f"encoded {done}/{len(texts)} {name} ({rate:.1f}/s, ~{(len(texts) - done) / rate / 60:.1f} min left)")
    matrix = np.vstack(chunks)
    np.save(path, matrix)
    return matrix


def pkg(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N test queries")
    parser.add_argument("--rerank-limit", type=int, default=500, help="Queries scored with the cross-encoder")
    parser.add_argument("--rerank-depth", type=int, default=20, help="Candidates rescored per query")
    parser.add_argument("--max-length", type=int, default=512, help="Token truncation for E5 and cross-encoder")
    args = parser.parse_args()

    t0 = time.time()
    queries, corpus, qrels = load_coir(split="test")
    qids = sorted(q for q in qrels if q in queries)
    if args.limit:
        qids = qids[: args.limit]
    log(f"CoIR AppsRetrieval test: {len(qids)} queries, {len(corpus)} documents (load {time.time() - t0:.0f}s)")

    doc_ids = sorted(corpus)
    doc_texts = [corpus[d]["text"] or "" for d in doc_ids]
    q_texts = [queries[q]["text"] or "" for q in qids]
    timings = {}

    t = time.time()
    bm25 = BM25Retriever().fit(corpus)
    bm25_runs = {q: [d for d, _ in bm25.retrieve(text, top_k=100)] for q, text in zip(qids, q_texts)}
    timings["bm25"] = (time.time() - t) / len(qids) * 1000
    log(f"BM25 done ({timings['bm25']:.1f} ms/query incl. indexing)")

    encoder = OnnxE5Encoder(max_length=args.max_length)
    doc_matrix = encode_cached(encoder, doc_texts, "passage:", "corpus")
    t = time.time()
    q_matrix = encode_cached(encoder, q_texts, "query:", "queries")
    sims = q_matrix @ doc_matrix.T
    top = np.argsort(-sims, axis=1)[:, :100]
    dense_runs = {q: [doc_ids[i] for i in top[row]] for row, q in enumerate(qids)}
    timings["dense"] = (time.time() - t) / len(qids) * 1000
    log("dense done")

    hybrid_runs = {q: [d for d, _ in reciprocal_rank_fusion([dense_runs[q], bm25_runs[q]], top_k=100)] for q in qids}

    sub = qids[: args.rerank_limit]
    cross = OnnxCrossEncoder(max_length=args.max_length)
    rerank_runs, t = {}, time.time()
    for n, q in enumerate(sub, 1):
        head = hybrid_runs[q][: args.rerank_depth]
        scores = cross.predict([(queries[q]["text"], corpus[d]["text"]) for d in head])
        order = [d for d, _ in sorted(zip(head, scores), key=lambda x: (-float(x[1]), x[0]))]
        rerank_runs[q] = order + hybrid_runs[q][args.rerank_depth:]
        if n % 50 == 0:
            log(f"reranked {n}/{len(sub)} queries")
    timings["rerank_extra"] = (time.time() - t) / max(len(sub), 1) * 1000

    sub_qrels = {q: qrels[q] for q in sub}
    full_qrels = {q: qrels[q] for q in qids}
    report = {
        "dataset": "CoIR AppsRetrieval (CoIR-Retrieval/apps)",
        "split": "test",
        "queries_full": len(qids),
        "queries_rerank_subset": len(sub),
        "corpus_size": len(corpus),
        "stack": "ONNX Runtime int8: Xenova/e5-base-v2, Xenova/ms-marco-MiniLM-L-6-v2; rank-bm25",
        "settings": {"max_length": args.max_length, "rerank_depth": args.rerank_depth, "rrf_k": 60},
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "onnxruntime": pkg("onnxruntime"), "rank-bm25": pkg("rank-bm25"), "datasets": pkg("datasets")},
        "full": {
            "BM25": evaluate({q: bm25_runs[q] for q in qids}, full_qrels),
            "Dense E5": evaluate({q: dense_runs[q] for q in qids}, full_qrels),
            "Hybrid RRF": evaluate({q: hybrid_runs[q] for q in qids}, full_qrels),
        },
        "subset": {
            "BM25": evaluate({q: bm25_runs[q] for q in sub}, sub_qrels),
            "Dense E5": evaluate({q: dense_runs[q] for q in sub}, sub_qrels),
            "Hybrid RRF": evaluate({q: hybrid_runs[q] for q in sub}, sub_qrels),
            "Hybrid RRF + cross-encoder": evaluate(rerank_runs, sub_qrels),
        },
        "latency_ms_per_query": timings,
        "wall_clock_s": round(time.time() - t0, 1),
    }

    out_dir = get_settings().results_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "benchmark_onnx.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    def table(block):
        rows = ["| Method | NDCG@10 | MRR | Recall@10 | Recall@100 |", "| :--- | :---: | :---: | :---: | :---: |"]
        for name, m in block.items():
            rows.append(f"| {name} | {m['ndcg@10']:.4f} | {m['mrr']:.4f} | {m['recall@10']:.4f} | {m['recall@100']:.4f} |")
        return "\n".join(rows)

    md = (f"# CoIR AppsRetrieval benchmark (ONNX stack)\n\n{report['stack']}. Truncation {args.max_length} tokens.\n\n"
          f"## All {len(qids)} test queries\n\n{table(report['full'])}\n\n"
          f"## First {len(sub)} test queries (with cross-encoder, depth {args.rerank_depth})\n\n{table(report['subset'])}\n")
    (out_dir / "benchmark_onnx.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
