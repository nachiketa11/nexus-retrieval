import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

from .metrics import evaluate as evaluate_metrics
from ..data.coir import load_coir
from ..data.samsung_demo import load_samsung_demo_data
from ..retrieval.dense import DenseE5Retriever
from ..retrieval.bm25 import BM25Retriever
from ..retrieval.hybrid import hybrid_retrieve
from ..retrieval.version_aware import VersionAwareFilter
from ..ranking.reranker import CodeReranker
from ..config.settings import get_settings
from ..utils.logging import get_logger

logger = get_logger("nexus.benchmark")


class BenchmarkRunner:
    """Benchmark evaluation pipeline for Nexus Retrieval."""

    def __init__(self, dataset_name: str = "coir", split: str = "test", limit: Optional[int] = None):
        self.dataset_name = dataset_name
        self.split = split
        self.limit = limit
        self.settings = get_settings()

        # Data placeholders
        self.queries: Dict[str, Dict] = {}
        self.corpus: Dict[str, Dict] = {}
        self.qrels: Dict[str, List[str]] = {}

        # Retrievers
        self.dense_retriever: Optional[DenseE5Retriever] = None
        self.bm25_retriever: Optional[BM25Retriever] = None
        self.reranker: Optional[CodeReranker] = None

    def load_dataset(self):
        """Load evaluation dataset (CoIR or Samsung Demo)."""
        start = time.time()
        if self.dataset_name == "samsung_demo":
            self.queries, self.corpus = load_samsung_demo_data()
            self.qrels = {qid: [data["expected_doc"]] for qid, data in self.queries.items()}
        else:
            self.queries, self.corpus, self.qrels = load_coir(split=self.split)

        # Filter queries to only those that have relevance judgments
        self.queries = {qid: self.queries[qid] for qid in self.qrels if qid in self.queries}

        if self.limit is not None and self.limit > 0:
            logger.info(f"Limiting benchmark evaluation to first {self.limit} queries.")
            selected_qids = sorted(self.queries.keys())[: self.limit]
            self.queries = {qid: self.queries[qid] for qid in selected_qids}
            self.qrels = {qid: self.qrels[qid] for qid in selected_qids if qid in self.qrels}

        logger.info(
            f"Dataset loaded ({self.dataset_name}): {len(self.queries)} queries, "
            f"{len(self.corpus)} corpus documents in {time.time() - start:.2f}s."
        )

    def run_all(self, force_rebuild: bool = False) -> Dict[str, Any]:
        """Run benchmark for dense, bm25, hybrid, and hybrid-rerank methods."""
        if not self.queries or not self.corpus:
            self.load_dataset()

        report: Dict[str, Any] = {
            "dataset": self.dataset_name,
            "split": self.split,
            "queries_count": len(self.queries),
            "corpus_count": len(self.corpus),
            "limit": self.limit,
            "methods": {},
        }

        # 1. Indexing / Setup phase timing
        logger.info("Initializing retrievers...")

        # BM25 indexing
        bm25_start = time.time()
        self.bm25_retriever = BM25Retriever(k1=self.settings.bm25_k1, b=self.settings.bm25_b)
        self.bm25_retriever.fit(self.corpus)
        bm25_indexing_time = time.time() - bm25_start

        # Dense indexing
        dense_start = time.time()
        self.dense_retriever = DenseE5Retriever(
            model_name=self.settings.dense_model_name,
            batch_size=self.settings.dense_batch_size,
            split=self.split,
        )
        self.dense_retriever.build_index(self.corpus, force_rebuild=force_rebuild)
        dense_indexing_time = time.time() - dense_start

        # --- Method 1: Dense Retrieval ---
        logger.info("Evaluating Dense E5 Retrieval...")
        t0 = time.time()
        dense_results = self.dense_retriever.retrieve(self.queries, top_k=self.settings.candidate_count)
        dense_retrieval_latency = (time.time() - t0) / len(self.queries) * 1000  # ms per query
        dense_metrics = evaluate_metrics(
            {qid: res[: self.settings.top_k] for qid, res in dense_results.items()},
            self.qrels,
        )
        report["methods"]["dense"] = {
            "metrics": dense_metrics,
            "indexing_time_s": round(dense_indexing_time, 3),
            "latency_per_query_ms": round(dense_retrieval_latency, 2),
        }

        # --- Method 2: BM25 Lexical Retrieval ---
        logger.info("Evaluating BM25 Retrieval...")
        t0 = time.time()
        bm25_results = self.bm25_retriever.retrieve_batch(self.queries, top_k=self.settings.candidate_count)
        bm25_retrieval_latency = (time.time() - t0) / len(self.queries) * 1000
        bm25_metrics = evaluate_metrics(
            {qid: res[: self.settings.top_k] for qid, res in bm25_results.items()},
            self.qrels,
        )
        report["methods"]["bm25"] = {
            "metrics": bm25_metrics,
            "indexing_time_s": round(bm25_indexing_time, 3),
            "latency_per_query_ms": round(bm25_retrieval_latency, 2),
        }

        # --- Method 3: Hybrid RRF Retrieval ---
        logger.info("Evaluating Hybrid RRF Retrieval...")
        t0 = time.time()
        hybrid_results = hybrid_retrieve(
            dense_results,
            bm25_results,
            k=self.settings.rrf_k,
            top_k=self.settings.candidate_count,
        )
        hybrid_retrieval_latency = (time.time() - t0) / len(self.queries) * 1000 + (
            dense_retrieval_latency + bm25_retrieval_latency
        )
        hybrid_metrics = evaluate_metrics(
            {qid: res[: self.settings.top_k] for qid, res in hybrid_results.items()},
            self.qrels,
        )
        report["methods"]["hybrid"] = {
            "metrics": hybrid_metrics,
            "latency_per_query_ms": round(hybrid_retrieval_latency, 2),
        }

        # --- Method 4: Hybrid RRF + Reranker ---
        logger.info("Evaluating Hybrid RRF + Reranker...")
        self.reranker = CodeReranker(model_name=self.settings.reranker_model_name)
        t0 = time.time()
        reranked_results = self.reranker.rerank_batch(
            self.queries,
            hybrid_results,
            self.corpus,
            top_k=self.settings.top_k,
        )
        rerank_latency = (time.time() - t0) / len(self.queries) * 1000
        rerank_metrics = evaluate_metrics(reranked_results, self.qrels)
        report["methods"]["hybrid-rerank"] = {
            "metrics": rerank_metrics,
            "rerank_latency_ms": round(rerank_latency, 2),
            "total_latency_per_query_ms": round(hybrid_retrieval_latency + rerank_latency, 2),
        }

        return report

    def save_reports(self, report: Dict[str, Any], json_path: Path, md_path: Path):
        """Save benchmark report in JSON and Markdown formats."""
        # Save JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        # Save Markdown
        md_lines = [
            "# Nexus Retrieval Benchmark Report",
            "",
            f"- **Dataset**: `{report['dataset']}`",
            f"- **Split**: `{report['split']}`",
            f"- **Queries Evaluated**: {report['queries_count']}",
            f"- **Corpus Documents**: {report['corpus_count']}",
            "",
            "## Evaluation Metrics Summary",
            "",
            "| Method | NDCG@10 | MRR | Recall@10 | Recall@100 | Warm Latency (ms) |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
        ]

        for method, data in report["methods"].items():
            m = data["metrics"]
            lat = data.get("total_latency_per_query_ms") or data.get("latency_per_query_ms", 0.0)
            md_lines.append(
                f"| `{method}` | {m['ndcg@10']:.4f} | {m['mrr']:.4f} | "
                f"{m['recall@10']:.4f} | {m['recall@100']:.4f} | {lat:.2f} ms |"
            )

        md_lines.extend(
            [
                "",
                "## Methodology & Configuration",
                "- **Dense Model**: `intfloat/e5-base-v2` with FAISS inner-product similarity.",
                "- **BM25 Parameters**: $k_1 = 1.5, b = 0.75$ with code symbol tokenization.",
                "- **Hybrid Fusion**: Reciprocal Rank Fusion (RRF) with $k = 60$.",
                "- **Cross-Encoder Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2` top-candidate reranking.",
            ]
        )

        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))

        logger.info(f"Saved benchmark JSON to {json_path}")
        logger.info(f"Saved benchmark Markdown to {md_path}")
