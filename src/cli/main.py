import argparse
import sys
import time
from typing import Dict, List, Any

from ..data.samsung_demo import load_samsung_demo_data
from ..retrieval.bm25 import BM25Retriever
from ..retrieval.hybrid import hybrid_retrieve
from ..retrieval.version_aware import VersionAwareFilter, parse_version_intent
from ..config.settings import get_settings
from ..utils.logging import get_logger

logger = get_logger("nexus.cli")


# Heavy dependencies (torch, sentence-transformers, FAISS, datasets/MTEB) load only when a
# command needs them, so lexical and agent commands on the demo corpus stay lightweight.
def load_coir(*args, **kwargs):
    from ..data.coir import load_coir as _load_coir
    return _load_coir(*args, **kwargs)


def DenseE5Retriever(*args, **kwargs):  # noqa: N802 - keeps the original call sites unchanged
    from ..retrieval.dense import DenseE5Retriever as _Dense
    return _Dense(*args, **kwargs)


def CodeReranker(*args, **kwargs):  # noqa: N802
    from ..ranking.reranker import CodeReranker as _Reranker
    return _Reranker(*args, **kwargs)


def BenchmarkRunner(*args, **kwargs):  # noqa: N802
    from ..evaluation.benchmark import BenchmarkRunner as _Runner
    return _Runner(*args, **kwargs)


def handle_agent(args: argparse.Namespace):
    """Run the NEXUS agent on the demo corpus and print its reasoning trace."""
    from ..serving import NexusEngine

    engine = NexusEngine(enable_models=not args.lexical)
    out = engine.agent(args.query, top_k=args.top_k, version=args.version, use_models=not args.lexical)
    neural = engine.models_enabled and not args.lexical
    print()
    print(f"NEXUS agent  |  tools: {'BM25 + E5 + cross-encoder (ONNX)' if neural else 'BM25 + metadata policy'}")
    print("=" * 78)
    for step in out["trace"]:
        print(f"[{step['phase'].upper():<7}] {step['tool']:<20} {step['thought']}")
    print("=" * 78)
    print(f"Confidence {out['confidence']:.2f} after {out['iterations']} iteration(s), {out['latency_ms']:.0f} ms")
    print(out["answer"]["summary"])
    for r in out["results"]:
        meta = r["metadata"]
        note = f"  <- {r['note']}" if r.get("note") else ""
        print(f"  {r['rank']:>2}. {r['doc_id']:<28} {meta.get('file', '')}  v{meta.get('version', '?')}{note}")


def handle_index(args: argparse.Namespace):
    """Index dataset for dense and BM25 retrieval."""
    print(f"Indexing dataset: {args.dataset}...")
    start = time.time()
    if args.dataset == "samsung_demo":
        _, corpus = load_samsung_demo_data()
    else:
        _, corpus, _ = load_coir(split="test")

    # Build Dense Index
    print("Building FAISS dense index...")
    dense_retriever = DenseE5Retriever(split="test")
    dense_retriever.build_index(corpus, force_rebuild=args.force)

    # Build BM25 Index
    print("Building BM25 index...")
    bm25_retriever = BM25Retriever()
    bm25_retriever.fit(corpus)

    print(f"Indexing completed successfully in {time.time() - start:.2f}s!")


def handle_retrieve(args: argparse.Namespace):
    """Retrieve top-K code snippets for a query."""
    query_text = args.query
    method = args.method.lower()
    rerank = args.rerank or (method == "hybrid-rerank")
    top_k = args.top_k or 10
    version = args.version

    print(f"\n[Nexus Retrieval] Query: \"{query_text}\" | Method: {method} | Top-K: {top_k}\n")
    start = time.time()

    # Load corpus
    if args.dataset == "samsung_demo":
        queries_dict, corpus = load_samsung_demo_data()
    else:
        queries_dict, corpus, _ = load_coir(split="test")

    query_obj = {"q1": {"text": query_text}}

    # 1. Base Retrieval
    candidate_cids: List[str] = []

    if method == "dense":
        dense_retriever = DenseE5Retriever()
        dense_retriever.build_index(corpus)
        res = dense_retriever.retrieve(query_obj, top_k=50)
        candidate_cids = res.get("q1", [])

    elif method == "bm25":
        bm25 = BM25Retriever()
        bm25.fit(corpus)
        res = bm25.retrieve_batch(query_obj, top_k=50)
        candidate_cids = res.get("q1", [])

    else:  # hybrid or hybrid-rerank
        dense_retriever = DenseE5Retriever()
        dense_retriever.build_index(corpus)
        dense_res = dense_retriever.retrieve(query_obj, top_k=50)

        bm25 = BM25Retriever()
        bm25.fit(corpus)
        bm25_res = bm25.retrieve_batch(query_obj, top_k=50)

        fused_res = hybrid_retrieve(dense_res, bm25_res, top_k=50)
        candidate_cids = fused_res.get("q1", [])

    # 2. Version filtering / boosting if version or metadata intent present
    v_filter = VersionAwareFilter()
    if version:
        candidate_cids = v_filter.filter_candidates(candidate_cids, corpus, version=version)
    elif parse_version_intent(query_text):
        boosted = v_filter.rerank_by_metadata_match(candidate_cids, corpus, query_text)
        candidate_cids = [cid for cid, _ in boosted]

    # 3. Reranking if enabled
    if rerank:
        reranker = CodeReranker()
        reranked_pairs = reranker.rerank_query(query_text, candidate_cids, corpus, top_k=top_k)
        final_cids = [cid for cid, _ in reranked_pairs]
    else:
        final_cids = candidate_cids[:top_k]

    elapsed = time.time() - start

    # Display Output
    print(f"Found {len(final_cids)} results in {elapsed*1000:.1f} ms:\n" + "-" * 70)

    for rank, cid in enumerate(final_cids, start=1):
        doc = corpus.get(cid, {})
        text = doc.get("text", "").strip()
        meta = doc.get("meta_information") or doc.get("metadata") or {}
        lang = meta.get("language") or doc.get("language") or "code"
        ver = meta.get("version") or "N/A"
        file_path = meta.get("file") or doc.get("title") or "N/A"

        preview = text[:180].replace("\n", " ") + ("..." if len(text) > 180 else "")

        print(f"Rank {rank:2d} | Doc ID: {cid}")
        print(f"        File: {file_path} | Language: {lang} | Version: {ver}")
        print(f"        Code: {preview}")
        print("-" * 70)


def handle_evaluate(args: argparse.Namespace):
    """Evaluate retrieval pipeline and output benchmark metrics."""
    print(f"Running benchmark evaluation (Dataset: {args.dataset}, Limit: {args.limit})...")
    runner = BenchmarkRunner(dataset_name=args.dataset, split="test", limit=args.limit)
    report = runner.run_all(force_rebuild=args.force)

    settings = get_settings()
    json_path = settings.results_dir / "benchmark.json"
    md_path = settings.results_dir / "benchmark.md"

    runner.save_reports(report, json_path, md_path)

    print("\nBenchmark Evaluation Summary:")
    print("=" * 65)
    print(f"{'Method':<16} | {'NDCG@10':<9} | {'MRR':<9} | {'Recall@10':<9} | {'Latency':<8}")
    print("-" * 65)

    for method, data in report["methods"].items():
        m = data["metrics"]
        lat = data.get("total_latency_per_query_ms") or data.get("latency_per_query_ms", 0.0)
        print(f"{method:<16} | {m['ndcg@10']:<9.4f} | {m['mrr']:<9.4f} | {m['recall@10']:<9.4f} | {lat:<6.1f} ms")
    print("=" * 65)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nexus",
        description="Nexus Code Retrieval System CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Index command
    idx_parser = subparsers.add_parser("index", help="Build dense FAISS and BM25 indices")
    idx_parser.add_argument(
        "--dataset",
        choices=["coir", "samsung_demo"],
        default="coir",
        help="Dataset to index",
    )
    idx_parser.add_argument("--force", action="store_true", help="Force rebuild indices without cache")

    # Retrieve command
    ret_parser = subparsers.add_parser("retrieve", help="Retrieve code snippets for a query")
    ret_parser.add_argument("--query", type=str, required=True, help="Natural language query")
    ret_parser.add_argument(
        "--method",
        choices=["dense", "bm25", "hybrid", "hybrid-rerank"],
        default="hybrid-rerank",
        help="Retrieval method",
    )
    ret_parser.add_argument("--top-k", type=int, default=10, help="Number of snippets to retrieve")
    ret_parser.add_argument("--rerank", action="store_true", help="Enable cross-encoder reranking")
    ret_parser.add_argument("--version", type=str, help="Specific version context (e.g. '3.0')")
    ret_parser.add_argument(
        "--dataset",
        choices=["coir", "samsung_demo"],
        default="coir",
        help="Target corpus dataset",
    )

    # Agent command
    agent_parser = subparsers.add_parser("agent", help="Agentic retrieval with a reasoning trace (demo corpus)")
    agent_parser.add_argument("--query", type=str, required=True, help="Natural language query")
    agent_parser.add_argument("--top-k", type=int, default=5, help="Number of snippets to return")
    agent_parser.add_argument("--version", type=str, help="Hard version constraint (e.g. '3.0')")
    agent_parser.add_argument("--lexical", action="store_true", help="Use BM25 only (no ONNX models)")

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate retrieval pipeline benchmark")
    eval_parser.add_argument(
        "--dataset",
        choices=["coir", "samsung_demo"],
        default="coir",
        help="Dataset split for evaluation",
    )
    eval_parser.add_argument("--limit", type=int, default=None, help="Limit to first N queries for fast evaluation")
    eval_parser.add_argument("--force", action="store_true", help="Force re-encode embeddings")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "index":
        handle_index(args)
    elif args.command == "retrieve":
        handle_retrieve(args)
    elif args.command == "agent":
        handle_agent(args)
    elif args.command == "evaluate":
        handle_evaluate(args)


if __name__ == "__main__":
    main()
