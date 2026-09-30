import argparse
from typing import Tuple, Dict, List

from .dense_e0 import evaluate_dense_e0, build_corpus_index, retrieve_top_k
from ..data.coir import load_coir
from .e0_cached import evaluate_cached_e0

def main() -> None:
    parser = argparse.ArgumentParser(description="Run dense E0 baseline with optional smoke‑test limit.")
    parser.add_argument("--limit", type=int, default=None, help="If set, run a quick smoke‑test using only the first N queries.")
    parser.add_argument("--top_k", type=int, default=100, help="Number of retrieved documents per query.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for encoding.")
    parser.add_argument("--cached", action="store_true", help="Run evaluation using already cached embeddings only (no model loading).")
    args = parser.parse_args()

    if args.cached:
        # Cached‑only evaluation (uses existing .npy files, no model loading)
        retrieved, metrics = evaluate_cached_e0(top_k=args.top_k)
        print("\nEvaluation Metrics (cached):")
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")
        print(f"Queries processed: {len(retrieved)}")
        sample_q = next(iter(retrieved))
        print(f"Sample query ID: {sample_q}, retrieved docs (top 5): {retrieved[sample_q][:5]}")
        return

    if args.limit is not None:
        # Load full data then truncate for smoke‑test
        queries, corpus, _ = load_coir(split="test")
        # Truncate queries
        selected_qids = sorted(queries.keys())[: args.limit]
        queries = {qid: queries[qid] for qid in selected_qids}
        # Truncate corpus to IDs referenced by these queries (fallback to first N)
        needed_cids = set()
        for qid in selected_qids:
            # qrels not needed for retrieval; just take a subset of corpus
            needed_cids.update([cid for cid in corpus.keys()][: args.limit])
        if len(needed_cids) < args.limit:
            extra = [cid for cid in sorted(corpus.keys()) if cid not in needed_cids][: args.limit - len(needed_cids)]
            needed_cids.update(extra)
        corpus = {cid: corpus[cid] for cid in needed_cids}
        index, id_map = build_corpus_index(corpus, batch_size=args.batch_size)
        retrieved = retrieve_top_k(queries, index, id_map, batch_size=args.batch_size, top_k=args.top_k)
    else:
        # Full evaluation
        queries, corpus, retrieved = evaluate_dense_e0(load_coir, batch_size=args.batch_size, top_k=args.top_k)
        # Load qrels for evaluation
        _, _, qrels = load_coir(split="test")
        # Compute metrics
        from ..evaluation.metrics import evaluate as eval_metrics
        metrics = eval_metrics(retrieved, qrels)
        print("\nEvaluation Metrics:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")

    # Simple summary output
    print(f"Queries processed: {len(queries)}")
    print(f"Corpus documents indexed: {len(corpus)}")
    sample_q = next(iter(retrieved))
    print(f"Sample query ID: {sample_q}, retrieved docs (top 5): {retrieved[sample_q][:5]}")

if __name__ == "__main__":
    main()
