"""Evaluate the NEXUS agent against plain retrieval on the labelled Samsung/demo queries.

Usage:
    python scripts/eval_agent.py              # BM25-only agent vs BM25
    python scripts/eval_agent.py --onnx       # + ONNX E5 dense and cross-encoder (needs models/)
    python scripts/eval_agent.py --onnx --out results/agent_eval.md
"""

import argparse
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.agent import NexusAgent  # noqa: E402
from src.data.samsung_demo import load_samsung_demo_data  # noqa: E402
from src.evaluation.metrics import mrr, ndcg_at_k, recall_at_k  # noqa: E402
from src.retrieval.bm25 import BM25Retriever  # noqa: E402
from src.retrieval.hybrid import reciprocal_rank_fusion  # noqa: E402


def _score(rankings, qrels):
    n = len(qrels)
    hit1 = sum(1 for q, r in rankings.items() if r[:1] and r[0] in qrels[q]) / n
    return {
        "Hit@1": hit1,
        "MRR": mrr([rankings[q] for q in qrels], [set(qrels[q]) for q in qrels]),
        "NDCG@10": sum(ndcg_at_k(rankings[q], set(qrels[q]), 10) for q in qrels) / n,
        "Recall@5": sum(recall_at_k(rankings[q], set(qrels[q]), 5) for q in qrels) / n,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", action="store_true", help="Use ONNX E5 dense retrieval and cross-encoder")
    parser.add_argument("--out", type=str, default=None, help="Write a Markdown report to this path")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    queries, corpus = load_samsung_demo_data()
    qrels = {qid: [q["expected_doc"]] for qid, q in queries.items()}
    bm25 = BM25Retriever().fit(corpus)

    dense_fn = rerank_fn = None
    if args.onnx:
        from src.retrieval.onnx_models import OnnxCrossEncoder, OnnxDenseRetriever, OnnxE5Encoder
        dense = OnnxDenseRetriever(OnnxE5Encoder()).fit(corpus)
        cross = OnnxCrossEncoder()
        dense_fn = dense.retrieve

        def rerank_fn(query, ids):
            scores = cross.predict([(query, corpus[c]["text"]) for c in ids])
            return sorted(zip(ids, map(float, scores)), key=lambda x: (-x[1], x[0]))

    systems = {"BM25": {}}
    if dense_fn:
        systems["Dense E5"] = {}
        systems["Hybrid RRF"] = {}
        systems["Hybrid RRF + cross-encoder"] = {}
    systems["NEXUS agent"] = {}
    latencies = {name: [] for name in systems}

    agent = NexusAgent(corpus, bm25=bm25, dense=dense_fn, reranker=rerank_fn)
    for qid, q in queries.items():
        text = q["text"]
        t = time.perf_counter()
        systems["BM25"][qid] = [c for c, _ in bm25.retrieve(text, 50)]
        latencies["BM25"].append(time.perf_counter() - t)
        if dense_fn:
            t = time.perf_counter()
            d = [c for c, _ in dense_fn(text, 50)]
            latencies["Dense E5"].append(time.perf_counter() - t)
            systems["Dense E5"][qid] = d
            fused = [c for c, _ in reciprocal_rank_fusion([d, systems["BM25"][qid]], top_k=50)]
            systems["Hybrid RRF"][qid] = fused
            latencies["Hybrid RRF"].append(latencies["Dense E5"][-1] + latencies["BM25"][-1])
            t = time.perf_counter()
            head = [c for c, _ in rerank_fn(text, fused[:15])]
            systems["Hybrid RRF + cross-encoder"][qid] = head + fused[15:]
            latencies["Hybrid RRF + cross-encoder"].append(latencies["Hybrid RRF"][-1] + time.perf_counter() - t)
        t = time.perf_counter()
        out = agent.run(text, top_k=10)
        latencies["NEXUS agent"].append(time.perf_counter() - t)
        systems["NEXUS agent"][qid] = [r["doc_id"] for r in out["results"]]
        if args.verbose:
            ok = "OK " if out["results"] and out["results"][0]["doc_id"] == q["expected_doc"] else "MISS"
            print(f"{ok} {qid:5} conf={out['confidence']:.2f} it={out['iterations']} "
                  f"{text!r} -> {systems['NEXUS agent'][qid][:3]}")

    lines = ["| System | Hit@1 | MRR | NDCG@10 | Recall@5 | Mean latency |",
             "| :--- | :---: | :---: | :---: | :---: | :---: |"]
    for name, rankings in systems.items():
        m = _score(rankings, qrels)
        lat = 1000 * sum(latencies[name]) / max(len(latencies[name]), 1)
        lines.append(f"| {name} | {m['Hit@1']:.3f} | {m['MRR']:.3f} | {m['NDCG@10']:.3f} | {m['Recall@5']:.3f} "
                     f"| {lat:.1f} ms |")
    table = "\n".join(lines)
    print(f"\nSamsung/demo labelled queries: {len(queries)} queries, {len(corpus)} documents\n")
    print(table)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(f"# NEXUS agent evaluation (Samsung/demo)\n\n{len(queries)} queries, "
                            f"{len(corpus)} documents.\n\n{table}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
