import sys
from src.data.coir import load_coir
from src.evaluation.metrics import evaluate

def main():
    # Load test split data
    queries, corpus, qrels = load_coir(split="test")
    # Build a perfect ranking: for each query, list its relevant docs first (any order)
    ranked_results = {qid: qrels[qid][:] for qid in qrels}
    metrics = evaluate(ranked_results, qrels)
    print("Perfect ranking metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()
