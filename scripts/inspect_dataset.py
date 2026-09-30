import sys
import platform
import importlib
import pkgutil
import json
from pathlib import Path

def get_version_and_path(package_name: str):
    try:
        pkg = importlib.import_module(package_name)
        version = getattr(pkg, "__version__", "unknown")
        # Try to get the file path of the package
        file = getattr(pkg, "__file__", None)
        if file:
            path = Path(file).parent
        else:
            # Fallback: locate spec
            spec = importlib.util.find_spec(package_name)
            path = Path(spec.origin).parent if spec and spec.origin else Path("<not found>")
        return version, str(path)
    except Exception as e:
        return "<error>", str(e)

def main():
    # 1. Python version
    print("Python version:", sys.version.split()[0])

    # 2-4. MTEB version and path, availability of AppsRetrieval
    try:
        import mteb
    except ImportError as e:
        print("MTEB not installed:", e)
        sys.exit(1)
    mteb_version, mteb_path = get_version_and_path("mteb")
    print("MTEB version:", mteb_version)
    print("MTEB installation path:", mteb_path)

    # Try to load the AppsRetrieval task via MTEB get_tasks helper
    try:
        from mteb import get_tasks
        tasks = get_tasks(tasks=["AppsRetrieval"])
        task = tasks[0]
        print("AppsRetrieval is available: Yes")
    except Exception as e:
        print("AppsRetrieval is available: No (", e, ")")
        sys.exit(1)

    # 5. Actual task class/type
    print("Task class:", task.__class__.__module__ + "." + task.__class__.__name__)

    # 6. Task metadata (if present)
    metadata = getattr(task, "metadata", {})
    print("Task metadata:")
    print(json.dumps(metadata, indent=2, default=str))

    # 7. Available splits
    splits = getattr(task, "splits", getattr(task, "corpus", {})).keys() if hasattr(task, "corpus") else []
    print("Available splits:", list(splits))

    # 8. Public attributes/methods of the task object
    public_attrs = [a for a in dir(task) if not a.startswith("_")]
    print("Public attributes/methods:")
    print(public_attrs)

    # Load data for the 'test' split (default) if present
    split_to_use = "test" if "test" in splits else next(iter(splits), None)
    if not split_to_use:
        print("No splits found in the task.")
        sys.exit(0)

    data = task.load_data(split=split_to_use)
    queries = data.get("queries", {})
    corpus = data.get("corpus", {})
    qrels = data.get("qrels", {})

    # 9. Query structure (sample)
    sample_query_id, sample_query = next(iter(queries.items()))
    print("Query structure (sample):")
    print(json.dumps(sample_query, indent=2, default=str))

    # 10. Corpus structure (sample)
    sample_doc_id, sample_doc = next(iter(corpus.items()))
    print("Corpus structure (sample):")
    print(json.dumps(sample_doc, indent=2, default=str))

    # 11. Qrels structure (sample)
    sample_qrel = {sample_query_id: qrels.get(sample_query_id, [])}
    print("Qrels structure (sample for above query):")
    print(json.dumps(sample_qrel, indent=2, default=str))

    # 12-14. Counts
    print("Query count:", len(queries))
    print("Corpus count:", len(corpus))
    print("Qrels count (unique queries with relevance judgments):", len(qrels))

    # 15. One-to-one or one-to-many relevance
    many = any(len(v) > 1 for v in qrels.values())
    print("Relevance mapping:", "one-to-many" if many else "one-to-one")

    # 16. One sample query (already have sample_query_id)
    print("Sample query ID:", sample_query_id)
    print("Sample query text:", sample_query.get("text") if isinstance(sample_query, dict) else sample_query)

    # 17. Its relevant document ID(s)
    relevant_ids = qrels.get(sample_query_id, [])
    print("Relevant document IDs for the sample query:", relevant_ids)

    # 18. Whether a usable train split exists
    train_exists = "train" in splits
    print("Usable train split exists:", train_exists)

    # Inspect the underlying HuggingFace CoIR-Retrieval/apps dataset
    try:
        from datasets import load_dataset
        hf_dataset = load_dataset("coir/apps")
        print("\n--- HuggingFace CoIR-Retrieval/apps dataset inspection ---")
        # 1. Dataset/config/subset names (if any)
        print("Dataset info:", hf_dataset)
        # 2. Available splits
        splits = list(hf_dataset.keys())
        print("Available splits:", splits)

        total_queries = 0
        total_qrels = 0
        doc_id_set = set()
        multiple_rel = False

        for split_name in splits:
            ds = hf_dataset[split_name]
            print(f"\nSplit: {split_name}")
            print("Number of rows:", len(ds))
            print("Column names:", ds.column_names)
            if len(ds) > 0:
                sample = ds[0]
                print("Sample row:", sample)
                query_fields = [k for k in sample.keys() if "query" in k.lower()]
                doc_fields = [k for k in sample.keys() if "positive_passages" in k.lower() or "passage" in k.lower() or "doc" in k.lower()]
                print("Detected query field(s):", query_fields)
                print("Detected document field(s):", doc_fields)
            # Accumulate stats
            total_queries += len(ds)
            for row in ds:
                pos = row.get("positive_passages", [])
                ids = [p.get("id") for p in pos if isinstance(p, dict) and "id" in p]
                doc_id_set.update(ids)
                total_qrels += len(ids)
                if len(ids) > 1:
                    multiple_rel = True

        print("\nTotal query count (across all splits):", total_queries)
        print("Total corpus/document count (unique IDs from positive passages):", len(doc_id_set))
        print("Total qrels count (total relevance judgments):", total_qrels)
        print("Relevance mapping (one-to-many if any query has >1 relevant doc):", "one-to-many" if multiple_rel else "one-to-one")

        # Concrete sample query and its relevant document IDs from first split
        if splits:
            first_ds = hf_dataset[splits[0]]
            if len(first_ds) > 0:
                row0 = first_ds[0]
                sample_query = next((v for k, v in row0.items() if "query" in k.lower()), None)
                sample_ids = [p.get("id") for p in row0.get("positive_passages", []) if isinstance(p, dict) and "id" in p]
                print("\nConcrete sample query:", sample_query)
                print("Relevant document ID(s) for the sample query:", sample_ids)
    except Exception as e:
        print("Failed to inspect HuggingFace CoIR dataset:", e)

if __name__ == "__main__":
    main()
