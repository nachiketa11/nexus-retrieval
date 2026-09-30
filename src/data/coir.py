import os
from typing import Dict, Tuple, List

from datasets import load_dataset
from mteb import get_tasks


def _detect_dataset_identifier(task) -> str:
    """Extract the HuggingFace dataset identifier from MTEB task metadata.

    The task metadata (task.metadata) contains a 'dataset' dict with a 'path' key.
    This function prefers that source but falls back to the older 'dataset' attribute.
    """
    # Prefer metadata attribute (newer MTEB versions)
    if hasattr(task, "metadata"):
        meta = getattr(task, "metadata")
        # meta may be a dict or a pydantic model
        dataset_info = None
        if isinstance(meta, dict):
            dataset_info = meta.get("dataset")
        else:
            dataset_info = getattr(meta, "dataset", None)
        if isinstance(dataset_info, dict) and "path" in dataset_info:
            return dataset_info["path"]
    # Fallback to legacy attribute
    if hasattr(task, "dataset"):
        dataset_info = getattr(task, "dataset")
        if isinstance(dataset_info, dict) and "path" in dataset_info:
            return dataset_info["path"]
    raise ValueError("Unable to determine dataset identifier from task metadata.")


def load_coir(split: str = "test") -> Tuple[Dict[str, Dict], Dict[str, Dict], Dict[str, List[str]]]:
    """Load the CoIR‑Retrieval/apps dataset.

    Returns three dictionaries:
        queries:  query_id -> {"text": ..., other metadata}
        corpus:   doc_id   -> {"text": ..., other metadata}
        qrels:    query_id -> [doc_id, ...]
    The loader respects the requested split (train/test) in the default config.
    """
    # 1️⃣ Resolve the task and dataset identifier via MTEB
    tasks = get_tasks(tasks=["AppsRetrieval"])
    if not tasks:
        raise RuntimeError("AppsRetrieval task not found via mteb.get_tasks")
    task = tasks[0]
    dataset_path = _detect_dataset_identifier(task)

    # 2️⃣ Load the three configs of the dataset
    #    • default config holds relevance pairs (qrels) with a 'split' column
    #    • queries config holds natural‑language queries
    #    • corpus config holds code/document snippets
    default_ds = load_dataset(dataset_path, name="default")
    if split not in default_ds:
        raise ValueError(
            f"Requested split '{split}' not found in default config. Available splits: {list(default_ds.keys())}"
        )
    default_split = default_ds[split]

    # Load all splits of the queries config and build a dict keyed by the stable _id
    queries_ds = load_dataset(dataset_path, name="queries")
    queries: Dict[str, Dict] = {}
    for cfg_split in queries_ds.keys():
        for row in queries_ds[cfg_split]:
            qid = row["_id"]
            queries[qid] = {
                "text": row.get("text", ""),
                "language": row.get("language"),
                "title": row.get("title"),
                "meta_information": row.get("meta_information"),
                "split": row.get("split"),
            }

    # Load all splits of the corpus config and build a dict keyed by the stable _id
    corpus_ds = load_dataset(dataset_path, name="corpus")
    corpus: Dict[str, Dict] = {}
    for cfg_split in corpus_ds.keys():
        for row in corpus_ds[cfg_split]:
            doc_id = row["_id"]
            corpus[doc_id] = {
                "text": row.get("text", ""),
                "language": row.get("language"),
                "title": row.get("title"),
                "meta_information": row.get("meta_information"),
                "split": row.get("split"),
            }

    # 3️⃣ Build qrels from the default config, respecting the requested split
    qrels: Dict[str, List[str]] = {}
    for row in default_split:
        qid = row["query-id"]
        doc_id = row["corpus-id"]
        # Verify IDs exist in the previously loaded dicts – this gives a clear error if the dataset is malformed
        if qid not in queries:
            raise ValueError(f"Query id {qid} present in qrels but not found in queries config")
        if doc_id not in corpus:
            raise ValueError(f"Corpus id {doc_id} present in qrels but not found in corpus config")
        qrels.setdefault(qid, []).append(doc_id)

    return queries, corpus, qrels
