import re
from typing import Dict, List, Tuple, Optional
from rank_bm25 import BM25Okapi


def tokenize_code(text: str) -> List[str]:
    """Code-aware tokenizer preserving full identifiers and split sub-words.

    Examples:
        "getUserInfo_v2" -> ["getuserinfo_v2", "getuserinfo", "v2", "get", "user", "info"]
        "parse_json_response" -> ["parse_json_response", "parse", "json", "response"]
    """
    if not text:
        return []

    tokens: List[str] = []
    # Extract all word tokens preserving underscores
    raw_words = re.findall(r"[a-zA-Z0-9_]+", text)

    for word in raw_words:
        w_lower = word.lower()
        if w_lower and w_lower not in tokens:
            tokens.append(w_lower)

        # Split snake_case / underscores
        if "_" in w_lower:
            parts = w_lower.split("_")
            for part in parts:
                if part and part not in tokens:
                    tokens.append(part)

        # Split camelCase / PascalCase
        camel_parts = re.findall(r"[a-z0-9]+|[A-Z][a-z0-9]*", word)
        if len(camel_parts) > 1:
            for cp in camel_parts:
                cp_lower = cp.lower()
                if cp_lower and cp_lower not in tokens:
                    tokens.append(cp_lower)

    return tokens


class BM25Retriever:
    """Deterministic BM25 retriever for code search."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.bm25: Optional[BM25Okapi] = None
        self.corpus_ids: List[str] = []
        self.corpus_dict: Dict[str, Dict] = {}

    def fit(self, corpus: Dict[str, Dict]) -> "BM25Retriever":
        """Index the provided corpus with deterministic ID sorting."""
        self.corpus_dict = corpus
        self.corpus_ids = sorted(corpus.keys())
        tokenized_corpus = [tokenize_code(corpus[cid].get("text", "")) for cid in self.corpus_ids]
        self.bm25 = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)
        return self

    def retrieve(self, query: str, top_k: int = 100) -> List[Tuple[str, float]]:
        """Retrieve top-k corpus IDs and BM25 scores for a single query."""
        if not self.bm25 or not self.corpus_ids:
            raise RuntimeError("BM25Retriever must be fit before calling retrieve.")

        tokenized_query = tokenize_code(query)
        scores = self.bm25.get_scores(tokenized_query)

        id_score_pairs = list(zip(self.corpus_ids, [float(s) for s in scores]))
        sorted_pairs = sorted(id_score_pairs, key=lambda x: (-x[1], x[0]))

        return sorted_pairs[:top_k]

    def retrieve_batch(self, queries: Dict[str, Dict], top_k: int = 100) -> Dict[str, List[str]]:
        """Retrieve top-k corpus IDs for a dictionary of queries."""
        results: Dict[str, List[str]] = {}
        for qid, q_data in queries.items():
            query_text = q_data.get("text", "")
            ranked_pairs = self.retrieve(query_text, top_k=top_k)
            results[qid] = [doc_id for doc_id, _ in ranked_pairs]
        return results
