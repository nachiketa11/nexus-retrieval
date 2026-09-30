import re
from typing import Dict, List, Any, Optional, Tuple


def parse_version_intent(query: str) -> Dict[str, Any]:
    """Extract version and deprecation intent keywords from natural language queries.

    Examples:
        "authentication using SDK version 3" -> {"version": "3", "is_sdk": True}
        "replacement for deprecated authentication API" -> {"deprecated": True, "replacement": True}
    """
    intent: Dict[str, Any] = {}
    query_lower = query.lower()

    # Version patterns like "version 3", "v3", "v2.1", "sdk version 3"
    ver_match = re.search(r"\b(?:version|v|sdk\s+version)\s*([0-9]+(?:\.[0-9]+)*)\b", query_lower)
    if ver_match:
        intent["version"] = ver_match.group(1)

    if "deprecated" in query_lower:
        intent["deprecated"] = True
    if "replacement" in query_lower or "replace" in query_lower or "migrate" in query_lower:
        intent["replacement"] = True

    return intent


class VersionAwareFilter:
    """Filters or boosts candidates based on version and metadata match."""

    def filter_candidates(
        self,
        candidate_ids: List[str],
        corpus: Dict[str, Dict],
        version: Optional[str] = None,
        library: Optional[str] = None,
        deprecated_only: Optional[bool] = None,
        min_version: Optional[str] = None,
    ) -> List[str]:
        """Strictly filter candidate documents by metadata criteria."""
        filtered = []

        for cid in candidate_ids:
            doc = corpus.get(cid, {})
            meta = doc.get("meta_information") or doc.get("metadata") or {}

            # Handle version check
            if version:
                doc_ver = str(meta.get("version", ""))
                target_ver = str(version).rstrip(".")
                if doc_ver and doc_ver != target_ver and not doc_ver.startswith(target_ver + "."):
                    continue

            # Handle library check
            if library:
                doc_lib = str(meta.get("library") or meta.get("package") or "").lower()
                if library.lower() not in doc_lib:
                    continue

            # Handle deprecated check
            if deprecated_only is not None:
                doc_dep = bool(meta.get("deprecated", False))
                if doc_dep != deprecated_only:
                    continue

            filtered.append(cid)

        return filtered

    def rerank_by_metadata_match(
        self,
        candidate_ids: List[str],
        corpus: Dict[str, Dict],
        query: str,
        version_override: Optional[str] = None,
    ) -> List[Tuple[str, float]]:
        """Re-weight candidate documents according to metadata relevance to query.

        Returns pairs of (doc_id, metadata_boost_score).
        """
        intent = parse_version_intent(query)
        target_version = version_override or intent.get("version")
        look_for_replacement = intent.get("replacement", False)
        look_for_deprecated = intent.get("deprecated", False)

        results = []
        for rank, cid in enumerate(candidate_ids):
            base_score = 1.0 / (rank + 1.0)
            doc = corpus.get(cid, {})
            meta = doc.get("meta_information") or doc.get("metadata") or {}

            boost = 1.0

            # Version match boost
            doc_ver = str(meta.get("version", ""))
            if target_version and doc_ver:
                if doc_ver == str(target_version):
                    boost += 1.5
                elif doc_ver.startswith(str(target_version)):
                    boost += 0.8
                else:
                    boost -= 0.5

            # Deprecation / Replacement intent match
            if look_for_replacement and meta.get("replacement"):
                boost += 2.0
            if look_for_deprecated and meta.get("deprecated"):
                boost += 1.0

            final_score = base_score * boost
            results.append((cid, final_score))

        # Sort by boosted score descending
        sorted_results = sorted(results, key=lambda x: (-x[1], x[0]))
        return sorted_results
