import re
from typing import Dict, List, Any, Optional, Tuple

_VERSION_RE = re.compile(r"\b(?:sdk\s+version|version|ver\.?|v)\s*([0-9]+(?:\.[0-9]+)*)\b")
_MIGRATION_RE = re.compile(r"\b(?:migrat\w*|upgrad\w*|port(?:ing)?\s+from|move\s+from|switch\s+from)\b")
_REPLACEMENT_RE = re.compile(r"\b(?:replacement|replace[sd]?|successor|instead\s+of|alternative\s+to)\b")
_CURRENT_RE = re.compile(r"\b(?:current|latest|newest|recommended|modern|up[- ]to[- ]date|supported)\b")
_DEPRECATED_RE = re.compile(r"\b(?:deprecated|legacy|obsolete|old)\b")


def _meta(doc: Dict[str, Any]) -> Dict[str, Any]:
    return doc.get("meta_information") or doc.get("metadata") or {}


def version_matches(doc_version: str, target: str) -> bool:
    """True when ``doc_version`` equals ``target`` or refines it ("3.1" matches "3")."""
    doc_version, target = str(doc_version).strip(), str(target).strip().rstrip(".")
    if not doc_version or not target:
        return False
    doc_parts, target_parts = doc_version.split("."), target.split(".")
    # Treat trailing ".0" as equivalent: "3" == "3.0" == "3.0.0".
    while len(target_parts) > 1 and target_parts[-1] == "0":
        target_parts.pop()
    return doc_parts[: len(target_parts)] == target_parts


def parse_version_intent(query: str) -> Dict[str, Any]:
    """Extract version, migration and deprecation intent from a natural-language query.

    Examples:
        "authentication using SDK version 3" -> {"version": "3"}
        "replacement for deprecated authentication API" -> {"deprecated": True, "replacement": True}
        "migrate legacy v1 authentication to the current version"
            -> {"source_version": "1", "replacement": True, "migration": True, "current": True, "deprecated": True}

    In a migration query the mentioned version is the version being migrated *from*, so it is
    reported as ``source_version`` rather than as a target ``version``.
    """
    intent: Dict[str, Any] = {}
    query_lower = query.lower()

    migration = bool(_MIGRATION_RE.search(query_lower))
    ver_match = _VERSION_RE.search(query_lower)
    if ver_match:
        intent["source_version" if migration else "version"] = ver_match.group(1)

    if _DEPRECATED_RE.search(query_lower):
        intent["deprecated"] = True
    if migration:
        intent["migration"] = True
        intent["replacement"] = True
    if _REPLACEMENT_RE.search(query_lower):
        intent["replacement"] = True
    if _CURRENT_RE.search(query_lower):
        intent["current"] = True

    return intent


def replacement_chain(doc_id: str, corpus: Dict[str, Dict], max_hops: int = 5) -> List[str]:
    """Follow ``metadata.replacement`` links from a document to its newest successor."""
    chain: List[str] = []
    current = doc_id
    for _ in range(max_hops):
        nxt = _meta(corpus.get(current, {})).get("replacement")
        if not nxt or nxt not in corpus or nxt in chain or nxt == doc_id:
            break
        chain.append(nxt)
        current = nxt
    return chain


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
        """Strictly filter candidate documents by metadata criteria.

        Documents without version metadata are kept for a version filter (they are not known
        to conflict); documents with a different version are removed.
        """
        filtered = []

        for cid in candidate_ids:
            meta = _meta(corpus.get(cid, {}))

            if version:
                doc_ver = str(meta.get("version", "") or "")
                if doc_ver and not version_matches(doc_ver, version):
                    continue

            if library:
                doc_lib = str(meta.get("library") or meta.get("package") or "").lower()
                if library.lower() not in doc_lib:
                    continue

            if deprecated_only is not None:
                if bool(meta.get("deprecated", False)) != deprecated_only:
                    continue

            filtered.append(cid)

        return filtered

    def rerank_by_metadata_match(
        self,
        candidate_ids: List[str],
        corpus: Dict[str, Dict],
        query: str,
        version_override: Optional[str] = None,
        base_scores: Optional[Dict[str, float]] = None,
    ) -> List[Tuple[str, float]]:
        """Re-weight candidates by how well their metadata matches the query's intent.

        * explicit target version: boost exact/refining matches, demote other versions;
        * replacement / migration / "current" intent: promote non-deprecated documents that are
          the declared replacement of a retrieved deprecated document, and demote deprecated ones;
        * deprecated intent (without replacement intent): promote deprecated documents.

        ``base_scores`` (doc_id -> relevance in (0, 1]) lets callers keep how close candidates were;
        without it the base relevance is ``1 / rank``.

        Returns ``(doc_id, boosted_score)`` pairs sorted by score descending.
        """
        intent = parse_version_intent(query)
        target_version = version_override or intent.get("version")
        source_version = intent.get("source_version")
        wants_replacement = intent.get("replacement", False)
        wants_current = intent.get("current", False) or wants_replacement
        wants_deprecated = intent.get("deprecated", False)

        def version_score(rank: int, meta: Dict[str, Any]) -> float:
            doc_ver = str(meta.get("version", "") or "")
            boost = 1.0
            if target_version and doc_ver:
                boost += 1.5 if version_matches(doc_ver, target_version) else -0.5
            if source_version and doc_ver and version_matches(doc_ver, source_version):
                # The version being migrated from is useful context, but not the answer.
                boost += 0.4
            base = base_scores.get(candidate_ids[rank]) if base_scores else None
            return (base if base is not None else 1.0 / (rank + 1.0)) * max(boost, 0.05)

        scores: Dict[str, float] = {}
        for rank, cid in enumerate(candidate_ids):
            meta = _meta(corpus.get(cid, {}))
            score = version_score(rank, meta)
            if meta.get("deprecated"):
                if wants_current and not wants_deprecated:
                    score *= 0.4          # the user wants current code; keep legacy as context only
                elif wants_deprecated and not wants_current:
                    score *= 2.0          # the user explicitly asked for the deprecated API
            scores[cid] = score

        if wants_current:
            # A replacement is placed just above the deprecated code it replaces, carrying that
            # document's relevance (before any deprecation penalty) — never above unrelated hits.
            for rank, cid in enumerate(candidate_ids):
                meta = _meta(corpus.get(cid, {}))
                if not meta.get("deprecated"):
                    continue
                predecessor_score = version_score(rank, meta)
                for successor in replacement_chain(cid, corpus):
                    if successor in scores and not _meta(corpus.get(successor, {})).get("deprecated"):
                        scores[successor] = max(scores[successor], predecessor_score * 1.05)

        results = list(scores.items())
        return sorted(results, key=lambda x: (-x[1], x[0]))
