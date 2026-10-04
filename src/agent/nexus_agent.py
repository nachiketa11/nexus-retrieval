"""NEXUS retrieval agent.

A deterministic, tool-using agent that wraps the retrieval pipeline in an
analyse -> plan -> act -> reflect -> refine loop:

1. **Analyse** the query: version / migration / deprecation intent, programming
   language, SDK family, and code identifiers.
2. **Plan** which tools to run (lexical, dense, fusion, reranker, version policy,
   deprecation-link resolution) from that analysis and the tools available.
3. **Act**: execute the plan, recording every tool call with its inputs, outputs and latency.
4. **Reflect**: score its own confidence from the retrieval signals (margin between
   the top candidates, dense/lexical agreement, metadata consistency).
5. **Refine**: when confidence is low, rewrite the query (pseudo-relevance feedback or
   relaxing an over-strict version constraint) and run another iteration.

Every decision is returned in ``trace`` so a user can audit why a snippet was chosen.
Tools are injected as callables, so the same agent runs on the PyTorch/FAISS stack,
on the ONNX serverless stack, or with BM25 only.
"""

import math
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..retrieval.bm25 import BM25Retriever, tokenize_code
from ..retrieval.hybrid import reciprocal_rank_fusion
from ..retrieval.version_aware import (
    VersionAwareFilter,
    parse_version_intent,
    replacement_chain,
    version_matches,
)

ScoredIds = List[Tuple[str, float]]
DenseFn = Callable[[str, int], ScoredIds]
RerankFn = Callable[[str, Sequence[str]], ScoredIds]

LANGUAGE_HINTS = {
    "python": ("python", "py"),
    "java": ("java",),
    "kotlin": ("kotlin", "kt"),
    "typescript": ("typescript", "ts", "node", "javascript", "js"),
    "cpp": ("c++", "cpp"),
    "c": ("tizen native", " c "),
}
LIBRARY_HINTS = {
    "samsung_health_sdk": ("samsung health", "health sdk", "step", "heart rate", "sleep"),
    "knox_sdk": ("knox",),
    "knox_sensor_sdk": ("knox sensor", "sensor stream"),
    "smartthings_sdk": ("smartthings", "smartapp"),
    "samsung_iap": ("in-app purchase", "iap", "galaxy store"),
    "tizen_native": ("tizen",),
    "wear_tiles": ("wear os", "tile"),
}
_STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "for", "in", "on", "with", "using", "use", "how", "do", "i",
    "find", "code", "api", "sdk", "version", "current", "latest", "new", "old", "legacy", "is", "it",
    "that", "this", "from", "by", "as", "be", "its", "what", "which", "get", "set", "return", "self",
    "import", "def", "fun", "public", "val", "var", "const", "await", "async", "string", "int", "void",
    "static", "class", "new", "true", "false", "none", "null", "if", "else", "return", "try", "catch",
}


@dataclass
class AgentStep:
    phase: str
    tool: str
    thought: str
    detail: Dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "phase": self.phase,
            "tool": self.tool,
            "thought": self.thought,
            "detail": self.detail,
            "latency_ms": round(self.latency_ms, 2),
        }


class NexusAgent:
    def __init__(
        self,
        corpus: Dict[str, Dict],
        bm25: Optional[BM25Retriever] = None,
        dense: Optional[DenseFn] = None,
        reranker: Optional[RerankFn] = None,
        max_iterations: int = 2,
        candidate_pool: int = 50,
        confidence_threshold: float = 0.55,
    ):
        self.corpus = corpus
        self.bm25 = bm25 or BM25Retriever().fit(corpus)
        self.dense = dense
        self.reranker = reranker
        self.max_iterations = max(1, max_iterations)
        self.candidate_pool = candidate_pool
        self.confidence_threshold = confidence_threshold
        self.version_filter = VersionAwareFilter()

    # ------------------------------------------------------------------ analysis
    @staticmethod
    def analyse(query: str) -> Dict[str, Any]:
        lowered = f" {query.lower()} "
        intent = parse_version_intent(query)
        languages = [lang for lang, keys in LANGUAGE_HINTS.items()
                     if any(re.search(rf"(?<![\w+]){re.escape(k.strip())}(?![\w+])", lowered) for k in keys)]
        libraries = [lib for lib, keys in LIBRARY_HINTS.items() if any(k in lowered for k in keys)]
        identifiers = [tok for tok in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", query)
                       if "_" in tok or re.search(r"[a-z][A-Z]", tok) or tok.endswith("()")]
        words = re.findall(r"[a-z0-9]+", lowered)
        return {
            "intent": intent,
            "languages": languages,
            "libraries": libraries,
            "identifiers": identifiers,
            "is_keyword_query": len(words) <= 3 or bool(identifiers),
            "version_sensitive": bool(intent),
        }

    def plan(self, analysis: Dict[str, Any], explicit_version: Optional[str], use_rerank: Optional[bool]) -> List[str]:
        steps = ["bm25"]
        if self.dense is not None:
            steps += ["dense", "rrf_fusion"]
        rerank_wanted = use_rerank if use_rerank is not None else not analysis["is_keyword_query"]
        if self.reranker is not None and rerank_wanted:
            steps.append("cross_encoder_rerank")
        if explicit_version or analysis["version_sensitive"]:
            steps.append("version_policy")
        if analysis["languages"] or analysis["libraries"]:
            steps.append("context_boost")
        steps += ["deprecation_resolver", "self_check"]
        return steps

    # ------------------------------------------------------------------ main loop
    def run(
        self,
        query: str,
        top_k: int = 5,
        version: Optional[str] = None,
        use_dense: bool = True,
        use_rerank: Optional[bool] = None,
    ) -> Dict[str, Any]:
        started = time.perf_counter()
        trace: List[AgentStep] = []
        dense_fn = self.dense if use_dense else None
        saved_dense, self.dense = self.dense, dense_fn
        try:
            analysis = self.analyse(query)
            trace.append(AgentStep(
                "analyse", "query_analyzer", self._describe_analysis(analysis, version), {"analysis": analysis}))

            plan = self.plan(analysis, version, use_rerank)
            trace.append(AgentStep("plan", "planner", "Execution plan: " + " → ".join(plan) + ".", {"plan": plan}))

            current_query, current_version = query, version
            best: Optional[Dict[str, Any]] = None
            for iteration in range(1, self.max_iterations + 1):
                outcome = self._iterate(current_query, query, analysis, plan, current_version, top_k, trace, iteration)
                if best is None or outcome["confidence"] > best["confidence"]:
                    best = outcome
                if outcome["confidence"] >= self.confidence_threshold or iteration == self.max_iterations:
                    break
                refined_query, refined_version, reason = self._refine(current_query, current_version, outcome, analysis)
                if refined_query == current_query and refined_version == current_version:
                    trace.append(AgentStep("refine", "query_rewriter",
                                           "No useful refinement available; keeping the best result so far."))
                    break
                trace.append(AgentStep("refine", "query_rewriter", reason,
                                       {"query": refined_query, "version": refined_version}))
                current_query, current_version = refined_query, refined_version

            assert best is not None
            answer = self._compose_answer(query, best["results"], analysis, best["confidence"])
            trace.append(AgentStep("respond", "answer_composer", answer["summary"]))
            return {
                "query": query,
                "final_query": best["query"],
                "analysis": analysis,
                "plan": plan,
                "iterations": best["iteration"],
                "confidence": round(best["confidence"], 3),
                "confidence_signals": best["signals"],
                "answer": answer,
                "results": best["results"],
                "trace": [s.as_dict() for s in trace],
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            }
        finally:
            self.dense = saved_dense

    # ------------------------------------------------------------------ one iteration
    def _iterate(self, query: str, original_query: str, analysis: Dict[str, Any], plan: List[str],
                 version: Optional[str], top_k: int, trace: List[AgentStep], iteration: int) -> Dict[str, Any]:
        signals: Dict[str, Dict[str, Any]] = {}
        pool = self.candidate_pool

        t = time.perf_counter()
        lexical = self.bm25.retrieve(query, top_k=pool)
        lexical = [(cid, s) for cid, s in lexical if s > 0] or lexical[: min(pool, 10)]
        for rank, (cid, score) in enumerate(lexical, 1):
            signals.setdefault(cid, {})["bm25"] = round(score, 4)
            signals[cid]["bm25_rank"] = rank
        trace.append(AgentStep("act", "bm25", f"[iter {iteration}] Lexical search matched {len(lexical)} candidates "
                               f"on code-aware tokens.", {"top": self._top(lexical)}, self._ms(t)))
        ranking = [cid for cid, _ in lexical]

        if "dense" in plan and self.dense is not None:
            t = time.perf_counter()
            semantic = self.dense(query, pool)
            for rank, (cid, score) in enumerate(semantic, 1):
                signals.setdefault(cid, {})["dense"] = round(score, 4)
                signals[cid]["dense_rank"] = rank
            trace.append(AgentStep("act", "dense_e5", f"[iter {iteration}] Semantic search with E5 embeddings.",
                                   {"top": self._top(semantic)}, self._ms(t)))
            t = time.perf_counter()
            fused = reciprocal_rank_fusion([[c for c, _ in semantic], ranking], top_k=pool)
            for cid, score in fused:
                signals[cid]["rrf"] = round(score, 5)
            ranking = [cid for cid, _ in fused]
            trace.append(AgentStep("act", "rrf_fusion", "Fused lexical and semantic rankings with reciprocal-rank fusion.",
                                   {"top": self._top(fused)}, self._ms(t)))

        if "cross_encoder_rerank" in plan and self.reranker is not None and ranking:
            t = time.perf_counter()
            head = ranking[: max(top_k * 3, 15)]
            reranked = self.reranker(query, head)
            for cid, score in reranked:
                signals[cid]["rerank"] = round(score, 4)
            ranking = [cid for cid, _ in reranked] + ranking[len(head):]
            trace.append(AgentStep("act", "cross_encoder", f"Re-scored the top {len(head)} candidates jointly with "
                                   "the query using the MS MARCO cross-encoder.", {"top": self._top(reranked)}, self._ms(t)))

        intent = parse_version_intent(original_query)
        if version:
            t = time.perf_counter()
            kept = self.version_filter.filter_candidates(ranking, self.corpus, version=version)
            trace.append(AgentStep("act", "version_filter", f"Applied explicit version constraint {version}: kept "
                                   f"{len(kept)} of {len(ranking)} candidates.", {"version": version}, self._ms(t)))
            ranking = kept
        elif "version_policy" in plan:
            t = time.perf_counter()
            boosted = self.version_filter.rerank_by_metadata_match(
                ranking, self.corpus, original_query, base_scores=self._relevance(ranking, signals))
            ranking = [cid for cid, _ in boosted]
            trace.append(AgentStep("act", "version_policy", "Re-weighted candidates by version/deprecation intent "
                                   f"{self._intent_text(intent)}.", {"top": self._top(boosted)}, self._ms(t)))

        if "context_boost" in plan:
            ranking = self._context_boost(ranking, analysis, trace)

        ranking, notes = self._resolve_deprecations(ranking, intent, version, analysis, trace)

        results = self._materialise(ranking[:top_k], signals, notes)
        confidence, conf_signals = self._self_check(original_query, results, signals, analysis, version, intent)
        trace.append(AgentStep("reflect", "self_check",
                               f"Confidence {confidence:.2f} "
                               + ("≥" if confidence >= self.confidence_threshold else "<")
                               + f" threshold {self.confidence_threshold:.2f}. " + conf_signals["explanation"],
                               {k: v for k, v in conf_signals.items() if k != "explanation"}))
        return {"iteration": iteration, "query": query, "results": results,
                "confidence": confidence, "signals": conf_signals}

    # ------------------------------------------------------------------ tools
    @staticmethod
    def _relevance(ranking: List[str], signals: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """Min-max normalised relevance from the strongest signal available for every candidate.

        Uses cross-encoder > RRF > BM25 > dense; candidates lacking that signal (e.g. below the
        reranked head) are scaled to sit under the lowest scored one.
        """
        for key in ("rerank", "rrf", "bm25", "dense"):
            scored = {cid: signals[cid][key] for cid in ranking if key in signals.get(cid, {})}
            if len(scored) >= 2:
                lo, hi = min(scored.values()), max(scored.values())
                span = (hi - lo) or 1.0
                rel = {cid: 0.1 + 0.9 * (s - lo) / span for cid, s in scored.items()}
                floor = 0.1
                for rank, cid in enumerate(ranking):
                    if cid not in rel:
                        floor *= 0.95
                        rel[cid] = floor
                return rel
        return {}

    def _context_boost(self, ranking: List[str], analysis: Dict[str, Any], trace: List[AgentStep]) -> List[str]:
        t = time.perf_counter()
        langs, libs = set(analysis["languages"]), set(analysis["libraries"])

        def key(item):
            idx, cid = item
            meta = self._meta(cid)
            bonus = 0
            if langs and str(meta.get("language", "")).lower() in langs:
                bonus += 1
            if libs and str(meta.get("library", "")).lower() in libs:
                bonus += 1
            return idx - bonus * 2.5

        boosted = [cid for _, cid in sorted(enumerate(ranking), key=key)]
        what = ", ".join(sorted(langs | libs))
        trace.append(AgentStep("act", "context_boost", f"Promoted candidates matching detected context ({what}).",
                               {"languages": sorted(langs), "libraries": sorted(libs)}, self._ms(t)))
        return boosted

    def _resolve_deprecations(self, ranking: List[str], intent: Dict[str, Any], version: Optional[str],
                              analysis: Dict[str, Any], trace: List[AgentStep]) -> Tuple[List[str], Dict[str, str]]:
        t = time.perf_counter()
        notes: Dict[str, str] = {}
        explicitly_deprecated = intent.get("deprecated") and not (intent.get("replacement") or intent.get("current"))
        wants_successor = bool(intent.get("replacement") or intent.get("current") or intent.get("migration"))
        target_version = None if wants_successor else intent.get("version")
        identifiers = [] if wants_successor else [i.lower() for i in analysis.get("identifiers", [])]
        result = list(ranking)
        hops, kept, ahead = [], [], []
        for cid in ranking[:5]:
            meta = self._meta(cid)
            if not meta.get("deprecated"):
                continue
            chain = replacement_chain(cid, self.corpus)
            if not chain:
                continue
            successor = chain[-1]
            notes[cid] = f"Deprecated — superseded by {successor}"
            notes.setdefault(successor, f"Replacement for deprecated {cid}")
            if version and not version_matches(self._meta(successor).get("version", ""), version):
                continue
            # Respect the user: an explicitly requested deprecated API, a target version that the
            # deprecated document satisfies, or an exact symbol lookup keeps the original order.
            text = self.corpus.get(cid, {}).get("text", "").lower()
            if explicitly_deprecated:
                kept.append(f"{cid} (query asks for deprecated code)")
                continue
            if target_version and version_matches(meta.get("version", ""), target_version):
                kept.append(f"{cid} (matches requested version {target_version})")
                continue
            if any(ident in text for ident in identifiers):
                kept.append(f"{cid} (exact identifier match)")
                continue
            if successor in result and result.index(successor) < result.index(cid):
                ahead.append(f"{cid} → {successor}")
                continue
            if successor in result:
                result.remove(successor)
            result.insert(result.index(cid), successor)
            hops.append({"from": cid, "to": successor, "chain": chain})
        parts = []
        if hops:
            parts.append("Followed deprecation links: " + "; ".join(f"{h['from']} → {h['to']}" for h in hops)
                         + " — successors now rank above the code they replace")
        if ahead:
            parts.append("Successors already rank above their deprecated predecessors: " + "; ".join(ahead))
        if kept:
            parts.append("Kept deprecated code in place (annotated with its successor): " + "; ".join(kept))
        if notes and not parts:
            parts.append("Deprecated candidates found; their successors fall outside the version constraint")
        thought = (". ".join(parts) + ".") if parts else "No deprecated candidates in the top results."
        trace.append(AgentStep("act", "deprecation_resolver", thought, {"hops": hops}, self._ms(t)))
        return result, notes

    def _self_check(self, query: str, results: List[Dict[str, Any]], signals: Dict[str, Dict[str, Any]],
                    analysis: Dict[str, Any], version: Optional[str], intent: Dict[str, Any]
                    ) -> Tuple[float, Dict[str, Any]]:
        if not results:
            return 0.0, {"explanation": "No candidates survived; the query or constraint is too narrow."}
        top = results[0]
        sig = signals.get(top["doc_id"], {})
        parts, reasons = [], []

        # 1. lexical evidence: how many query terms appear in the top document
        q_terms = {t for t in tokenize_code(query) if t not in _STOPWORDS}
        doc_terms = set(tokenize_code(self.corpus.get(top["doc_id"], {}).get("text", "")))
        coverage = len(q_terms & doc_terms) / max(len(q_terms), 1)
        parts.append(min(1.0, coverage * 1.4))
        reasons.append(f"{coverage:.0%} query-term coverage")

        # 2. agreement between retrievers
        if "dense_rank" in sig or "bm25_rank" in sig:
            best_rank = min(sig.get("dense_rank", 99), sig.get("bm25_rank", 99))
            agree = 1.0 if sig.get("dense_rank", 99) <= 3 and sig.get("bm25_rank", 99) <= 3 else \
                0.6 if best_rank <= 3 else 0.3
            parts.append(agree)
            reasons.append("retrievers agree" if agree == 1.0 else "partial retriever agreement" if agree > 0.5
                           else "weak retriever agreement")

        # 3. margin over the best competing candidate (cross-encoder if available, else BM25).
        # Deprecated predecessors of the top result are not competitors: they are its migration source.
        if len(results) > 1:
            key = "rerank" if "rerank" in sig else "bm25" if "bm25" in sig else None
            if key:
                top_id = top["doc_id"]
                rivals = [signals.get(r["doc_id"], {}).get(key) for r in results[1:]
                          if top_id not in replacement_chain(r["doc_id"], self.corpus)]
                rivals = [s for s in rivals if s is not None]
                if rivals:
                    s1, s2 = sig.get(key, 0.0), max(rivals)
                    if key == "rerank":   # logits: a 2-logit lead is a clear win
                        margin = 1.0 - math.exp(-max(0.0, s1 - s2) / 2.0)
                    else:                 # BM25: relative lead over the runner-up
                        margin = min(1.0, max(0.0, (s1 - s2) / max(abs(s1), 1e-9)) * 2.0)
                    parts.append(margin)
                    reasons.append(f"{key} margin {margin:.2f}")

        # 4. metadata consistency
        meta = top.get("metadata", {})
        meta_ok = 1.0
        if version and not version_matches(meta.get("version", ""), version):
            meta_ok = 0.0
        if (intent.get("current") or intent.get("replacement")) and meta.get("deprecated"):
            meta_ok = min(meta_ok, 0.2)
        if analysis["languages"] and str(meta.get("language", "")).lower() not in analysis["languages"]:
            meta_ok = min(meta_ok, 0.6)
        parts.append(meta_ok)
        reasons.append("metadata consistent" if meta_ok == 1.0 else "metadata partially inconsistent")

        confidence = sum(parts) / len(parts)
        return confidence, {"coverage": round(coverage, 3), "components": [round(p, 3) for p in parts],
                            "explanation": "Signals: " + ", ".join(reasons) + "."}

    def _refine(self, query: str, version: Optional[str], outcome: Dict[str, Any],
                analysis: Dict[str, Any]) -> Tuple[str, Optional[str], str]:
        if version and not outcome["results"]:
            return query, None, f"Version {version} eliminated every candidate; relaxing the hard filter and " \
                                f"letting the version policy boost matches instead."
        top_docs = [r["doc_id"] for r in outcome["results"][:3]]
        counts: Counter = Counter()
        q_tokens = set(tokenize_code(query))
        for cid in top_docs:
            toks = {t for t in tokenize_code(self.corpus.get(cid, {}).get("text", ""))
                    if len(t) > 3 and t not in _STOPWORDS and t not in q_tokens and not t.isdigit()}
            counts.update(toks)
        expansion = [tok for tok, n in counts.most_common(12) if n >= 2][:4]
        if not expansion and top_docs:
            meta = self._meta(top_docs[0])
            expansion = [t for t in tokenize_code(str(meta.get("library", ""))) if t not in q_tokens][:2]
        if not expansion:
            return query, version, ""
        new_query = f"{query} {' '.join(expansion)}"
        return new_query, version, "Low confidence: expanding the query with terms shared by the top candidates " \
                                   f"(pseudo-relevance feedback): {', '.join(expansion)}."

    # ------------------------------------------------------------------ helpers
    def _compose_answer(self, query: str, results: List[Dict[str, Any]], analysis: Dict[str, Any],
                        confidence: float) -> Dict[str, Any]:
        if not results:
            return {"summary": "No matching code found. Try broadening the query or removing the version filter.",
                    "best": None, "migration": None, "warnings": []}
        best = results[0]
        meta = best["metadata"]
        loc = meta.get("file") or best["doc_id"]
        ver = f" v{meta['version']}" if meta.get("version") else ""
        lib = meta.get("library", "")
        lead = "Best match" if confidence >= self.confidence_threshold else             "Low confidence — closest candidate, verify before use"
        summary = f"{lead}: {loc} ({lib}{ver}, {meta.get('language', 'code')})."
        warnings, migration = [], None
        if meta.get("deprecated"):
            warnings.append(f"{loc} is deprecated.")
        # Show a migration path only when it involves the chosen answer: either the answer is
        # deprecated (path forward from it) or the answer is the successor of a deprecated document.
        path: List[str] = []
        if meta.get("deprecated"):
            path = [best["doc_id"]] + replacement_chain(best["doc_id"], self.corpus)
        else:
            predecessors = [cid for cid in self.corpus
                            if best["doc_id"] in replacement_chain(cid, self.corpus)]
            if predecessors:
                # Prefer the predecessor the agent actually retrieved (best rank), then the oldest version.
                ranked = {r["doc_id"]: r["rank"] for r in results}
                source = min(predecessors, key=lambda c: (ranked.get(c, 10 ** 6),
                                                          -len(replacement_chain(c, self.corpus)),
                                                          str(self._meta(c).get("version", "")), c))
                path = [source] + replacement_chain(source, self.corpus)
                path = path[: path.index(best["doc_id"]) + 1]
        if len(path) > 1:
            versions = [f"v{self._meta(c).get('version', '?')}" for c in path]
            migration = {"path": path, "versions": versions}
            summary += f" Migration path: {' → '.join(versions)} ({' → '.join(path)})."
        return {"summary": summary, "best": best["doc_id"], "migration": migration, "warnings": warnings}

    def _materialise(self, ids: List[str], signals: Dict[str, Dict[str, Any]], notes: Dict[str, str]
                     ) -> List[Dict[str, Any]]:
        out = []
        for rank, cid in enumerate(ids, 1):
            doc = self.corpus.get(cid, {})
            out.append({
                "doc_id": cid,
                "rank": rank,
                "text": doc.get("text", ""),
                "metadata": self._meta(cid),
                "signals": signals.get(cid, {}),
                "note": notes.get(cid),
            })
        return out

    def _meta(self, cid: str) -> Dict[str, Any]:
        doc = self.corpus.get(cid, {})
        return doc.get("meta_information") or doc.get("metadata") or {}

    @staticmethod
    def _top(pairs: Sequence[Tuple[str, float]], n: int = 3) -> List[Dict[str, Any]]:
        return [{"doc_id": c, "score": round(float(s), 4)} for c, s in list(pairs)[:n]]

    @staticmethod
    def _ms(start: float) -> float:
        return (time.perf_counter() - start) * 1000

    @staticmethod
    def _intent_text(intent: Dict[str, Any]) -> str:
        return "(" + ", ".join(f"{k}={v}" for k, v in intent.items()) + ")" if intent else "(none)"

    def _describe_analysis(self, analysis: Dict[str, Any], version: Optional[str]) -> str:
        bits = []
        intent = analysis["intent"]
        if intent.get("migration"):
            bits.append(f"migration request from v{intent.get('source_version', '?')}")
        if intent.get("version"):
            bits.append(f"targets version {intent['version']}")
        if intent.get("replacement") and not intent.get("migration"):
            bits.append("asks for the replacement API")
        if intent.get("current"):
            bits.append("wants current/non-deprecated API")
        if intent.get("deprecated"):
            bits.append("mentions deprecated/legacy code")
        if version:
            bits.append(f"explicit version filter {version}")
        if analysis["languages"]:
            bits.append("language: " + ", ".join(analysis["languages"]))
        if analysis["libraries"]:
            bits.append("SDK: " + ", ".join(analysis["libraries"]))
        if analysis["identifiers"]:
            bits.append("identifiers: " + ", ".join(analysis["identifiers"]))
        bits.append("keyword-style query" if analysis["is_keyword_query"] else "natural-language query")
        return "Query analysis — " + "; ".join(bits) + "."

