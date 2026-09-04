from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable

TOKEN_RE = re.compile(r"[a-z0-9_][a-z0-9_./:+#-]*|[\u4e00-\u9fff]", re.IGNORECASE)
EXACT_ANCHOR_RE = re.compile(
    r"(?:\b[A-Z]{2,}[A-Z0-9_-]*-?\d+\b|\bv?\d+(?:\.\d+){1,3}\b|\b\d+(?:\.\d+)?%?\b|"
    r"\b\d{4}-\d{2}-\d{2}\b|`[^`]+`)",
    re.IGNORECASE,
)
MULTIPART_RE = re.compile(
    r"(?:\band\b|\balso\b|\bincluding\b|\bas well as\b|\brespectively\b|"
    r"\bwhat caused\b.*\bwhat\b|\bwhich\b.*\band\b|以及|并且|分别|同时|还要|包括)",
    re.IGNORECASE | re.DOTALL,
)
CONFLICT_RE = re.compile(r"(?:conflict|disagree|different version|which is correct|冲突|不一致|哪个为准)", re.IGNORECASE)

CONTENT_STOPWORDS = {
    "a", "about", "after", "all", "an", "and", "are", "as", "at", "be", "before",
    "by", "can", "did", "do", "does", "during", "for", "from", "has", "have", "how",
    "in", "into", "is", "it", "its", "of", "on", "or", "that", "the", "their", "this",
    "to", "was", "were", "what", "when", "where", "which", "who", "why", "with",
}


@dataclass(frozen=True)
class RouterFeatures:
    query_token_count: int
    exact_anchor_count: int
    subquestion_count: int
    conflict_language: bool
    context_count: int
    context_chars: int
    distinct_document_count: int
    distinct_source_count: int
    route_count: int
    top1_route_support: int
    top1_route_agreement: float
    top1_margin: float | None
    average_query_coverage: float | None
    maximum_query_coverage: float | None
    average_redundancy: float
    evidence_conflict_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RouterDecision:
    mode: str
    risk_score: float
    reasons: tuple[str, ...]
    features: RouterFeatures

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "risk_score": self.risk_score,
            "reasons": list(self.reasons),
            "features": self.features.to_dict(),
        }


@dataclass(frozen=True)
class RouterConfig:
    quality_risk_threshold: float = 0.34
    deep_risk_threshold: float = 0.68
    low_margin_threshold: float = 0.012
    low_query_coverage_threshold: float = 0.22
    high_redundancy_threshold: float = 0.58


class AdaptiveRouter:
    """Deterministic router using only online-observable retrieval/query signals."""

    def __init__(self, config: RouterConfig | None = None) -> None:
        self.config = config or RouterConfig()

    def decide(
        self,
        row: dict[str, Any],
        details: dict[str, Any] | None = None,
        *,
        forced_mode: str | None = None,
    ) -> RouterDecision:
        features = extract_features(row, details)
        if forced_mode is not None:
            if forced_mode not in {"fast", "quality", "deep"}:
                raise ValueError(f"unsupported forced mode: {forced_mode}")
            return RouterDecision(forced_mode, 1.0 if forced_mode == "deep" else 0.0, ("forced",), features)

        score = 0.0
        reasons: list[str] = []

        if features.evidence_conflict_count > 0 or features.conflict_language:
            score += 0.48
            reasons.append("conflicting_evidence_or_query")
        if features.subquestion_count >= 3:
            score += 0.38
            reasons.append("three_or_more_requirements")
        elif features.subquestion_count == 2:
            score += 0.20
            reasons.append("multi_part_question")
        if features.exact_anchor_count >= 2:
            # Exact identifiers make lexical retrieval strong but increase the cost of
            # silently dropping a requested constraint.
            score += 0.06
            reasons.append("multiple_exact_constraints")
        if features.route_count >= 2 and features.top1_route_agreement < 0.75:
            score += 0.18
            reasons.append("retrieval_routes_disagree")
        if features.route_count >= 3 and features.top1_route_support < 2:
            score += 0.12
            reasons.append("top_document_has_weak_route_support")
        if features.top1_margin is not None and features.top1_margin < self.config.low_margin_threshold:
            score += 0.15
            reasons.append("low_top1_margin")
        if (
            features.average_query_coverage is not None
            and features.average_query_coverage < self.config.low_query_coverage_threshold
        ):
            score += 0.10
            reasons.append("low_evidence_query_coverage")
        if features.average_redundancy > self.config.high_redundancy_threshold:
            score += 0.09
            reasons.append("high_context_redundancy")
        if features.distinct_source_count >= 3 and features.subquestion_count >= 2:
            score += 0.12
            reasons.append("cross_source_multi_part")
        # Raw candidate size is not a quality signal by itself: the Java
        # EvidenceBuilder emits a large pool for almost every benchmark row.
        # Budget pressure is handled later by DynamicEvidenceBudget instead of
        # forcing nearly all queries into Quality mode.
        score = min(1.0, score)
        low_confidence = (
            (features.route_count >= 2 and features.top1_route_agreement < 0.75)
            or (features.top1_margin is not None and features.top1_margin < self.config.low_margin_threshold)
        )
        if (
            score >= self.config.deep_risk_threshold
            or features.evidence_conflict_count > 0
            or (features.subquestion_count >= 3 and low_confidence)
        ):
            mode = "deep"
        elif score >= self.config.quality_risk_threshold:
            mode = "quality"
        else:
            mode = "fast"
        if not reasons:
            reasons.append("high_confidence_single_requirement")
        return RouterDecision(mode, score, tuple(reasons), features)


def extract_features(
    row: dict[str, Any],
    details: dict[str, Any] | None = None,
) -> RouterFeatures:
    question = str(row.get("question") or "")
    tokens = content_tokens(question)
    contexts = [value for value in row.get("contexts") or [] if isinstance(value, dict)]
    distinct_docs = {str(context.get("doc_id") or "") for context in contexts if context.get("doc_id")}
    distinct_sources = {
        str(context.get("source_type") or "unknown") for context in contexts if context.get("source_type")
    }
    coverages = [
        float(context["query_coverage"])
        for context in contexts
        if isinstance(context.get("query_coverage"), (int, float))
        and math.isfinite(float(context["query_coverage"]))
    ]

    route_top_docs: dict[str, str] = {}
    all_routes: set[str] = set()
    top_document = _top_document_id(contexts)
    top1_route_support = 0
    for context in contexts:
        doc_id = str(context.get("doc_id") or "")
        for signal in context.get("route_signals") or []:
            if not isinstance(signal, dict):
                continue
            route = str(signal.get("route") or "")
            if not route:
                continue
            all_routes.add(route)
            if int(signal.get("rank") or 0) == 1 and route not in route_top_docs:
                route_top_docs[route] = doc_id
            if doc_id == top_document:
                top1_route_support += 1
    # Multiple chunks from the same document can carry the same signal.
    if top_document:
        top1_route_support = len(
            {
                str(signal.get("route"))
                for context in contexts
                if str(context.get("doc_id") or "") == top_document
                for signal in context.get("route_signals") or []
                if isinstance(signal, dict) and signal.get("route")
            }
        )
    agreement = 1.0
    if route_top_docs:
        counts: dict[str, int] = {}
        for doc_id in route_top_docs.values():
            counts[doc_id] = counts.get(doc_id, 0) + 1
        agreement = max(counts.values()) / len(route_top_docs)

    ranked = None
    if details:
        ranked = details.get("ranked_documents")
    if not ranked:
        ranked = row.get("ranked_documents")
    top1_margin = _normalized_margin(ranked)

    subquestions = estimate_subquestion_count(question)
    conflicts = row.get("evidence_conflicts") or []
    return RouterFeatures(
        query_token_count=len(tokens),
        exact_anchor_count=len(EXACT_ANCHOR_RE.findall(question)),
        subquestion_count=subquestions,
        conflict_language=bool(CONFLICT_RE.search(question)),
        context_count=len(contexts),
        context_chars=sum(len(str(context.get("text") or "")) for context in contexts),
        distinct_document_count=len(distinct_docs),
        distinct_source_count=len(distinct_sources),
        route_count=len(all_routes),
        top1_route_support=top1_route_support,
        top1_route_agreement=agreement,
        top1_margin=top1_margin,
        average_query_coverage=(sum(coverages) / len(coverages)) if coverages else None,
        maximum_query_coverage=max(coverages) if coverages else None,
        average_redundancy=average_redundancy(contexts[:12]),
        evidence_conflict_count=len(conflicts) if isinstance(conflicts, list) else 0,
    )


def estimate_subquestion_count(question: str) -> int:
    if not question.strip():
        return 1
    count = max(1, question.count("?") + question.count("？"))
    markers = len(MULTIPART_RE.findall(question))
    if markers:
        count = max(count, min(4, 1 + markers))
    # Colon/semicolon-separated requests are common in runbook questions.
    count = max(count, min(4, 1 + question.count(";") + question.count("；")))
    return count


def content_tokens(text: str) -> list[str]:
    return [token for token in TOKEN_RE.findall(text.lower()) if token not in CONTENT_STOPWORDS]


def average_redundancy(contexts: list[dict[str, Any]]) -> float:
    token_sets = [set(content_tokens(str(context.get("text") or ""))) for context in contexts]
    token_sets = [tokens for tokens in token_sets if tokens]
    if len(token_sets) < 2:
        return 0.0
    values: list[float] = []
    for index, left in enumerate(token_sets):
        for right in token_sets[index + 1 :]:
            union = left | right
            if union:
                values.append(len(left & right) / len(union))
    return sum(values) / len(values) if values else 0.0


def _top_document_id(contexts: list[dict[str, Any]]) -> str:
    ranked = sorted(
        contexts,
        key=lambda context: (
            int(context.get("document_rank") or context.get("rank") or 10**9),
            int(context.get("rank") or 10**9),
        ),
    )
    return str(ranked[0].get("doc_id") or "") if ranked else ""


def _normalized_margin(ranked: Any) -> float | None:
    if not isinstance(ranked, list) or len(ranked) < 2:
        return None
    first = ranked[0] if isinstance(ranked[0], dict) else {}
    second = ranked[1] if isinstance(ranked[1], dict) else {}
    if not isinstance(first.get("score"), (int, float)) or not isinstance(second.get("score"), (int, float)):
        return None
    first_score = float(first["score"])
    second_score = float(second["score"])
    denominator = max(abs(first_score), 1e-9)
    return max(0.0, (first_score - second_score) / denominator)
