from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from tools.qwen_plus_rag_pipeline import fact_scores, token_recall

FAILURE_STAGES = (
    "R1_RETRIEVAL_MISS",
    "R2_EVIDENCE_MISS",
    "R3_WINDOW_MISS",
    "R4_GENERATION_MISS",
    "R5_CITATION_OR_CONFLICT_ERROR",
    "UNANSWERABLE_FALSE_POSITIVE",
    "UNANSWERABLE_OK",
    "OK",
    "UNLABELED",
)


@dataclass(frozen=True)
class FactAttribution:
    fact: str
    stage: str
    full_evidence_recall: float
    final_prompt_recall: float
    answer_recall: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "fact": self.fact,
            "stage": self.stage,
            "full_evidence_recall": self.full_evidence_recall,
            "final_prompt_recall": self.final_prompt_recall,
            "answer_recall": self.answer_recall,
            "reason": self.reason,
        }


def attribute_row(
    source_row: dict[str, Any],
    result_row: dict[str, Any],
    details_row: dict[str, Any] | None = None,
    *,
    coverage_threshold: float = 0.60,
) -> dict[str, Any]:
    facts = [str(value) for value in source_row.get("answer_facts") or [] if str(value).strip()]
    expected = {str(value) for value in source_row.get("expected_doc_ids") or []}
    ranked = _ranked_doc_ids(source_row, result_row, details_row)
    retrieval_hit = not expected or bool(expected & set(ranked[:10]))
    full_contexts = [value for value in source_row.get("contexts") or [] if isinstance(value, dict)]
    final_contexts = [value for value in result_row.get("selected_contexts") or [] if isinstance(value, dict)]
    full_text = "\n".join(str(context.get("text") or "") for context in full_contexts)
    final_text = "\n".join(str(context.get("text") or "") for context in final_contexts)
    answer = str((result_row.get("generation") or {}).get("answer") or result_row.get("answer") or "")
    metrics = result_row.get("metrics") or {}
    invalid_citations = list(metrics.get("invalid_citations") or [])
    verification = result_row.get("verification") or {}
    unsupported_claims = [
        claim
        for claim in verification.get("claims") or []
        if isinstance(claim, dict) and claim.get("status") in {"partial", "unsupported"}
    ]
    conflict_error = _conflict_error(source_row, result_row)

    unanswerable = str(source_row.get("question_type") or "") == "info_not_found"
    model_answerable = bool((result_row.get("generation") or {}).get("answerable"))
    if unanswerable:
        stage = "UNANSWERABLE_FALSE_POSITIVE" if model_answerable else "UNANSWERABLE_OK"
        return {
            "qid": source_row.get("qid") or source_row.get("id"),
            "question": source_row.get("question"),
            "question_type": source_row.get("question_type"),
            "expected_doc_ids": sorted(expected),
            "retrieved_doc_ids": ranked[:10],
            "retrieval_hit_at_10": None,
            "full_evidence_context_count": len(full_contexts),
            "final_prompt_context_count": len(final_contexts),
            "router_mode": (result_row.get("router") or {}).get("mode"),
            "requirement_coverage": (result_row.get("requirements") or {}).get("coverage"),
            "verification_status": verification.get("status"),
            "invalid_citations": invalid_citations,
            "dominant_failure_stage": stage,
            "fact_attributions": [],
            "error": result_row.get("error"),
        }

    attributions: list[FactAttribution] = []
    for fact in facts:
        full_recall = token_recall(full_text, fact)
        prompt_recall = token_recall(final_text, fact)
        answer_recall = token_recall(answer, fact)
        if expected and not retrieval_hit:
            stage = "R1_RETRIEVAL_MISS"
            reason = "no expected document appears in the first ten retrieved documents"
        elif full_recall < coverage_threshold:
            stage = "R2_EVIDENCE_MISS"
            reason = "the retrieved document set exists, but Java evidence does not cover the fact"
        elif prompt_recall < coverage_threshold:
            stage = "R3_WINDOW_MISS"
            reason = "the fact exists in available evidence but was excluded from the final prompt"
        elif answer_recall < coverage_threshold:
            stage = "R4_GENERATION_MISS"
            reason = "the final prompt contains the fact but the answer did not express it"
        elif invalid_citations or unsupported_claims or conflict_error:
            stage = "R5_CITATION_OR_CONFLICT_ERROR"
            reason = "the answer fact is present, but citation support or conflict handling failed"
        else:
            stage = "OK"
            reason = "retrieval, evidence, prompt, answer, and citation checks all passed"
        attributions.append(
            FactAttribution(
                fact=fact,
                stage=stage,
                full_evidence_recall=full_recall,
                final_prompt_recall=prompt_recall,
                answer_recall=answer_recall,
                reason=reason,
            )
        )

    if not facts:
        dominant = "UNLABELED"
    else:
        dominant = _dominant_stage([value.stage for value in attributions])
    return {
        "qid": source_row.get("qid") or source_row.get("id"),
        "question": source_row.get("question"),
        "question_type": source_row.get("question_type"),
        "expected_doc_ids": sorted(expected),
        "retrieved_doc_ids": ranked[:10],
        "retrieval_hit_at_10": retrieval_hit if expected else None,
        "full_evidence_context_count": len(full_contexts),
        "final_prompt_context_count": len(final_contexts),
        "router_mode": (result_row.get("router") or {}).get("mode"),
        "requirement_coverage": (result_row.get("requirements") or {}).get("coverage"),
        "verification_status": verification.get("status"),
        "invalid_citations": invalid_citations,
        "dominant_failure_stage": dominant,
        "fact_attributions": [value.to_dict() for value in attributions],
        "error": result_row.get("error"),
    }


def summarize_attributions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    dominant = Counter(str(row.get("dominant_failure_stage") or "UNLABELED") for row in rows)
    facts = Counter(
        str(fact.get("stage") or "UNLABELED")
        for row in rows
        for fact in row.get("fact_attributions") or []
        if isinstance(fact, dict)
    )
    by_type: dict[str, Counter[str]] = defaultdict(Counter)
    by_mode: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        stage = str(row.get("dominant_failure_stage") or "UNLABELED")
        by_type[str(row.get("question_type") or "unknown")][stage] += 1
        by_mode[str(row.get("router_mode") or "unknown")][stage] += 1
    return {
        "questions_total": len(rows),
        "dominant_failure_stage_counts": _ordered_counts(dominant),
        "fact_failure_stage_counts": _ordered_counts(facts),
        "by_question_type": {
            name: _ordered_counts(values) for name, values in sorted(by_type.items())
        },
        "by_router_mode": {
            name: _ordered_counts(values) for name, values in sorted(by_mode.items())
        },
    }


def _ranked_doc_ids(
    source: dict[str, Any],
    result: dict[str, Any],
    details: dict[str, Any] | None,
) -> list[str]:
    direct = source.get("ranked_doc_ids") or result.get("ranked_doc_ids")
    if isinstance(direct, list):
        return [str(value) for value in direct]
    ranked = (details or {}).get("ranked_documents") or source.get("ranked_documents") or []
    return [str(value.get("doc_id") or "") for value in ranked if isinstance(value, dict)]


def _conflict_error(source: dict[str, Any], result: dict[str, Any]) -> bool:
    conflicts = source.get("evidence_conflicts") or []
    if not conflicts:
        return False
    verification = result.get("verification") or {}
    claims = verification.get("claims") or []
    if any(isinstance(claim, dict) and claim.get("status") == "conflicting" for claim in claims):
        return False
    answer = str((result.get("generation") or {}).get("answer") or "").lower()
    return not any(marker in answer for marker in ("conflict", "disagree", "different", "冲突", "不一致"))


def _dominant_stage(stages: list[str]) -> str:
    priority = {stage: index for index, stage in enumerate(FAILURE_STAGES)}
    failures = [stage for stage in stages if stage != "OK"]
    return min(failures, key=lambda stage: priority.get(stage, 10**9)) if failures else "OK"


def _ordered_counts(values: Counter[str]) -> dict[str, int]:
    return {stage: int(values.get(stage, 0)) for stage in FAILURE_STAGES if values.get(stage, 0)}
