from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Protocol

from tools.qwen_plus_rag_pipeline import ApiResult, PipelineError, QwenClient, REFUSAL_TEXT, UNTRUSTED_EVIDENCE_RULE, score_result, validate_generation

from .budget import BudgetedEvidence, DynamicEvidenceBudget
from .claims import split_cited_segments
from .features import AdaptiveRouter, RouterDecision
from .requirements import (
    Requirement,
    RequirementMapper,
    RequirementPlan,
    deterministic_requirement_plan,
)
from .retrieval import SearchPrincipal, SecondaryRetrievalClient, SecondaryRetrievalResult
from .verifier import ClaimCitationVerifier, VerificationResult

INLINE_CITATION_RE = re.compile(r"\[S[1-9][0-9]*\]")
UNCITED_CAVEAT_RE = re.compile(
    r"(?:evidence (?:does not|doesn't|cannot|can't)|not established|not provided|"
    r"insufficient evidence|cannot determine|cannot verify)",
    re.IGNORECASE,
)

GENERATION_SYSTEM_PROMPT = UNTRUSTED_EVIDENCE_RULE + "\n\n" + """You are the answer stage of an enterprise RAG system.
Use only the supplied authorized evidence. Follow the requirement map and answer every supported requirement.
For missing requirements, explicitly say the supplied evidence does not establish that part; do not invent it.
When sources materially conflict, state the conflict and cite both sides instead of choosing silently.
Preserve exact identifiers, numbers, dates, units, conditions, exceptions, and negations.
Every factual sentence must end with one or more exact source markers such as [S1] or [S1][S2].

Return exactly one JSON object:
{
  "answerable": true,
  "answer": "grounded answer with inline [S1] citations",
  "citations": ["S1"],
  "covered_requirements": ["R1"],
  "missing_requirements": []
}
or
{
  "answerable": false,
  "answer": "INSUFFICIENT_EVIDENCE",
  "citations": [],
  "covered_requirements": [],
  "missing_requirements": ["R1"]
}
Return JSON only."""


class JsonClient(Protocol):
    model: str

    def complete_json(self, **kwargs: Any) -> ApiResult:
        ...


@dataclass(frozen=True)
class AdaptiveRagConfig:
    map_fast_mode: bool = False
    requirements_max_contexts: int = 30
    requirements_max_chars: int = 48_000
    requirements_max_count: int = 12
    requirements_max_selected: int = 16
    requirements_max_tokens: int = 1_024
    generation_max_tokens: int = 1_024
    verifier_mode: str = "conditional"
    verifier_max_chars: int = 24_000
    verifier_max_tokens: int = 1_024
    temperature: float = 0.0
    secondary_max_queries: int = 4


class AdaptiveRagController:
    def __init__(
        self,
        *,
        mapper_client: JsonClient,
        generator_client: JsonClient,
        verifier_client: JsonClient,
        router: AdaptiveRouter | None = None,
        budget: DynamicEvidenceBudget | None = None,
        secondary_retrieval: SecondaryRetrievalClient | None = None,
        config: AdaptiveRagConfig | None = None,
    ) -> None:
        self.mapper = RequirementMapper(mapper_client)
        self.generator_client = generator_client
        self.verifier = ClaimCitationVerifier(verifier_client)
        self.router = router or AdaptiveRouter()
        self.budget = budget or DynamicEvidenceBudget()
        self.secondary_retrieval = secondary_retrieval
        self.config = config or AdaptiveRagConfig()

    def process(
        self,
        row: dict[str, Any],
        *,
        details: dict[str, Any] | None = None,
        principal: SearchPrincipal | None = None,
        forced_mode: str | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        route = self.router.decide(row, details, forced_mode=forced_mode)
        original_contexts = [dict(value) for value in row.get("contexts") or [] if isinstance(value, dict)]
        if not original_contexts:
            return self._no_evidence(row, route, started)

        try:
            plan, requirement_inputs, requirement_chars = self._plan(row, original_contexts, route)
        except Exception as exc:
            return self._error(row, route, "requirements", exc)

        secondary = SecondaryRetrievalResult(
            attempted=False,
            queries=tuple(),
            added_contexts=0,
            merged_contexts=tuple(original_contexts),
            responses=tuple(),
        )
        working_contexts = original_contexts
        remapped = False
        if self.secondary_retrieval is not None and principal is not None and self._needs_secondary(route, plan):
            secondary = self.secondary_retrieval.retrieve_for_plan(
                row={**row, "contexts": working_contexts},
                route=route,
                plan=plan,
                principal=principal,
                max_queries=self.config.secondary_max_queries,
            )
            working_contexts = list(secondary.merged_contexts)
            if secondary.added_contexts > 0:
                try:
                    plan, requirement_inputs, requirement_chars = self._plan(
                        row,
                        working_contexts,
                        route,
                    )
                    remapped = True
                except Exception as exc:
                    secondary = SecondaryRetrievalResult(
                        attempted=secondary.attempted,
                        queries=secondary.queries,
                        added_contexts=secondary.added_contexts,
                        merged_contexts=secondary.merged_contexts,
                        responses=secondary.responses,
                        error=f"requirement remap failed: {exc}",
                    )

        budget_decision = self.budget.decide(route, plan)
        budgeted = self.budget.build(working_contexts, plan=plan, decision=budget_decision)
        if not budgeted.contexts:
            return self._error(row, route, "budget", PipelineError("dynamic budget produced no contexts"))

        generation_api: ApiResult | None = None
        try:
            if plan.answerability == "insufficient" and not secondary.added_contexts:
                generation_value = {
                    "answerable": False,
                    "answer": REFUSAL_TEXT,
                    "citations": [],
                    "covered_requirements": [],
                    "missing_requirements": [requirement.id for requirement in plan.requirements],
                    "citation_normalized": False,
                }
            else:
                valid_citations = {
                    str(context.get("citation_id") or "") for context in budgeted.contexts
                }
                generation_api = self.generator_client.complete_json(
                    messages=build_generation_messages(
                        question=str(row.get("question") or ""),
                        plan=plan,
                        rendered_contexts=budgeted.rendered,
                    ),
                    max_tokens=self.config.generation_max_tokens,
                    temperature=self.config.temperature,
                    validator=lambda payload: validate_adaptive_generation(
                        payload,
                        valid_citations=valid_citations,
                        requirement_ids={requirement.id for requirement in plan.requirements},
                    ),
                )
                generation_value = generation_api.value
        except Exception as exc:
            return self._error(row, route, "generation", exc)

        pre_verification = dict(generation_value)
        try:
            verification = self.verifier.verify(
                question=str(row.get("question") or ""),
                answerable=bool(generation_value["answerable"]),
                answer=str(generation_value["answer"]),
                citations=list(generation_value["citations"]),
                contexts=list(budgeted.contexts),
                plan=plan,
                route=route,
                mode=self.config.verifier_mode,
                max_input_chars=self.config.verifier_max_chars,
                max_tokens=self.config.verifier_max_tokens,
                temperature=self.config.temperature,
            )
        except Exception as exc:
            # A verifier outage must be visible, but it does not discard an already
            # grounded generation. Online policy can choose fail-closed separately.
            verification = VerificationResult(
                triggered=True,
                trigger_reasons=("verifier_error",),
                status="error",
                claims=tuple(),
                answerable=bool(generation_value["answerable"]),
                answer=str(generation_value["answer"]),
                citations=tuple(generation_value["citations"]),
                model=getattr(self.verifier.client, "model", "unknown"),
                latency_ms=float(getattr(exc, "latency_ms", 0.0)),
                usage=dict(getattr(exc, "usage", zero_usage())),
                request_id="",
                attempts=int(getattr(exc, "attempts", 0)),
            )
            verifier_error = str(exc)
        else:
            verifier_error = None

        final_answerable = verification.answerable
        final_answer = verification.answer
        final_citations = list(verification.citations)
        metrics = score_result(
            row,
            selected_contexts=list(budgeted.contexts),
            answerable=final_answerable,
            answer=final_answer,
            citations=final_citations,
        )
        total_latency_ms = (time.perf_counter() - started) * 1000.0
        return {
            "qid": row.get("qid") or row.get("id"),
            "question": row.get("question"),
            "question_type": row.get("question_type"),
            "source_types": row.get("source_types") or [],
            "expected_doc_ids": row.get("expected_doc_ids") or [],
            "expected_accessible_doc_ids": row.get("expected_accessible_doc_ids") or [],
            "gold_answer": row.get("gold_answer"),
            "answer_facts": row.get("answer_facts") or [],
            "is_evaluable": bool(row.get("is_evaluable")),
            "retrieval_hit_at_10": bool(row.get("retrieval_hit_at_10")),
            "router": route.to_dict(),
            "requirements": {
                **plan.to_dict(),
                "input_context_count": len(requirement_inputs),
                "input_context_chars": requirement_chars,
                "remapped_after_secondary_retrieval": remapped,
            },
            "secondary_retrieval": secondary.to_dict(),
            "budget": {
                **budget_decision.to_dict(),
                **budgeted.to_dict(),
            },
            "selected_contexts": list(budgeted.contexts),
            "pre_verification_generation": {
                **pre_verification,
                "model": getattr(self.generator_client, "model", "unknown"),
                "latency_ms": generation_api.latency_ms if generation_api else 0.0,
                "usage": generation_api.usage if generation_api else zero_usage(),
                "request_id": generation_api.request_id if generation_api else "",
                "attempts": generation_api.attempts if generation_api else 0,
            },
            "verification": {
                **verification.to_dict(),
                "error": verifier_error,
            },
            "generation": {
                "answerable": final_answerable,
                "answer": final_answer,
                "citations": final_citations,
                "covered_requirements": pre_verification.get("covered_requirements") or [],
                "missing_requirements": pre_verification.get("missing_requirements") or [],
            },
            "metrics": metrics,
            "usage": aggregate_usage(plan, generation_api, verification),
            "total_latency_ms": total_latency_ms,
            "error": None,
        }

    def _plan(
        self,
        row: dict[str, Any],
        contexts: list[dict[str, Any]],
        route: RouterDecision,
    ) -> tuple[RequirementPlan, list[dict[str, Any]], int]:
        if route.mode == "fast" and not self.config.map_fast_mode:
            return deterministic_requirement_plan(str(row.get("question") or "")), [], 0
        deep = route.mode == "deep"
        return self.mapper.map(
            row,
            contexts,
            max_contexts=self.config.requirements_max_contexts,
            max_input_chars=self.config.requirements_max_chars,
            max_requirements=max(self.config.requirements_max_count, 16) if deep
            else self.config.requirements_max_count,
            max_selected=max(self.config.requirements_max_selected, 24) if deep
            else self.config.requirements_max_selected,
            max_tokens=max(self.config.requirements_max_tokens, 1_536) if deep
            else self.config.requirements_max_tokens,
            temperature=self.config.temperature,
        )

    @staticmethod
    def _needs_secondary(route: RouterDecision, plan: RequirementPlan) -> bool:
        return route.mode == "deep" or (plan.missing_count > 0 and route.risk_score >= 0.35)

    @staticmethod
    def _no_evidence(row: dict[str, Any], route: RouterDecision, started: float) -> dict[str, Any]:
        question = str(row.get("question") or "").strip()
        requirement = Requirement(
            id="R1",
            requirement=question or "Answer the user question",
            status="missing",
            citations=tuple(),
            search_query=question,
        )
        plan = RequirementPlan(
            answerability="insufficient",
            requirements=(requirement,),
            selected_citations=tuple(),
            conflict_citations=tuple(),
            model="no-evidence",
            latency_ms=0.0,
            usage=zero_usage(),
            request_id="",
            attempts=0,
            reported_answerability="insufficient",
            answerability_normalized=False,
        )
        metrics = score_result(
            row,
            selected_contexts=[],
            answerable=False,
            answer=REFUSAL_TEXT,
            citations=[],
        )
        return {
            "qid": row.get("qid") or row.get("id"),
            "question": row.get("question"),
            "question_type": row.get("question_type"),
            "source_types": row.get("source_types") or [],
            "expected_doc_ids": row.get("expected_doc_ids") or [],
            "expected_accessible_doc_ids": row.get("expected_accessible_doc_ids") or [],
            "gold_answer": row.get("gold_answer"),
            "answer_facts": row.get("answer_facts") or [],
            "is_evaluable": bool(row.get("is_evaluable")),
            "retrieval_hit_at_10": bool(row.get("retrieval_hit_at_10")),
            "router": route.to_dict(),
            "requirements": {
                **plan.to_dict(),
                "input_context_count": 0,
                "input_context_chars": 0,
                "remapped_after_secondary_retrieval": False,
            },
            "secondary_retrieval": {
                "attempted": False,
                "queries": [],
                "added_contexts": 0,
                "responses": [],
                "error": None,
            },
            "budget": {
                "mode": route.mode,
                "initial_chars": 0,
                "maximum_chars": 0,
                "max_contexts": 0,
                "max_contexts_per_document": 0,
                "reasons": ["no_authorized_evidence"],
                "rendered_chars": 0,
                "budget_chars": 0,
                "expanded_to_maximum": False,
                "context_count": 0,
                "selected_citations_present": [],
                "selected_citations_missing": [],
            },
            "selected_contexts": [],
            "pre_verification_generation": {
                "answerable": False,
                "answer": REFUSAL_TEXT,
                "citations": [],
                "covered_requirements": [],
                "missing_requirements": ["R1"],
                "model": "deterministic-refusal",
                "latency_ms": 0.0,
                "usage": zero_usage(),
                "request_id": "",
                "attempts": 0,
            },
            "verification": {
                "triggered": False,
                "trigger_reasons": ["no_authorized_evidence"],
                "status": "skipped",
                "claims": [],
                "unsupported_count": 0,
                "answerable": False,
                "answer": REFUSAL_TEXT,
                "citations": [],
                "model": "not_run",
                "latency_ms": 0.0,
                "usage": zero_usage(),
                "request_id": "",
                "attempts": 0,
                "error": None,
            },
            "generation": {
                "answerable": False,
                "answer": REFUSAL_TEXT,
                "citations": [],
                "covered_requirements": [],
                "missing_requirements": ["R1"],
            },
            "metrics": metrics,
            "usage": zero_usage(),
            "total_latency_ms": (time.perf_counter() - started) * 1000.0,
            "error": None,
        }

    @staticmethod
    def _error(
        row: dict[str, Any],
        route: RouterDecision,
        stage: str,
        error: Exception,
    ) -> dict[str, Any]:
        usage = dict(getattr(error, "usage", zero_usage()))
        latency_ms = float(getattr(error, "latency_ms", 0.0))
        attempts = int(getattr(error, "attempts", 0))
        return {
            "qid": row.get("qid") or row.get("id"),
            "question": row.get("question"),
            "question_type": row.get("question_type"),
            "router": route.to_dict(),
            "requirements": None,
            "secondary_retrieval": None,
            "budget": None,
            "selected_contexts": [],
            "generation": {"answerable": False, "answer": "", "citations": []},
            "metrics": {},
            "usage": usage,
            "total_latency_ms": latency_ms,
            "error": {
                "stage": stage,
                "message": str(error),
                "attempts": attempts,
                "max_tokens_used": int(getattr(error, "max_tokens_used", 0)),
                "usage": usage,
                "latency_ms": latency_ms,
            },
        }


def build_generation_messages(
    *,
    question: str,
    plan: RequirementPlan,
    rendered_contexts: str,
) -> list[dict[str, str]]:
    requirements = "\n".join(
        f"{requirement.id}. {requirement.requirement} | status={requirement.status} | "
        f"evidence={','.join(requirement.citations) or 'none'}"
        for requirement in plan.requirements
    )
    user = (
        f"Question:\n{question}\n\n"
        f"Requirement map:\n{requirements}\n\n"
        f"Authorized evidence:\n{rendered_contexts}"
    )
    return [
        {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def validate_adaptive_generation(
    payload: dict[str, Any],
    *,
    valid_citations: set[str],
    requirement_ids: set[str],
) -> dict[str, Any]:
    base = validate_generation(payload, valid_citations=valid_citations)
    covered = _requirement_ids(payload.get("covered_requirements") or [], requirement_ids)
    missing = _requirement_ids(payload.get("missing_requirements") or [], requirement_ids)
    if set(covered) & set(missing):
        raise PipelineError("a requirement cannot be both covered and missing")
    requirement_coverage_normalized = False
    if base["answerable"] and not covered and requirement_ids:
        covered = sorted(requirement_ids - set(missing), key=requirement_sort_key)
        requirement_coverage_normalized = True
    sentence_citation_normalized = False
    sentence_citation_normalized_count = 0
    if base["answerable"]:
        normalized_answer, sentence_citation_normalized_count = normalize_sentence_citations(
            str(base["answer"]),
            list(base["citations"]),
        )
        if sentence_citation_normalized_count:
            base["answer"] = normalized_answer
            sentence_citation_normalized = True
    if not base["answerable"]:
        covered = []
        missing = sorted(requirement_ids, key=requirement_sort_key)
    return {
        **base,
        "covered_requirements": covered,
        "missing_requirements": missing,
        "requirement_coverage_normalized": requirement_coverage_normalized,
        "sentence_citation_normalized": sentence_citation_normalized,
        "sentence_citation_normalized_count": sentence_citation_normalized_count,
    }


def normalize_sentence_citations(answer: str, citations: list[str]) -> tuple[str, int]:
    if not citations:
        return answer, 0
    suffix = "".join(f"[{citation}]" for citation in citations)
    parts: list[str] = []
    cursor = 0
    normalized = 0
    for value in split_cited_segments(answer):
        end = answer.index(value, cursor) + len(value)
        parts.append(answer[cursor:end])
        cursor = end
        segment = value.lstrip("-*•0123456789. )")
        factual = bool(re.search(r"[A-Za-z0-9\u4e00-\u9fff]", segment))
        if (factual and not INLINE_CITATION_RE.search(segment)
                and not UNCITED_CAVEAT_RE.search(segment)):
            # Legacy syntactic recovery only, NOT a claim-support judgment.
            parts.append(" " + suffix)
            normalized += 1
    parts.append(answer[cursor:])
    return "".join(parts), normalized


def uncited_factual_segments(answer: str) -> list[str]:
    raw_segments = split_cited_segments(answer)
    segments: list[str] = []
    citation_only = re.compile(r"(?:\[S[1-9][0-9]*\])+[.,;:!?。！？；：]*")
    for value in raw_segments:
        if citation_only.fullmatch(value) and segments:
            segments[-1] = segments[-1] + " " + value
        else:
            segments.append(value.strip().lstrip("-*•0123456789. )"))
    return [
        segment
        for segment in segments
        if not INLINE_CITATION_RE.search(segment)
        and not UNCITED_CAVEAT_RE.search(segment)
        and re.search(r"[A-Za-z0-9\u4e00-\u9fff]", segment)
    ]


def _requirement_ids(raw: Any, valid: set[str]) -> list[str]:
    if not isinstance(raw, list):
        raise PipelineError("requirement coverage fields must be arrays")
    output: list[str] = []
    for value in raw:
        requirement = str(value).strip().upper()
        if requirement not in valid:
            raise PipelineError(f"unknown requirement ID: {requirement!r}")
        if requirement not in output:
            output.append(requirement)
    return output


def requirement_sort_key(value: str) -> tuple[int, str]:
    if value.startswith("R") and value[1:].isdigit():
        return int(value[1:]), value
    return 10**9, value


def aggregate_usage(
    plan: RequirementPlan,
    generation: ApiResult | None,
    verification: VerificationResult,
) -> dict[str, int]:
    values = [dict(plan.usage), dict(generation.usage) if generation else zero_usage(), dict(verification.usage)]
    return {
        key: sum(int(value.get(key) or 0) for value in values)
        for key in zero_usage()
    }


def zero_usage() -> dict[str, int]:
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cached_tokens": 0}
