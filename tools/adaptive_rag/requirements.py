from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Protocol

from tools.qwen_plus_rag_pipeline import ApiResult, PipelineError, UNTRUSTED_EVIDENCE_RULE, render_contexts

REQUIREMENT_SYSTEM_PROMPT = UNTRUSTED_EVIDENCE_RULE + "\n\n" + """You are the requirement and evidence planner for an enterprise RAG system.
Use only the supplied authorized evidence. Do not answer the user's question and do not use outside knowledge.
Break the question into the smallest independently checkable requirements. For every requirement, decide whether
it is supported, missing, or materially conflicting, and cite only supplied S* evidence IDs.

Return exactly one JSON object:
{
  "answerability": "answerable" | "partial" | "insufficient" | "conflicting",
  "requirements": [
    {
      "id": "R1",
      "requirement": "what must be answered",
      "status": "supported" | "missing" | "conflicting",
      "citations": ["S1"],
      "search_query": "standalone retrieval query if more evidence is needed"
    }
  ],
  "selected_citations": ["S1"],
  "conflict_citations": []
}

Rules:
- Keep requirements faithful to the question; never add benchmark labels or unstated requirements.
- Respect the stated maximums. If a question has many closely related list items, group them into one independently checkable requirement instead of exceeding the limit.
- A supported requirement needs at least one direct citation that contains the requested value itself.
- For complete/exact/exhaustive mappings, lists, thresholds, procedures, or account sets, a pointer to another file,
  team, registry, ticket, or source-of-truth location does NOT support the value requirement unless the supplied
  evidence explicitly enumerates every requested item. Treat the location as a separate requirement.
- A missing requirement has no citations and should include a concise standalone search_query.
- A conflicting requirement needs at least two citations showing the disagreement.
- selected_citations is the ordered union of evidence needed by supported/conflicting requirements.
- Use partial when some requirements are supported and some are missing.
- Use insufficient when none of the requirements is supported.
- Use conflicting when a material conflict prevents one unqualified answer.
- Return JSON only."""


class JsonClient(Protocol):
    model: str

    def complete_json(self, **kwargs: Any) -> ApiResult:
        ...


@dataclass(frozen=True)
class Requirement:
    id: str
    requirement: str
    status: str
    citations: tuple[str, ...]
    search_query: str

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["citations"] = list(self.citations)
        return value


@dataclass(frozen=True)
class RequirementPlan:
    answerability: str
    requirements: tuple[Requirement, ...]
    selected_citations: tuple[str, ...]
    conflict_citations: tuple[str, ...]
    model: str
    latency_ms: float
    usage: dict[str, int]
    request_id: str
    attempts: int
    reported_answerability: str = ""
    answerability_normalized: bool = False
    selection_compacted: bool = False
    selected_citations_before_compaction: int = 0

    @property
    def supported_count(self) -> int:
        return sum(requirement.status == "supported" for requirement in self.requirements)

    @property
    def missing_count(self) -> int:
        return sum(requirement.status == "missing" for requirement in self.requirements)

    @property
    def conflicting_count(self) -> int:
        return sum(requirement.status == "conflicting" for requirement in self.requirements)

    @property
    def coverage(self) -> float:
        if not self.requirements:
            return 0.0
        covered = self.supported_count + self.conflicting_count
        return covered / len(self.requirements)

    def to_dict(self) -> dict[str, Any]:
        return {
            "answerability": self.answerability,
            "requirements": [requirement.to_dict() for requirement in self.requirements],
            "selected_citations": list(self.selected_citations),
            "conflict_citations": list(self.conflict_citations),
            "supported_count": self.supported_count,
            "missing_count": self.missing_count,
            "conflicting_count": self.conflicting_count,
            "coverage": self.coverage,
            "model": self.model,
            "latency_ms": self.latency_ms,
            "usage": dict(self.usage),
            "request_id": self.request_id,
            "attempts": self.attempts,
            "reported_answerability": self.reported_answerability or self.answerability,
            "answerability_normalized": self.answerability_normalized,
            "selection_compacted": self.selection_compacted,
            "selected_citations_before_compaction": self.selected_citations_before_compaction,
        }


class RequirementMapper:
    def __init__(self, client: JsonClient) -> None:
        self.client = client

    def map(
        self,
        row: dict[str, Any],
        contexts: list[dict[str, Any]],
        *,
        max_contexts: int = 30,
        max_input_chars: int = 48_000,
        max_requirements: int = 12,
        max_selected: int = 16,
        max_tokens: int = 1_024,
        temperature: float = 0.0,
    ) -> tuple[RequirementPlan, list[dict[str, Any]], int]:
        rendered, included, rendered_chars = render_contexts(
            contexts,
            max_contexts=max_contexts,
            max_chars=max_input_chars,
        )
        valid_citations = {
            str(context.get("citation_id") or "")
            for context in included
            if context.get("citation_id")
        }
        if not valid_citations:
            return deterministic_requirement_plan(str(row.get("question") or ""), model="no-evidence"), [], 0
        result = self.client.complete_json(
            messages=build_requirement_messages(
                question=str(row.get("question") or ""),
                rendered_contexts=rendered,
                max_requirements=max_requirements,
                max_selected=max_selected,
            ),
            max_tokens=max_tokens,
            temperature=temperature,
            validator=lambda payload: validate_requirement_plan(
                payload,
                valid_citations=valid_citations,
                max_requirements=max_requirements,
                max_selected=max_selected,
            ),
        )
        plan = plan_from_payload(result.value, result)
        return plan, included, rendered_chars


def deterministic_requirement_plan(question: str, *, model: str = "deterministic") -> RequirementPlan:
    requirement = Requirement(
        id="R1",
        requirement=question.strip() or "Answer the user question",
        status="supported",
        citations=tuple(),
        search_query=question.strip(),
    )
    return RequirementPlan(
        answerability="answerable",
        requirements=(requirement,),
        selected_citations=tuple(),
        conflict_citations=tuple(),
        model=model,
        latency_ms=0.0,
        usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cached_tokens": 0},
        request_id="",
        attempts=0,
        reported_answerability="answerable",
        answerability_normalized=False,
        selection_compacted=False,
        selected_citations_before_compaction=0,
    )


def build_requirement_messages(
    *,
    question: str,
    rendered_contexts: str,
    max_requirements: int,
    max_selected: int,
) -> list[dict[str, str]]:
    user = (
        f"Maximum requirements: {max_requirements}\n"
        f"Maximum selected evidence IDs: {max_selected}\n\n"
        f"Question:\n{question}\n\n"
        f"Authorized evidence:\n{rendered_contexts}"
    )
    return [
        {"role": "system", "content": REQUIREMENT_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def validate_requirement_plan(
    payload: dict[str, Any],
    *,
    valid_citations: set[str],
    max_requirements: int,
    max_selected: int,
) -> dict[str, Any]:
    answerability = str(payload.get("answerability") or "").strip().lower()
    if answerability not in {"answerable", "partial", "insufficient", "conflicting"}:
        raise PipelineError(f"invalid requirement answerability: {answerability!r}")
    raw_requirements = payload.get("requirements")
    if not isinstance(raw_requirements, list) or not raw_requirements:
        raise PipelineError("requirements must be a non-empty array")
    if len(raw_requirements) > max_requirements:
        raise PipelineError(f"requirements exceed limit {max_requirements}")

    requirements: list[dict[str, Any]] = []
    observed_ids: set[str] = set()
    inferred_selected: list[str] = []
    inferred_conflicts: list[str] = []
    for index, raw in enumerate(raw_requirements, start=1):
        if not isinstance(raw, dict):
            raise PipelineError("each requirement must be an object")
        requirement_id = str(raw.get("id") or f"R{index}").strip().upper()
        expected_id = f"R{index}"
        if requirement_id != expected_id or requirement_id in observed_ids:
            raise PipelineError(f"requirement IDs must be ordered R1..Rn; got {requirement_id!r}")
        observed_ids.add(requirement_id)
        text = str(raw.get("requirement") or "").strip()
        if not text:
            raise PipelineError(f"{requirement_id} has an empty requirement")
        status = str(raw.get("status") or "").strip().lower()
        if status not in {"supported", "missing", "conflicting"}:
            raise PipelineError(f"invalid status for {requirement_id}: {status!r}")
        citations = _citations(raw.get("citations") or [], valid_citations)
        search_query = str(raw.get("search_query") or "").strip()
        if status == "supported" and not citations:
            raise PipelineError(f"supported {requirement_id} must cite evidence")
        if status == "missing" and citations:
            raise PipelineError(f"missing {requirement_id} must not cite evidence")
        if status == "missing" and not search_query:
            search_query = text
        if status == "conflicting" and len(citations) < 2:
            raise PipelineError(f"conflicting {requirement_id} needs at least two citations")
        for citation in citations:
            if citation not in inferred_selected:
                inferred_selected.append(citation)
            if status == "conflicting" and citation not in inferred_conflicts:
                inferred_conflicts.append(citation)
        requirements.append(
            {
                "id": requirement_id,
                "requirement": text,
                "status": status,
                "citations": citations,
                "search_query": search_query,
            }
        )

    # The selected list is a derived projection of requirement citations, not an
    # independent place where the model may smuggle in extra context. Preserve
    # the model's order for required IDs, drop unrelated extras, then append any
    # required citation it forgot to list.
    reported_selected = _citations(payload.get("selected_citations") or inferred_selected, valid_citations)
    inferred_selected_set = set(inferred_selected)
    selected = [citation for citation in reported_selected if citation in inferred_selected_set]
    for citation in inferred_selected:
        if citation not in selected:
            selected.append(citation)
    selected_before_compaction = len(selected)
    selection_compacted = selected_before_compaction > max_selected
    if selection_compacted:
        selected = compact_requirement_citations(
            requirements,
            preferred_order=selected,
            max_selected=max_selected,
        )
        selected_set = set(selected)
        for requirement in requirements:
            requirement["citations"] = [
                citation for citation in requirement["citations"] if citation in selected_set
            ]

    # Conflict IDs are also a derived projection. The compacting algorithm
    # reserves two citations for every conflicting requirement before filling
    # optional evidence slots, so conflict semantics cannot be truncated away.
    reported_conflicts = _citations(payload.get("conflict_citations") or inferred_conflicts, valid_citations)
    inferred_conflict_set = {
        citation
        for requirement in requirements
        if requirement["status"] == "conflicting"
        for citation in requirement["citations"]
    }
    conflicts = [
        citation for citation in reported_conflicts
        if citation in inferred_conflict_set and citation in selected
    ]
    for citation in selected:
        if citation in inferred_conflict_set and citation not in conflicts:
            conflicts.append(citation)

    statuses = {requirement["status"] for requirement in requirements}
    if statuses == {"supported"}:
        inferred_answerability = "answerable"
    elif statuses == {"missing"}:
        inferred_answerability = "insufficient"
    elif "conflicting" in statuses:
        inferred_answerability = "conflicting"
    else:
        inferred_answerability = "partial"
    return {
        "answerability": inferred_answerability,
        "reported_answerability": answerability,
        "answerability_normalized": inferred_answerability != answerability,
        "requirements": requirements,
        "selected_citations": selected,
        "conflict_citations": conflicts,
        "selection_compacted": selection_compacted,
        "selected_citations_before_compaction": selected_before_compaction,
    }


def compact_requirement_citations(
    requirements: list[dict[str, Any]],
    *,
    preferred_order: list[str],
    max_selected: int,
) -> list[str]:
    """Bound evidence while preserving minimum coverage for every requirement."""
    required: list[str] = []
    all_relevant: list[str] = []
    for requirement in requirements:
        citations = list(requirement.get("citations") or [])
        minimum = 2 if requirement.get("status") == "conflicting" else 1 if citations else 0
        for citation in citations[:minimum]:
            if citation not in required:
                required.append(citation)
        for citation in citations:
            if citation not in all_relevant:
                all_relevant.append(citation)
    if len(required) > max_selected:
        raise PipelineError(
            "minimum evidence needed to preserve requirement coverage exceeds "
            f"selected citation limit {max_selected}"
        )

    order = list(dict.fromkeys(preferred_order + all_relevant))
    required_set = set(required)
    selected = [citation for citation in order if citation in required_set]
    for citation in required:
        if citation not in selected:
            selected.append(citation)
    relevant_set = set(all_relevant)
    for citation in order:
        if len(selected) >= max_selected:
            break
        if citation in relevant_set and citation not in selected:
            selected.append(citation)
    return selected


def plan_from_payload(payload: dict[str, Any], api_result: ApiResult) -> RequirementPlan:
    requirements = tuple(
        Requirement(
            id=str(value["id"]),
            requirement=str(value["requirement"]),
            status=str(value["status"]),
            citations=tuple(str(citation) for citation in value["citations"]),
            search_query=str(value.get("search_query") or ""),
        )
        for value in payload["requirements"]
    )
    return RequirementPlan(
        answerability=str(payload["answerability"]),
        requirements=requirements,
        selected_citations=tuple(str(value) for value in payload["selected_citations"]),
        conflict_citations=tuple(str(value) for value in payload["conflict_citations"]),
        model=api_result.returned_model or api_result.value.get("model", "") or "unknown",
        latency_ms=api_result.latency_ms,
        usage=dict(api_result.usage),
        request_id=api_result.request_id,
        attempts=api_result.attempts,
        reported_answerability=str(payload.get("reported_answerability") or payload["answerability"]),
        answerability_normalized=bool(payload.get("answerability_normalized")),
        selection_compacted=bool(payload.get("selection_compacted")),
        selected_citations_before_compaction=int(
            payload.get("selected_citations_before_compaction") or len(payload["selected_citations"])
        ),
    )


def _citations(raw: Any, valid_citations: set[str]) -> list[str]:
    if not isinstance(raw, list):
        raise PipelineError("citations must be an array")
    result: list[str] = []
    for value in raw:
        citation = str(value).strip().upper()
        if citation not in valid_citations:
            raise PipelineError(f"unknown evidence citation: {citation!r}")
        if citation not in result:
            result.append(citation)
    return result
