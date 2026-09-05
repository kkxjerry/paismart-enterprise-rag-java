"""Requirement- and canonical-document-aware generation (E2).

The generator receives explicit document roles instead of a flat S* list. For
single-artifact questions, only the selected canonical document may support the
answer. Supplemental documents are either omitted or clearly isolated. This is
an opt-in experiment and does not alter the legacy generation path.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Protocol, Sequence

from tools.qwen_plus_rag_pipeline import ApiResult, PipelineError, UNTRUSTED_EVIDENCE_RULE

from .hierarchical_evidence import HierarchicalEvidence, requires_canonical_document

_REQUIREMENT_SPLIT = re.compile(
    r"\s*(?:;|\band\s+(?=(?:what|which|when|where|who|how|whether)\b))\s*",
    re.I,
)
_COMPARATIVE = re.compile(r"\b(?:compare|versus|vs\.?|difference|conflict|disagree|across)\b|比较|冲突|差异", re.I)
_CITATION = re.compile(r"S[1-9][0-9]*\Z")

CANONICAL_GENERATION_SYSTEM_PROMPT = UNTRUSTED_EVIDENCE_RULE + "\n\n" + """You answer enterprise questions from authorized evidence grouped by requirement and document role.

Rules:
- Answer every requirement independently before composing the final answer.
- For a single-artifact requirement, facts and identity MUST come from its CANONICAL document. Do not merge names, actions, fields, versions, or paths from similar supplemental artifacts.
- Supplemental evidence may add context only when the requirement explicitly allows it. It never overrides canonical identity.
- Preserve exact identifiers, numbers, dates, units, list items, conditions, exceptions, and negations.
- A requirement marked missing must be reported as missing, not guessed.
- Every supported requirement answer must cite the exact supplied S* evidence used.

Return exactly one JSON object:
{
  "requirements": [
    {
      "id": "R1",
      "status": "supported" | "missing" | "conflicting",
      "answer": "short complete answer",
      "citations": ["S1"],
      "source_doc_ids": ["doc-id"]
    }
  ]
}
Return JSON only."""


class JsonClient(Protocol):
    model: str

    def complete_json(self, **kwargs: Any) -> ApiResult:
        ...


@dataclass(frozen=True)
class GenerationRequirement:
    id: str
    text: str
    canonical_doc_id: str
    canonical_title: str
    allowed_doc_ids: tuple[str, ...]
    canonical_only: bool


@dataclass(frozen=True)
class CanonicalGenerationPlan:
    question: str
    requirements: tuple[GenerationRequirement, ...]
    citation_doc_map: dict[str, str]
    canonical_doc_ids: tuple[str, ...]
    supplemental_doc_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "requirements": [asdict(value) for value in self.requirements],
            "citation_doc_map": dict(self.citation_doc_map),
            "canonical_doc_ids": list(self.canonical_doc_ids),
            "supplemental_doc_ids": list(self.supplemental_doc_ids),
        }


@dataclass(frozen=True)
class CanonicalGenerationResult:
    answerable: bool
    answer: str
    citations: tuple[str, ...]
    requirement_results: tuple[dict[str, Any], ...]
    canonical_source_accuracy_proxy: float
    source_contamination_count: int
    source_contamination_rate: float
    plan: CanonicalGenerationPlan
    api: ApiResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "answerable": self.answerable,
            "answer": self.answer,
            "citations": list(self.citations),
            "requirements": list(self.requirement_results),
            "canonical_source_accuracy_proxy": self.canonical_source_accuracy_proxy,
            "source_contamination_count": self.source_contamination_count,
            "source_contamination_rate": self.source_contamination_rate,
            "plan": self.plan.to_dict(),
            "model": self.api.returned_model or self.api.value.get("model") or "unknown",
            "latency_ms": self.api.latency_ms,
            "usage": dict(self.api.usage),
            "request_id": self.api.request_id,
            "attempts": self.api.attempts,
        }


def decompose_question(question: str, *, maximum: int = 6) -> tuple[str, ...]:
    """Conservative deterministic decomposition; never invents benchmark labels."""
    normalized = re.sub(r"\s+", " ", question).strip()
    if not normalized:
        return ("Answer the user question",)
    parts = [value.strip(" ,;?") for value in _REQUIREMENT_SPLIT.split(normalized) if value.strip(" ,;?")]
    if len(parts) <= 1 or len(parts) > maximum:
        return (normalized,)
    # A fragment must remain independently intelligible; otherwise keep one requirement.
    question_words = ("what", "which", "when", "where", "who", "how", "whether")
    if not all(index == 0 or part.casefold().startswith(question_words) for index, part in enumerate(parts)):
        return (normalized,)
    return tuple(parts)


def build_canonical_plan(
    question: str,
    evidence: HierarchicalEvidence,
    *,
    requirements: Sequence[str] = (),
) -> CanonicalGenerationPlan:
    contexts = list(evidence.contexts)
    citation_doc_map = {
        str(context.get("citation_id") or ""): str(context.get("doc_id") or "")
        for context in contexts
        if _CITATION.fullmatch(str(context.get("citation_id") or ""))
    }
    if not citation_doc_map:
        raise PipelineError("canonical generation requires at least one valid evidence citation")
    ranked_docs = list(evidence.canonical_documents)
    if not ranked_docs:
        raise PipelineError("canonical generation has no ranked documents")
    requirement_texts = tuple(requirements) or decompose_question(question)
    comparative = bool(_COMPARATIVE.search(question)) or evidence.route_mode == "global"
    single_artifact = requires_canonical_document(question) and not comparative
    canonical_doc = str(ranked_docs[0]["doc_id"])
    canonical_title = str(ranked_docs[0].get("title") or "")
    all_docs = tuple(dict.fromkeys(citation_doc_map.values()))
    allowed = all_docs if comparative else (canonical_doc,)
    planned = tuple(
        GenerationRequirement(
            id=f"R{index}",
            text=text,
            canonical_doc_id=canonical_doc,
            canonical_title=canonical_title,
            allowed_doc_ids=allowed,
            canonical_only=single_artifact,
        )
        for index, text in enumerate(requirement_texts, start=1)
    )
    return CanonicalGenerationPlan(
        question=question,
        requirements=planned,
        citation_doc_map=citation_doc_map,
        canonical_doc_ids=(canonical_doc,),
        supplemental_doc_ids=tuple(doc for doc in all_docs if doc != canonical_doc),
    )


def build_canonical_messages(
    plan: CanonicalGenerationPlan,
    evidence: HierarchicalEvidence,
) -> list[dict[str, str]]:
    by_doc: dict[str, list[dict[str, Any]]] = {}
    for context in evidence.contexts:
        by_doc.setdefault(str(context.get("doc_id") or ""), []).append(context)
    blocks: list[str] = []
    for requirement in plan.requirements:
        blocks.append(
            f"<requirement id=\"{requirement.id}\" canonical_only=\"{str(requirement.canonical_only).lower()}\">\n"
            f"Need: {requirement.text}\n"
            f"CANONICAL DOCUMENT: doc_id={requirement.canonical_doc_id} title={requirement.canonical_title}\n"
            f"{_render_contexts(by_doc.get(requirement.canonical_doc_id, []))}"
        )
        supplements = [doc for doc in requirement.allowed_doc_ids if doc != requirement.canonical_doc_id]
        if supplements:
            blocks.append(
                "SUPPLEMENTAL/COMPARATIVE DOCUMENTS (keep identities separate):\n"
                + "\n".join(
                    f"doc_id={doc}\n{_render_contexts(by_doc.get(doc, []))}" for doc in supplements
                )
            )
        blocks.append("</requirement>")
    user = f"Question:\n{plan.question}\n\n" + "\n\n".join(blocks)
    return [
        {"role": "system", "content": CANONICAL_GENERATION_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def validate_canonical_payload(
    payload: dict[str, Any],
    *,
    plan: CanonicalGenerationPlan,
) -> dict[str, Any]:
    raw = payload.get("requirements")
    if not isinstance(raw, list) or len(raw) != len(plan.requirements):
        raise PipelineError("generation must return exactly one result per requirement")
    expected = {value.id: value for value in plan.requirements}
    output: list[dict[str, Any]] = []
    observed: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            raise PipelineError("requirement result must be an object")
        requirement_id = str(item.get("id") or "").strip().upper()
        if requirement_id not in expected or requirement_id in observed:
            raise PipelineError(f"invalid or duplicate requirement ID: {requirement_id!r}")
        observed.add(requirement_id)
        requirement = expected[requirement_id]
        status = str(item.get("status") or "").strip().lower()
        if status not in {"supported", "missing", "conflicting"}:
            raise PipelineError(f"invalid status for {requirement_id}: {status!r}")
        answer = str(item.get("answer") or "").strip()
        citations = _unique_strings(item.get("citations"), "citations")
        unknown = [citation for citation in citations if citation not in plan.citation_doc_map]
        if unknown:
            raise PipelineError(f"unknown citations for {requirement_id}: {unknown}")
        source_docs = _unique_strings(item.get("source_doc_ids"), "source_doc_ids")
        actual_docs = tuple(dict.fromkeys(plan.citation_doc_map[citation] for citation in citations))
        if set(source_docs) != set(actual_docs):
            raise PipelineError(f"source_doc_ids must match citation provenance for {requirement_id}")
        if status == "missing":
            if citations or source_docs:
                raise PipelineError(f"missing {requirement_id} cannot cite evidence")
        elif not answer or not citations:
            raise PipelineError(f"{status} {requirement_id} requires answer and citations")
        disallowed = set(actual_docs) - set(requirement.allowed_doc_ids)
        if disallowed:
            raise PipelineError(
                f"source contamination for {requirement_id}: {sorted(disallowed)} outside allowed documents"
            )
        if requirement.canonical_only and status == "supported" and set(actual_docs) != {requirement.canonical_doc_id}:
            raise PipelineError(f"single-artifact {requirement_id} must cite only its canonical document")
        output.append(
            {
                "id": requirement_id,
                "status": status,
                "answer": answer,
                "citations": citations,
                "source_doc_ids": list(actual_docs),
            }
        )
    output.sort(key=lambda value: int(value["id"][1:]))
    return {"requirements": output}


def generate_canonical_answer(
    client: JsonClient,
    *,
    question: str,
    evidence: HierarchicalEvidence,
    requirements: Sequence[str] = (),
    max_tokens: int = 1_024,
    temperature: float = 0.0,
) -> CanonicalGenerationResult:
    plan = build_canonical_plan(question, evidence, requirements=requirements)
    api = client.complete_json(
        messages=build_canonical_messages(plan, evidence),
        max_tokens=max_tokens,
        temperature=temperature,
        validator=lambda payload: validate_canonical_payload(payload, plan=plan),
    )
    rows = tuple(dict(value) for value in api.value["requirements"])
    supported = [value for value in rows if value["status"] in {"supported", "conflicting"}]
    answer = "\n".join(value["answer"] for value in rows if value["answer"])
    citations = tuple(
        dict.fromkeys(citation for value in supported for citation in value["citations"])
    )
    contaminated = sum(
        bool(set(value["source_doc_ids"]) - set(plan.requirements[index].allowed_doc_ids))
        for index, value in enumerate(rows)
    )
    canonical_correct = sum(
        not value["source_doc_ids"]
        or plan.requirements[index].canonical_doc_id in value["source_doc_ids"]
        for index, value in enumerate(rows)
    )
    return CanonicalGenerationResult(
        answerable=bool(supported),
        answer=answer,
        citations=citations,
        requirement_results=rows,
        canonical_source_accuracy_proxy=canonical_correct / len(rows) if rows else 0.0,
        source_contamination_count=contaminated,
        source_contamination_rate=contaminated / len(rows) if rows else 0.0,
        plan=plan,
        api=api,
    )


def _render_contexts(contexts: Sequence[dict[str, Any]]) -> str:
    blocks = []
    for context in contexts:
        citation = str(context.get("citation_id") or "")
        blocks.append(
            f"[{citation}] doc_id={context.get('doc_id') or ''} title={context.get('title') or ''} "
            f"section={(context.get('hierarchy') or {}).get('section_path') or context.get('section_path') or ''}\n"
            f"{context.get('text') or ''}"
        )
    return "\n\n".join(blocks) or "NO EVIDENCE"


def _unique_strings(value: Any, name: str) -> list[str]:
    if not isinstance(value, list):
        raise PipelineError(f"{name} must be an array")
    output = []
    for item in value:
        text = str(item).strip()
        if not text:
            raise PipelineError(f"{name} must not contain empty values")
        if text not in output:
            output.append(text)
    return output
