"""E2 canonical-source, requirement-level generation contracts.

The generator must bind each requirement to one evidence document before writing
its answer. The validator rejects citations from another document and constructs
the final answer from validated requirement outputs. This prevents a fluent top-
level answer from bypassing source decisions.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Sequence

from tools.qwen_plus_rag_pipeline import PipelineError, UNTRUSTED_EVIDENCE_RULE, validate_generation

from .hierarchical import build_global_hierarchy, is_global_question, is_single_artifact_question
from .requirements import RequirementPlan

_CANONICAL_PROMPT = UNTRUSTED_EVIDENCE_RULE + "\n\n" + """You are the source-binding answer stage of an enterprise RAG system.
Use only the authorized evidence cards below. First bind every requirement to one canonical document, then answer from that document. Never merge fields, names, actions, paths, versions, or policies from merely similar documents.

Rules:
- Every supported requirement must name one canonical_doc_id and cite only S* evidence from that document.
- If the question explicitly compares sources or the requirement is conflicting, use multiple source records instead of silently synthesizing them.
- A similarly named page, card, service, policy, or runbook is not interchangeable with the requested artifact.
- Preserve exact identifiers, numbers, dates, units, list items, conditions, exceptions, and negations.
- Each requirement answer must contain inline citations and be independently usable.
- Missing requirements must use answer="INSUFFICIENT_EVIDENCE", canonical_doc_id="", citations=[] and missing=true.
- Do not put unsupported facts in the top-level answer. The final answer will be reconstructed from requirement answers.

Return exactly one JSON object:
{
  "answerable": true,
  "requirements": [
    {
      "id": "R1",
      "canonical_doc_id": "document-id",
      "answer": "requirement answer [S1]",
      "citations": ["S1"],
      "missing": false
    }
  ],
  "answer": "optional draft; validator reconstructs it",
  "citations": ["S1"],
  "covered_requirements": ["R1"],
  "missing_requirements": []
}
Return JSON only."""

_SOURCE_BINDING_PROMPT = UNTRUSTED_EVIDENCE_RULE + "\n\n" + """You are the canonical-source binding stage of an enterprise RAG system.
Do not answer the user question. Compare the supplied document cards and bind each requirement to the one document that directly matches all requested artifact identity, scope, values, and qualifiers.

Rules:
- Retrieval rank is only a candidate prior, not proof.
- Similar pages, cards, policies, dashboards, or services are distractors unless their own text directly matches the requirement.
- Prefer explicit identity language and exact requested actions/fields/paths over broad topical overlap.
- One requirement gets one canonical_doc_id. A conflicting requirement may set conflicting=true and name multiple documents.
- evidence_citations must all belong to the bound document and must show why it is the canonical source.
- If no document directly supports the requirement, set missing=true with no document or citations.

Return exactly one JSON object:
{
  "bindings": [
    {
      "id": "R1",
      "canonical_doc_id": "document-id",
      "evidence_citations": ["S1"],
      "missing": false,
      "conflicting": false
    }
  ]
}
Return JSON only."""

_GLOBAL_PROMPT = UNTRUSTED_EVIDENCE_RULE + "\n\n" + """You are the global synthesis stage of an enterprise RAG system.
Use only the extractive document hierarchy. Each bullet is copied from source evidence and carries its S* citation. Synthesize cross-document themes only when at least two supplied documents support them. Preserve disagreements and scope qualifiers. Do not invent a global trend from one local example.

Return exactly one JSON object:
{
  "answerable": true,
  "answer": "global answer with inline S* citations",
  "citations": ["S1", "S2"],
  "covered_requirements": ["R1"],
  "missing_requirements": [],
  "documents_used": ["document-id"]
}
Return JSON only."""


def _context_block(context: dict[str, Any]) -> str:
    citation = str(context.get("citation_id") or "")
    values = [
        f"[{citation}]",
        f"title={context.get('title') or ''}",
        f"source_type={context.get('source_type') or 'unknown'}",
        f"doc_id={context.get('doc_id') or ''}",
    ]
    section = str(context.get("section_path") or "").strip()
    if section:
        values.append(f"section={section}")
    return " ".join(values) + "\n" + str(context.get("text") or "")


def group_contexts_by_document(
    contexts: Sequence[dict[str, Any]],
    *,
    question: str,
) -> tuple[str, dict[str, str], dict[str, list[str]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for context in contexts:
        doc_id = str(context.get("doc_id") or context.get("citation_id") or "")
        if doc_id:
            grouped[doc_id].append(dict(context))
    ordered = sorted(
        grouped.items(),
        key=lambda item: (
            min(int(value.get("document_rank") or 10**9) for value in item[1]),
            -max(float(value.get("query_coverage") or 0.0) for value in item[1]),
            item[0],
        ),
    )
    citation_to_doc: dict[str, str] = {}
    doc_to_citations: dict[str, list[str]] = {}
    cards = []
    for index, (doc_id, values) in enumerate(ordered, start=1):
        title = str(values[0].get("title") or "")
        source = str(values[0].get("source_type") or "unknown")
        citations = [str(value.get("citation_id") or "") for value in values]
        doc_to_citations[doc_id] = citations
        for citation in citations:
            citation_to_doc[citation] = doc_id
        cards.append(
            f"Document D{index}: doc_id={doc_id} title={title} source_type={source}\n"
            + "\n\n".join(_context_block(value) for value in values)
        )
    return "\n\n===== NEXT DOCUMENT =====\n\n".join(cards), citation_to_doc, doc_to_citations


def build_source_binding_messages(
    *,
    question: str,
    plan: RequirementPlan,
    contexts: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, str]], dict[str, str], bool]:
    rendered, citation_to_doc, _ = group_contexts_by_document(contexts, question=question)
    requirements = "\n".join(
        f"{value.id}. {value.requirement} | planner_status={value.status}"
        for value in plan.requirements
    )
    single_source = is_single_artifact_question(question) and not any(
        value.status == "conflicting" for value in plan.requirements
    )
    mode = (
        "All supported requirements describe one artifact; bind them to the same document."
        if single_source
        else "Bind each requirement independently; only explicit conflicts may use multiple documents."
    )
    user = (
        f"Question:\n{question}\n\n"
        f"Binding mode:\n{mode}\n\n"
        f"Requirements:\n{requirements}\n\n"
        f"Candidate document cards:\n{rendered}"
    )
    return [
        {"role": "system", "content": _SOURCE_BINDING_PROMPT},
        {"role": "user", "content": user},
    ], citation_to_doc, single_source


def validate_source_bindings(
    payload: dict[str, Any],
    *,
    citation_to_doc: dict[str, str],
    requirement_ids: set[str],
    single_source: bool,
) -> dict[str, Any]:
    raw = payload.get("bindings")
    if not isinstance(raw, list) or len(raw) != len(requirement_ids):
        raise PipelineError("source binding must return one row per requirement")
    observed: set[str] = set()
    bindings = []
    supported_docs: list[str] = []
    for value in raw:
        if not isinstance(value, dict):
            raise PipelineError("source binding row must be an object")
        requirement_id = str(value.get("id") or "").strip().upper()
        if requirement_id not in requirement_ids or requirement_id in observed:
            raise PipelineError(f"invalid or duplicate binding id: {requirement_id!r}")
        observed.add(requirement_id)
        missing = value.get("missing")
        conflicting = value.get("conflicting")
        if not isinstance(missing, bool) or not isinstance(conflicting, bool):
            raise PipelineError("binding missing/conflicting fields must be boolean")
        doc_id = str(value.get("canonical_doc_id") or "").strip()
        citations_raw = value.get("evidence_citations")
        if not isinstance(citations_raw, list):
            raise PipelineError("binding evidence_citations must be an array")
        citations: list[str] = []
        for citation_value in citations_raw:
            citation = str(citation_value).strip()
            if citation not in citation_to_doc:
                raise PipelineError(f"unknown binding citation: {citation!r}")
            if citation not in citations:
                citations.append(citation)
        if missing:
            if doc_id or citations or conflicting:
                raise PipelineError("missing binding must not name documents or citations")
        else:
            if not doc_id or not citations:
                raise PipelineError("supported binding requires a document and evidence")
            wrong = [citation for citation in citations if citation_to_doc[citation] != doc_id]
            if wrong:
                raise PipelineError(f"binding citations do not belong to {doc_id}: {wrong}")
            supported_docs.append(doc_id)
        bindings.append({
            "id": requirement_id,
            "canonical_doc_id": doc_id,
            "evidence_citations": citations,
            "missing": missing,
            "conflicting": conflicting,
        })
    if observed != requirement_ids:
        raise PipelineError("source binding omitted requirements")
    if single_source and len(set(supported_docs)) > 1:
        raise PipelineError("single-artifact source binding selected multiple documents")
    return {
        "bindings": sorted(bindings, key=lambda value: value["id"]),
        "canonical_doc_ids": list(dict.fromkeys(supported_docs)),
        "missing_requirements": [value["id"] for value in bindings if value["missing"]],
        "single_source_enforced": single_source,
    }


def contexts_for_bindings(
    contexts: Sequence[dict[str, Any]],
    bindings: dict[str, Any],
) -> list[dict[str, Any]]:
    selected_docs = set(str(value) for value in bindings.get("canonical_doc_ids") or [])
    return [dict(value) for value in contexts if str(value.get("doc_id") or "") in selected_docs]


def build_canonical_generation_messages(
    *,
    question: str,
    plan: RequirementPlan,
    contexts: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, str]], dict[str, str], bool]:
    rendered, citation_to_doc, _ = group_contexts_by_document(contexts, question=question)
    requirements = "\n".join(
        f"{value.id}. {value.requirement} | planner_status={value.status}"
        for value in plan.requirements
    )
    single_source = is_single_artifact_question(question) and not any(
        value.status == "conflicting" for value in plan.requirements
    )
    mode = (
        "This is likely a single-artifact question. Use one canonical document across supported requirements."
        if single_source
        else "Use one canonical document per requirement; different requirements may use different documents."
    )
    user = (
        f"Question:\n{question}\n\n"
        f"Source binding mode:\n{mode}\n\n"
        f"Requirements:\n{requirements}\n\n"
        f"Authorized evidence grouped by document:\n{rendered}"
    )
    return [
        {"role": "system", "content": _CANONICAL_PROMPT},
        {"role": "user", "content": user},
    ], citation_to_doc, single_source


def _ids(raw: Any, valid: set[str], label: str) -> list[str]:
    if not isinstance(raw, list):
        raise PipelineError(f"{label} must be an array")
    output: list[str] = []
    for value in raw:
        item = str(value).strip().upper()
        if item not in valid:
            raise PipelineError(f"unknown requirement ID in {label}: {item!r}")
        if item not in output:
            output.append(item)
    return output


def validate_canonical_generation(
    payload: dict[str, Any],
    *,
    valid_citations: set[str],
    citation_to_doc: dict[str, str],
    requirement_ids: set[str],
    single_source: bool,
) -> dict[str, Any]:
    raw_requirements = payload.get("requirements")
    if not isinstance(raw_requirements, list) or not raw_requirements:
        raise PipelineError("canonical generation requires non-empty requirements")
    if len(raw_requirements) != len(requirement_ids):
        raise PipelineError("canonical generation must return exactly one row per requirement")
    observed: set[str] = set()
    validated_rows = []
    covered = []
    missing = []
    all_citations: list[str] = []
    canonical_docs: list[str] = []
    final_parts: list[str] = []
    for raw in raw_requirements:
        if not isinstance(raw, dict):
            raise PipelineError("canonical requirement row must be an object")
        requirement_id = str(raw.get("id") or "").strip().upper()
        if requirement_id not in requirement_ids or requirement_id in observed:
            raise PipelineError(f"invalid or duplicate requirement id: {requirement_id!r}")
        observed.add(requirement_id)
        is_missing = raw.get("missing")
        if not isinstance(is_missing, bool):
            raise PipelineError(f"{requirement_id} missing must be boolean")
        answer = str(raw.get("answer") or "").strip()
        canonical_doc = str(raw.get("canonical_doc_id") or "").strip()
        citations = raw.get("citations")
        probe = validate_generation(
            {
                "answerable": not is_missing,
                "answer": answer,
                "citations": citations,
            },
            valid_citations=valid_citations,
        )
        answer = str(probe["answer"])
        values = list(probe["citations"])
        if is_missing:
            if canonical_doc or values or answer != "INSUFFICIENT_EVIDENCE":
                raise PipelineError(f"missing {requirement_id} must not bind a source")
            missing.append(requirement_id)
        else:
            if not canonical_doc:
                raise PipelineError(f"supported {requirement_id} requires canonical_doc_id")
            wrong = [citation for citation in values if citation_to_doc.get(citation) != canonical_doc]
            if wrong:
                raise PipelineError(
                    f"{requirement_id} cites evidence outside canonical document {canonical_doc}: {wrong}"
                )
            if not values:
                raise PipelineError(f"supported {requirement_id} requires citations")
            if not re.search(r"\[S[1-9][0-9]*\]", answer):
                raise PipelineError(f"supported {requirement_id} answer requires inline citation")
            covered.append(requirement_id)
            canonical_docs.append(canonical_doc)
            final_parts.append(answer)
            for citation in values:
                if citation not in all_citations:
                    all_citations.append(citation)
        validated_rows.append(
            {
                "id": requirement_id,
                "canonical_doc_id": canonical_doc,
                "answer": answer,
                "citations": values,
                "missing": is_missing,
            }
        )
    if observed != requirement_ids:
        raise PipelineError("canonical generation omitted requirements")
    if single_source and len(set(canonical_docs)) > 1:
        raise PipelineError("single-artifact question used more than one canonical document")
    answerable = bool(covered)
    final_answer = "\n".join(final_parts) if final_parts else "INSUFFICIENT_EVIDENCE"
    # The model's top-level fields are intentionally ignored after validating rows.
    return {
        "answerable": answerable,
        "answer": final_answer,
        "citations": all_citations,
        "covered_requirements": sorted(covered),
        "missing_requirements": sorted(missing),
        "requirements": validated_rows,
        "canonical_doc_ids": list(dict.fromkeys(canonical_docs)),
        "single_source_enforced": single_source,
        "top_level_answer_ignored": True,
    }


def build_global_generation_messages(
    *,
    question: str,
    contexts: Sequence[dict[str, Any]],
    max_documents: int = 8,
    leaves_per_document: int = 3,
    max_chars: int = 14_000,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    hierarchy = build_global_hierarchy(
        contexts,
        question=question,
        max_documents=max_documents,
        leaves_per_document=leaves_per_document,
        max_chars=max_chars,
    )
    user = f"Question:\n{question}\n\nExtractive document hierarchy:\n{hierarchy['rendered']}"
    return [
        {"role": "system", "content": _GLOBAL_PROMPT},
        {"role": "user", "content": user},
    ], hierarchy


def validate_global_generation(
    payload: dict[str, Any],
    *,
    valid_citations: set[str],
    valid_documents: set[str],
) -> dict[str, Any]:
    base = validate_generation(payload, valid_citations=valid_citations)
    documents = payload.get("documents_used")
    if not isinstance(documents, list):
        raise PipelineError("documents_used must be an array")
    selected: list[str] = []
    for value in documents:
        doc_id = str(value).strip()
        if doc_id not in valid_documents:
            raise PipelineError(f"unknown global document: {doc_id!r}")
        if doc_id not in selected:
            selected.append(doc_id)
    if base["answerable"] and not selected:
        raise PipelineError("answerable global response must identify documents_used")
    if base["answerable"] and len(selected) < 2:
        # A global claim from one document is too weak; return visible insufficient
        # evidence rather than quietly promoting a local example into a global fact.
        return {
            **base,
            "answerable": False,
            "answer": "INSUFFICIENT_EVIDENCE",
            "citations": [],
            "covered_requirements": [],
            "missing_requirements": ["R1"],
            "documents_used": selected,
            "global_support_insufficient": True,
        }
    return {
        **base,
        "covered_requirements": ["R1"] if base["answerable"] else [],
        "missing_requirements": [] if base["answerable"] else ["R1"],
        "documents_used": selected,
        "global_support_insufficient": False,
    }


def should_use_global_route(question: str, question_type: str | None) -> bool:
    return is_global_question(question, question_type)
