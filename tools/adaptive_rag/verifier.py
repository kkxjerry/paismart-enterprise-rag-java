from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Protocol

from tools.qwen_plus_rag_pipeline import ApiResult, PipelineError, REFUSAL_TEXT, render_contexts, validate_generation

from .features import EXACT_ANCHOR_RE, RouterDecision
from .requirements import RequirementPlan

VERIFY_SYSTEM_PROMPT = """You are a strict claim-to-citation verifier for an enterprise RAG answer.
Check every numbered claim only against the supplied authorized evidence. Do not use outside knowledge.
A claim is supported only when its cited text directly supports the exact names, numbers, dates, conditions,
exceptions, and negations in the claim. A relevant document is not enough.

Return exactly one JSON object:
{
  "status": "pass" | "repair" | "reject",
  "claims": [
    {
      "id": "C1",
      "citations": ["S1"],
      "status": "supported" | "partial" | "unsupported" | "conflicting"
    }
  ],
  "answerable": true,
  "revised_answer": "",
  "citations": []
}

Rules:
- Return one claims entry for every supplied C* claim, in the same order. Never repeat the claim text.
- pass: every claim is supported or conflicting. Leave revised_answer empty and citations empty; the system keeps the original answer.
- repair: remove or qualify unsupported material while preserving supported requirements. Only then return a complete revised_answer and its citations.
- reject: no useful supported answer remains; set answerable=false, revised_answer=INSUFFICIENT_EVIDENCE, citations=[].
- Every factual sentence in a repaired answer must end with one or more supplied citation markers.
- If sources materially disagree, say so and cite both sides.
- Never add facts not present in evidence. Return JSON only."""


class JsonClient(Protocol):
    model: str

    def complete_json(self, **kwargs: Any) -> ApiResult:
        ...


@dataclass(frozen=True)
class ClaimCheck:
    id: str
    claim: str
    citations: tuple[str, ...]
    status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "claim": self.claim,
            "citations": list(self.citations),
            "status": self.status,
        }


@dataclass(frozen=True)
class VerificationResult:
    triggered: bool
    trigger_reasons: tuple[str, ...]
    status: str
    claims: tuple[ClaimCheck, ...]
    answerable: bool
    answer: str
    citations: tuple[str, ...]
    model: str
    latency_ms: float
    usage: dict[str, int]
    request_id: str
    attempts: int
    status_normalized: bool = False

    @property
    def unsupported_count(self) -> int:
        return sum(claim.status in {"unsupported", "partial"} for claim in self.claims)

    def to_dict(self) -> dict[str, Any]:
        return {
            "triggered": self.triggered,
            "trigger_reasons": list(self.trigger_reasons),
            "status": self.status,
            "claims": [claim.to_dict() for claim in self.claims],
            "unsupported_count": self.unsupported_count,
            "answerable": self.answerable,
            "answer": self.answer,
            "citations": list(self.citations),
            "model": self.model,
            "latency_ms": self.latency_ms,
            "usage": dict(self.usage),
            "request_id": self.request_id,
            "attempts": self.attempts,
            "status_normalized": self.status_normalized,
        }


class ClaimCitationVerifier:
    def __init__(self, client: JsonClient) -> None:
        self.client = client

    def verify(
        self,
        *,
        question: str,
        answerable: bool,
        answer: str,
        citations: list[str],
        contexts: list[dict[str, Any]],
        plan: RequirementPlan,
        route: RouterDecision,
        mode: str = "conditional",
        max_input_chars: int = 24_000,
        max_tokens: int = 1_024,
        temperature: float = 0.0,
    ) -> VerificationResult:
        reasons = verification_trigger_reasons(
            answer=answer,
            citations=citations,
            contexts=contexts,
            plan=plan,
            route=route,
        )
        triggered = mode == "always" or (mode == "conditional" and bool(reasons))
        if mode == "off" or not triggered or not answerable:
            return VerificationResult(
                triggered=False,
                trigger_reasons=tuple(reasons),
                status="skipped",
                claims=tuple(),
                answerable=answerable,
                answer=answer,
                citations=tuple(citations),
                model="not_run",
                latency_ms=0.0,
                usage=_zero_usage(),
                request_id="",
                attempts=0,
            )

        context_map = {
            str(context.get("citation_id") or ""): context
            for context in contexts
            if context.get("citation_id")
        }
        cited_contexts = [context_map[citation] for citation in citations if citation in context_map]
        # Include conflict citations and requirement evidence as a verifier safety net.
        for citation in list(plan.conflict_citations) + list(plan.selected_citations):
            if citation in context_map and context_map[citation] not in cited_contexts:
                cited_contexts.append(context_map[citation])
        rendered, included, _ = render_contexts(
            cited_contexts,
            max_contexts=max(4, len(cited_contexts)),
            max_chars=max_input_chars,
        )
        valid = {str(context.get("citation_id") or "") for context in included}
        claim_inputs = extract_numbered_claims(answer)
        claim_text_by_id = {claim["id"]: claim["claim"] for claim in claim_inputs}
        claim_citations_by_id = {
            claim["id"]: list(claim["citations"]) for claim in claim_inputs
        }
        result = self.client.complete_json(
            messages=build_verification_messages(
                question=question,
                claims=claim_inputs,
                requirements=plan,
                rendered_contexts=rendered,
            ),
            max_tokens=max_tokens,
            temperature=temperature,
            validator=lambda payload: validate_verification(
                payload,
                valid_citations=valid,
                claim_text_by_id=claim_text_by_id,
                claim_citations_by_id=claim_citations_by_id,
                original_answer=answer,
                original_citations=citations,
            ),
        )
        value = result.value
        claims = tuple(
            ClaimCheck(
                id=str(claim["id"]),
                claim=str(claim["claim"]),
                citations=tuple(str(citation) for citation in claim["citations"]),
                status=str(claim["status"]),
            )
            for claim in value["claims"]
        )
        return VerificationResult(
            triggered=True,
            trigger_reasons=tuple(reasons),
            status=str(value["status"]),
            claims=claims,
            answerable=bool(value["answerable"]),
            answer=str(value["revised_answer"]),
            citations=tuple(str(citation) for citation in value["citations"]),
            model=result.returned_model or self.client.model,
            latency_ms=result.latency_ms,
            usage=dict(result.usage),
            request_id=result.request_id,
            attempts=result.attempts,
            status_normalized=bool(value.get("status_normalized")),
        )


def verification_trigger_reasons(
    *,
    answer: str,
    citations: list[str],
    contexts: list[dict[str, Any]],
    plan: RequirementPlan,
    route: RouterDecision,
) -> list[str]:
    reasons: list[str] = []
    if route.mode == "deep":
        reasons.append("route=deep")
    if plan.answerability in {"partial", "conflicting"} or plan.missing_count or plan.conflicting_count:
        reasons.append("requirements_not_fully_supported")
    # Exact values are high risk only when the answer composes more than one
    # evidence span. A single-citation fast answer does not justify another
    # model call solely because it contains a number or version.
    cited_sources = {
        str(context.get("source_type") or "")
        for context in contexts
        if str(context.get("citation_id") or "") in citations
    }
    cross_source = len(cited_sources) >= 2
    if len(citations) >= 6:
        reasons.append("many_citations")
    if EXACT_ANCHOR_RE.search(answer) and cross_source and len(citations) >= 3:
        reasons.append("cross_source_exact_values")
    return list(dict.fromkeys(reasons))


def extract_numbered_claims(answer: str) -> list[dict[str, Any]]:
    """Split an answer into stable claim IDs without asking the model to echo text."""
    raw_segments = [
        value.strip()
        for value in re.split(r"\n+|(?<=[.!?。！？])\s+", answer)
        if value.strip()
    ]
    citation_only = re.compile(r"(?:\[S[1-9][0-9]*\])+[.,;:!?。！？；：]*")
    claims: list[dict[str, Any]] = []
    for segment in raw_segments:
        if citation_only.fullmatch(segment):
            if claims:
                claims[-1]["claim"] = claims[-1]["claim"].rstrip() + " " + segment
                claims[-1]["citations"] = _inline_citations(claims[-1]["claim"])
            continue
        claims.append({
            "id": f"C{len(claims) + 1}",
            "claim": segment,
            "citations": _inline_citations(segment),
        })
    if not claims and answer.strip():
        claims.append({"id": "C1", "claim": answer.strip(), "citations": _inline_citations(answer)})
    return claims


def build_verification_messages(
    *,
    question: str,
    claims: list[dict[str, Any]],
    requirements: RequirementPlan,
    rendered_contexts: str,
) -> list[dict[str, str]]:
    requirements_text = "\n".join(
        f"{requirement.id}: {requirement.requirement} [{requirement.status}]"
        for requirement in requirements.requirements
    )
    claims_text = "\n".join(
        f"{claim['id']} | cited={','.join(claim['citations']) or 'none'} | {claim['claim']}"
        for claim in claims
    )
    user = (
        f"Question:\n{question}\n\n"
        f"Required coverage:\n{requirements_text}\n\n"
        f"Numbered claims to verify:\n{claims_text}\n\n"
        f"Authorized cited evidence:\n{rendered_contexts}"
    )
    return [
        {"role": "system", "content": VERIFY_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def validate_verification(
    payload: dict[str, Any],
    *,
    valid_citations: set[str],
    claim_text_by_id: dict[str, str] | None = None,
    claim_citations_by_id: dict[str, list[str]] | None = None,
    original_answer: str = "",
    original_citations: list[str] | None = None,
) -> dict[str, Any]:
    status = str(payload.get("status") or "").strip().lower()
    if status not in {"pass", "repair", "reject"}:
        raise PipelineError(f"invalid verifier status: {status!r}")
    raw_claims = payload.get("claims") or []
    if not isinstance(raw_claims, list):
        raise PipelineError("verifier claims must be an array")

    # Direct unit callers may still supply the legacy claim-text form. Runtime
    # calls always pass a stable C* -> text map, so the model only emits IDs.
    expected = dict(claim_text_by_id or {})
    if not expected:
        for index, raw in enumerate(raw_claims, start=1):
            if isinstance(raw, dict):
                text = str(raw.get("claim") or "").strip()
                if text:
                    expected[f"C{index}"] = text

    claims: list[dict[str, Any]] = []
    observed_ids: list[str] = []
    for index, raw in enumerate(raw_claims, start=1):
        if not isinstance(raw, dict):
            raise PipelineError("verifier claim must be an object")
        claim_id = str(raw.get("id") or f"C{index}").strip().upper()
        if claim_id not in expected:
            raise PipelineError(f"unknown verifier claim ID: {claim_id!r}")
        if claim_id in observed_ids:
            raise PipelineError(f"duplicate verifier claim ID: {claim_id!r}")
        observed_ids.append(claim_id)
        claim_status = str(raw.get("status") or "").strip().lower()
        if claim_status not in {"supported", "partial", "unsupported", "conflicting"}:
            raise PipelineError(f"invalid claim status: {claim_status!r}")
        citations = _citations(
            raw.get("citations") or [],
            valid_citations,
            allow_none=claim_status in {"partial", "unsupported"},
        )
        citation_normalized = False
        original_claim_citations = [
            citation
            for citation in (claim_citations_by_id or {}).get(claim_id, [])
            if citation in valid_citations
        ]
        if claim_status == "supported" and not citations and original_claim_citations:
            citations = list(dict.fromkeys(original_claim_citations))
            citation_normalized = True
        if claim_status == "conflicting" and len(citations) < 2:
            for citation in original_claim_citations:
                if citation not in citations:
                    citations.append(citation)
            citation_normalized = len(citations) >= 2
        if claim_status == "supported" and not citations:
            raise PipelineError("supported claim must cite evidence")
        if claim_status == "conflicting" and len(citations) < 2:
            raise PipelineError("conflicting claim must cite both sides")
        claims.append({
            "id": claim_id,
            "claim": expected[claim_id],
            "citations": citations,
            "status": claim_status,
            "citation_normalized": citation_normalized,
        })

    if expected and observed_ids != list(expected):
        raise PipelineError(
            "verifier must return every claim in order: expected "
            f"{list(expected)}, got {observed_ids}"
        )
    answerable = payload.get("answerable")
    if not isinstance(answerable, bool):
        raise PipelineError("verifier answerable must be boolean")
    unsupported = any(claim["status"] in {"partial", "unsupported"} for claim in claims)
    if status == "pass" and not answerable:
        raise PipelineError("pass verifier result must remain answerable")

    if status == "reject" or not answerable:
        return {
            "status": "reject",
            "claims": claims,
            "answerable": False,
            "revised_answer": REFUSAL_TEXT,
            "citations": [],
            "status_normalized": status != "reject",
        }

    status_normalized = False
    if status == "pass" and unsupported:
        # This is a common small-model protocol inconsistency. Do not pay for
        # identical retries. Deterministically keep only claims the verifier
        # itself marked supported/conflicting and turn the result into repair.
        status = "repair"
        status_normalized = True

    if status == "pass":
        answer_to_keep = original_answer or str(payload.get("revised_answer") or "").strip()
        citations_to_keep = list(original_citations or [])
        if not citations_to_keep:
            citations_to_keep = _citations(payload.get("citations") or [], valid_citations)
    else:
        answer_to_keep = str(payload.get("revised_answer") or "").strip()
        citations_to_keep = _citations(payload.get("citations") or [], valid_citations)
        if not answer_to_keep or unsupported:
            answer_to_keep, citations_to_keep = _supported_claim_answer(claims)
        if not answer_to_keep or not citations_to_keep:
            return {
                "status": "reject",
                "claims": claims,
                "answerable": False,
                "revised_answer": REFUSAL_TEXT,
                "citations": [],
                "status_normalized": True,
            }
    validated = validate_generation(
        {"answerable": True, "answer": answer_to_keep, "citations": citations_to_keep},
        valid_citations=valid_citations,
    )
    return {
        "status": status,
        "claims": claims,
        "answerable": True,
        "revised_answer": validated["answer"],
        "citations": validated["citations"],
        "status_normalized": status_normalized,
    }


def _supported_claim_answer(
    claims: list[dict[str, Any]],
) -> tuple[str, list[str]]:
    rendered: list[str] = []
    citations: list[str] = []
    for claim in claims:
        if claim.get("status") not in {"supported", "conflicting"}:
            continue
        claim_citations = [str(value) for value in claim.get("citations") or []]
        if not claim_citations:
            continue
        text = re.sub(r"\s*\[S[1-9][0-9]*\]", "", str(claim.get("claim") or "")).strip()
        if not text:
            continue
        rendered.append(text.rstrip() + " " + "".join(f"[{value}]" for value in claim_citations))
        for citation in claim_citations:
            if citation not in citations:
                citations.append(citation)
    return "\n".join(rendered), citations


def _inline_citations(text: str) -> list[str]:
    return list(dict.fromkeys(re.findall(r"\[(S[1-9][0-9]*)\]", text)))


def _citations(raw: Any, valid: set[str], *, allow_none: bool = False) -> list[str]:
    if not isinstance(raw, list):
        raise PipelineError("citations must be an array")
    output: list[str] = []
    for value in raw:
        citation = str(value).strip().upper()
        if allow_none and citation in {"", "NONE", "N/A", "NA", "NULL"}:
            continue
        if citation not in valid:
            raise PipelineError(f"unknown verifier citation: {citation!r}")
        if citation not in output:
            output.append(citation)
    return output


def _zero_usage() -> dict[str, int]:
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cached_tokens": 0}
