#!/usr/bin/env python3
"""Run evidence enhancement and grounded answer generation with Qwen Plus.

The enhancement stage is deliberately selection-only: Qwen Plus chooses original
``S*`` evidence IDs, while the generator receives the untouched source text for
those IDs. This prevents an LLM-written summary from becoming an unaudited new
source of truth.

The script uses the OpenAI-compatible Alibaba Cloud Model Studio endpoint and
reads the API key from an environment variable. It never writes the key to an
output or command-line manifest.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import random
import re
import statistics
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

DEFAULT_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen-plus"
REFUSAL_TEXT = "INSUFFICIENT_EVIDENCE"
CITATION_RE = re.compile(r"^S[1-9][0-9]*$")
INLINE_CITATION_RE = re.compile(r"\[(S[1-9][0-9]*)\]")
TOKEN_RE = re.compile(r"[a-z0-9_][a-z0-9_./:+#-]*|[\u4e00-\u9fff]", re.IGNORECASE)
CONTENT_STOPWORDS = {
    "a", "about", "according", "after", "all", "an", "and", "are", "as", "at",
    "be", "before", "by", "can", "did", "do", "does", "during", "for", "from",
    "has", "have", "how", "i", "in", "into", "is", "it", "its", "me", "of", "on",
    "or", "our", "should", "that", "the", "their", "they", "this", "to", "was", "we",
    "were", "what", "when", "where", "which", "who", "why", "with",
}
INSUFFICIENT_CAVEAT_RE = re.compile(
    r"(?:insufficient (?:evidence|information)|"
    r"not (?:explicitly )?(?:available|found|included|mentioned|provided|specified|stated)|"
    r"(?:do|does) not (?:contain|include|mention|provide|specify|state)|"
    r"cannot be (?:answered|determined|found|verified)|"
    r"no (?:available |relevant )?(?:evidence|information))",
    re.IGNORECASE,
)

ENHANCEMENT_SYSTEM_PROMPT = """You are an evidence selector for an enterprise RAG system.
Select the smallest set of supplied evidence IDs needed to answer every part of the question.
Do not answer the question, rewrite evidence, infer from outside knowledge, or use any benchmark labels.
Prefer direct statements over background material. Preserve both sides when sources conflict.
Return exactly one JSON object with this shape:
{
  "answerability": "answerable" | "insufficient" | "conflicting",
  "selected_citations": ["S1", "S2"],
  "conflict_citations": ["S3", "S4"]
}
Rules:
- IDs must come from the supplied evidence.
- selected_citations must be unique, ordered from most useful to least useful, and no longer than the stated limit.
- If one evidence ID contains every requested fact, select exactly that one ID.
- Do not select related background chunks from the same document when they add no required fact.
- Multi-part questions must retain all evidence needed for all subparts.
- Use insufficient only when the supplied evidence cannot support a grounded answer.
- Use conflicting when material sources disagree, and include all conflicting IDs in selected_citations.
- conflict_citations is empty unless answerability is conflicting.
Return JSON only."""

GENERATION_SYSTEM_PROMPT = """You are an enterprise RAG answer generator.
Use only the supplied authorized evidence. Do not use outside knowledge or invent missing facts.
Answer every requested subpart, preserving exact names, numbers, dates, conditions, exceptions, and negations.
If evidence conflicts, state the conflict instead of silently choosing one side.
Every factual sentence must end with one or more exact source markers such as [S1] or [S1][S2].
Return exactly one JSON object in one of these shapes:
{"answerable":true,"answer":"grounded answer with inline [S1] citations","citations":["S1"]}
{"answerable":false,"answer":"INSUFFICIENT_EVIDENCE","citations":[]}
The citations array must contain every source marker used in the answer, with no unsupported IDs.
Return JSON only."""


class PipelineError(RuntimeError):
    """Raised for a retryable or terminal pipeline validation failure."""


class QwenRequestError(PipelineError):
    """Terminal model error that preserves consumed usage and latency."""

    def __init__(
        self,
        message: str,
        *,
        usage: dict[str, int],
        latency_ms: float,
        attempts: int,
        max_tokens_used: int,
    ) -> None:
        super().__init__(message)
        self.usage = dict(usage)
        self.latency_ms = float(latency_ms)
        self.attempts = int(attempts)
        self.max_tokens_used = int(max_tokens_used)


@dataclass(frozen=True)
class ApiResult:
    value: dict[str, Any]
    latency_ms: float
    usage: dict[str, int]
    request_id: str
    returned_model: str
    attempts: int
    finish_reason: str
    max_tokens_used: int


class QwenClient:
    def __init__(
        self,
        *,
        api_base: str,
        api_key: str,
        model: str,
        timeout_seconds: float,
        retries: int,
    ) -> None:
        if not api_key:
            raise ValueError("Qwen API key must not be empty")
        self.endpoint = api_base.rstrip("/") + "/chat/completions"
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.retries = retries

    def complete_json(
        self,
        *,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        validator: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> ApiResult:
        last_error: Exception | None = None
        current_max_tokens = max_tokens
        cumulative_usage = zero_usage()
        total_latency_ms = 0.0
        attempts_made = 0
        deadline = time.monotonic() + self.timeout_seconds
        for attempt in range(1, self.retries + 2):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                last_error = TimeoutError(
                    f"Qwen request exceeded total deadline of {self.timeout_seconds}s"
                )
                break
            attempts_made = attempt
            started = time.perf_counter()
            try:
                payload = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": current_max_tokens,
                    "response_format": {"type": "json_object"},
                }
                request = urllib.request.Request(
                    self.endpoint,
                    data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}",
                    },
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=max(1.0, remaining)) as response:
                    response_payload = json.load(response)
                    request_id = response.headers.get("x-request-id", "")
                latency_ms = (time.perf_counter() - started) * 1000.0
                total_latency_ms += latency_ms
                usage_node = response_payload.get("usage") or {}
                attempt_usage = {
                    "prompt_tokens": int(usage_node.get("prompt_tokens") or 0),
                    "completion_tokens": int(usage_node.get("completion_tokens") or 0),
                    "total_tokens": int(usage_node.get("total_tokens") or 0),
                    "cached_tokens": int(
                        (usage_node.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
                    ),
                }
                cumulative_usage = add_usage(cumulative_usage, attempt_usage)
                choice = response_payload["choices"][0]
                finish_reason = str(choice.get("finish_reason") or "")
                if finish_reason == "length":
                    raise PipelineError(
                        f"model output hit max_tokens={current_max_tokens} before completing JSON"
                    )
                raw_content = choice["message"]["content"]
                parsed = parse_json_object(str(raw_content))
                validated = validator(parsed)
                return ApiResult(
                    value=validated,
                    latency_ms=total_latency_ms,
                    usage=cumulative_usage,
                    request_id=request_id,
                    returned_model=str(response_payload.get("model") or ""),
                    attempts=attempt,
                    finish_reason=finish_reason,
                    max_tokens_used=current_max_tokens,
                )
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")[:2000]
                last_error = PipelineError(f"HTTP {exc.code}: {body}")
                retryable = exc.code == 429 or exc.code >= 500
                if not retryable or attempt > self.retries:
                    break
            except (json.JSONDecodeError, PipelineError, KeyError, ValueError, TypeError) as exc:
                last_error = exc
                current_max_tokens = min(4096, max(current_max_tokens + 1, current_max_tokens * 2))
                if attempt > self.retries:
                    break
            except (urllib.error.URLError, TimeoutError) as exc:
                last_error = exc
                if attempt > self.retries:
                    break
            delay = min(20.0, 0.75 * (2 ** (attempt - 1))) + random.random() * 0.25
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(delay, remaining))
        raise QwenRequestError(
            f"Qwen request failed after {attempts_made} attempts within "
            f"{self.timeout_seconds}s: {last_error}",
            usage=cumulative_usage,
            latency_ms=total_latency_ms,
            attempts=attempts_made,
            max_tokens_used=current_max_tokens,
        )


def parse_json_object(content: str) -> dict[str, Any]:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    parsed = json.loads(stripped)
    if not isinstance(parsed, dict):
        raise PipelineError("model response must be a JSON object")
    return parsed


def unique_citations(values: Any) -> list[str]:
    if not isinstance(values, list):
        raise PipelineError("citation field must be an array")
    result: list[str] = []
    for value in values:
        citation = str(value).strip()
        if not CITATION_RE.fullmatch(citation):
            raise PipelineError(f"invalid citation ID: {citation!r}")
        if citation not in result:
            result.append(citation)
    return result


def validate_enhancement(
    payload: dict[str, Any],
    *,
    valid_citations: set[str],
    max_selected: int,
) -> dict[str, Any]:
    answerability = str(payload.get("answerability") or "").strip().lower()
    if answerability not in {"answerable", "insufficient", "conflicting"}:
        raise PipelineError(f"invalid enhancement answerability: {answerability!r}")
    selected = unique_citations(payload.get("selected_citations"))
    conflicts = unique_citations(payload.get("conflict_citations") or [])
    unknown = [citation for citation in selected + conflicts if citation not in valid_citations]
    if unknown:
        raise PipelineError(f"enhancement returned unknown citations: {unknown}")
    if len(selected) > max_selected:
        raise PipelineError(
            f"enhancement selected {len(selected)} citations, exceeding limit {max_selected}"
        )
    if answerability in {"answerable", "conflicting"} and not selected:
        raise PipelineError("answerable/conflicting enhancement must select evidence")
    if answerability == "conflicting":
        if len(conflicts) < 2:
            raise PipelineError("conflicting enhancement must identify at least two citations")
        if any(citation not in selected for citation in conflicts):
            raise PipelineError("all conflict citations must also be selected")
    elif conflicts:
        raise PipelineError("conflict_citations must be empty unless answerability is conflicting")
    return {
        "answerability": answerability,
        "selected_citations": selected,
        "conflict_citations": conflicts,
    }


def validate_generation(
    payload: dict[str, Any],
    *,
    valid_citations: set[str],
) -> dict[str, Any]:
    answerable = payload.get("answerable")
    if not isinstance(answerable, bool):
        raise PipelineError("generation answerable must be a boolean")
    answer = str(payload.get("answer") or "").strip()
    citations = unique_citations(payload.get("citations"))
    unknown = [citation for citation in citations if citation not in valid_citations]
    if unknown:
        raise PipelineError(f"generation returned unknown citations: {unknown}")
    if not answerable:
        if citations:
            raise PipelineError("unanswerable response must not contain citations")
        return {
            "answerable": False,
            "answer": REFUSAL_TEXT,
            "citations": [],
            "citation_normalized": answer != REFUSAL_TEXT,
        }
    if not answer:
        raise PipelineError("answerable response must contain an answer")
    if not citations:
        raise PipelineError("answerable response must contain citations")
    inline = unique_preserving_order(INLINE_CITATION_RE.findall(answer))
    unknown_inline = [citation for citation in inline if citation not in valid_citations]
    if unknown_inline:
        raise PipelineError(f"answer contains unknown inline citations: {unknown_inline}")
    normalized = False
    for citation in inline:
        if citation not in citations:
            citations.append(citation)
            normalized = True
    missing_inline = [citation for citation in citations if citation not in inline]
    if missing_inline:
        answer = answer.rstrip() + " " + "".join(f"[{citation}]" for citation in missing_inline)
        normalized = True
    return {
        "answerable": True,
        "answer": answer,
        "citations": citations,
        "citation_normalized": normalized,
    }


def unique_preserving_order(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def validate_output_paths(input_paths: Iterable[Path], output_paths: Iterable[Path]) -> None:
    inputs = {path.expanduser().resolve() for path in input_paths if path is not None}
    outputs = [path.expanduser().resolve() for path in output_paths if path is not None]
    if len(set(outputs)) != len(outputs):
        raise ValueError("output paths must be distinct")
    collisions = sorted(str(path) for path in outputs if path in inputs)
    if collisions:
        raise ValueError(f"outputs must not overwrite inputs: {collisions}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_run_signature(args: argparse.Namespace) -> str:
    payload = {
        "contexts_sha256": sha256_file(args.contexts),
        "qid_file_sha256": sha256_file(args.qid_file) if args.qid_file else None,
        "pipeline": args.pipeline,
        "api_base": args.api_base.rstrip("/"),
        "model": args.model,
        "limit": args.limit,
        "stratified": args.stratified,
        "enhance_max_contexts": args.enhance_max_contexts,
        "enhance_max_input_chars": args.enhance_max_input_chars,
        "enhance_max_selected": args.enhance_max_selected,
        "enhance_max_tokens": args.enhance_max_tokens,
        "selection_expansion": args.selection_expansion,
        "expansion_max_per_document": args.expansion_max_per_document,
        "generation_max_contexts": args.generation_max_contexts,
        "generation_max_input_chars": args.generation_max_input_chars,
        "generation_max_tokens": args.generation_max_tokens,
        "temperature": args.temperature,
        "enhancement_prompt_sha256": hashlib.sha256(
            ENHANCEMENT_SYSTEM_PROMPT.encode("utf-8")
        ).hexdigest(),
        "generation_prompt_sha256": hashlib.sha256(
            GENERATION_SYSTEM_PROMPT.encode("utf-8")
        ).hexdigest(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row at {path}:{line_number} must be an object")
            rows.append(row)
    return rows


def select_rows(
    rows: list[dict[str, Any]],
    *,
    limit: int | None,
    stratified: bool,
    qids: set[str] | None,
) -> list[dict[str, Any]]:
    selected = [row for row in rows if qids is None or str(row.get("qid")) in qids]
    if limit is None or limit <= 0 or limit >= len(selected):
        return selected
    if not stratified:
        return selected[:limit]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        grouped[str(row.get("question_type") or "unknown")].append(row)
    output: list[dict[str, Any]] = []
    offsets = {name: 0 for name in grouped}
    names = sorted(grouped)
    while len(output) < limit:
        progress = False
        for name in names:
            offset = offsets[name]
            if offset >= len(grouped[name]):
                continue
            output.append(grouped[name][offset])
            offsets[name] += 1
            progress = True
            if len(output) >= limit:
                break
        if not progress:
            break
    return output


def render_contexts(
    contexts: list[dict[str, Any]],
    *,
    max_contexts: int,
    max_chars: int,
) -> tuple[str, list[dict[str, Any]], int]:
    rendered: list[str] = []
    included: list[dict[str, Any]] = []
    remaining = max_chars
    for context in contexts[:max_contexts]:
        citation_id = str(context.get("citation_id") or "").strip()
        if not CITATION_RE.fullmatch(citation_id):
            continue
        header = (
            f"[{citation_id}] title={context.get('title') or ''} "
            f"source_type={context.get('source_type') or 'unknown'} "
            f"doc_id={context.get('doc_id') or ''}\n"
        )
        if remaining <= len(header):
            break
        text = str(context.get("text") or "")
        allowed = max(0, remaining - len(header))
        emitted = text[:allowed]
        block = header + emitted
        rendered.append(block)
        copied = dict(context)
        copied["text"] = emitted
        included.append(copied)
        remaining -= len(block) + 2
        if len(emitted) < len(text):
            break
    output = "\n\n".join(rendered)
    return output, included, len(output)


def build_enhancement_messages(
    row: dict[str, Any],
    *,
    rendered_contexts: str,
    max_selected: int,
) -> list[dict[str, str]]:
    user = (
        f"Maximum selected evidence IDs: {max_selected}\n\n"
        f"Question:\n{row.get('question') or ''}\n\n"
        f"Authorized evidence:\n{rendered_contexts}"
    )
    return [
        {"role": "system", "content": ENHANCEMENT_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def build_generation_messages(
    row: dict[str, Any],
    *,
    rendered_contexts: str,
    answerability_hint: str,
) -> list[dict[str, str]]:
    hint = (
        "The evidence selector marked the evidence as materially conflicting. Preserve the conflict."
        if answerability_hint == "conflicting"
        else "The evidence selector marked the evidence as potentially insufficient. Independently verify the supplied fallback evidence before refusing."
        if answerability_hint == "insufficient"
        else "The evidence selector did not provide an answerability decision. Inspect the evidence yourself."
        if answerability_hint == "unknown"
        else "The evidence selector marked the evidence as potentially answerable. Verify it before answering."
    )
    user = (
        f"Question:\n{row.get('question') or ''}\n\n"
        f"Selector note:\n{hint}\n\n"
        f"Authorized evidence:\n{rendered_contexts}"
    )
    return [
        {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def context_by_citation(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for context in row.get("contexts") or []:
        citation_id = str(context.get("citation_id") or "").strip()
        if CITATION_RE.fullmatch(citation_id) and citation_id not in result:
            result[citation_id] = context
    return result


def expand_selected_contexts(
    original_contexts: list[dict[str, Any]],
    selected_citations: list[str],
    *,
    mode: str,
    max_per_document: int,
) -> list[dict[str, Any]]:
    source_map = {
        str(context.get("citation_id")): context
        for context in original_contexts
        if CITATION_RE.fullmatch(str(context.get("citation_id") or ""))
    }
    selected = [source_map[citation] for citation in selected_citations if citation in source_map]
    if mode == "none":
        return selected
    if mode == "rerank":
        selected_ids = {str(context.get("citation_id") or "") for context in selected}
        return selected + [
            context
            for context in original_contexts
            if str(context.get("citation_id") or "") not in selected_ids
        ]
    if mode != "document":
        raise ValueError(f"unsupported selection expansion mode: {mode}")

    document_order: list[str] = []
    for context in selected:
        doc_id = str(context.get("doc_id") or "")
        if doc_id and doc_id not in document_order:
            document_order.append(doc_id)

    output: list[dict[str, Any]] = []
    emitted: set[str] = set()
    for doc_id in document_order:
        selected_for_doc = [
            context for context in selected if str(context.get("doc_id") or "") == doc_id
        ]
        siblings = [
            context
            for context in original_contexts
            if str(context.get("doc_id") or "") == doc_id
            and str(context.get("citation_id") or "") not in {
                str(value.get("citation_id") or "") for value in selected_for_doc
            }
        ]
        for context in selected_for_doc + siblings:
            citation_id = str(context.get("citation_id") or "")
            if citation_id in emitted:
                continue
            output.append(context)
            emitted.add(citation_id)
            if sum(str(value.get("doc_id") or "") == doc_id for value in output) >= max_per_document:
                break
    return output


def content_tokens(text: str) -> list[str]:
    return [token for token in TOKEN_RE.findall(text.lower()) if token not in CONTENT_STOPWORDS]


def token_recall(candidate: str, reference: str) -> float:
    expected = set(content_tokens(reference))
    if not expected:
        return 1.0 if not content_tokens(candidate) else 0.0
    actual = set(content_tokens(candidate))
    return len(expected & actual) / len(expected)


def token_f1(candidate: str, reference: str) -> float:
    actual = set(content_tokens(candidate))
    expected = set(content_tokens(reference))
    if not actual or not expected:
        return float(actual == expected)
    overlap = len(actual & expected)
    if not overlap:
        return 0.0
    precision = overlap / len(actual)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def fact_scores(candidate: str, facts: Iterable[str], threshold: float = 0.6) -> tuple[float, float]:
    values = [str(fact) for fact in facts if str(fact).strip()]
    if not values:
        return 0.0, 0.0
    recalls = [token_recall(candidate, fact) for fact in values]
    return statistics.fmean(recalls), sum(value >= threshold for value in recalls) / len(recalls)


def score_result(
    row: dict[str, Any],
    *,
    selected_contexts: list[dict[str, Any]],
    answerable: bool,
    answer: str,
    citations: list[str],
) -> dict[str, Any]:
    facts = [str(value) for value in row.get("answer_facts") or [] if str(value).strip()]
    question_type = str(row.get("question_type") or "unknown")
    unanswerable = question_type == "info_not_found"
    answer_evaluable = bool(facts) and not unanswerable
    context_text = "\n".join(str(context.get("text") or "") for context in selected_contexts)
    answer_fact_recall, answer_fact_coverage = fact_scores(answer, facts)
    context_fact_recall, context_fact_coverage = fact_scores(context_text, facts)
    gold_answer = str(row.get("gold_answer") or "")
    context_map = {str(context.get("citation_id")): context for context in selected_contexts}
    cited_doc_ids = {
        str(context_map[citation].get("doc_id") or "")
        for citation in citations
        if citation in context_map
    }
    expected_doc_ids = {str(value) for value in row.get("expected_doc_ids") or []}
    valid_citations = set(context_map)
    invalid = [citation for citation in citations if citation not in valid_citations]
    caveat = bool(INSUFFICIENT_CAVEAT_RE.search(answer)) or answer == REFUSAL_TEXT
    return {
        "is_answer_evaluable": answer_evaluable,
        "is_unanswerable": unanswerable,
        "answer_fact_token_recall": answer_fact_recall if answer_evaluable else None,
        "answer_fact_coverage_proxy": answer_fact_coverage if answer_evaluable else None,
        "context_fact_token_recall_upper_bound": context_fact_recall if answer_evaluable else None,
        "context_fact_coverage_proxy": context_fact_coverage if answer_evaluable else None,
        "gold_answer_token_f1": token_f1(answer, gold_answer) if answer_evaluable else None,
        "gold_answer_token_recall": token_recall(answer, gold_answer) if answer_evaluable else None,
        "model_answerable": answerable,
        "abstained": not answerable,
        "unanswerable_abstain_correct": (not answerable) if unanswerable else None,
        "unanswerable_caveat_correct": caveat if unanswerable else None,
        "citation_coverage": float(bool(citations)) if answerable else 0.0,
        "citation_precision": 1.0 if not citations else (len(citations) - len(invalid)) / len(citations),
        "invalid_citations": invalid,
        "gold_doc_cited": bool(expected_doc_ids & cited_doc_ids) if expected_doc_ids else None,
        "grounded_answer_proxy": bool(
            answer_evaluable
            and answerable
            and answer_fact_coverage >= 0.6
            and citations
            and not invalid
        ),
    }


def process_row(
    row: dict[str, Any],
    *,
    client: QwenClient,
    pipeline: str,
    enhance_max_contexts: int,
    enhance_max_input_chars: int,
    enhance_max_selected: int,
    enhance_max_tokens: int,
    selection_expansion: str,
    expansion_max_per_document: int,
    generation_max_contexts: int,
    generation_max_input_chars: int,
    generation_max_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    qid = str(row.get("qid") or row.get("id") or "")
    original_contexts = list(row.get("contexts") or [])
    source_map = context_by_citation(row)
    if not source_map:
        return error_row(row, pipeline, "input", "no valid evidence citations")

    enhancement_payload: dict[str, Any] | None = None
    enhancement_api: ApiResult | None = None
    if pipeline == "enhance-generate":
        rendered_for_enhancement, enhancement_inputs, enhancement_input_chars = render_contexts(
            original_contexts,
            max_contexts=enhance_max_contexts,
            max_chars=enhance_max_input_chars,
        )
        valid_ids = {str(context.get("citation_id")) for context in enhancement_inputs}
        try:
            enhancement_api = client.complete_json(
                messages=build_enhancement_messages(
                    row,
                    rendered_contexts=rendered_for_enhancement,
                    max_selected=enhance_max_selected,
                ),
                max_tokens=enhance_max_tokens,
                temperature=temperature,
                validator=lambda payload: validate_enhancement(
                    payload,
                    valid_citations=valid_ids,
                    max_selected=enhance_max_selected,
                ),
            )
        except Exception as exc:
            return error_row(row, pipeline, "enhancement", str(exc))
        enhancement_payload = enhancement_api.value
        selected_ids = list(enhancement_payload["selected_citations"])
        selected_original = expand_selected_contexts(
            original_contexts,
            selected_ids,
            mode=selection_expansion,
            max_per_document=expansion_max_per_document,
        )
        answerability_hint = str(enhancement_payload["answerability"])
    elif pipeline == "generate-only":
        enhancement_inputs = []
        enhancement_input_chars = 0
        selected_original = original_contexts
        answerability_hint = "unknown"
    else:
        return error_row(row, pipeline, "input", f"unsupported pipeline: {pipeline}")

    if pipeline == "enhance-generate" and not selected_original:
        if answerability_hint != "insufficient":
            return error_row(row, pipeline, "enhancement", "selector returned no usable contexts")
        generation_value = {
            "answerable": False,
            "answer": REFUSAL_TEXT,
            "citations": [],
            "citation_normalized": False,
        }
        generation_api = None
        rendered_for_generation = ""
        generation_contexts: list[dict[str, Any]] = []
        generation_input_chars = 0
    else:
        rendered_for_generation, generation_contexts, generation_input_chars = render_contexts(
            selected_original,
            max_contexts=generation_max_contexts,
            max_chars=generation_max_input_chars,
        )
        generation_ids = {str(context.get("citation_id")) for context in generation_contexts}
        try:
            generation_api = client.complete_json(
                messages=build_generation_messages(
                    row,
                    rendered_contexts=rendered_for_generation,
                    answerability_hint=answerability_hint,
                ),
                max_tokens=generation_max_tokens,
                temperature=temperature,
                validator=lambda payload: validate_generation(
                    payload,
                    valid_citations=generation_ids,
                ),
            )
        except Exception as exc:
            return error_row(row, pipeline, "generation", str(exc))
        generation_value = generation_api.value

    metrics = score_result(
        row,
        selected_contexts=generation_contexts,
        answerable=bool(generation_value["answerable"]),
        answer=str(generation_value["answer"]),
        citations=list(generation_value["citations"]),
    )
    original_chars = sum(len(str(context.get("text") or "")) for context in original_contexts)
    total_latency_ms = (time.perf_counter() - started) * 1000.0
    return {
        "qid": qid,
        "question": row.get("question"),
        "question_type": row.get("question_type"),
        "source_types": row.get("source_types") or [],
        "pipeline": pipeline,
        "model": client.model,
        "expected_doc_ids": row.get("expected_doc_ids") or [],
        "expected_accessible_doc_ids": row.get("expected_accessible_doc_ids") or [],
        "gold_answer": row.get("gold_answer"),
        "answer_facts": row.get("answer_facts") or [],
        "is_evaluable": bool(row.get("is_evaluable")),
        "retrieval_hit_at_10": bool(row.get("retrieval_hit_at_10")),
        "enhancement": {
            "status": "ok" if enhancement_payload is not None else "not_run",
            "answerability": enhancement_payload.get("answerability") if enhancement_payload else None,
            "selected_citations": enhancement_payload.get("selected_citations") if enhancement_payload else None,
            "expanded_citations": [
                str(context.get("citation_id") or "") for context in selected_original
            ] if enhancement_payload else None,
            "selection_expansion": selection_expansion if enhancement_payload else None,
            "conflict_citations": enhancement_payload.get("conflict_citations") if enhancement_payload else None,
            "input_context_count": len(enhancement_inputs),
            "input_context_chars": enhancement_input_chars,
            "latency_ms": enhancement_api.latency_ms if enhancement_api else 0.0,
            "usage": enhancement_api.usage if enhancement_api else zero_usage(),
            "request_id": enhancement_api.request_id if enhancement_api else "",
            "returned_model": enhancement_api.returned_model if enhancement_api else "",
            "attempts": enhancement_api.attempts if enhancement_api else 0,
            "finish_reason": enhancement_api.finish_reason if enhancement_api else "",
            "max_tokens_used": enhancement_api.max_tokens_used if enhancement_api else 0,
        },
        "selected_contexts": generation_contexts,
        "context_stats": {
            "original_context_count": len(original_contexts),
            "original_context_chars": original_chars,
            "selected_context_count": len(generation_contexts),
            "selected_context_chars": sum(
                len(str(context.get("text") or "")) for context in generation_contexts
            ),
            "generation_prompt_context_chars": generation_input_chars,
            "context_count_ratio": len(generation_contexts) / len(original_contexts)
            if original_contexts
            else 0.0,
            "context_char_ratio": generation_input_chars / original_chars if original_chars else 0.0,
        },
        "generation": {
            "status": "ok",
            "answerable": generation_value["answerable"],
            "answer": generation_value["answer"],
            "citations": generation_value["citations"],
            "citation_normalized": generation_value["citation_normalized"],
            "latency_ms": generation_api.latency_ms if generation_api else 0.0,
            "usage": generation_api.usage if generation_api else zero_usage(),
            "request_id": generation_api.request_id if generation_api else "",
            "returned_model": generation_api.returned_model if generation_api else "",
            "attempts": generation_api.attempts if generation_api else 0,
            "finish_reason": generation_api.finish_reason if generation_api else "",
            "max_tokens_used": generation_api.max_tokens_used if generation_api else 0,
        },
        "metrics": metrics,
        "total_latency_ms": total_latency_ms,
        "error": None,
    }


def error_row(row: dict[str, Any], pipeline: str, stage: str, message: str) -> dict[str, Any]:
    return {
        "qid": str(row.get("qid") or row.get("id") or ""),
        "question": row.get("question"),
        "question_type": row.get("question_type"),
        "source_types": row.get("source_types") or [],
        "pipeline": pipeline,
        "expected_doc_ids": row.get("expected_doc_ids") or [],
        "gold_answer": row.get("gold_answer"),
        "answer_facts": row.get("answer_facts") or [],
        "enhancement": {"status": "error" if stage == "enhancement" else "not_run"},
        "selected_contexts": [],
        "generation": {"status": "error" if stage == "generation" else "not_run"},
        "metrics": {},
        "total_latency_ms": 0.0,
        "error": {"stage": stage, "message": message},
    }


def zero_usage() -> dict[str, int]:
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cached_tokens": 0,
    }


def add_usage(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    return {key: int(left.get(key, 0)) + int(right.get(key, 0)) for key in zero_usage()}


def numeric_average(rows: list[dict[str, Any]], path: tuple[str, ...]) -> float | None:
    values: list[float] = []
    for row in rows:
        value: Any = row
        for key in path:
            if not isinstance(value, dict):
                value = None
                break
            value = value.get(key)
        if isinstance(value, bool):
            values.append(float(value))
        elif isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return statistics.fmean(values) if values else None


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * pct / 100.0) - 1))
    return ordered[index]


def summarize(rows: list[dict[str, Any]], *, input_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    successful = [row for row in rows if not row.get("error")]
    answer_evaluable = [
        row for row in successful if bool((row.get("metrics") or {}).get("is_answer_evaluable"))
    ]
    unanswerable = [
        row for row in successful if bool((row.get("metrics") or {}).get("is_unanswerable"))
    ]
    enhancement_rows = [
        row for row in successful if (row.get("enhancement") or {}).get("status") == "ok"
    ]
    latencies = [float(row.get("total_latency_ms") or 0.0) for row in successful]

    def usage_total(stage: str, key: str) -> int:
        return sum(
            int((((row.get(stage) or {}).get("usage") or {}).get(key)) or 0)
            for row in successful
        )

    summary: dict[str, Any] = {
        "input": str(input_path),
        "output": str(args.output),
        "pipeline": args.pipeline,
        "api_base": args.api_base,
        "model": args.model,
        "api_key_env": args.api_key_env,
        "run_signature": args.run_signature,
        "temperature": args.temperature,
        "questions_total": len(rows),
        "questions_successful": len(successful),
        "questions_errored": len(rows) - len(successful),
        "questions_answer_evaluable": len(answer_evaluable),
        "unanswerable_count": len(unanswerable),
        "retrieval_hit_at_10": numeric_average(
            [row for row in successful if row.get("is_evaluable")], ("retrieval_hit_at_10",)
        ),
        "enhancement": {
            "rows": len(enhancement_rows),
            "answerable_rate": numeric_average(
                [
                    {"value": (row.get("enhancement") or {}).get("answerability") == "answerable"}
                    for row in enhancement_rows
                ],
                ("value",),
            ),
            "conflicting_rate": numeric_average(
                [
                    {"value": (row.get("enhancement") or {}).get("answerability") == "conflicting"}
                    for row in enhancement_rows
                ],
                ("value",),
            ),
            "insufficient_rate": numeric_average(
                [
                    {"value": (row.get("enhancement") or {}).get("answerability") == "insufficient"}
                    for row in enhancement_rows
                ],
                ("value",),
            ),
            "avg_model_selected_citations": numeric_average(
                [
                    {
                        "count": len((row.get("enhancement") or {}).get("selected_citations") or [])
                    }
                    for row in enhancement_rows
                ],
                ("count",),
            ),
            "avg_selected_contexts": numeric_average(
                successful, ("context_stats", "selected_context_count")
            ),
            "avg_selected_context_chars": numeric_average(
                successful, ("context_stats", "selected_context_chars")
            ),
            "avg_generation_prompt_context_chars": numeric_average(
                successful, ("context_stats", "generation_prompt_context_chars")
            ),
            "avg_context_count_ratio": numeric_average(
                successful, ("context_stats", "context_count_ratio")
            ),
            "avg_context_char_ratio": numeric_average(
                successful, ("context_stats", "context_char_ratio")
            ),
            "avg_latency_ms": numeric_average(successful, ("enhancement", "latency_ms")),
            "prompt_tokens": usage_total("enhancement", "prompt_tokens"),
            "completion_tokens": usage_total("enhancement", "completion_tokens"),
            "total_tokens": usage_total("enhancement", "total_tokens"),
            "cached_tokens": usage_total("enhancement", "cached_tokens"),
        },
        "generation": {
            "answerable_rate": numeric_average(successful, ("generation", "answerable")),
            "answer_fact_token_recall_avg": numeric_average(
                answer_evaluable, ("metrics", "answer_fact_token_recall")
            ),
            "answer_fact_coverage_proxy_avg": numeric_average(
                answer_evaluable, ("metrics", "answer_fact_coverage_proxy")
            ),
            "context_fact_token_recall_upper_bound_avg": numeric_average(
                answer_evaluable, ("metrics", "context_fact_token_recall_upper_bound")
            ),
            "context_fact_coverage_proxy_avg": numeric_average(
                answer_evaluable, ("metrics", "context_fact_coverage_proxy")
            ),
            "gold_answer_token_f1_avg": numeric_average(
                answer_evaluable, ("metrics", "gold_answer_token_f1")
            ),
            "gold_answer_token_recall_avg": numeric_average(
                answer_evaluable, ("metrics", "gold_answer_token_recall")
            ),
            "citation_coverage_avg": numeric_average(successful, ("metrics", "citation_coverage")),
            "citation_precision_avg": numeric_average(successful, ("metrics", "citation_precision")),
            "gold_doc_citation_rate": numeric_average(
                [row for row in successful if row.get("expected_doc_ids")],
                ("metrics", "gold_doc_cited"),
            ),
            "grounded_answer_proxy_rate": numeric_average(
                answer_evaluable, ("metrics", "grounded_answer_proxy")
            ),
            "unanswerable_abstain_accuracy": numeric_average(
                unanswerable, ("metrics", "unanswerable_abstain_correct")
            ),
            "unanswerable_caveat_accuracy": numeric_average(
                unanswerable, ("metrics", "unanswerable_caveat_correct")
            ),
            "invalid_citation_count": sum(
                len((row.get("metrics") or {}).get("invalid_citations") or []) for row in successful
            ),
            "citation_normalized_count": sum(
                bool((row.get("generation") or {}).get("citation_normalized")) for row in successful
            ),
            "avg_latency_ms": numeric_average(successful, ("generation", "latency_ms")),
            "prompt_tokens": usage_total("generation", "prompt_tokens"),
            "completion_tokens": usage_total("generation", "completion_tokens"),
            "total_tokens": usage_total("generation", "total_tokens"),
            "cached_tokens": usage_total("generation", "cached_tokens"),
        },
        "avg_total_latency_ms": statistics.fmean(latencies) if latencies else None,
        "p95_total_latency_ms": percentile(latencies, 95),
        "wall_time_seconds": args.wall_time_seconds,
        "configuration": {
            "concurrency": args.concurrency,
            "enhance_max_contexts": args.enhance_max_contexts,
            "enhance_max_input_chars": args.enhance_max_input_chars,
            "enhance_max_selected": args.enhance_max_selected,
            "enhance_max_tokens": args.enhance_max_tokens,
            "selection_expansion": args.selection_expansion,
            "expansion_max_per_document": args.expansion_max_per_document,
            "generation_max_contexts": args.generation_max_contexts,
            "generation_max_input_chars": args.generation_max_input_chars,
            "generation_max_tokens": args.generation_max_tokens,
            "timeout_seconds": args.timeout_seconds,
            "retries": args.retries,
        },
    }
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in successful:
        grouped[str(row.get("question_type") or "unknown")].append(row)
    summary["by_question_type"] = {
        name: {
            "questions": len(group),
            "answer_fact_token_recall_avg": numeric_average(
                [row for row in group if (row.get("metrics") or {}).get("is_answer_evaluable")],
                ("metrics", "answer_fact_token_recall"),
            ),
            "context_fact_token_recall_upper_bound_avg": numeric_average(
                [row for row in group if (row.get("metrics") or {}).get("is_answer_evaluable")],
                ("metrics", "context_fact_token_recall_upper_bound"),
            ),
            "gold_doc_citation_rate": numeric_average(
                [row for row in group if row.get("expected_doc_ids")],
                ("metrics", "gold_doc_cited"),
            ),
            "grounded_answer_proxy_rate": numeric_average(
                [row for row in group if (row.get("metrics") or {}).get("is_answer_evaluable")],
                ("metrics", "grounded_answer_proxy"),
            ),
        }
        for name, group in sorted(grouped.items())
    }
    return summary


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as sink:
        for row in rows:
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def load_resume(path: Path, expected_signature: str) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = load_jsonl(path)
    mismatched = [
        str(row.get("qid") or "<unknown>")
        for row in rows
        if row.get("run_signature") != expected_signature
    ]
    if mismatched:
        raise ValueError(
            "resume output was created by another input/configuration; "
            f"mismatched rows include {mismatched[:5]}"
        )
    return {
        str(row.get("qid")): row
        for row in rows
        if str(row.get("qid") or "") and not row.get("error")
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary-output", required=True, type=Path)
    parser.add_argument(
        "--pipeline",
        choices=("generate-only", "enhance-generate"),
        default="generate-only",
    )
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--stratified", action="store_true")
    parser.add_argument("--qid-file", type=Path)
    parser.add_argument("--enhance-max-contexts", type=int, default=30)
    parser.add_argument("--enhance-max-input-chars", type=int, default=48000)
    parser.add_argument("--enhance-max-selected", type=int, default=8)
    parser.add_argument("--enhance-max-tokens", type=int, default=256)
    parser.add_argument(
        "--selection-expansion",
        choices=("none", "document", "rerank"),
        default="rerank",
        help="Use selected IDs only, expand their documents, or promote selected IDs before the original fallback ranking.",
    )
    parser.add_argument("--expansion-max-per-document", type=int, default=3)
    parser.add_argument("--generation-max-contexts", type=int, default=30)
    parser.add_argument("--generation-max-input-chars", type=int, default=16000)
    parser.add_argument("--generation-max-tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    positive = {
        "concurrency": args.concurrency,
        "enhance_max_contexts": args.enhance_max_contexts,
        "enhance_max_input_chars": args.enhance_max_input_chars,
        "enhance_max_selected": args.enhance_max_selected,
        "enhance_max_tokens": args.enhance_max_tokens,
        "expansion_max_per_document": args.expansion_max_per_document,
        "generation_max_contexts": args.generation_max_contexts,
        "generation_max_input_chars": args.generation_max_input_chars,
        "generation_max_tokens": args.generation_max_tokens,
        "timeout_seconds": args.timeout_seconds,
    }
    for name, value in positive.items():
        if value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.retries < 0:
        parser.error("--retries must not be negative")
    if not 0.0 <= args.temperature <= 2.0:
        parser.error("--temperature must be between 0 and 2")
    return args


def main() -> int:
    args = parse_args()
    validate_output_paths(
        [args.contexts, args.qid_file] if args.qid_file else [args.contexts],
        [args.output, args.summary_output],
    )
    api_key = os.getenv(args.api_key_env, "")
    if not api_key:
        raise SystemExit(f"missing API key environment variable: {args.api_key_env}")
    qids: set[str] | None = None
    if args.qid_file:
        raw = args.qid_file.read_text(encoding="utf-8")
        try:
            payload = json.loads(raw)
            values = payload if isinstance(payload, list) else payload.get("qids")
            if not isinstance(values, list):
                raise ValueError
            qids = {str(value) for value in values}
        except (json.JSONDecodeError, AttributeError, ValueError) as exc:
            raise SystemExit("--qid-file must contain a JSON array or {\"qids\": [...]} object") from exc

    source_rows = select_rows(
        load_jsonl(args.contexts),
        limit=args.limit,
        stratified=args.stratified,
        qids=qids,
    )
    if not source_rows:
        raise SystemExit("no input rows selected")
    args.run_signature = build_run_signature(args)
    resumed = load_resume(args.output, args.run_signature) if args.resume else {}
    client = QwenClient(
        api_base=args.api_base,
        api_key=api_key,
        model=args.model,
        timeout_seconds=args.timeout_seconds,
        retries=args.retries,
    )
    lock = threading.Lock()
    completed = 0
    started = time.perf_counter()

    def evaluate(row: dict[str, Any]) -> dict[str, Any]:
        qid = str(row.get("qid") or row.get("id") or "")
        if qid in resumed:
            return resumed[qid]
        result = process_row(
            row,
            client=client,
            pipeline=args.pipeline,
            enhance_max_contexts=args.enhance_max_contexts,
            enhance_max_input_chars=args.enhance_max_input_chars,
            enhance_max_selected=args.enhance_max_selected,
            enhance_max_tokens=args.enhance_max_tokens,
            selection_expansion=args.selection_expansion,
            expansion_max_per_document=args.expansion_max_per_document,
            generation_max_contexts=args.generation_max_contexts,
            generation_max_input_chars=args.generation_max_input_chars,
            generation_max_tokens=args.generation_max_tokens,
            temperature=args.temperature,
        )
        result["run_signature"] = args.run_signature
        return result

    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(evaluate, row) for row in source_rows]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            with lock:
                results.append(result)
                completed += 1
                if args.progress_every > 0 and (
                    completed % args.progress_every == 0 or completed == len(source_rows)
                ):
                    elapsed = max(time.perf_counter() - started, 0.001)
                    print(
                        json.dumps(
                            {
                                "event": "qwen_plus_rag_progress",
                                "completed": completed,
                                "total": len(source_rows),
                                "qps": round(completed / elapsed, 3),
                                "errors": sum(bool(row.get("error")) for row in results),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                if args.progress_every > 0 and completed % args.progress_every == 0:
                    order = {
                        str(row.get("qid") or row.get("id") or ""): index
                        for index, row in enumerate(source_rows)
                    }
                    atomic_write_jsonl(
                        args.output,
                        sorted(results, key=lambda row: order.get(str(row.get("qid")), 10**9)),
                    )

    order = {str(row.get("qid") or row.get("id") or ""): index for index, row in enumerate(source_rows)}
    results.sort(key=lambda row: order.get(str(row.get("qid")), 10**9))
    atomic_write_jsonl(args.output, results)
    args.wall_time_seconds = time.perf_counter() - started
    summary = summarize(results, input_path=args.contexts, args=args)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
