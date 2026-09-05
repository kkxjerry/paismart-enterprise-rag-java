"""Experimental, label-blind evidence packing; no model calls or answerability claims.

Every emitted body is one contiguous slice of its authorized input chunk. Offsets
are Python character offsets within that chunk, NOT byte or document offsets.
Selected/conflicting citations stay whole; other sources use paragraph windows.
Budgets include serialized headers and separators. Character caps are not token caps.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from functools import lru_cache
from dataclasses import dataclass
from typing import Any, Sequence

STRATEGY_VERSION = "query-spans-v3"
_CITATION = re.compile(r"S[1-9][0-9]*\Z")
_WORDS = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)*|[\u3400-\u9fff]+", re.IGNORECASE)
_STOP = frozenset(
    "a an the of to in on at for and or is are was were be been with by as "
    "what which who how when where does do did can could would should it its "
    "this that these those from about please give me tell us".split()
)
_STRUCTURED = re.compile(r"(?m)^\s*(?:[-*+]\s|\d+[.)]\s|\|)|```|~~~")
_DURATION_QUESTION = re.compile(r"how long|duration|recovery time|restore time|多久|多长时间", re.I)
_DURATION_VALUE = re.compile(
    r"\b(?:\d+(?:\.\d+)?|tens? of|several|a few|one|two|three)\s*"
    r"(?:seconds?|minutes?|hours?|days?)\b|(?:\d+|几十|几)\s*(?:秒|分钟|小时|天)", re.I
)
_LIST_QUESTION = re.compile(r"\b(?:fields?|enumerate|list|mandatory|required)\b|字段|列出", re.I)


@lru_cache(maxsize=32768)
def terms(text: str) -> frozenset[str]:
    """Small bilingual lexical feature set, not a semantic fact validator."""
    result: set[str] = set()
    for value in _WORDS.findall(text.casefold()):
        if value in _STOP:
            continue
        if re.fullmatch(r"[\u3400-\u9fff]+", value):
            result.update(value[i:i + 2] for i in range(max(1, len(value) - 1)))
        else:
            result.add(value[:-1] if len(value) > 4 and value.endswith("s")
                       and not value.endswith("ss") else value)
    return frozenset(result)


@lru_cache(maxsize=8192)
def paragraph_windows(
    text: str,
    *,
    whole_chunk_chars: int = 2400,
    neighbor_paragraphs: int = 1,
) -> list[tuple[int, int, int, int]]:
    """Preserve small chunks and indivisible lists/tables/code without slicing lines.

For longer prose retain the matching paragraph and its immediate neighbours.
This cannot prove that dependencies farther away are unnecessary; omitted ranges
remain explicit in the trace and this strategy is opt-in pending paired evaluation.
"""
    if len(text) <= whole_chunk_chars or _STRUCTURED.search(text):
        return [(0, len(text), 0, len(text))]
    boundaries = [0] + [m.end() for m in re.finditer(r"\n[ \t]*\n", text)]
    if boundaries[-1] != len(text):
        boundaries.append(len(text))
    ranges = [(a, b) for a, b in zip(boundaries, boundaries[1:]) if text[a:b].strip()]
    if not ranges:
        return [(0, len(text), 0, len(text))]
    return list(dict.fromkeys(
        (
            ranges[max(0, i - neighbor_paragraphs)][0],
            ranges[min(len(ranges) - 1, i + neighbor_paragraphs)][1],
            ranges[i][0],
            ranges[i][1],
        )
        for i in range(len(ranges))
    ))


@dataclass(frozen=True)
class PackingConfig:
    whole_chunk_chars: int = 2400
    neighbor_paragraphs: int = 1
    rank_exponent: float = 0.75
    diversity_penalty: float = 0.15
    density_exponent: float = 0.5
    density_scale_chars: float = 600.0
    duration_boost: float = 2.0
    list_boost: float = 1.5
    evidence_score_weight: float = 0.0
    query_coverage_weight: float = 0.0
    saturation_power: float = 1.0


DEFAULT_PACKING_CONFIG = PackingConfig(
    density_exponent=0.4,
    query_coverage_weight=0.2,
)


@dataclass(frozen=True)
class PackedEvidence:
    rendered: str
    contexts: tuple[dict[str, Any], ...]
    trace: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class _Candidate:
    citation: str
    context: dict[str, Any]
    start: int
    end: int
    ordinal: int
    affinities: tuple[float, ...]

    @property
    def body(self) -> str:
        return str(self.context.get("text") or "")[self.start:self.end]

    @property
    def block(self) -> str:
        return (
            f"[{self.citation}] title={self.context.get('title') or ''} "
            f"source_type={self.context.get('source_type') or 'unknown'} "
            f"doc_id={self.context.get('doc_id') or ''} "
            f"chunk_chars={self.start}:{self.end}/{len(str(self.context.get('text') or ''))}\n"
            + self.body
        )


def pack_evidence(
    contexts: Sequence[dict[str, Any]],
    *,
    question: str,
    requirements: Sequence[tuple[str, str]] = (),
    required_citations: Sequence[str] = (),
    max_chars: int,
    max_contexts: int,
    config: PackingConfig | None = None,
) -> PackedEvidence:
    """Soft source diversity replaces hard per-document deletion.

Only the question, requirement text, citation identities and input evidence enter
selection. Gold answers, answer facts and expected document IDs have no interface.
Returned lexical affinities MUST NOT be reported as factual support/coverage.
"""
    if max_chars <= 0 or max_contexts <= 0:
        raise ValueError("evidence budgets must be positive")
    selected_config = config or DEFAULT_PACKING_CONFIG
    if selected_config.whole_chunk_chars <= 0 or selected_config.neighbor_paragraphs < 0:
        raise ValueError("invalid packing window configuration")
    sources: dict[str, dict[str, Any]] = {}
    ordinals: dict[str, int] = {}
    diagnostics: list[dict[str, Any]] = []
    for ordinal, original in enumerate(contexts):
        citation = str(original.get("citation_id") or "").strip()
        if not _CITATION.fullmatch(citation):
            diagnostics.append({"citation_id": citation, "status": "excluded", "reason": "invalid_citation"})
            continue
        if citation in sources:
            raise ValueError(f"duplicate evidence citation: {citation}")
        sources[citation] = dict(original, citation_id=citation)
        ordinals[citation] = ordinal
    required = tuple(dict.fromkeys(required_citations))
    required_set = set(required)
    queries = list(requirements) or [("question", question)]
    query_terms = [terms(text) or terms(question) for _, text in queries]
    document_terms = [terms(str(source.get("text") or "")) for source in sources.values()]
    frequency: Counter[str] = Counter(term for value in document_terms for term in value)
    weights = {term: 1.0 + math.log1p(len(sources) / (1 + count)) for term, count in frequency.items()}
    candidates: list[_Candidate] = []
    for citation, source in sources.items():
        text = str(source.get("text") or "")
        if not text.strip():
            continue
        spans = (
            [(0, len(text), 0, len(text))]
            if citation in required_set
            else paragraph_windows(
                text,
                whole_chunk_chars=selected_config.whole_chunk_chars,
                neighbor_paragraphs=selected_config.neighbor_paragraphs,
            )
        )
        for start, end, focus_start, focus_end in spans:
            body = text[focus_start:focus_end]
            body_terms = terms(body)
            affinities = []
            for (_, query), tokens in zip(queries, query_terms):
                score = sum(weights.get(term, 1.0) for term in sorted(body_terms & tokens))
                # Generic answer-shape hints are ranking signals, never proof.
                if score and _DURATION_QUESTION.search(query + " " + question) and _DURATION_VALUE.search(body):
                    score += selected_config.duration_boost
                if score and _LIST_QUESTION.search(query + " " + question) and (
                    _STRUCTURED.search(body) or (":" in body and body.count(",") >= 2)
                ):
                    score += selected_config.list_boost
                affinities.append(score)
            candidates.append(_Candidate(citation, source, start, end, ordinals[citation], tuple(affinities)))

    selected: list[_Candidate] = []
    chosen: set[str] = set()
    used_chars = 0
    per_document: Counter[str] = Counter()
    lexical_hits = [0] * len(queries)
    reasons: dict[str, str] = {}

    def fits(candidate: _Candidate) -> bool:
        return len(selected) < max_contexts and used_chars + len(candidate.block) + (2 if selected else 0) <= max_chars

    def add(candidate: _Candidate) -> None:
        nonlocal used_chars
        used_chars += len(candidate.block) + (2 if selected else 0)
        selected.append(candidate)
        chosen.add(candidate.citation)
        per_document[str(candidate.context.get("doc_id") or candidate.citation)] += 1
        for i, affinity in enumerate(candidate.affinities):
            lexical_hits[i] += int(affinity > 0)

    # Reserve mapped evidence first, whole. Oversized required blocks are explicitly
    # missing, not silently cropped or falsely declared complete by citation ID.
    for citation in required:
        candidate = next((c for c in candidates if c.citation == citation), None)
        if candidate is None:
            reasons[citation] = "required_not_in_input_or_empty"
        elif fits(candidate):
            add(candidate)
        else:
            reasons[citation] = "required_exceeds_budget_or_context_limit"

    while len(selected) < max_contexts:
        available = [c for c in candidates if c.citation not in chosen
                     and c.citation not in required_set and fits(c)]
        if not available:
            break

        def utility(candidate: _Candidate) -> tuple[float, int, int]:
            relevance = sum(
                score / ((1 + lexical_hits[i]) ** selected_config.saturation_power)
                for i, score in enumerate(candidate.affinities)
            )
            diversity = 1 + selected_config.diversity_penalty * per_document[
                str(candidate.context.get("doc_id") or candidate.citation)
            ]
            density = max(1.0, len(candidate.block) / selected_config.density_scale_chars) ** selected_config.density_exponent
            # Retrieval ranks contain information absent from local lexical overlap.
            # Without this prior, keyword-dense unrelated chunks displace the very
            # document the retriever correctly found (observed on real replay).
            raw_rank = candidate.context.get("document_rank")
            rank = float(raw_rank) if isinstance(raw_rank, (int, float)) and not isinstance(raw_rank, bool) else 1.0
            if not math.isfinite(rank) or rank < 1:
                rank = 1.0
            source_prior = rank ** -selected_config.rank_exponent
            raw_evidence_score = candidate.context.get("evidence_score")
            evidence_score = (
                float(raw_evidence_score)
                if isinstance(raw_evidence_score, (int, float)) and not isinstance(raw_evidence_score, bool)
                else 0.0
            )
            raw_query_coverage = candidate.context.get("query_coverage")
            query_coverage = (
                float(raw_query_coverage)
                if isinstance(raw_query_coverage, (int, float)) and not isinstance(raw_query_coverage, bool)
                else 0.0
            )
            metadata_prior = (
                1.0
                + selected_config.evidence_score_weight * max(0.0, min(1.5, evidence_score))
                + selected_config.query_coverage_weight * max(0.0, min(1.0, query_coverage))
            )
            return (
                (relevance + 0.001 / (1 + candidate.ordinal))
                * source_prior
                * metadata_prior
                / (diversity * density),
                -candidate.ordinal,
                -candidate.start,
            )

        add(max(available, key=utility))

    packed: list[dict[str, Any]] = []
    selected_map = {candidate.citation: candidate for candidate in selected}
    for candidate in selected:
        copied = dict(candidate.context)
        copied["text"] = candidate.body
        copied["evidence_span"] = {
            "start_char": candidate.start,
            "end_char": candidate.end,
            "original_chars": len(str(candidate.context.get("text") or "")),
            "offset_unit": "python_character_within_input_chunk",
            "strategy": STRATEGY_VERSION,
        }
        packed.append(copied)
    for citation in dict.fromkeys([*sources, *required]):
        candidate = selected_map.get(citation)
        entry: dict[str, Any] = {
            "citation_id": citation,
            "doc_id": sources.get(citation, {}).get("doc_id"),
            "status": "included" if candidate else "excluded",
            "required": citation in required_set,
            "reason": ("required_whole_chunk" if citation in required_set else "lexical_utility_soft_diversity")
            if candidate else reasons.get(citation, "empty_text" if not str(sources.get(citation, {}).get("text") or "").strip()
                                         else "budget_or_context_limit"),
        }
        if candidate:
            entry.update({"start_char": candidate.start, "end_char": candidate.end,
                          "original_chars": len(str(candidate.context.get("text") or "")),
                          "lexical_affinity_not_support": dict(zip((key for key, _ in queries), candidate.affinities))})
        diagnostics.append(entry)
    rendered = "\n\n".join(candidate.block for candidate in selected)
    assert len(rendered) == used_chars and used_chars <= max_chars
    return PackedEvidence(rendered, tuple(packed), tuple(diagnostics))
