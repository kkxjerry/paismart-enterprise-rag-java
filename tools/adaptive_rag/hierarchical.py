"""Hierarchical Evidence experiments for E1/E4/E5/E6.

The module is deliberately label blind: selection only receives the question,
optional requirement text, authorized contexts and retrieval metadata. Gold answers,
answer facts and expected document IDs are evaluation-only data and have no input
parameter here.

E1 leaf-parent: score sentence/list/field leaves, return a contiguous parent span.
E4 contextual prefix: add deterministic document/section/speaker/time context to the
retrieval key while keeping the returned citation text unchanged.
E5 proposition-parent: use a deterministic proposition-like retrieval key. This is
not LLM proposition extraction and must not be called semantic entailment.
E6 global hierarchy: build an extractive document hierarchy for global generation;
it never invents summaries and every bullet retains an original S* citation.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Sequence

from .evidence_spans import terms

Strategy = Literal["leaf-parent", "contextual-leaf-parent", "proposition-parent"]

_CITATION_RE = re.compile(r"S[1-9][0-9]*\Z")
_SENTENCE_END = re.compile(r"(?<=[.!?。！？])(?:\s+|(?=[A-Z\u3400-\u9fff]))")
_LIST_LINE = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+|[A-Za-z_][\w.-]*\s*:\s+)")
_TABLE_LINE = re.compile(r"^\s*\|.*\|\s*$")
_HEADING_LINE = re.compile(r"^\s*#{1,6}\s+(.+?)\s*$")
_FIRE_SECTION = re.compile(
    r"^\s*(summary|transcript|topics?|next[_ ]steps?|action[_ ]items?|decisions?|"
    r"meeting header|attendees|timeline)\s*:\s*$",
    re.IGNORECASE,
)
_GMAIL_HEADER = re.compile(r"^\s*(from|to|cc|date|subject)\s*:\s*(.+)$", re.IGNORECASE)
_QUOTED_REPLY = re.compile(r"^\s*(?:>|On .+ wrote:|-{2,}\s*Original Message\s*-{2,})", re.I)
_EXACT_ANCHOR = re.compile(
    r"(?:\b\d{1,2}:\d{2}(?:\s*(?:UTC|GMT|[AP]M))?\b|"
    r"\b\d+(?:\.\d+)?\s*(?:%|ms|seconds?|minutes?|hours?|days?|MiB|GiB|MB|GB)\b|"
    r"\b[vV]?\d+(?:\.\d+){1,3}\b|"
    r"\b[A-Za-z][A-Za-z0-9]+(?:_[A-Za-z0-9]+)+\b|"
    r"\b[A-Z][a-z0-9]+(?:[A-Z][A-Za-z0-9]+)+\b|"
    r"/[A-Za-z0-9._~!$&'()*+,;=:@%/-]+|"
    r"\"[^\"\n]{3,120}\")",
    re.IGNORECASE,
)
_CONDITION = re.compile(
    r"\b(?:if|when|unless|only if|provided|assuming|depends? on|except|otherwise|"
    r"before|after|at least|at most|must not|cannot|not supported)\b|"
    r"如果|仅当|除非|取决于|例外|否则|之前|之后|至少|至多|不得|不能",
    re.IGNORECASE,
)
_GLOBAL_QUERY = re.compile(
    r"\b(?:overall|across|company-wide|organization-wide|landscape|themes?|trends?|"
    r"summarize all|compare across|portfolio|mission|strategy)\b|"
    r"总体|全局|跨文档|所有项目|主题|趋势|战略|使命",
    re.IGNORECASE,
)
_SINGLE_ARTIFACT = re.compile(
    r"\b(?:which|what|where|name of|title of|page|document|runbook|mechanism|service|"
    r"dashboard|tile|card|endpoint|field|fields)\b|"
    r"哪个|名称|页面|文档|手册|机制|服务|字段",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class HierarchyConfig:
    strategy: Strategy = "leaf-parent"
    max_parent_chars: int = 2200
    neighbor_sentences: int = 1
    rank_exponent: float = 0.70
    diversity_penalty: float = 0.12
    density_exponent: float = 0.35
    query_coverage_weight: float = 0.25
    evidence_score_weight: float = 0.10
    exact_anchor_boost: float = 2.5
    condition_parent_boost: float = 0.75
    prefix_weight: float = 0.35
    proposition_weight: float = 0.40
    max_leaves_per_parent: int = 3

    def validated(self) -> "HierarchyConfig":
        if self.strategy not in {"leaf-parent", "contextual-leaf-parent", "proposition-parent"}:
            raise ValueError(f"unsupported hierarchy strategy: {self.strategy}")
        if self.max_parent_chars <= 0 or self.neighbor_sentences < 0 or self.max_leaves_per_parent <= 0:
            raise ValueError("invalid hierarchy window limits")
        for value in (
            self.rank_exponent,
            self.diversity_penalty,
            self.density_exponent,
            self.query_coverage_weight,
            self.evidence_score_weight,
            self.exact_anchor_boost,
            self.condition_parent_boost,
            self.prefix_weight,
            self.proposition_weight,
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError("hierarchy weights must be finite and non-negative")
        return self


@dataclass(frozen=True)
class ParentSpan:
    citation_id: str
    context: dict[str, Any]
    start: int
    end: int
    kind: str
    parent_key: str

    @property
    def text(self) -> str:
        return str(self.context.get("text") or "")[self.start:self.end]


@dataclass(frozen=True)
class Leaf:
    parent: ParentSpan
    start: int
    end: int
    text: str
    retrieval_key: str
    proposition_key: str
    kind: str
    ordinal: int


@dataclass(frozen=True)
class HierarchicalEvidence:
    rendered: str
    contexts: tuple[dict[str, Any], ...]
    trace: tuple[dict[str, Any], ...]
    selected_leaves: tuple[dict[str, Any], ...]
    strategy: str


def contextual_prefix(context: dict[str, Any]) -> str:
    """Deterministic E4 prefix. It is used for retrieval only, not citation text."""
    fields = [
        ("title", context.get("title")),
        ("source", context.get("source_type")),
        ("section", context.get("section_path")),
        ("artifact", context.get("source_path")),
        ("speaker", context.get("speaker")),
        ("time", context.get("event_time") or context.get("source_updated_at")),
        ("kind", context.get("chunk_kind")),
    ]
    values = [f"{name}={str(value).strip()}" for name, value in fields if str(value or "").strip()]
    return " | ".join(values)


def exact_anchors(text: str) -> tuple[str, ...]:
    output: list[str] = []
    for value in _EXACT_ANCHOR.findall(text or ""):
        normalized = re.sub(r"\s+", " ", value.casefold()).strip().strip('"').rstrip(".,;:!?")
        if normalized and normalized not in output:
            output.append(normalized)
    return tuple(output)


def is_global_question(question: str, question_type: str | None = None) -> bool:
    return question_type == "high_level" or bool(_GLOBAL_QUERY.search(question or ""))


def is_single_artifact_question(question: str) -> bool:
    return bool(_SINGLE_ARTIFACT.search(question or "")) and not is_global_question(question)


def _line_spans(text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    cursor = 0
    for raw in text.splitlines(keepends=True):
        end = cursor + len(raw)
        spans.append((cursor, end, raw.rstrip("\r\n")))
        cursor = end
    if cursor < len(text):
        spans.append((cursor, len(text), text[cursor:]))
    return spans


def _paragraph_spans(text: str) -> list[tuple[int, int]]:
    boundaries = [0]
    boundaries.extend(match.end() for match in re.finditer(r"\n[ \t]*\n", text))
    if boundaries[-1] != len(text):
        boundaries.append(len(text))
    result = [(left, right) for left, right in zip(boundaries, boundaries[1:]) if text[left:right].strip()]
    return result or ([(0, len(text))] if text.strip() else [])


def _sentence_spans(text: str, offset: int) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    cursor = 0
    for match in _SENTENCE_END.finditer(text):
        end = match.start()
        if text[cursor:end].strip():
            result.append((offset + cursor, offset + end))
        cursor = match.end()
    if text[cursor:].strip():
        result.append((offset + cursor, offset + len(text)))
    return result or ([(offset, offset + len(text))] if text.strip() else [])


def _structured_blocks(text: str, source_type: str) -> list[tuple[int, int, str]]:
    lines = _line_spans(text)
    if not lines:
        return []
    blocks: list[tuple[int, int, str]] = []
    start: int | None = None
    kind = "paragraph"

    def flush(end: int) -> None:
        nonlocal start, kind
        if start is not None and text[start:end].strip():
            blocks.append((start, end, kind))
        start = None
        kind = "paragraph"

    active_section = ""
    for line_start, line_end, line in lines:
        stripped = line.strip()
        if not stripped:
            flush(line_start)
            continue
        heading = _HEADING_LINE.match(line)
        fire = _FIRE_SECTION.match(line) if source_type == "fireflies" else None
        mail = _GMAIL_HEADER.match(line) if source_type == "gmail" else None
        if heading or fire or (source_type == "gmail" and mail and mail.group(1).casefold() == "from"):
            flush(line_start)
            start = line_start
            active_section = (
                heading.group(1).strip() if heading else fire.group(1).strip() if fire else "email_message"
            )
            kind = "section" if source_type != "gmail" else "email_message"
            continue
        line_kind = (
            "table" if _TABLE_LINE.match(line) else
            "list" if _LIST_LINE.match(line) else
            "quoted_reply" if source_type == "gmail" and _QUOTED_REPLY.match(line) else
            "transcript_turn" if source_type in {"slack", "fireflies"} and re.match(r"^\s*\[[^]]+\]\s*[^:]{1,80}:", line) else
            "paragraph"
        )
        if start is None:
            start, kind = line_start, line_kind
        elif line_kind != kind and line_kind in {"table", "list", "quoted_reply", "transcript_turn"}:
            flush(line_start)
            start, kind = line_start, line_kind
        elif kind in {"table", "list", "transcript_turn"} and line_kind != kind:
            flush(line_start)
            start, kind = line_start, line_kind
        if active_section and kind == "paragraph":
            kind = f"{active_section.casefold().replace(' ', '_')}_paragraph"
    flush(len(text))
    return blocks or [(start, end, "paragraph") for start, end in _paragraph_spans(text)]


def _parent_spans(context: dict[str, Any], config: HierarchyConfig) -> list[ParentSpan]:
    text = str(context.get("text") or "")
    citation = str(context.get("citation_id") or "")
    source = str(context.get("source_type") or "").casefold()
    if not text.strip() or not _CITATION_RE.fullmatch(citation):
        return []
    # E1 works on already-selected Java Evidence chunks. A leaf chooses which
    # input chunk matters; the input chunk is the available parent. Splitting it
    # again into tiny paragraphs made the comparison unfair and dropped context.
    # E3 supplies true source parents later through parent_text metadata.
    parent_text = str(context.get("parent_text") or "")
    if parent_text.strip():
        copied = dict(context)
        copied["text"] = parent_text
        kind = str(context.get("parent_kind") or "source_parent")
        return [ParentSpan(citation, copied, 0, len(parent_text), kind, f"{citation}:source-parent")]
    if len(text) <= config.max_parent_chars:
        structured = bool(_TABLE_LINE.search(text) or re.search(r"(?m)^\s*(?:[-*+]\s+|\d+[.)]\s+)", text))
        kind = "structured_chunk" if structured else "input_chunk"
        return [ParentSpan(citation, context, 0, len(text), kind, f"{citation}:input-chunk")]

    blocks = _structured_blocks(text, source)
    parents: list[ParentSpan] = []
    for index, (start, end, kind) in enumerate(blocks):
        if end - start > config.max_parent_chars and kind not in {"table", "list", "email_message"}:
            paragraph = text[start:end]
            sentence_spans = _sentence_spans(paragraph, start)
            window = max(1, config.neighbor_sentences * 2 + 1)
            for sentence_index in range(0, len(sentence_spans), window):
                left = sentence_spans[sentence_index][0]
                right = sentence_spans[min(len(sentence_spans) - 1, sentence_index + window - 1)][1]
                parents.append(ParentSpan(citation, context, left, right, kind, f"{citation}:{index}:{sentence_index}"))
        else:
            parents.append(ParentSpan(citation, context, start, end, kind, f"{citation}:{index}"))
    return parents


def _leaf_spans(parent: ParentSpan) -> list[tuple[int, int, str]]:
    text = parent.text
    absolute = parent.start
    if parent.kind in {"list", "table", "transcript_turn", "email_message", "structured_chunk"}:
        values = []
        for start, end, line in _line_spans(text):
            if line.strip():
                values.append((absolute + start, absolute + end, parent.kind + "_item"))
        return values or [(parent.start, parent.end, parent.kind)]
    return [(start, end, "sentence") for start, end in _sentence_spans(text, absolute)]


def proposition_key(context: dict[str, Any], leaf_text: str) -> str:
    """Deterministic standalone key; no new factual content is generated."""
    prefix = contextual_prefix(context)
    stripped = re.sub(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)", "", leaf_text.strip())
    if re.match(r"^[A-Za-z_][\w.-]*\s*:\s*", stripped):
        proposition = stripped
    elif prefix:
        proposition = f"Within {prefix}, {stripped}"
    else:
        proposition = stripped
    return re.sub(r"\s+", " ", proposition).strip()


def build_leaves(contexts: Sequence[dict[str, Any]], config: HierarchyConfig) -> list[Leaf]:
    selected = config.validated()
    leaves: list[Leaf] = []
    ordinal = 0
    for context in contexts:
        for parent in _parent_spans(dict(context), selected):
            parent_leaves = _leaf_spans(parent)
            if len(parent_leaves) > selected.max_leaves_per_parent * 8:
                # Avoid a giant table dominating candidate memory; all original lines
                # remain in the parent and only evenly sampled leaf keys are scored.
                stride = max(1, len(parent_leaves) // (selected.max_leaves_per_parent * 8))
                parent_leaves = parent_leaves[::stride]
            for start, end, kind in parent_leaves:
                leaf_text = str(context.get("text") or "")[start:end].strip()
                if not leaf_text:
                    continue
                prefix = contextual_prefix(context)
                retrieval = leaf_text
                if selected.strategy == "contextual-leaf-parent" and prefix:
                    retrieval = f"{prefix}\n{leaf_text}"
                proposition = proposition_key(context, leaf_text)
                if selected.strategy == "proposition-parent":
                    retrieval = proposition
                leaves.append(
                    Leaf(parent, start, end, leaf_text, retrieval, proposition, kind, ordinal)
                )
                ordinal += 1
    return leaves


def _finite(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    selected = float(value)
    return selected if math.isfinite(selected) else default


def _block(parent: ParentSpan) -> str:
    context = parent.context
    metadata = [
        f"[{parent.citation_id}]",
        f"title={context.get('title') or ''}",
        f"source_type={context.get('source_type') or 'unknown'}",
        f"doc_id={context.get('doc_id') or ''}",
        f"parent_kind={parent.kind}",
        f"chunk_chars={parent.start}:{parent.end}/{len(str(context.get('text') or ''))}",
    ]
    section = str(context.get("section_path") or "").strip()
    if section:
        metadata.append(f"section={section}")
    return " ".join(metadata) + "\n" + parent.text


def pack_hierarchical(
    contexts: Sequence[dict[str, Any]],
    *,
    question: str,
    requirements: Sequence[tuple[str, str]] = (),
    required_citations: Sequence[str] = (),
    max_chars: int,
    max_contexts: int,
    config: HierarchyConfig | None = None,
) -> HierarchicalEvidence:
    selected_config = (config or HierarchyConfig()).validated()
    if max_chars <= 0 or max_contexts <= 0:
        raise ValueError("hierarchical budgets must be positive")
    context_values = [dict(value) for value in contexts]
    citations = [str(value.get("citation_id") or "") for value in context_values]
    if len([value for value in citations if _CITATION_RE.fullmatch(value)]) != len(set(value for value in citations if _CITATION_RE.fullmatch(value))):
        raise ValueError("duplicate evidence citation")
    leaves = build_leaves(context_values, selected_config)
    queries = list(requirements) or [("question", question)]
    query_terms = [terms(value) or terms(question) for _, value in queries]
    anchors = exact_anchors(question + "\n" + "\n".join(value for _, value in queries))
    leaf_terms = {leaf.ordinal: terms(leaf.retrieval_key) for leaf in leaves}
    document_frequency = Counter(term for value in leaf_terms.values() for term in value)
    idf = {term: 1.0 + math.log1p(len(leaves) / (1 + count)) for term, count in document_frequency.items()}
    block_by_parent = {leaf.parent.parent_key: _block(leaf.parent) for leaf in leaves}
    question_has_condition = bool(_CONDITION.search(question))
    features: dict[int, tuple[tuple[float, ...], float, str, float]] = {}
    for leaf in leaves:
        key_terms = leaf_terms[leaf.ordinal]
        overlaps = tuple(
            sum(idf.get(term, 1.0) for term in key_terms & tokens)
            for tokens in query_terms
        )
        normalized = " ".join(leaf.retrieval_key.casefold().split())
        static = selected_config.exact_anchor_boost * sum(anchor in normalized for anchor in anchors)
        if question_has_condition and _CONDITION.search(leaf.parent.text):
            static += selected_config.condition_parent_boost
        context = leaf.parent.context
        static_prior = max(1.0, _finite(context.get("document_rank"), 1.0)) ** -selected_config.rank_exponent
        static_prior *= 1.0 + selected_config.query_coverage_weight * max(
            0.0, min(1.0, _finite(context.get("query_coverage")))
        )
        static_prior *= 1.0 + selected_config.evidence_score_weight * max(
            0.0, min(1.5, _finite(context.get("evidence_score")))
        )
        if selected_config.strategy == "contextual-leaf-parent":
            static_prior *= 1.0 + selected_config.prefix_weight
        if selected_config.strategy == "proposition-parent":
            static_prior *= 1.0 + selected_config.proposition_weight
        doc = str(context.get("doc_id") or leaf.parent.citation_id)
        density = max(1.0, len(block_by_parent[leaf.parent.parent_key]) / 700.0) ** selected_config.density_exponent
        features[leaf.ordinal] = (overlaps, static, doc, static_prior / density)

    per_doc: Counter[str] = Counter()
    coverage = [0] * len(queries)
    selected_parents: list[ParentSpan] = []
    selected_parent_keys: set[str] = set()
    selected_citations: set[str] = set()
    leaf_trace: list[dict[str, Any]] = []
    used = 0

    def score(leaf: Leaf) -> tuple[float, int, int]:
        overlaps, static, doc, prior = features[leaf.ordinal]
        relevance = sum(
            overlap / ((1 + coverage[index]) ** 0.85)
            for index, overlap in enumerate(overlaps)
        )
        raw = (relevance + static) * prior
        raw /= 1.0 + selected_config.diversity_penalty * per_doc[doc]
        return raw, -leaf.ordinal, -leaf.start

    required = tuple(dict.fromkeys(required_citations))
    by_citation: dict[str, list[Leaf]] = defaultdict(list)
    for leaf in leaves:
        by_citation[leaf.parent.citation_id].append(leaf)

    def add(leaf: Leaf, reason: str) -> bool:
        nonlocal used
        parent = leaf.parent
        if parent.parent_key in selected_parent_keys or parent.citation_id in selected_citations:
            return False
        block = block_by_parent[parent.parent_key]
        cost = len(block) + (2 if selected_parents else 0)
        if len(selected_parents) >= max_contexts or used + cost > max_chars:
            return False
        selected_parents.append(parent)
        selected_parent_keys.add(parent.parent_key)
        selected_citations.add(parent.citation_id)
        used += cost
        doc = str(parent.context.get("doc_id") or parent.citation_id)
        per_doc[doc] += 1
        selected_terms = leaf_terms[leaf.ordinal]
        for index, tokens in enumerate(query_terms):
            coverage[index] += int(bool(selected_terms & tokens))
        leaf_trace.append({
            "citation_id": parent.citation_id,
            "doc_id": parent.context.get("doc_id"),
            "leaf_text": leaf.text,
            "leaf_kind": leaf.kind,
            "parent_kind": parent.kind,
            "parent_start": parent.start,
            "parent_end": parent.end,
            "selection_reason": reason,
            "score": score(leaf)[0],
        })
        return True

    for citation in required:
        options = by_citation.get(citation, [])
        if options:
            add(max(options, key=score), "required_citation")

    while len(selected_parents) < max_contexts:
        available = [
            leaf for leaf in leaves
            if leaf.parent.parent_key not in selected_parent_keys
            and leaf.parent.citation_id not in selected_citations
            and used + len(block_by_parent[leaf.parent.parent_key]) + (2 if selected_parents else 0) <= max_chars
        ]
        if not available:
            break
        best = max(available, key=score)
        if not add(best, "leaf_score"):
            break

    selected_contexts: list[dict[str, Any]] = []
    for parent in selected_parents:
        copied = dict(parent.context)
        copied["text"] = parent.text
        copied["hierarchy"] = {
            "strategy": selected_config.strategy,
            "parent_key": parent.parent_key,
            "parent_kind": parent.kind,
            "start_char": parent.start,
            "end_char": parent.end,
            "offset_unit": "python_character_within_input_chunk",
        }
        selected_contexts.append(copied)
    trace = []
    chosen_by_citation = {value["citation_id"]: value for value in leaf_trace}
    for context in context_values:
        citation = str(context.get("citation_id") or "")
        selected = chosen_by_citation.get(citation)
        trace.append({
            "citation_id": citation,
            "doc_id": context.get("doc_id"),
            "status": "included" if selected else "excluded",
            "reason": selected["selection_reason"] if selected else "budget_or_lower_leaf_score",
            "parent_start": selected.get("parent_start") if selected else None,
            "parent_end": selected.get("parent_end") if selected else None,
            "strategy": selected_config.strategy,
        })
    rendered = "\n\n".join(block_by_parent[parent.parent_key] for parent in selected_parents)
    if len(rendered) != used or len(rendered) > max_chars:
        raise AssertionError("hierarchical renderer violated exact character budget")
    return HierarchicalEvidence(
        rendered,
        tuple(selected_contexts),
        tuple(trace),
        tuple(leaf_trace),
        selected_config.strategy,
    )


def build_global_hierarchy(
    contexts: Sequence[dict[str, Any]],
    *,
    question: str,
    max_documents: int = 8,
    leaves_per_document: int = 3,
    max_chars: int = 14_000,
) -> dict[str, Any]:
    """E6 extractive document hierarchy for global/high-level generation.

    The synopsis text is copied verbatim from leaves and carries the original
    citation next to every bullet. It is not an abstractive RAPTOR summary.
    """
    if max_documents <= 0 or leaves_per_document <= 0 or max_chars <= 0:
        raise ValueError("global hierarchy limits must be positive")
    config = HierarchyConfig(strategy="contextual-leaf-parent", max_parent_chars=1800)
    leaves = build_leaves(contexts, config)
    query_terms = terms(question)
    grouped: dict[str, list[Leaf]] = defaultdict(list)
    for leaf in leaves:
        doc = str(leaf.parent.context.get("doc_id") or leaf.parent.citation_id)
        grouped[doc].append(leaf)

    def leaf_score(leaf: Leaf) -> float:
        overlap = len(terms(leaf.retrieval_key) & query_terms)
        context = leaf.parent.context
        rank = max(1.0, _finite(context.get("document_rank"), 1.0))
        return (overlap + 0.25 + _finite(context.get("query_coverage"))) / rank ** 0.65

    documents = []
    for doc_id, values in grouped.items():
        best = sorted(values, key=leaf_score, reverse=True)[:leaves_per_document]
        if not best:
            continue
        context = best[0].parent.context
        documents.append({
            "doc_id": doc_id,
            "title": str(context.get("title") or ""),
            "source_type": str(context.get("source_type") or "unknown"),
            "document_rank": _finite(context.get("document_rank"), 999.0),
            "score": sum(leaf_score(value) for value in best),
            "leaves": [
                {"citation_id": value.parent.citation_id, "text": value.text, "kind": value.kind}
                for value in best
            ],
        })
    documents.sort(key=lambda value: (-value["score"], value["document_rank"], value["doc_id"]))
    selected = []
    rendered_parts = []
    used = 0
    for index, document in enumerate(documents[:max_documents], start=1):
        body = [
            f"Document D{index}: title={document['title']} source_type={document['source_type']} "
            f"doc_id={document['doc_id']}"
        ]
        for leaf in document["leaves"]:
            body.append(f"- [{leaf['citation_id']}] {leaf['text']}")
        block = "\n".join(body)
        cost = len(block) + (2 if rendered_parts else 0)
        if used + cost > max_chars:
            continue
        rendered_parts.append(block)
        selected.append(document)
        used += cost
    return {
        "rendered": "\n\n".join(rendered_parts),
        "documents": selected,
        "rendered_chars": used,
        "source_citations": [
            leaf["citation_id"] for document in selected for leaf in document["leaves"]
        ],
        "mode": "extractive-document-hierarchy-v1",
    }
