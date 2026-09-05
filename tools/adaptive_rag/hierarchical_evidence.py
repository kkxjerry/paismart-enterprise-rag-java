"""Hierarchical, source-aware evidence selection experiments (E1/E3/E4/E5/E6).

The module is deliberately label-blind: gold answers, answer facts and expected
IDs are not accepted by any selection API. Existing authorized evidence is split
into source-shaped parents and atomic leaves. Leaves are ranked; contiguous parent
text is returned for generation and citation. Deterministic contextual prefixes
are search-only and never replace quoted source text.

This is an opt-in research path. It does not mutate an index or relax ACLs.
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Literal, Sequence

from .evidence_spans import terms

LeafMode = Literal["sentence", "proposition"]
RouteMode = Literal["local", "global"]

_CITATION = re.compile(r"S[1-9][0-9]*\Z")
_MARKDOWN_HEADING = re.compile(r"(?m)^(#{1,6})[ \t]+(.+?)[ \t]*$")
_LABELED_SECTION = re.compile(
    r"(?im)^(summary|topics?|next[_ ]steps?|decisions?|action[_ ]items?|timeline|"
    r"root[_ ]cause|impact|resolution|workaround|acceptance[_ ]criteria|notes?)\s*:\s*$"
)
_EMAIL_BOUNDARY = re.compile(
    r"(?im)^(?:from|sent|date|subject|to|cc):\s+.+$|^on .{3,160} wrote:\s*$"
)
_CODE_FENCE = re.compile(r"(?ms)^```.*?^```[ \t]*$")
_TABLE_LINE = re.compile(r"^[ \t]*\|.*\|[ \t]*$")
_LIST_LINE = re.compile(r"^[ \t]*(?:[-*+] |\d+[.)] ).+")
_KEY_VALUE = re.compile(r"^[ \t]*[A-Za-z][A-Za-z0-9 _./-]{0,80}:\s+.+")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?。！？])(?:[ \t]+|\n+)")
_PROPOSITION_BOUNDARY = re.compile(r"(?<=[;；])\s+|\s+[—–]\s+|(?<=,)\s+(?=(?:and|but|while|whereas)\b)", re.I)
_VALUE = re.compile(
    r"\b\d{1,2}:\d{2}(?:\s*(?:UTC|GMT|[AP]M))?\b|"
    r"\b\d+(?:\.\d+)?\s*(?:%|ms|seconds?|minutes?|hours?|days?|GB|MB|KB|MiB|GiB)\b|"
    r"\b[vV]?\d+(?:\.\d+){1,3}\b|\btens? of minutes\b|"
    r"\b[A-Za-z][A-Za-z0-9]+(?:_[A-Za-z0-9]+)+\b|"
    r"\b[A-Z][a-z0-9]+(?:[A-Z][A-Za-z0-9]+)+\b|"
    r"/[A-Za-z0-9._~!$&'()*+,;=:@%/-]+",
    re.I,
)
_DURATION_QUERY = re.compile(r"how long|duration|recovery time|restore time|多久|多长时间", re.I)
_LIST_QUERY = re.compile(r"\b(?:which|what|list|enumerate|required|mandatory|fields?|items?|steps?)\b|列出|字段|步骤", re.I)
_GLOBAL_QUERY = re.compile(
    r"\b(?:overall|across|themes?|landscape|portfolio|company-wide|mission|strategy|"
    r"summari[sz]e|high[- ]level|policy dimensions?|all projects?)\b|整体|全局|总结|主题|使命|战略",
    re.I,
)
_CANONICAL_QUERY = re.compile(
    r"\b(?:what is the name|where can i find|which (?:page|document|runbook|mechanism|service)|"
    r"the (?:page|document|runbook|mechanism|service) (?:that|which))\b|"
    r"哪个(?:页面|文档|机制|服务)|名称是什么|在哪里",
    re.I,
)


@dataclass(frozen=True)
class HierarchyConfig:
    leaf_mode: LeafMode = "sentence"
    route_mode: RouteMode = "local"
    contextual_prefix: bool = True
    whole_parent_chars: int = 2_400
    parent_window_chars: int = 1_800
    neighbor_units: int = 1
    max_leaf_candidates: int = 160
    max_selected_leaves: int = 32
    rank_exponent: float = 0.75
    evidence_score_weight: float = 0.10
    query_coverage_weight: float = 0.25
    contextual_match_weight: float = 0.30
    diversity_penalty: float = 0.10
    global_document_bonus: float = 0.75
    canonical_margin: float = 0.12

    def validated(self) -> "HierarchyConfig":
        if self.leaf_mode not in {"sentence", "proposition"}:
            raise ValueError(f"unsupported leaf mode: {self.leaf_mode}")
        if self.route_mode not in {"local", "global"}:
            raise ValueError(f"unsupported route mode: {self.route_mode}")
        for name in (
            "whole_parent_chars",
            "parent_window_chars",
            "neighbor_units",
            "max_leaf_candidates",
            "max_selected_leaves",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in (
            "rank_exponent",
            "evidence_score_weight",
            "query_coverage_weight",
            "contextual_match_weight",
            "diversity_penalty",
            "global_document_bonus",
            "canonical_margin",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        return self


DEFAULT_HIERARCHY_CONFIG = HierarchyConfig()


@dataclass(frozen=True)
class ParentNode:
    id: str
    citation_id: str
    doc_id: str
    source_type: str
    title: str
    section_path: str
    kind: str
    speaker: str
    event_time: str
    start_char: int
    end_char: int
    text: str
    contextual_prefix: str
    context: dict[str, Any]


@dataclass(frozen=True)
class LeafNode:
    id: str
    parent_id: str
    citation_id: str
    doc_id: str
    kind: str
    start_char: int
    end_char: int
    text: str
    search_text: str
    ordinal: int


@dataclass(frozen=True)
class ScoredLeaf:
    leaf: LeafNode
    score: float
    lexical_score: float
    contextual_score: float
    exact_anchor_score: float
    rank_prior: float
    evidence_prior: float
    query_coverage_prior: float


@dataclass(frozen=True)
class HierarchicalEvidence:
    rendered: str
    contexts: tuple[dict[str, Any], ...]
    trace: tuple[dict[str, Any], ...]
    canonical_documents: tuple[dict[str, Any], ...]
    route_mode: str
    leaf_mode: str
    contextual_prefix_enabled: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "rendered_chars": len(self.rendered),
            "context_count": len(self.contexts),
            "route_mode": self.route_mode,
            "leaf_mode": self.leaf_mode,
            "contextual_prefix_enabled": self.contextual_prefix_enabled,
            "canonical_documents": list(self.canonical_documents),
            "trace": list(self.trace),
        }


def inferred_route(question: str, question_type: str | None = None) -> RouteMode:
    if question_type == "high_level" or _GLOBAL_QUERY.search(question):
        return "global"
    return "local"


def requires_canonical_document(question: str) -> bool:
    return bool(_CANONICAL_QUERY.search(question))


def contextual_prefix(context: dict[str, Any], *, section_path: str = "", kind: str = "") -> str:
    values = [
        ("source", context.get("source_type")),
        ("title", context.get("title")),
        ("section", section_path or context.get("section_path")),
        ("kind", kind or context.get("chunk_kind")),
        ("speaker", context.get("speaker")),
        ("time", context.get("event_time") or context.get("source_updated_at")),
        ("path", context.get("source_path")),
        ("artifact", context.get("doc_id")),
    ]
    rendered = [f"{key}={_compact(value, 160)}" for key, value in values if str(value or "").strip()]
    return "[" + " | ".join(rendered) + "]" if rendered else ""


def build_hierarchy(contexts: Sequence[dict[str, Any]], config: HierarchyConfig | None = None) -> tuple[list[ParentNode], list[LeafNode]]:
    selected = (config or DEFAULT_HIERARCHY_CONFIG).validated()
    parents: list[ParentNode] = []
    leaves: list[LeafNode] = []
    observed: set[str] = set()
    for context_ordinal, original in enumerate(contexts):
        context = dict(original)
        citation = str(context.get("citation_id") or "").strip()
        if not _CITATION.fullmatch(citation):
            continue
        if citation in observed:
            raise ValueError(f"duplicate citation ID: {citation}")
        observed.add(citation)
        text = str(context.get("text") or "")
        if not text.strip():
            continue
        source = str(context.get("source_type") or "unknown").casefold()
        ranges = _parent_ranges(source, text, context)
        for parent_index, (start, end, kind, section) in enumerate(ranges, start=1):
            body = text[start:end]
            if not body.strip():
                continue
            parent_id = _node_id(citation, "parent", parent_index, start, end)
            prefix = contextual_prefix(context, section_path=section, kind=kind)
            parent = ParentNode(
                id=parent_id,
                citation_id=citation,
                doc_id=str(context.get("doc_id") or citation),
                source_type=source,
                title=str(context.get("title") or ""),
                section_path=section,
                kind=kind,
                speaker=str(context.get("speaker") or ""),
                event_time=str(context.get("event_time") or ""),
                start_char=start,
                end_char=end,
                text=body,
                contextual_prefix=prefix,
                context=context,
            )
            parents.append(parent)
            for unit_index, (relative_start, relative_end, leaf_kind) in enumerate(
                _leaf_ranges(body, source=source, mode=selected.leaf_mode), start=1
            ):
                absolute_start = start + relative_start
                absolute_end = start + relative_end
                leaf_text = text[absolute_start:absolute_end]
                if not leaf_text.strip():
                    continue
                search_text = (
                    f"{prefix}\n{leaf_text}" if selected.contextual_prefix and prefix else leaf_text
                )
                leaves.append(
                    LeafNode(
                        id=_node_id(citation, "leaf", parent_index, unit_index, absolute_start, absolute_end),
                        parent_id=parent_id,
                        citation_id=citation,
                        doc_id=parent.doc_id,
                        kind=leaf_kind,
                        start_char=absolute_start,
                        end_char=absolute_end,
                        text=leaf_text,
                        search_text=search_text,
                        ordinal=context_ordinal * 10_000 + parent_index * 100 + unit_index,
                    )
                )
    return parents, leaves


def pack_hierarchical_evidence(
    contexts: Sequence[dict[str, Any]],
    *,
    question: str,
    requirements: Sequence[tuple[str, str]] = (),
    max_chars: int,
    max_contexts: int,
    config: HierarchyConfig | None = None,
) -> HierarchicalEvidence:
    selected = (config or DEFAULT_HIERARCHY_CONFIG).validated()
    if max_chars <= 0 or max_contexts <= 0:
        raise ValueError("evidence budgets must be positive")
    parents, leaves = build_hierarchy(contexts, selected)
    parent_by_id = {parent.id: parent for parent in parents}
    queries = list(requirements) or [("R1", question)]
    scored = _score_leaves(leaves, parent_by_id, queries, question, selected)
    chosen = _select_leaves(scored, queries, selected)
    ranges = _expand_to_parent_ranges(chosen, parents, leaves, selected)
    candidates = _range_candidates(ranges, parent_by_id, chosen)
    canonical = _canonical_documents(chosen, parent_by_id, queries, selected)
    if selected.route_mode == "global":
        candidates = _global_order(candidates, canonical)
    else:
        candidates.sort(key=lambda value: (-value["score"], value["document_rank"], value["ordinal"]))

    included: list[dict[str, Any]] = []
    rendered_blocks: list[str] = []
    used = 0
    seen_citations: set[str] = set()
    trace: list[dict[str, Any]] = []
    for value in candidates:
        citation = value["citation_id"]
        if citation in seen_citations:
            trace.append(_trace(value, "excluded", "lower_scoring_range_for_same_citation"))
            continue
        block = _render_candidate(value)
        separator = 2 if rendered_blocks else 0
        if len(included) >= max_contexts or used + separator + len(block) > max_chars:
            trace.append(_trace(value, "excluded", "character_or_context_budget"))
            continue
        copied = dict(value["context"])
        copied["text"] = value["text"]
        copied["hierarchy"] = {
            "parent_id": value["parent_id"],
            "parent_kind": value["parent_kind"],
            "section_path": value["section_path"],
            "start_char": value["start_char"],
            "end_char": value["end_char"],
            "offset_unit": "python_character_within_input_evidence_chunk",
            "leaf_ids": value["leaf_ids"],
            "leaf_kinds": value["leaf_kinds"],
            "search_context_prefix": value["contextual_prefix"],
            "search_context_prefix_returned_to_generator": False,
            "strategy": "hierarchical-evidence-v1",
            "route_mode": selected.route_mode,
            "leaf_mode": selected.leaf_mode,
        }
        included.append(copied)
        rendered_blocks.append(block)
        used += separator + len(block)
        seen_citations.add(citation)
        trace.append(_trace(value, "included", "ranked_leaf_expanded_to_contiguous_parent"))

    rendered = "\n\n".join(rendered_blocks)
    if len(rendered) != used or len(rendered) > max_chars:
        raise AssertionError("hierarchical evidence budget accounting mismatch")
    return HierarchicalEvidence(
        rendered=rendered,
        contexts=tuple(included),
        trace=tuple(trace),
        canonical_documents=tuple(canonical),
        route_mode=selected.route_mode,
        leaf_mode=selected.leaf_mode,
        contextual_prefix_enabled=selected.contextual_prefix,
    )


def summary_tree(contexts: Sequence[dict[str, Any]], *, max_sentences_per_document: int = 3) -> list[dict[str, Any]]:
    """Build deterministic extractive document summaries for the E6 global route.

    This is a shallow RAPTOR-like POC: it creates document-level parents from
    existing evidence, without claiming an LLM-generated semantic hierarchy.
    """
    parents, leaves = build_hierarchy(contexts, HierarchyConfig(route_mode="global"))
    parent_by_id = {parent.id: parent for parent in parents}
    by_doc: dict[str, list[LeafNode]] = defaultdict(list)
    for leaf in leaves:
        by_doc[leaf.doc_id].append(leaf)
    output: list[dict[str, Any]] = []
    for doc_id, values in by_doc.items():
        document_terms = Counter(term for leaf in values for term in terms(leaf.text))
        ranked = sorted(
            values,
            key=lambda leaf: (
                -sum(1.0 / max(1, document_terms[token]) for token in terms(leaf.text)),
                leaf.ordinal,
            ),
        )[:max_sentences_per_document]
        if not ranked:
            continue
        parent = parent_by_id[ranked[0].parent_id]
        output.append(
            {
                "doc_id": doc_id,
                "title": parent.title,
                "source_type": parent.source_type,
                "summary": " ".join(leaf.text.strip() for leaf in sorted(ranked, key=lambda item: item.ordinal)),
                "leaf_ids": [leaf.id for leaf in ranked],
                "document_rank": _number(parent.context.get("document_rank"), 10**9),
            }
        )
    return sorted(output, key=lambda value: (value["document_rank"], value["doc_id"]))


def _parent_ranges(source: str, text: str, context: dict[str, Any]) -> list[tuple[int, int, str, str]]:
    if source == "fireflies":
        ranges = _labeled_ranges(text, "meeting")
        if ranges:
            return ranges
    if source == "gmail":
        ranges = _email_ranges(text)
        if ranges:
            return ranges
    if source in {"confluence", "google_drive", "github"}:
        ranges = _markdown_ranges(text, "section" if source != "github" else "github_section")
        if ranges:
            return ranges
    if source in {"jira", "linear"}:
        section = str(context.get("section_path") or context.get("chunk_kind") or "issue")
        return [(0, len(text), str(context.get("chunk_kind") or "issue_section"), section)]
    if source == "slack":
        section = str(context.get("section_path") or context.get("thread_id") or "thread")
        return [(0, len(text), str(context.get("chunk_kind") or "conversation_window"), section)]
    ranges = _paragraph_ranges(text)
    return ranges or [(0, len(text), "body", str(context.get("section_path") or "body"))]


def _markdown_ranges(text: str, default_kind: str) -> list[tuple[int, int, str, str]]:
    matches = list(_MARKDOWN_HEADING.finditer(text))
    if not matches:
        return _paragraph_ranges(text)
    output: list[tuple[int, int, str, str]] = []
    stack: list[tuple[int, str]] = []
    if matches[0].start() > 0 and text[: matches[0].start()].strip():
        output.append((0, matches[0].start(), f"{default_kind}_preamble", "preamble"))
    for index, match in enumerate(matches):
        level = len(match.group(1))
        label = match.group(2).strip()
        stack = [value for value in stack if value[0] < level]
        stack.append((level, label))
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        if text[match.start():end].strip():
            output.append((match.start(), end, default_kind, " / ".join(item[1] for item in stack)))
    return output


def _labeled_ranges(text: str, prefix: str) -> list[tuple[int, int, str, str]]:
    matches = list(_LABELED_SECTION.finditer(text))
    if not matches:
        return []
    output: list[tuple[int, int, str, str]] = []
    if matches[0].start() > 0 and text[: matches[0].start()].strip():
        output.append((0, matches[0].start(), f"{prefix}_preamble", "preamble"))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        label = re.sub(r"\s+", "_", match.group(1).casefold())
        output.append((match.start(), end, f"{prefix}_{label}", label))
    return output


def _email_ranges(text: str) -> list[tuple[int, int, str, str]]:
    matches = list(_EMAIL_BOUNDARY.finditer(text))
    starts = sorted({match.start() for match in matches if match.group(0).casefold().startswith(("from:", "on "))})
    if not starts:
        return _paragraph_ranges(text)
    if starts[0] != 0:
        starts.insert(0, 0)
    output = []
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(text)
        body = text[start:end]
        subject = re.search(r"(?im)^subject:\s*(.+)$", body)
        section = subject.group(1).strip() if subject else f"message-{index + 1}"
        output.append((start, end, "email_message", section))
    return output


def _paragraph_ranges(text: str) -> list[tuple[int, int, str, str]]:
    boundaries = [0] + [match.end() for match in re.finditer(r"\n[ \t]*\n", text)]
    if boundaries[-1] != len(text):
        boundaries.append(len(text))
    output = []
    for index, (start, end) in enumerate(zip(boundaries, boundaries[1:]), start=1):
        if text[start:end].strip():
            output.append((start, end, "paragraph", f"paragraph-{index}"))
    return output


def _leaf_ranges(text: str, *, source: str, mode: LeafMode) -> list[tuple[int, int, str]]:
    # Code blocks and tables are retrieval-addressable but returned through their parent.
    protected: list[tuple[int, int, str]] = [
        (match.start(), match.end(), "code_block") for match in _CODE_FENCE.finditer(text)
    ]
    lines = list(re.finditer(r"(?m)^.*(?:\n|$)", text))
    table_group: list[re.Match[str]] = []
    output: list[tuple[int, int, str]] = []

    def flush_table() -> None:
        nonlocal table_group
        if table_group:
            output.extend((match.start(), match.end(), "table_row") for match in table_group if match.group().strip())
            table_group = []

    for match in lines:
        value = match.group().rstrip("\n")
        if not value.strip():
            flush_table()
            continue
        if any(start <= match.start() < end for start, end, _ in protected):
            continue
        if _TABLE_LINE.match(value):
            table_group.append(match)
            continue
        flush_table()
        if _LIST_LINE.match(value):
            output.append((match.start(), match.start() + len(value), "list_item"))
        elif _KEY_VALUE.match(value) or (source == "fireflies" and ":" in value[:80]):
            output.append((match.start(), match.start() + len(value), "key_value"))
    flush_table()
    output.extend(protected)

    occupied = [(start, end) for start, end, _ in output]
    for segment_start, segment_end in _unoccupied_ranges(len(text), occupied):
        segment = text[segment_start:segment_end]
        cursor = 0
        for part in _split_with_offsets(segment, _SENTENCE_BOUNDARY):
            start, end = segment_start + part[0], segment_start + part[1]
            if not text[start:end].strip():
                continue
            if mode == "proposition" and end - start > 80:
                proposition_parts = _split_with_offsets(text[start:end], _PROPOSITION_BOUNDARY)
                if len(proposition_parts) > 1:
                    for p_start, p_end in proposition_parts:
                        if text[start + p_start:start + p_end].strip():
                            output.append((start + p_start, start + p_end, "proposition"))
                    continue
            output.append((start, end, "sentence"))
            cursor = end
        del cursor
    # Exact duplicates can arise when a key-value line is also a whole sentence.
    return sorted(dict.fromkeys(output), key=lambda value: (value[0], value[1], value[2]))


def _score_leaves(
    leaves: Sequence[LeafNode],
    parents: dict[str, ParentNode],
    queries: Sequence[tuple[str, str]],
    question: str,
    config: HierarchyConfig,
) -> list[ScoredLeaf]:
    if not leaves:
        return []
    query_tokens = [terms(value) or terms(question) for _, value in queries]
    body_sets = [terms(leaf.text) for leaf in leaves]
    search_sets = [terms(leaf.search_text) for leaf in leaves]
    frequencies = Counter(token for values in body_sets for token in values)
    weights = {
        token: 1.0 + math.log1p(len(leaves) / (1 + count)) for token, count in frequencies.items()
    }
    anchors = {value.casefold() for value in _VALUE.findall(question)}
    output: list[ScoredLeaf] = []
    for leaf, body, searchable in zip(leaves, body_sets, search_sets):
        parent = parents[leaf.parent_id]
        per_query = []
        contextual = []
        for tokens in query_tokens:
            denominator = sum(weights.get(token, 1.0) for token in tokens) or 1.0
            per_query.append(sum(weights.get(token, 1.0) for token in body & tokens) / denominator)
            contextual.append(sum(weights.get(token, 1.0) for token in searchable & tokens) / denominator)
        lexical = max(per_query, default=0.0)
        contextual_score = max(contextual, default=0.0) - lexical
        lower = leaf.text.casefold()
        exact = (sum(anchor in lower for anchor in anchors) / len(anchors)) if anchors else 0.0
        if _DURATION_QUERY.search(question) and _VALUE.search(leaf.text):
            exact += 0.15
        if _LIST_QUERY.search(question) and leaf.kind in {"list_item", "table_row", "key_value"}:
            exact += 0.10
        rank = max(1.0, _number(parent.context.get("document_rank"), 1.0))
        rank_prior = rank ** -config.rank_exponent
        evidence = _bounded(parent.context.get("evidence_score"))
        query_coverage = _bounded(parent.context.get("query_coverage"))
        score = (
            2.5 * lexical
            + config.contextual_match_weight * max(0.0, contextual_score)
            + 1.25 * exact
            + 0.50 * rank_prior
            + config.evidence_score_weight * evidence
            + config.query_coverage_weight * query_coverage
        )
        output.append(
            ScoredLeaf(
                leaf=leaf,
                score=score,
                lexical_score=lexical,
                contextual_score=max(0.0, contextual_score),
                exact_anchor_score=exact,
                rank_prior=rank_prior,
                evidence_prior=evidence,
                query_coverage_prior=query_coverage,
            )
        )
    return sorted(output, key=lambda value: (-value.score, value.leaf.ordinal))[: config.max_leaf_candidates]


def _select_leaves(
    candidates: Sequence[ScoredLeaf],
    queries: Sequence[tuple[str, str]],
    config: HierarchyConfig,
) -> list[ScoredLeaf]:
    del queries
    selected: list[ScoredLeaf] = []
    selected_ids: set[str] = set()
    per_doc: Counter[str] = Counter()
    observed_docs: set[str] = set()
    while len(selected) < config.max_selected_leaves:
        best: tuple[float, ScoredLeaf] | None = None
        for candidate in candidates:
            leaf = candidate.leaf
            if leaf.id in selected_ids:
                continue
            same_doc_penalty = 1.0 + config.diversity_penalty * per_doc[leaf.doc_id]
            global_bonus = (
                config.global_document_bonus if config.route_mode == "global" and leaf.doc_id not in observed_docs else 0.0
            )
            redundancy = max(
                (_jaccard(terms(leaf.text), terms(value.leaf.text)) for value in selected),
                default=0.0,
            )
            utility = (candidate.score + global_bonus - 0.35 * redundancy) / same_doc_penalty
            key = (utility, candidate)
            if best is None or key[0] > best[0] or (
                math.isclose(key[0], best[0]) and candidate.leaf.ordinal < best[1].leaf.ordinal
            ):
                best = key
        if best is None or (best[0] <= 0 and selected):
            break
        chosen = best[1]
        selected.append(chosen)
        selected_ids.add(chosen.leaf.id)
        per_doc[chosen.leaf.doc_id] += 1
        observed_docs.add(chosen.leaf.doc_id)
    return selected


def _expand_to_parent_ranges(
    selected: Sequence[ScoredLeaf],
    parents: Sequence[ParentNode],
    leaves: Sequence[LeafNode],
    config: HierarchyConfig,
) -> dict[str, tuple[int, int]]:
    parent_by_id = {parent.id: parent for parent in parents}
    leaves_by_parent: dict[str, list[LeafNode]] = defaultdict(list)
    selected_by_parent: dict[str, list[LeafNode]] = defaultdict(list)
    for leaf in leaves:
        leaves_by_parent[leaf.parent_id].append(leaf)
    for value in selected:
        selected_by_parent[value.leaf.parent_id].append(value.leaf)
    ranges: dict[str, tuple[int, int]] = {}
    for parent_id, selected_leaves in selected_by_parent.items():
        parent = parent_by_id[parent_id]
        if (
            len(parent.text) <= config.whole_parent_chars
            or parent.kind in {"code_block", "table", "meeting_next_steps", "meeting_decision", "email_message"}
            or len(selected_leaves) >= 2
        ):
            ranges[parent_id] = (parent.start_char, parent.end_char)
            continue
        ordered = sorted(leaves_by_parent[parent_id], key=lambda leaf: (leaf.start_char, leaf.end_char))
        indices = [ordered.index(leaf) for leaf in selected_leaves if leaf in ordered]
        if not indices:
            continue
        low = max(0, min(indices) - config.neighbor_units)
        high = min(len(ordered) - 1, max(indices) + config.neighbor_units)
        start = ordered[low].start_char
        end = ordered[high].end_char
        if end - start > config.parent_window_chars:
            strongest = min(selected_leaves, key=lambda leaf: leaf.ordinal)
            center = (strongest.start_char + strongest.end_char) // 2
            start = max(parent.start_char, center - config.parent_window_chars // 2)
            end = min(parent.end_char, start + config.parent_window_chars)
            start = max(parent.start_char, end - config.parent_window_chars)
        ranges[parent_id] = (start, end)
    return ranges


def _range_candidates(
    ranges: dict[str, tuple[int, int]],
    parents: dict[str, ParentNode],
    selected: Sequence[ScoredLeaf],
) -> list[dict[str, Any]]:
    by_parent: dict[str, list[ScoredLeaf]] = defaultdict(list)
    for value in selected:
        by_parent[value.leaf.parent_id].append(value)
    output = []
    for parent_id, (start, end) in ranges.items():
        parent = parents[parent_id]
        values = by_parent[parent_id]
        source_text = str(parent.context.get("text") or "")
        output.append(
            {
                "citation_id": parent.citation_id,
                "doc_id": parent.doc_id,
                "parent_id": parent.id,
                "parent_kind": parent.kind,
                "section_path": parent.section_path,
                "start_char": start,
                "end_char": end,
                "text": source_text[start:end],
                "contextual_prefix": parent.contextual_prefix,
                "leaf_ids": [value.leaf.id for value in values],
                "leaf_kinds": sorted({value.leaf.kind for value in values}),
                "score": max(value.score for value in values) + 0.10 * sum(value.score for value in values),
                "document_rank": int(_number(parent.context.get("document_rank"), 10**9)),
                "ordinal": min(value.leaf.ordinal for value in values),
                "context": parent.context,
            }
        )
    return output


def _canonical_documents(
    selected: Sequence[ScoredLeaf],
    parents: dict[str, ParentNode],
    queries: Sequence[tuple[str, str]],
    config: HierarchyConfig,
) -> list[dict[str, Any]]:
    by_doc: dict[str, list[ScoredLeaf]] = defaultdict(list)
    for value in selected:
        by_doc[value.leaf.doc_id].append(value)
    ranked = []
    for doc_id, values in by_doc.items():
        parent = parents[values[0].leaf.parent_id]
        top = sorted((value.score for value in values), reverse=True)
        score = top[0] + 0.25 * sum(top[1:3])
        ranked.append(
            {
                "doc_id": doc_id,
                "title": parent.title,
                "source_type": parent.source_type,
                "score": score,
                "best_document_rank": min(
                    int(_number(parents[value.leaf.parent_id].context.get("document_rank"), 10**9))
                    for value in values
                ),
                "citation_ids": sorted({value.leaf.citation_id for value in values}, key=_citation_sort),
                "leaf_ids": [value.leaf.id for value in values],
            }
        )
    ranked.sort(key=lambda value: (-value["score"], value["best_document_rank"], value["doc_id"]))
    if not ranked:
        return []
    best = ranked[0]["score"]
    for index, value in enumerate(ranked):
        value["role"] = "canonical" if index == 0 else (
            "near_tie" if best - value["score"] <= config.canonical_margin else "supplemental"
        )
        value["requirements"] = [requirement_id for requirement_id, _ in queries]
    return ranked


def _global_order(candidates: list[dict[str, Any]], canonical: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    role = {value["doc_id"]: value["role"] for value in canonical}
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for value in candidates:
        buckets[value["doc_id"]].append(value)
    for values in buckets.values():
        values.sort(key=lambda item: (-item["score"], item["ordinal"]))
    ordered = []
    while buckets:
        docs = sorted(
            buckets,
            key=lambda doc_id: (
                {"canonical": 0, "near_tie": 1, "supplemental": 2}.get(role.get(doc_id), 3),
                -buckets[doc_id][0]["score"],
                doc_id,
            ),
        )
        for doc_id in docs:
            ordered.append(buckets[doc_id].pop(0))
            if not buckets[doc_id]:
                del buckets[doc_id]
    return ordered


def _render_candidate(value: dict[str, Any]) -> str:
    context = value["context"]
    return (
        f"[{value['citation_id']}] title={context.get('title') or ''} "
        f"source_type={context.get('source_type') or 'unknown'} "
        f"doc_id={value['doc_id']} section={value['section_path']} "
        f"chunk_chars={value['start_char']}:{value['end_char']}/{len(str(context.get('text') or ''))}\n"
        f"{value['text']}"
    )


def _trace(value: dict[str, Any], status: str, reason: str) -> dict[str, Any]:
    return {
        "citation_id": value["citation_id"],
        "doc_id": value["doc_id"],
        "parent_id": value["parent_id"],
        "status": status,
        "reason": reason,
        "score": value["score"],
        "start_char": value["start_char"],
        "end_char": value["end_char"],
        "leaf_ids": value["leaf_ids"],
        "leaf_kinds": value["leaf_kinds"],
    }


def _unoccupied_ranges(length: int, occupied: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(occupied):
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    output = []
    cursor = 0
    for start, end in merged:
        if cursor < start:
            output.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < length:
        output.append((cursor, length))
    return output


def _split_with_offsets(text: str, boundary: re.Pattern[str]) -> list[tuple[int, int]]:
    output = []
    start = 0
    for match in boundary.finditer(text):
        end = match.start()
        if text[start:end].strip():
            output.append((start, end))
        start = match.end()
    if text[start:].strip():
        output.append((start, len(text)))
    return output or ([(0, len(text))] if text.strip() else [])


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 0.0
    return len(left & right) / len(left | right)


def _node_id(citation: str, kind: str, *parts: int) -> str:
    suffix = hashlib.sha256((citation + ":" + kind + ":" + ":".join(map(str, parts))).encode()).hexdigest()[:12]
    return f"{citation}:{kind}:{suffix}"


def _compact(value: Any, maximum: int) -> str:
    normalized = re.sub(r"\s+", " ", str(value or "")).strip()
    return normalized[:maximum]


def _number(value: Any, default: float) -> float:
    if isinstance(value, bool):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _bounded(value: Any) -> float:
    return max(0.0, min(1.0, _number(value, 0.0)))


def _citation_sort(value: str) -> tuple[int, str]:
    return (int(value[1:]), value) if value.startswith("S") and value[1:].isdigit() else (10**9, value)
