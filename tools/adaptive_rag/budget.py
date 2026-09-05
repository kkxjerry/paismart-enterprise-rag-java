from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from tools.qwen_plus_rag_pipeline import render_contexts

from .features import RouterDecision
from .requirements import RequirementPlan


@dataclass(frozen=True)
class BudgetDecision:
    mode: str
    initial_chars: int
    maximum_chars: int
    max_contexts: int
    max_contexts_per_document: int
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["reasons"] = list(self.reasons)
        return value


@dataclass(frozen=True)
class BudgetedEvidence:
    rendered: str
    contexts: tuple[dict[str, Any], ...]
    rendered_chars: int
    budget_chars: int
    expanded_to_maximum: bool
    selected_citations_present: tuple[str, ...]
    selected_citations_missing: tuple[str, ...]
    selected_citations_truncated: tuple[str, ...] = tuple()
    selection_strategy: str = "legacy"
    selection_trace: tuple[dict[str, Any], ...] = tuple()

    def to_dict(self) -> dict[str, Any]:
        return {
            "rendered_chars": self.rendered_chars,
            "budget_chars": self.budget_chars,
            "expanded_to_maximum": self.expanded_to_maximum,
            "context_count": len(self.contexts),
            "selected_citations_present": list(self.selected_citations_present),
            "selected_citations_missing": list(self.selected_citations_missing),
            "selected_citations_truncated": list(self.selected_citations_truncated),
            "selection_strategy": self.selection_strategy,
            "selection_trace": list(self.selection_trace),
            "per_document_limit_applied": self.selection_strategy == "legacy",
        }


class DynamicEvidenceBudget:
    """Allocate evidence by route/risk instead of one global prompt size."""

    def __init__(self, strategy: str = "legacy") -> None:
        if strategy not in {"legacy", "query-spans"}:
            raise ValueError(f"unsupported evidence strategy: {strategy}")
        self.strategy = strategy

    def decide(self, route: RouterDecision, plan: RequirementPlan) -> BudgetDecision:
        reasons: list[str] = [f"router_mode={route.mode}"]
        if route.mode == "fast":
            initial, maximum, max_contexts, per_doc = 10_000, 12_000, 12, 2
        elif route.mode == "quality":
            initial, maximum, max_contexts, per_doc = 11_000, 16_000, 18, 3
        elif route.mode == "deep":
            initial, maximum, max_contexts, per_doc = 16_000, 24_000, 30, 4
        else:
            raise ValueError(f"unsupported route mode: {route.mode}")

        extra_requirements = max(0, len(plan.requirements) - 1)
        if extra_requirements:
            initial += min(4_000, extra_requirements * 1_000)
            reasons.append(f"requirements={len(plan.requirements)}")
        if plan.conflicting_count:
            maximum = max(maximum, 24_000)
            max_contexts = max(max_contexts, 24)
            per_doc = max(per_doc, 4)
            reasons.append("preserve_conflicting_sources")
        if plan.missing_count:
            maximum = max(maximum, 20_000 if route.mode != "deep" else 28_000)
            reasons.append("missing_requirements_allow_expansion")
        # A selected citation is a contract with the generator. Reserve enough
        # room for every selected evidence block before adding fallback chunks;
        # otherwise the plan can mention S27 while the generation window ends at
        # S18. Source chunks are ~1.2K characters plus metadata, so 1.5K is a
        # conservative per-citation envelope. The 32K cap keeps rare exhaustive
        # questions bounded.
        if plan.selected_citations:
            selected_capacity = min(32_000, len(plan.selected_citations) * 1_500)
            if selected_capacity > maximum:
                maximum = selected_capacity
                reasons.append(f"selected_evidence={len(plan.selected_citations)}")
            max_contexts = max(max_contexts, len(plan.selected_citations))
        if route.features.exact_anchor_count >= 2:
            initial += 1_000
            reasons.append("protect_exact_constraints")
        initial = min(initial, maximum)
        return BudgetDecision(
            mode=route.mode,
            initial_chars=initial,
            maximum_chars=maximum,
            max_contexts=max_contexts,
            max_contexts_per_document=per_doc,
            reasons=tuple(reasons),
        )

    def build(
        self,
        contexts: list[dict[str, Any]],
        *,
        plan: RequirementPlan,
        decision: BudgetDecision,
        question: str = "",
    ) -> BudgetedEvidence:
        if self.strategy == "query-spans":
            return self._build_query_spans(contexts, plan=plan, decision=decision, question=question)
        ordered = prioritize_contexts(
            contexts,
            selected_citations=list(plan.selected_citations),
            conflict_citations=list(plan.conflict_citations),
            max_per_document=decision.max_contexts_per_document,
        )
        rendered, included, rendered_chars = render_contexts(
            ordered,
            max_contexts=decision.max_contexts,
            max_chars=decision.initial_chars,
        )
        present = {
            str(context.get("citation_id") or "")
            for context in included
            if context.get("citation_id")
        }
        required = set(plan.selected_citations)
        missing = required - present
        original_text = {str(c.get("citation_id") or ""): str(c.get("text") or "") for c in ordered}
        def truncated_required(values: list[dict[str, Any]]) -> set[str]:
            return {str(c.get("citation_id") or "") for c in values
                    if str(c.get("citation_id") or "") in required
                    and str(c.get("text") or "") != original_text[str(c.get("citation_id") or "")]}
        truncated = truncated_required(included)
        expand = (bool(missing or truncated) or plan.missing_count > 0) and decision.maximum_chars > decision.initial_chars
        budget = decision.initial_chars
        if expand:
            budget = decision.maximum_chars
            rendered, included, rendered_chars = render_contexts(
                ordered,
                max_contexts=decision.max_contexts,
                max_chars=budget,
            )
            present = {
                str(context.get("citation_id") or "")
                for context in included
                if context.get("citation_id")
            }
            missing = required - present
            truncated = truncated_required(included)
        return BudgetedEvidence(
            rendered=rendered,
            contexts=tuple(included),
            rendered_chars=rendered_chars,
            budget_chars=budget,
            expanded_to_maximum=expand,
            selected_citations_present=tuple(sorted(required & present, key=citation_sort_key)),
            selected_citations_missing=tuple(sorted(missing, key=citation_sort_key)),
            selected_citations_truncated=tuple(sorted(truncated, key=citation_sort_key)),
        )

    @staticmethod
    def _build_query_spans(
        contexts: list[dict[str, Any]],
        *,
        plan: RequirementPlan,
        decision: BudgetDecision,
        question: str,
    ) -> BudgetedEvidence:
        from .evidence_spans import pack_evidence

        required = tuple(dict.fromkeys(plan.conflict_citations + plan.selected_citations))
        requirements = tuple((item.id, item.requirement) for item in plan.requirements)

        def pack(limit: int):
            return pack_evidence(
                contexts, question=question, requirements=requirements,
                required_citations=required, max_chars=limit,
                max_contexts=decision.max_contexts,
            )

        budget = decision.initial_chars
        packed = pack(budget)
        present = {str(item.get("citation_id") or "") for item in packed.contexts}
        missing = set(required) - present
        expand = bool(missing or plan.missing_count) and decision.maximum_chars > budget
        if expand:
            budget = decision.maximum_chars
            packed = pack(budget)
            present = {str(item.get("citation_id") or "") for item in packed.contexts}
            missing = set(required) - present
        return BudgetedEvidence(
            rendered=packed.rendered, contexts=packed.contexts,
            rendered_chars=len(packed.rendered), budget_chars=budget,
            expanded_to_maximum=expand,
            selected_citations_present=tuple(sorted(set(required) & present, key=citation_sort_key)),
            selected_citations_missing=tuple(sorted(missing, key=citation_sort_key)),
            selected_citations_truncated=(), selection_strategy="query-spans",
            selection_trace=packed.trace,
        )


def prioritize_contexts(
    contexts: list[dict[str, Any]],
    *,
    selected_citations: list[str],
    conflict_citations: list[str],
    max_per_document: int,
) -> list[dict[str, Any]]:
    by_citation = {
        str(context.get("citation_id") or ""): context
        for context in contexts
        if context.get("citation_id")
    }
    priority_ids: list[str] = []
    for citation in conflict_citations + selected_citations:
        if citation in by_citation and citation not in priority_ids:
            priority_ids.append(citation)

    ranked_rest = sorted(
        [context for context in contexts if str(context.get("citation_id") or "") not in priority_ids],
        key=context_sort_key,
    )
    candidates = [by_citation[citation] for citation in priority_ids] + ranked_rest

    output: list[dict[str, Any]] = []
    seen_citations: set[str] = set()
    per_document: dict[str, int] = {}
    selected_set = set(priority_ids)
    for context in candidates:
        citation = str(context.get("citation_id") or "")
        if not citation or citation in seen_citations:
            continue
        doc_id = str(context.get("doc_id") or citation)
        count = per_document.get(doc_id, 0)
        # Selected/conflict evidence is never removed by the fairness cap. The cap
        # only prevents fallback chunks from one long document crowding out all
        # other sources.
        if citation not in selected_set and count >= max_per_document:
            continue
        output.append(context)
        seen_citations.add(citation)
        per_document[doc_id] = count + 1
    return output


def context_sort_key(context: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(context.get("document_rank") or 10**9),
        -float(context.get("evidence_score") or 0.0),
        int(context.get("rank") or 10**9),
        citation_sort_key(str(context.get("citation_id") or "")),
    )


def citation_sort_key(citation: str) -> tuple[int, str]:
    if citation.startswith("S") and citation[1:].isdigit():
        return int(citation[1:]), citation
    return 10**9, citation
