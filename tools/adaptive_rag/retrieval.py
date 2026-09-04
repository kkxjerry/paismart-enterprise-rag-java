from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any

from .features import RouterDecision
from .requirements import RequirementPlan


@dataclass(frozen=True)
class SearchPrincipal:
    tenant_id: str
    group_ids: tuple[str, ...]
    source_types: tuple[str, ...] = tuple()
    classifications: tuple[str, ...] = tuple()

    def __post_init__(self) -> None:
        tenant = str(self.tenant_id or "").strip()
        if not tenant:
            raise ValueError("tenant_id is required")
        object.__setattr__(self, "tenant_id", tenant)
        object.__setattr__(self, "group_ids", _normalized_tuple(self.group_ids))
        object.__setattr__(self, "source_types", _normalized_tuple(self.source_types))
        object.__setattr__(self, "classifications", _normalized_tuple(self.classifications))
        if not self.classifications:
            raise ValueError("at least one authorized classification is required")

    def to_dict(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "group_ids": list(self.group_ids),
            "source_types": list(self.source_types),
            "classifications": list(self.classifications),
        }


@dataclass(frozen=True)
class SecondaryRetrievalResult:
    attempted: bool
    queries: tuple[str, ...]
    added_contexts: int
    merged_contexts: tuple[dict[str, Any], ...]
    responses: tuple[dict[str, Any], ...]
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempted": self.attempted,
            "queries": list(self.queries),
            "added_contexts": self.added_contexts,
            "responses": list(self.responses),
            "error": self.error,
        }


class SecondaryRetrievalClient:
    def __init__(
        self,
        *,
        api_url: str,
        api_key: str = "",
        timeout_seconds: float = 60.0,
        max_contexts_per_query: int = 12,
    ) -> None:
        normalized_url = api_url.rstrip("/")
        self.api_url = normalized_url if normalized_url.endswith("/v1/search") else normalized_url + "/v1/search"
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.max_contexts_per_query = max_contexts_per_query

    def search(self, query: str, principal: SearchPrincipal) -> dict[str, Any]:
        payload = {
            "query": query,
            "max_contexts": self.max_contexts_per_query,
        }
        headers = {
            "Content-Type": "application/json",
            "X-RAG-Tenant-Id": principal.tenant_id,
            "X-RAG-Group-Ids": ",".join(principal.group_ids),
            "X-RAG-Classifications": ",".join(principal.classifications),
            "X-RAG-Source-Types": ",".join(principal.source_types),
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self.api_url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                result = json.load(response)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:2_000]
            raise RuntimeError(f"secondary retrieval HTTP {exc.code}: {body}") from exc
        if not isinstance(result, dict):
            raise RuntimeError("secondary retrieval response must be an object")
        return result

    def retrieve_for_plan(
        self,
        *,
        row: dict[str, Any],
        route: RouterDecision,
        plan: RequirementPlan,
        principal: SearchPrincipal,
        max_queries: int = 4,
    ) -> SecondaryRetrievalResult:
        queries = secondary_queries(str(row.get("question") or ""), route, plan, max_queries=max_queries)
        original = list(row.get("contexts") or [])
        if not queries:
            return SecondaryRetrievalResult(False, tuple(), 0, tuple(original), tuple())
        responses: list[dict[str, Any]] = []
        try:
            for query in queries:
                responses.append(self.search(query, principal))
            merged = merge_contexts(original, responses, queries)
            return SecondaryRetrievalResult(
                attempted=True,
                queries=tuple(queries),
                added_contexts=max(0, len(merged) - len(original)),
                merged_contexts=tuple(merged),
                responses=tuple(_response_audit(response) for response in responses),
            )
        except Exception as exc:
            return SecondaryRetrievalResult(
                attempted=True,
                queries=tuple(queries),
                added_contexts=0,
                merged_contexts=tuple(original),
                responses=tuple(_response_audit(response) for response in responses),
                error=str(exc),
            )


def secondary_queries(
    original_query: str,
    route: RouterDecision,
    plan: RequirementPlan,
    *,
    max_queries: int,
) -> list[str]:
    queries: list[str] = []
    for requirement in plan.requirements:
        if requirement.status == "missing" or route.mode == "deep":
            query = requirement.search_query.strip() or requirement.requirement.strip()
            if query and query.casefold() != original_query.strip().casefold() and query not in queries:
                queries.append(query)
        if len(queries) >= max_queries:
            break
    if route.mode == "deep" and not queries:
        for requirement in plan.requirements:
            query = requirement.search_query.strip() or requirement.requirement.strip()
            if query and query.casefold() != original_query.strip().casefold() and query not in queries:
                queries.append(query)
            if len(queries) >= max_queries:
                break
    return queries


def merge_contexts(
    original: list[dict[str, Any]],
    responses: list[dict[str, Any]],
    queries: list[str],
) -> list[dict[str, Any]]:
    output = [dict(context) for context in original]
    seen = {_context_identity(context) for context in output}
    next_citation = 1 + max(
        [int(str(context.get("citation_id"))[1:]) for context in output
         if str(context.get("citation_id") or "").startswith("S")
         and str(context.get("citation_id"))[1:].isdigit()]
        or [0]
    )
    for response_index, response in enumerate(responses):
        query = queries[response_index] if response_index < len(queries) else ""
        contexts = response.get("contexts") or response.get("evidence") or []
        if isinstance(contexts, dict):
            contexts = contexts.get("spans") or []
        if not isinstance(contexts, list):
            continue
        for raw in contexts:
            if not isinstance(raw, dict):
                continue
            context = dict(raw)
            identity = _context_identity(context)
            if identity in seen:
                continue
            context["citation_id"] = f"S{next_citation}"
            next_citation += 1
            context["secondary_retrieval"] = True
            context["secondary_query"] = query
            output.append(context)
            seen.add(identity)
    return output


def _context_identity(context: dict[str, Any]) -> tuple[str, str, str]:
    chunk = str(context.get("chunk_es_id") or context.get("content_hash") or "")
    doc = str(context.get("doc_id") or "")
    text = str(context.get("text") or "")
    return chunk, doc, text[:160]


def _normalized_tuple(values: Any) -> tuple[str, ...]:
    if values is None:
        return tuple()
    if isinstance(values, str):
        raise ValueError("principal list fields must be arrays, not strings")
    try:
        normalized = {str(value).strip() for value in values if str(value).strip()}
    except TypeError as exc:
        raise ValueError("principal list fields must be arrays") from exc
    return tuple(sorted(normalized))


def _response_audit(response: dict[str, Any]) -> dict[str, Any]:
    return {
        "trace_id": response.get("trace_id"),
        "query": response.get("query"),
        "route_features": response.get("route_features"),
        "context_count": len(response.get("contexts") or []),
        "latency_ms": response.get("latency_ms"),
    }
