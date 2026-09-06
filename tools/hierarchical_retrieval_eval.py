#!/usr/bin/env python3
"""Evaluate an isolated hierarchical leaf index and parent expansion.

The evaluator queries leaf vectors and BM25 searchText, fuses leaf ranks, then
fetches contiguous parent nodes. Gold document IDs and answer facts are used only
after retrieval. It does not update aliases or online services.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
import urllib.parse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from tools.hierarchical_index_build import embed, request_json, sha256_file
from tools.qwen_plus_rag_pipeline import fact_scores

EVALUATOR_VERSION = "hierarchical-retrieval-eval-v1"


def load_questions(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("["):
        value = json.loads(text)
        if not isinstance(value, list):
            raise ValueError("questions JSON must contain an array")
        rows = value
    else:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("every question must be an object")
    return [dict(row) for row in rows]


def query_text(question: str, instruction: str) -> str:
    return f"Instruct: {instruction}\nQuery: {question}" if instruction.strip() else question


def search_filter(row: dict[str, Any], source_types: Sequence[str]) -> list[dict[str, Any]]:
    filters: list[dict[str, Any]] = [{"term": {"nodeType": "leaf"}}]
    tenant = str(row.get("tenant_id") or "").strip()
    if tenant:
        filters.append({"term": {"tenantId": tenant}})
    allowed_sources = sorted({str(value).casefold() for value in source_types if str(value)})
    if allowed_sources:
        filters.append({"terms": {"sourceType": allowed_sources}})
    return filters


def dense_search(
    *,
    es_url: str,
    index: str,
    vector: Sequence[float],
    filters: Sequence[dict[str, Any]],
    k: int,
    num_candidates: int,
    timeout: float,
) -> list[dict[str, Any]]:
    payload = {
        "size": k,
        "_source": ["nodeId", "parentNodeId", "docId", "sourceType", "title", "sectionPath", "rawText"],
        "knn": {
            "field": "vector",
            "query_vector": list(vector),
            "k": k,
            "num_candidates": max(k, num_candidates),
            "filter": {"bool": {"filter": list(filters)}},
        },
    }
    return hits(request_json("POST", f"{es_url.rstrip('/')}/{quote(index)}/_search", payload=payload, timeout=timeout))


def bm25_search(
    *,
    es_url: str,
    index: str,
    question: str,
    filters: Sequence[dict[str, Any]],
    k: int,
    timeout: float,
) -> list[dict[str, Any]]:
    payload = {
        "size": k,
        "_source": ["nodeId", "parentNodeId", "docId", "sourceType", "title", "sectionPath", "rawText"],
        "query": {
            "bool": {
                "filter": list(filters),
                "must": [
                    {
                        "multi_match": {
                            "query": question,
                            "fields": ["searchText", "title^1.5", "sectionPath^1.25"],
                            "type": "best_fields",
                            "operator": "or",
                        }
                    }
                ],
            }
        },
    }
    return hits(request_json("POST", f"{es_url.rstrip('/')}/{quote(index)}/_search", payload=payload, timeout=timeout))


def hits(payload: Any) -> list[dict[str, Any]]:
    raw = payload.get("hits", {}).get("hits", []) if isinstance(payload, dict) else []
    output = []
    for rank, item in enumerate(raw, start=1):
        if not isinstance(item, dict) or not isinstance(item.get("_source"), dict):
            continue
        output.append(
            {
                "id": str(item.get("_id") or ""),
                "score": float(item.get("_score") or 0.0),
                "rank": rank,
                **item["_source"],
            }
        )
    return output


def weighted_rrf(routes: Sequence[tuple[str, Sequence[dict[str, Any]], float]], *, rrf_k: int) -> list[dict[str, Any]]:
    fused: dict[str, dict[str, Any]] = {}
    for route, values, weight in routes:
        for rank, value in enumerate(values, start=1):
            node_id = str(value.get("nodeId") or value.get("id") or "")
            if not node_id:
                continue
            current = fused.setdefault(node_id, {**value, "rrf": 0.0, "route_ranks": {}})
            current["rrf"] += weight / (rrf_k + rank)
            current["route_ranks"][route] = rank
    return sorted(fused.values(), key=lambda value: (-value["rrf"], str(value.get("nodeId") or "")))


def collapse_documents(leaves: Sequence[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    by_doc: dict[str, dict[str, Any]] = {}
    for leaf in leaves:
        doc_id = str(leaf.get("docId") or "")
        if not doc_id:
            continue
        current = by_doc.get(doc_id)
        if current is None:
            by_doc[doc_id] = {
                "doc_id": doc_id,
                "score": float(leaf.get("rrf") or 0.0),
                "best_leaf": leaf,
                "parent_ids": [str(leaf.get("parentNodeId") or "")],
                "leaf_ids": [str(leaf.get("nodeId") or leaf.get("id") or "")],
            }
        else:
            current["score"] += 0.20 * float(leaf.get("rrf") or 0.0)
            parent = str(leaf.get("parentNodeId") or "")
            node = str(leaf.get("nodeId") or leaf.get("id") or "")
            if parent and parent not in current["parent_ids"]:
                current["parent_ids"].append(parent)
            if node and node not in current["leaf_ids"]:
                current["leaf_ids"].append(node)
    return sorted(by_doc.values(), key=lambda value: (-value["score"], value["doc_id"]))[:limit]


def fetch_parents(es_url: str, index: str, parent_ids: Sequence[str], timeout: float) -> dict[str, dict[str, Any]]:
    ids = list(dict.fromkeys(value for value in parent_ids if value))
    if not ids:
        return {}
    payload = request_json(
        "POST",
        f"{es_url.rstrip('/')}/{quote(index)}/_mget",
        payload={"ids": ids, "_source": ["nodeId", "docId", "sourceType", "title", "sectionPath", "rawText"]},
        timeout=timeout,
    )
    output = {}
    for item in payload.get("docs", []) if isinstance(payload, dict) else []:
        if isinstance(item, dict) and item.get("found") and isinstance(item.get("_source"), dict):
            output[str(item.get("_id") or "")] = item["_source"]
    return output


def evaluate_row(args: argparse.Namespace, row: dict[str, Any], vector: Sequence[float]) -> dict[str, Any]:
    question = str(row.get("question") or "")
    filters = search_filter(row, args.source_type)
    dense = dense_search(
        es_url=args.es_url,
        index=args.index,
        vector=vector,
        filters=filters,
        k=args.route_k,
        num_candidates=args.num_candidates,
        timeout=args.timeout_seconds,
    )
    bm25 = bm25_search(
        es_url=args.es_url,
        index=args.index,
        question=question,
        filters=filters,
        k=args.route_k,
        timeout=args.timeout_seconds,
    )
    fused = weighted_rrf(
        [("dense", dense, args.dense_weight), ("bm25", bm25, args.bm25_weight)],
        rrf_k=args.rrf_k,
    )
    documents = collapse_documents(fused, args.top_documents)
    parent_ids = [parent for document in documents for parent in document["parent_ids"][: args.parents_per_document]]
    parent_map = fetch_parents(args.es_url, args.index, parent_ids, args.timeout_seconds)
    contexts = []
    for document in documents:
        for parent_id in document["parent_ids"][: args.parents_per_document]:
            source = parent_map.get(parent_id)
            if source:
                contexts.append({"parent_id": parent_id, **source})
    ranked_docs = [value["doc_id"] for value in documents]
    expected = [str(value) for value in row.get("expected_doc_ids") or [] if str(value)]
    accessible = [str(value) for value in row.get("expected_accessible_doc_ids") or expected if str(value)]
    target = accessible or expected
    ranks = [ranked_docs.index(doc_id) + 1 for doc_id in target if doc_id in ranked_docs]
    facts = [str(value) for value in row.get("answer_facts") or [] if str(value).strip()]
    parent_text = "\n".join(str(value.get("rawText") or "") for value in contexts)
    fact_recall, fact_coverage = fact_scores(parent_text, facts)
    return {
        "qid": str(row.get("qid") or row.get("id") or ""),
        "question": question,
        "question_type": row.get("question_type"),
        "source_types": row.get("source_types") or [],
        "expected_doc_ids": expected,
        "expected_accessible_doc_ids": accessible,
        "ranked_doc_ids": ranked_docs,
        "hit_at_1": bool(ranks and min(ranks) <= 1) if target else None,
        "hit_at_5": bool(ranks and min(ranks) <= 5) if target else None,
        "hit_at_10": bool(ranks and min(ranks) <= 10) if target else None,
        "all_expected_at_10": all(doc_id in ranked_docs[:10] for doc_id in target) if target else None,
        "reciprocal_rank": 1.0 / min(ranks) if ranks else (0.0 if target else None),
        "parent_fact_token_recall": fact_recall if facts else None,
        "parent_fact_coverage": fact_coverage if facts else None,
        "dense_leaf_hits": len(dense),
        "bm25_leaf_hits": len(bm25),
        "fused_leaf_hits": len(fused),
        "parent_contexts": contexts,
        "filters": filters,
    }


def summarize(rows: Sequence[dict[str, Any]], args: argparse.Namespace, wall_seconds: float) -> dict[str, Any]:
    eligible = [row for row in rows if row.get("hit_at_10") is not None]
    facts = [row for row in rows if row.get("parent_fact_token_recall") is not None]
    def average(name: str, selected: Sequence[dict[str, Any]]) -> float | None:
        values = [float(row[name]) for row in selected if isinstance(row.get(name), (int, float, bool))]
        return statistics.fmean(values) if values else None
    return {
        "schema_version": 1,
        "evaluator_version": EVALUATOR_VERSION,
        "index": args.index,
        "questions": len(rows),
        "retrieval_evaluable": len(eligible),
        "fact_evaluable": len(facts),
        "hit_at_1": average("hit_at_1", eligible),
        "hit_at_5": average("hit_at_5", eligible),
        "hit_at_10": average("hit_at_10", eligible),
        "all_expected_at_10": average("all_expected_at_10", eligible),
        "mrr": average("reciprocal_rank", eligible),
        "parent_fact_token_recall": average("parent_fact_token_recall", facts),
        "parent_fact_coverage": average("parent_fact_coverage", facts),
        "miss_qids": [row["qid"] for row in eligible if not row["hit_at_10"]],
        "errors": 0,
        "wall_seconds": wall_seconds,
        "configuration": {
            "source_types": list(args.source_type),
            "route_k": args.route_k,
            "num_candidates": args.num_candidates,
            "top_documents": args.top_documents,
            "parents_per_document": args.parents_per_document,
            "rrf_k": args.rrf_k,
            "dense_weight": args.dense_weight,
            "bm25_weight": args.bm25_weight,
            "embedding_model": args.embedding_model,
            "embedding_dimension": args.embedding_dimension,
            "query_instruction": args.query_instruction,
            "acl_scope": "tenant and source filters only; user-to-group expansion unavailable in question rows",
        },
    }


def select_questions(rows: Sequence[dict[str, Any]], source_types: Sequence[str], limit: int | None) -> list[dict[str, Any]]:
    sources = {value.casefold() for value in source_types}
    selected = []
    for row in rows:
        row_sources = {str(value).casefold() for value in row.get("source_types") or [] if str(value)}
        # A source-specific index can only be judged on questions whose complete
        # declared source set exists in that index. Intersection-based selection
        # would silently score cross-source questions against a partial corpus.
        if not sources or (row_sources and row_sources <= sources):
            selected.append(row)
    return selected[:limit] if limit else selected


def quote(value: str) -> str:
    return urllib.parse.quote(value, safe="-_.*")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--es-url", default="http://127.0.0.1:19200")
    parser.add_argument("--index", required=True)
    parser.add_argument("--embedding-url", default="http://127.0.0.1:18084/v1/embeddings")
    parser.add_argument("--embedding-model", default="Qwen/Qwen3-Embedding-4B")
    parser.add_argument("--embedding-dimension", type=int, default=2048)
    parser.add_argument("--query-instruction", default="Given an enterprise search query, retrieve relevant passages that answer the query")
    parser.add_argument("--source-type", action="append", default=[])
    parser.add_argument("--route-k", type=int, default=100)
    parser.add_argument("--num-candidates", type=int, default=500)
    parser.add_argument("--top-documents", type=int, default=10)
    parser.add_argument("--parents-per-document", type=int, default=2)
    parser.add_argument("--rrf-k", type=int, default=10)
    parser.add_argument("--dense-weight", type=float, default=0.75)
    parser.add_argument("--bm25-weight", type=float, default=1.0)
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--details", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.questions.is_file():
        parser.error(f"questions are missing: {args.questions}")
    for name in ("embedding_dimension", "route_k", "num_candidates", "top_documents", "parents_per_document", "rrf_k", "embedding_batch_size"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    questions = select_questions(load_questions(args.questions), args.source_type, args.limit)
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    args.details.parent.mkdir(parents=True, exist_ok=True)
    with args.details.open("x", encoding="utf-8") as output:
        for offset in range(0, len(questions), args.embedding_batch_size):
            batch = questions[offset:offset + args.embedding_batch_size]
            vectors = embed(
                [query_text(str(row.get("question") or ""), args.query_instruction) for row in batch],
                endpoint=args.embedding_url,
                model=args.embedding_model,
                dimension=args.embedding_dimension,
                timeout=args.timeout_seconds,
            )
            for row, vector in zip(batch, vectors):
                result = evaluate_row(args, row, vector)
                results.append(result)
                output.write(json.dumps(result, ensure_ascii=False) + "\n")
                output.flush()
            print(f"EVALUATED {min(offset + len(batch), len(questions))}/{len(questions)}", flush=True)
    summary = summarize(results, args, time.perf_counter() - started)
    summary["questions_sha256"] = sha256_file(args.questions)
    summary["details_sha256"] = sha256_file(args.details)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
