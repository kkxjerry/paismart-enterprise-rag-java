#!/usr/bin/env python3
"""Evaluate E4 context prefixes and E5 proposition keys with the real embedder.

The comparison is local to each row's authorized Java Evidence pool. It does not
claim full-index retrieval quality. Gold document IDs are used only after vectors
are produced to score ranking. Vector values are never written to disk.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

from tools.adaptive_rag.hierarchical import (
    HierarchyConfig,
    build_leaves,
    contextual_prefix,
)
from tools.qwen_plus_rag_pipeline import load_jsonl

QUERY_INSTRUCTION = "Given an enterprise search query, retrieve relevant passages that answer the query"


def embed(endpoint: str, model: str, inputs: list[str], timeout: float) -> tuple[list[list[float]], dict[str, int], float]:
    request = urllib.request.Request(
        endpoint,
        data=json.dumps({"model": model, "input": inputs}, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    latency = (time.perf_counter() - started) * 1000.0
    rows = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or len(rows) != len(inputs):
        raise ValueError("embedding response count does not match inputs")
    rows = sorted(rows, key=lambda value: int(value.get("index", 0)))
    vectors = [value.get("embedding") for value in rows]
    if not all(isinstance(value, list) and value for value in vectors):
        raise ValueError("embedding response lacks vectors")
    dimensions = {len(value) for value in vectors}
    if len(dimensions) != 1:
        raise ValueError("embedding dimensions are inconsistent")
    usage = payload.get("usage") or {}
    return vectors, {
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
    }, latency


def cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


def rank_docs(query: list[float], entries: list[tuple[str, list[float]]]) -> list[str]:
    scores: dict[str, float] = {}
    for doc_id, vector in entries:
        scores[doc_id] = max(scores.get(doc_id, -math.inf), cosine(query, vector))
    return [doc for doc, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))]


def reciprocal_rank(ranking: list[str], expected: set[str]) -> float:
    for index, doc_id in enumerate(ranking, start=1):
        if doc_id in expected:
            return 1.0 / index
    return 0.0


def summarize(records: list[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any]:
    arms = ("plain_chunk", "contextual_chunk", "proposition_leaf")
    aggregates = {}
    for arm in arms:
        values = [record for record in records if not record.get("error") and record.get("gold_available")]
        aggregates[arm] = {
            "questions": len(values),
            "hit_at_1": statistics.fmean(record[arm]["hit_at_1"] for record in values) if values else None,
            "hit_at_3": statistics.fmean(record[arm]["hit_at_3"] for record in values) if values else None,
            "mrr": statistics.fmean(record[arm]["reciprocal_rank"] for record in values) if values else None,
            "mean_gold_rank": statistics.fmean(record[arm]["gold_rank"] for record in values if record[arm]["gold_rank"] is not None) if values else None,
        }
    return {
        "metadata": metadata,
        "rows": len(records),
        "errors": sum(bool(record.get("error")) for record in records),
        "gold_available_rows": sum(bool(record.get("gold_available")) for record in records),
        "aggregates": aggregates,
        "usage": {
            key: sum(int((record.get("usage") or {}).get(key) or 0) for record in records)
            for key in ("prompt_tokens", "total_tokens")
        },
        "mean_api_latency_ms": statistics.fmean(record["latency_ms"] for record in records if not record.get("error")) if records else None,
        "failed_qids": [record["qid"] for record in records if record.get("error")],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-4B")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--max-proposition-leaves-per-context", type=int, default=4)
    args = parser.parse_args()
    if args.limit <= 0 or args.timeout <= 0 or args.max_proposition_leaves_per_context <= 0:
        parser.error("limit, timeout and leaf cap must be positive")
    rows = load_jsonl(args.contexts)[:args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        "input": str(args.contexts),
        "endpoint": args.endpoint,
        "model": args.model,
        "limit": args.limit,
        "query_instruction": QUERY_INSTRUCTION,
        "scope": "reranking within each existing authorized Java Evidence candidate pool",
        "vectors_logged": False,
        "label_boundary": "expected_doc_ids are read only after ranking",
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    records = []
    with (args.output_dir / "records.jsonl").open("x", encoding="utf-8") as output:
        for index, row in enumerate(rows, start=1):
            qid = str(row.get("qid") or row.get("id"))
            contexts = [dict(value) for value in row.get("contexts") or [] if value.get("doc_id")]
            record: dict[str, Any] = {"qid": qid, "error": None}
            try:
                query = QUERY_INSTRUCTION + "\n" + str(row.get("question") or "")
                plain: list[tuple[str, str]] = []
                contextual: list[tuple[str, str]] = []
                propositions: list[tuple[str, str]] = []
                proposition_config = HierarchyConfig(strategy="proposition-parent")
                for context in contexts:
                    doc_id = str(context.get("doc_id"))
                    title = str(context.get("title") or "")
                    text = str(context.get("text") or "")
                    plain.append((doc_id, title + "\n" + text))
                    prefix = contextual_prefix(context) or title
                    contextual.append((doc_id, prefix + "\n" + text))
                    leaves = build_leaves([context], proposition_config)
                    ranked_leaves = sorted(
                        leaves,
                        key=lambda value: (
                            -len(set(str(row.get("question") or "").casefold().split())
                                 & set(value.proposition_key.casefold().split())),
                            value.ordinal,
                        ),
                    )[: args.max_proposition_leaves_per_context]
                    propositions.extend((doc_id, value.proposition_key) for value in ranked_leaves)
                texts = [query] + [text for _, text in plain] + [text for _, text in contextual] + [text for _, text in propositions]
                vectors, usage, latency = embed(args.endpoint, args.model, texts, args.timeout)
                offset = 1
                plain_vectors = list(zip((doc for doc, _ in plain), vectors[offset:offset + len(plain)]))
                offset += len(plain)
                contextual_vectors = list(zip((doc for doc, _ in contextual), vectors[offset:offset + len(contextual)]))
                offset += len(contextual)
                proposition_vectors = list(zip((doc for doc, _ in propositions), vectors[offset:]))
                expected = {str(value) for value in row.get("expected_doc_ids") or [] if str(value)}
                available = {doc for doc, _ in plain}
                record["gold_available"] = bool(expected & available)
                for name, entries in (
                    ("plain_chunk", plain_vectors),
                    ("contextual_chunk", contextual_vectors),
                    ("proposition_leaf", proposition_vectors),
                ):
                    ranking = rank_docs(vectors[0], entries)
                    rank = next((i for i, doc in enumerate(ranking, start=1) if doc in expected), None)
                    record[name] = {
                        "ranking": ranking,
                        "gold_rank": rank,
                        "hit_at_1": float(rank == 1),
                        "hit_at_3": float(rank is not None and rank <= 3),
                        "reciprocal_rank": reciprocal_rank(ranking, expected),
                    }
                record["usage"] = usage
                record["latency_ms"] = latency
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            records.append(record)
            if index % 25 == 0:
                print(f"EMBED {index}/{len(rows)} errors={sum(bool(value.get('error')) for value in records)}", flush=True)
    summary = summarize(records, metadata)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if summary["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
