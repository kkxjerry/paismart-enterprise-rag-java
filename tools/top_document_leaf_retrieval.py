#!/usr/bin/env python3
"""E1 query-time leaf retrieval inside already authorized top documents.

The input ranked_doc_ids come from the frozen, ACL-filtered Java retrieval run.
They form the only document allow-list used by this experiment. The script fetches
additional chunks for those documents from the existing evidence index, creates
sentence/proposition leaves in memory, ranks them, and returns contiguous parent
text. It does not rebuild the index, change aliases, or consult gold labels before
selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
import urllib.parse
from pathlib import Path
from typing import Any, Iterable, Sequence

from tools.adaptive_rag.hierarchical_evidence import DEFAULT_HIERARCHY_CONFIG, pack_hierarchical_evidence
from tools.hierarchical_index_build import request_json, sha256_file
from tools.qwen_plus_rag_pipeline import load_jsonl
from tools.rag_e1_e6_experiment import aggregate_prompt, pack_arm, score_prompt

VERSION = "top-document-leaf-retrieval-v1"


def fetch_document_chunks(
    *,
    es_url: str,
    index: str,
    doc_ids: Sequence[str],
    tenant_id: str,
    chunks_per_document: int,
    timeout: float,
) -> list[dict[str, Any]]:
    allowed = list(dict.fromkeys(str(value) for value in doc_ids if str(value)))
    if not allowed:
        return []
    filters: list[dict[str, Any]] = [{"terms": {"benchmarkDocId": allowed}}]
    if tenant_id:
        filters.append({"term": {"tenantId": tenant_id}})
    fields = [
        "benchmarkDocId", "chunkId", "chunkKind", "sectionPath", "speaker", "threadId",
        "eventTime", "sourceType", "sourcePath", "title", "classification", "documentVersion",
        "documentHash", "sourceUpdatedAt", "contentHash", "textContent",
    ]
    payload = {
        "size": min(10_000, max(1, len(allowed) * chunks_per_document * 3)),
        "track_total_hits": False,
        "_source": fields,
        "query": {"bool": {"filter": filters}},
        "sort": [{"benchmarkDocId": "asc"}, {"chunkId": "asc"}],
    }
    response = request_json(
        "POST",
        f"{es_url.rstrip('/')}/{urllib.parse.quote(index, safe='-_.*')}/_search",
        payload=payload,
        timeout=timeout,
    )
    raw = response.get("hits", {}).get("hits", []) if isinstance(response, dict) else []
    by_doc: dict[str, list[dict[str, Any]]] = {doc_id: [] for doc_id in allowed}
    for item in raw:
        source = item.get("_source") if isinstance(item, dict) else None
        if not isinstance(source, dict):
            continue
        doc_id = str(source.get("benchmarkDocId") or "")
        if doc_id not in by_doc or len(by_doc[doc_id]) >= chunks_per_document:
            continue
        by_doc[doc_id].append({"chunk_es_id": str(item.get("_id") or ""), **source})
    return [chunk for doc_id in allowed for chunk in by_doc[doc_id]]


def contexts_from_chunks(chunks: Sequence[dict[str, Any]], ranked_doc_ids: Sequence[str]) -> list[dict[str, Any]]:
    rank = {str(doc_id): index for index, doc_id in enumerate(ranked_doc_ids, start=1)}
    output = []
    for index, chunk in enumerate(chunks, start=1):
        doc_id = str(chunk.get("benchmarkDocId") or "")
        if doc_id not in rank:
            continue
        output.append(
            {
                "citation_id": f"S{index}",
                "doc_id": doc_id,
                "document_rank": rank[doc_id],
                "rank": index,
                "chunk_es_id": str(chunk.get("chunk_es_id") or ""),
                "chunk_id": int(chunk.get("chunkId") or 0),
                "chunk_kind": str(chunk.get("chunkKind") or ""),
                "section_path": str(chunk.get("sectionPath") or ""),
                "speaker": str(chunk.get("speaker") or ""),
                "thread_id": str(chunk.get("threadId") or ""),
                "event_time": str(chunk.get("eventTime") or ""),
                "source_type": str(chunk.get("sourceType") or "unknown"),
                "source_path": str(chunk.get("sourcePath") or ""),
                "title": str(chunk.get("title") or ""),
                "classification": str(chunk.get("classification") or ""),
                "document_version": str(chunk.get("documentVersion") or ""),
                "document_hash": str(chunk.get("documentHash") or ""),
                "source_updated_at": str(chunk.get("sourceUpdatedAt") or ""),
                "content_hash": str(chunk.get("contentHash") or ""),
                "text": str(chunk.get("textContent") or ""),
            }
        )
    return output


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    rows = load_jsonl(args.contexts)
    details: list[dict[str, Any]] = []
    baseline_values: list[dict[str, Any]] = []
    candidate_values: list[dict[str, Any]] = []
    fetch_latencies: list[float] = []
    pack_latencies: list[float] = []
    args.details.parent.mkdir(parents=True, exist_ok=True)
    with args.details.open("x", encoding="utf-8") as output:
        for index, row in enumerate(rows, start=1):
            question = str(row.get("question") or "")
            ranked = [str(value) for value in row.get("ranked_doc_ids") or [] if str(value)][: args.top_documents]
            baseline_packed = pack_arm(row, "query-spans-v3", max_chars=args.max_chars, max_contexts=args.max_contexts)
            baseline = score_prompt(row, baseline_packed)
            baseline_values.append(baseline)
            fetch_started = time.perf_counter()
            chunks = fetch_document_chunks(
                es_url=args.es_url,
                index=args.index,
                doc_ids=ranked,
                tenant_id=str(row.get("tenant_id") or ""),
                chunks_per_document=args.chunks_per_document,
                timeout=args.timeout_seconds,
            )
            fetch_ms = (time.perf_counter() - fetch_started) * 1000.0
            fetch_latencies.append(fetch_ms)
            expanded = contexts_from_chunks(chunks, ranked)
            pack_started = time.perf_counter()
            packed = pack_hierarchical_evidence(
                expanded,
                question=question,
                max_chars=args.max_chars,
                max_contexts=args.max_contexts,
                config=DEFAULT_HIERARCHY_CONFIG,
            )
            pack_ms = (time.perf_counter() - pack_started) * 1000.0
            pack_latencies.append(pack_ms)
            wrapped = {
                "rendered": packed.rendered,
                "contexts": list(packed.contexts),
                "metadata": packed.to_dict(),
                "canonical_documents": list(packed.canonical_documents),
            }
            candidate = score_prompt(row, wrapped)
            candidate_values.append(candidate)
            record = {
                "qid": candidate["qid"],
                "question_type": candidate["question_type"],
                "source_types": candidate["source_types"],
                "authorized_top_doc_ids": ranked,
                "fetched_chunks": len(chunks),
                "fetched_documents": len({str(value.get("benchmarkDocId") or "") for value in chunks}),
                "fetch_ms": fetch_ms,
                "pack_ms": pack_ms,
                "baseline": baseline,
                "candidate": candidate,
                "selection": packed.to_dict(),
            }
            details.append(record)
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            if index % args.progress_every == 0 or index == len(rows):
                print(f"E1 {index}/{len(rows)}", flush=True)
    baseline_map = {value["qid"]: value for value in baseline_values}
    result = {
        "schema_version": 1,
        "version": VERSION,
        "input": str(args.contexts),
        "input_sha256": sha256_file(args.contexts),
        "index": args.index,
        "questions": len(rows),
        "top_documents": args.top_documents,
        "chunks_per_document": args.chunks_per_document,
        "max_chars": args.max_chars,
        "max_contexts": args.max_contexts,
        "authorization_scope": "document IDs from frozen ACL-filtered ranked_doc_ids plus tenant filter",
        "baseline": aggregate_prompt(baseline_values, baseline_map),
        "candidate": aggregate_prompt(candidate_values, baseline_map),
        "mean_fetched_chunks": _mean(value["fetched_chunks"] for value in details),
        "mean_fetched_documents": _mean(value["fetched_documents"] for value in details),
        "mean_fetch_ms": _mean(fetch_latencies),
        "p95_fetch_ms": percentile(fetch_latencies, 0.95),
        "mean_pack_ms": _mean(pack_latencies),
        "p95_pack_ms": percentile(pack_latencies, 0.95),
        "empty_fetch_qids": [value["qid"] for value in details if value["fetched_chunks"] == 0],
        "details_sha256": sha256_file(args.details),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def _mean(values: Iterable[float]) -> float | None:
    selected = [float(value) for value in values]
    return statistics.fmean(selected) if selected else None


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts", type=Path, required=True)
    parser.add_argument("--es-url", default="http://127.0.0.1:19200")
    parser.add_argument("--index", required=True)
    parser.add_argument("--top-documents", type=int, default=10)
    parser.add_argument("--chunks-per-document", type=int, default=20)
    parser.add_argument("--max-chars", type=int, default=10_000)
    parser.add_argument("--max-contexts", type=int, default=12)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--details", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.contexts.is_file():
        parser.error(f"contexts are missing: {args.contexts}")
    for name in ("top_documents", "chunks_per_document", "max_chars", "max_contexts", "progress_every"):
        if int(getattr(args, name)) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def main(argv: Iterable[str] | None = None) -> int:
    result = evaluate(parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not result["empty_fetch_qids"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
