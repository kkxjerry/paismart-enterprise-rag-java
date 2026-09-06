#!/usr/bin/env python3
"""Build an isolated Elasticsearch Parent/Leaf index for E3/E4/E5.

Parents retain raw source text and provenance. Leaves carry embeddings of
search_text, which may include a deterministic contextual prefix. The script
never updates aliases and refuses to overwrite an existing index.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

SCHEMA_VERSION = 1
BUILDER_VERSION = "hierarchical-parent-leaf-index-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def request_json(
    method: str,
    url: str,
    *,
    payload: Any | None = None,
    body: bytes | None = None,
    content_type: str = "application/json",
    timeout: float = 120.0,
    retries: int = 3,
) -> Any:
    if payload is not None and body is not None:
        raise ValueError("payload and body are mutually exclusive")
    data = body if body is not None else (
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if payload is not None
        else None
    )
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": content_type, "Accept": "application/json"},
                method=method,
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read()
            return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as exc:
            text = exc.read().decode("utf-8", errors="replace")[:4000]
            if exc.code not in {429, 500, 502, 503, 504} or attempt >= retries:
                raise RuntimeError(f"HTTP {exc.code} {method} {url}: {text}") from exc
            last = exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = exc
            if attempt >= retries:
                raise RuntimeError(f"request failed {method} {url}: {exc}") from exc
        time.sleep(min(8.0, 0.5 * 2**attempt) + random.random() * 0.1)
    raise RuntimeError(f"request failed: {last}")


def index_mapping(dimension: int) -> dict[str, Any]:
    if dimension <= 0:
        raise ValueError("embedding dimension must be positive")
    keyword = {"type": "keyword", "ignore_above": 2048}
    return {
        "settings": {
            "index": {"number_of_shards": 1, "number_of_replicas": 0},
            "analysis": {"analyzer": {"hier_text": {"type": "standard"}}},
        },
        "mappings": {
            "dynamic": "strict",
            "properties": {
                "nodeId": keyword,
                "nodeType": keyword,
                "parentNodeId": keyword,
                "docId": keyword,
                "sourceType": keyword,
                "sourcePath": keyword,
                "sourceDataset": keyword,
                "documentVersion": keyword,
                "sourceRevision": keyword,
                "title": {"type": "text", "analyzer": "hier_text", "fields": {"keyword": keyword}},
                "sectionPath": {"type": "text", "analyzer": "hier_text", "fields": {"keyword": keyword}},
                "nodeKind": keyword,
                "speaker": keyword,
                "eventTime": {"type": "keyword", "ignore_above": 512},
                "sourceUpdatedAt": {"type": "keyword", "ignore_above": 512},
                "startChar": {"type": "integer"},
                "endChar": {"type": "integer"},
                "rawText": {"type": "text", "index": False},
                "searchText": {"type": "text", "analyzer": "hier_text"},
                "contextualPrefix": {"type": "text", "index": False},
                "tenantId": keyword,
                "classification": keyword,
                "allowedUserIds": keyword,
                "allowedGroupIds": keyword,
                "deniedUserIds": keyword,
                "deniedGroupIds": keyword,
                "vector": {
                    "type": "dense_vector",
                    "dims": dimension,
                    "index": True,
                    "similarity": "cosine",
                    "index_options": {"type": "hnsw", "m": 16, "ef_construction": 100},
                },
            },
        },
    }


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must be an object")
            yield value


def batches(values: Iterable[Any], size: int) -> Iterator[list[Any]]:
    if size <= 0:
        raise ValueError("batch size must be positive")
    batch: list[Any] = []
    for value in values:
        batch.append(value)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def embed(
    texts: Sequence[str],
    *,
    endpoint: str,
    model: str,
    dimension: int,
    timeout: float,
) -> list[list[float]]:
    response = request_json(
        "POST",
        endpoint,
        payload={"model": model, "input": list(texts)},
        timeout=timeout,
    )
    data = response.get("data") if isinstance(response, dict) else None
    if not isinstance(data, list) or len(data) != len(texts):
        raise ValueError("embedding response count does not match request")
    ordered = sorted(data, key=lambda value: int(value.get("index", 0)))
    vectors: list[list[float]] = []
    for item in ordered:
        vector = item.get("embedding") if isinstance(item, dict) else None
        if not isinstance(vector, list) or len(vector) != dimension:
            raise ValueError(
                f"embedding dimension mismatch: expected {dimension}, got "
                f"{len(vector) if isinstance(vector, list) else type(vector).__name__}"
            )
        values = [float(value) for value in vector]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("embedding contains non-finite values")
        vectors.append(values)
    return vectors


def normalize_acl(value: Any) -> dict[str, Any]:
    acl = value if isinstance(value, dict) else {}
    return {
        "tenantId": str(acl.get("tenant_id") or acl.get("tenantId") or ""),
        "classification": str(acl.get("classification") or ""),
        "allowedUserIds": _strings(acl.get("allowed_user_ids") or acl.get("allowedUserIds") or []),
        "allowedGroupIds": _strings(acl.get("allowed_group_ids") or acl.get("allowedGroupIds") or []),
        "deniedUserIds": _strings(acl.get("denied_user_ids") or acl.get("deniedUserIds") or []),
        "deniedGroupIds": _strings(acl.get("denied_group_ids") or acl.get("deniedGroupIds") or []),
    }


def elastic_document(node: dict[str, Any], vector: Sequence[float] | None = None) -> dict[str, Any]:
    acl = normalize_acl(node.get("acl"))
    value = {
        "nodeId": str(node.get("node_id") or ""),
        "nodeType": str(node.get("node_type") or ""),
        "parentNodeId": str(node.get("parent_node_id") or ""),
        "docId": str(node.get("doc_id") or ""),
        "sourceType": str(node.get("source_type") or "unknown"),
        "sourcePath": str(node.get("source_path") or ""),
        "sourceDataset": str(node.get("source_dataset") or ""),
        "documentVersion": str(node.get("document_version") or ""),
        "sourceRevision": str(node.get("source_revision") or ""),
        "title": str(node.get("title") or ""),
        "sectionPath": str(node.get("section_path") or ""),
        "nodeKind": str(node.get("parent_kind") or node.get("leaf_kind") or ""),
        "speaker": str(node.get("speaker") or ""),
        "eventTime": str(node.get("event_time") or ""),
        "sourceUpdatedAt": str(node.get("source_updated_at") or ""),
        "startChar": int(node.get("start_char") or 0),
        "endChar": int(node.get("end_char") or 0),
        "rawText": str(node.get("raw_text") or ""),
        "searchText": str(node.get("search_text") or ""),
        "contextualPrefix": str(node.get("contextual_prefix") or ""),
        **acl,
    }
    if vector is not None:
        value["vector"] = list(vector)
    return value


def bulk_index(es_url: str, index: str, documents: Sequence[tuple[str, dict[str, Any]]], timeout: float) -> None:
    lines: list[str] = []
    for node_id, document in documents:
        lines.append(json.dumps({"index": {"_index": index, "_id": node_id}}, separators=(",", ":")))
        lines.append(json.dumps(document, ensure_ascii=False, separators=(",", ":")))
    body = ("\n".join(lines) + "\n").encode("utf-8")
    response = request_json(
        "POST",
        es_url.rstrip("/") + "/_bulk?refresh=false",
        body=body,
        content_type="application/x-ndjson",
        timeout=timeout,
    )
    if not isinstance(response, dict) or response.get("errors"):
        failures = []
        for item in (response.get("items") or [])[:20] if isinstance(response, dict) else []:
            result = item.get("index") if isinstance(item, dict) else None
            if isinstance(result, dict) and result.get("error"):
                failures.append(result.get("error"))
        raise RuntimeError(f"bulk indexing failed: {failures[:3]}")


def build(args: argparse.Namespace) -> dict[str, Any]:
    base = args.es_url.rstrip("/")
    quoted = urllib.parse.quote(args.index, safe="-_.*")
    exists = request_status("HEAD", f"{base}/{quoted}", timeout=args.timeout_seconds)
    if exists == 200:
        raise ValueError(f"refusing to overwrite existing index: {args.index}")
    if exists not in {404}:
        raise RuntimeError(f"unexpected index existence status {exists}")
    request_json("PUT", f"{base}/{quoted}", payload=index_mapping(args.embedding_dimension), timeout=args.timeout_seconds)
    started = time.perf_counter()
    counts: Counter[str] = Counter()
    try:
        for batch in batches(read_jsonl(args.parents), args.bulk_size):
            documents = [(str(node["node_id"]), elastic_document(node)) for node in batch]
            bulk_index(base, args.index, documents, args.timeout_seconds)
            counts["parents"] += len(documents)
        for batch in batches(read_jsonl(args.leaves), args.embedding_batch_size):
            texts = [str(node.get("search_text") or "") for node in batch]
            vectors = embed(
                texts,
                endpoint=args.embedding_url,
                model=args.embedding_model,
                dimension=args.embedding_dimension,
                timeout=args.timeout_seconds,
            )
            documents = [
                (str(node["node_id"]), elastic_document(node, vector))
                for node, vector in zip(batch, vectors)
            ]
            for sub_batch in batches(documents, args.bulk_size):
                bulk_index(base, args.index, sub_batch, args.timeout_seconds)
            counts["leaves"] += len(documents)
            if args.progress_every and counts["leaves"] % args.progress_every < len(documents):
                print(f"INDEXED leaves={counts['leaves']}", flush=True)
        request_json("POST", f"{base}/{quoted}/_refresh", payload={}, timeout=args.timeout_seconds)
        count_result = request_json("GET", f"{base}/{quoted}/_count", timeout=args.timeout_seconds)
        index_count = int(count_result.get("count") or 0)
        expected = counts["parents"] + counts["leaves"]
        if index_count != expected:
            raise RuntimeError(f"index count mismatch: expected {expected}, got {index_count}")
        return {
            "schema_version": SCHEMA_VERSION,
            "builder_version": BUILDER_VERSION,
            "index": args.index,
            "parents": counts["parents"],
            "leaves": counts["leaves"],
            "index_documents": index_count,
            "embedding_model": args.embedding_model,
            "embedding_dimension": args.embedding_dimension,
            "parents_sha256": sha256_file(args.parents),
            "leaves_sha256": sha256_file(args.leaves),
            "elapsed_seconds": time.perf_counter() - started,
            "alias_changed": False,
        }
    except Exception:
        # Preserve a failed index for audit unless explicitly requested otherwise.
        if args.delete_failed_index:
            request_json("DELETE", f"{base}/{quoted}", timeout=args.timeout_seconds)
        raise


def request_status(method: str, url: str, *, timeout: float) -> int:
    request = urllib.request.Request(url, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return int(response.status)
    except urllib.error.HTTPError as exc:
        return exc.code


def _strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item)]


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parents", type=Path, required=True)
    parser.add_argument("--leaves", type=Path, required=True)
    parser.add_argument("--es-url", default="http://127.0.0.1:19200")
    parser.add_argument("--index", required=True)
    parser.add_argument("--embedding-url", default="http://127.0.0.1:18084/v1/embeddings")
    parser.add_argument("--embedding-model", default="Qwen/Qwen3-Embedding-4B")
    parser.add_argument("--embedding-dimension", type=int, default=2048)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--bulk-size", type=int, default=200)
    parser.add_argument("--progress-every", type=int, default=5000)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--delete-failed-index", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    for path in (args.parents, args.leaves):
        if not path.is_file():
            parser.error(f"input is missing: {path}")
    for name in ("embedding_dimension", "embedding_batch_size", "bulk_size"):
        if int(getattr(args, name)) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    result = build(args)
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
