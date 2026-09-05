#!/usr/bin/env python3
"""Export E3/E4/E5 parent and leaf nodes without mutating Elasticsearch.

The exporter streams source documents, preserves raw text and character offsets,
attaches the original ACL record, and emits deterministic contextual search text.
It is an index-build input, not an online index migration. No embeddings or model
calls are made.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from tools.adaptive_rag.hierarchical_evidence import HierarchyConfig, build_hierarchy

SCHEMA_VERSION = 1
EXPORTER_VERSION = "hierarchical-index-export-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_id(*parts: Any) -> str:
    encoded = "\0".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_acl(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    result: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"ACL line {line_number} must be an object")
            doc_id = str(value.get("doc_id") or value.get("document_id") or "").strip()
            if not doc_id:
                raise ValueError(f"ACL line {line_number} is missing doc_id")
            if doc_id in result:
                raise ValueError(f"duplicate ACL doc_id: {doc_id}")
            result[doc_id] = value
    return result


def source_metadata(document: dict[str, Any]) -> dict[str, Any]:
    metadata = document.get("metadata") if isinstance(document.get("metadata"), dict) else {}
    return {
        "source_path": document.get("source_path") or metadata.get("source_path") or "",
        "source_dataset": document.get("source_dataset") or metadata.get("source_dataset") or "",
        "document_version": document.get("document_version") or document.get("version")
        or metadata.get("document_version") or metadata.get("version") or "",
        "source_updated_at": document.get("source_updated_at") or document.get("updated_at")
        or metadata.get("source_updated_at") or metadata.get("updated_at") or "",
        "source_revision": document.get("source_revision") or document.get("revision")
        or metadata.get("source_revision") or metadata.get("revision") or "",
    }


def percentile(values: list[int], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return float(ordered[low])
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def export(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise ValueError(f"output directory already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    acl_by_doc = load_acl(args.acl_docs)
    source_filter = {value.casefold() for value in args.source_type}
    config = HierarchyConfig(
        leaf_mode=args.leaf_mode,
        contextual_prefix=not args.no_contextual_prefix,
    )
    parents_path = args.output_dir / "parents.jsonl"
    leaves_path = args.output_dir / "leaves.jsonl"
    manifest_path = args.output_dir / "manifest.json"
    summary_path = args.output_dir / "summary.json"
    started = time.perf_counter()
    documents_seen = 0
    documents_exported = 0
    missing_acl = 0
    source_counts: Counter[str] = Counter()
    parent_kinds: Counter[str] = Counter()
    leaf_kinds: Counter[str] = Counter()
    parent_lengths: list[int] = []
    leaf_lengths: list[int] = []
    offset_violations = 0
    prefix_nodes = 0
    parent_count_by_source: Counter[str] = Counter()
    leaf_count_by_source: Counter[str] = Counter()

    with (
        args.docs.open(encoding="utf-8") as source,
        parents_path.open("x", encoding="utf-8") as parent_output,
        leaves_path.open("x", encoding="utf-8") as leaf_output,
    ):
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            documents_seen += 1
            document = json.loads(line)
            if not isinstance(document, dict):
                raise ValueError(f"document line {line_number} must be an object")
            doc_id = str(document.get("doc_id") or "").strip()
            if not doc_id:
                raise ValueError(f"document line {line_number} is missing doc_id")
            source_type = str(document.get("source_type") or "unknown").casefold()
            if source_filter and source_type not in source_filter:
                continue
            if args.limit is not None and documents_exported >= args.limit:
                break
            title = str(document.get("title") or "")
            text = str(document.get("text") or "")
            if not text.strip():
                continue
            acl = acl_by_doc.get(doc_id)
            if acl is None:
                missing_acl += 1
                if args.fail_on_missing_acl:
                    raise ValueError(f"missing ACL for document {doc_id}")
                acl = {}
            metadata = source_metadata(document)
            context = {
                "citation_id": "S1",
                "doc_id": doc_id,
                "title": title,
                "text": text,
                "source_type": source_type,
                **metadata,
            }
            parents, leaves = build_hierarchy([context], config)
            local_parent_ids: dict[str, str] = {}
            for parent in parents:
                node_id = stable_id(EXPORTER_VERSION, doc_id, "parent", parent.start_char, parent.end_char, parent.kind)
                local_parent_ids[parent.id] = node_id
                raw_slice = text[parent.start_char:parent.end_char]
                if raw_slice != parent.text:
                    offset_violations += 1
                search_text = f"{parent.contextual_prefix}\n{raw_slice}" if parent.contextual_prefix else raw_slice
                record = {
                    "schema_version": SCHEMA_VERSION,
                    "node_id": node_id,
                    "node_type": "parent",
                    "doc_id": doc_id,
                    "source_type": source_type,
                    "title": title,
                    "parent_kind": parent.kind,
                    "section_path": parent.section_path,
                    "speaker": parent.speaker,
                    "event_time": parent.event_time,
                    "start_char": parent.start_char,
                    "end_char": parent.end_char,
                    "offset_unit": "python_character_within_raw_document",
                    "raw_text": raw_slice,
                    "search_text": search_text,
                    "contextual_prefix": parent.contextual_prefix,
                    "acl": acl,
                    **metadata,
                }
                parent_output.write(json.dumps(record, ensure_ascii=False) + "\n")
                parent_lengths.append(len(raw_slice))
                parent_kinds[parent.kind] += 1
                parent_count_by_source[source_type] += 1
                prefix_nodes += int(bool(parent.contextual_prefix))
            for leaf in leaves:
                parent_node_id = local_parent_ids[leaf.parent_id]
                node_id = stable_id(EXPORTER_VERSION, doc_id, "leaf", leaf.start_char, leaf.end_char, leaf.kind)
                raw_slice = text[leaf.start_char:leaf.end_char]
                if raw_slice != leaf.text:
                    offset_violations += 1
                record = {
                    "schema_version": SCHEMA_VERSION,
                    "node_id": node_id,
                    "node_type": "leaf",
                    "parent_node_id": parent_node_id,
                    "doc_id": doc_id,
                    "source_type": source_type,
                    "title": title,
                    "leaf_kind": leaf.kind,
                    "start_char": leaf.start_char,
                    "end_char": leaf.end_char,
                    "offset_unit": "python_character_within_raw_document",
                    "raw_text": raw_slice,
                    "search_text": leaf.search_text,
                    "acl": acl,
                    **metadata,
                }
                leaf_output.write(json.dumps(record, ensure_ascii=False) + "\n")
                leaf_lengths.append(len(raw_slice))
                leaf_kinds[leaf.kind] += 1
                leaf_count_by_source[source_type] += 1
                prefix_nodes += int(leaf.search_text != leaf.text)
            documents_exported += 1
            source_counts[source_type] += 1
            if args.progress_every and documents_exported % args.progress_every == 0:
                print(f"EXPORTED {documents_exported} documents", flush=True)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "exporter_version": EXPORTER_VERSION,
        "documents_seen": documents_seen,
        "documents_exported": documents_exported,
        "missing_acl_documents": missing_acl,
        "parents": len(parent_lengths),
        "leaves": len(leaf_lengths),
        "offset_violations": offset_violations,
        "contextual_prefix_nodes": prefix_nodes,
        "source_documents": dict(sorted(source_counts.items())),
        "parents_by_source": dict(sorted(parent_count_by_source.items())),
        "leaves_by_source": dict(sorted(leaf_count_by_source.items())),
        "parent_kinds": dict(sorted(parent_kinds.items())),
        "leaf_kinds": dict(sorted(leaf_kinds.items())),
        "parent_characters": _distribution(parent_lengths),
        "leaf_characters": _distribution(leaf_lengths),
        "elapsed_seconds": time.perf_counter() - started,
        "leaf_mode": args.leaf_mode,
        "contextual_prefix": not args.no_contextual_prefix,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "exporter_version": EXPORTER_VERSION,
        "docs": str(args.docs.resolve()),
        "docs_sha256": sha256_file(args.docs),
        "acl_docs": str(args.acl_docs.resolve()) if args.acl_docs else None,
        "acl_docs_sha256": sha256_file(args.acl_docs) if args.acl_docs else None,
        "source_types": sorted(source_filter),
        "limit": args.limit,
        "leaf_mode": args.leaf_mode,
        "contextual_prefix": not args.no_contextual_prefix,
        "parents_sha256": sha256_file(parents_path),
        "leaves_sha256": sha256_file(leaves_path),
        "summary_sha256": sha256_file(summary_path),
        "environment": {"python": os.sys.version, "cwd": os.getcwd()},
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"summary": summary, "manifest": manifest}


def _distribution(values: list[int]) -> dict[str, Any]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else None,
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "max": max(values) if values else None,
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", type=Path, required=True)
    parser.add_argument("--acl-docs", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-type", action="append", default=[])
    parser.add_argument("--leaf-mode", choices=("sentence", "proposition"), default="sentence")
    parser.add_argument("--no-contextual-prefix", action="store_true")
    parser.add_argument("--fail-on-missing-acl", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--progress-every", type=int, default=1000)
    args = parser.parse_args(argv)
    if not args.docs.is_file():
        parser.error(f"docs file is missing: {args.docs}")
    if args.acl_docs is not None and not args.acl_docs.is_file():
        parser.error(f"ACL file is missing: {args.acl_docs}")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    return args


def main(argv: Iterable[str] | None = None) -> int:
    result = export(parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
