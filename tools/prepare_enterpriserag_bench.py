#!/usr/bin/env python3
"""Download EnterpriseRAG-Bench and emit the JSON/JSONL files used by the Java CLI."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def row_to_dict(row: Any) -> dict[str, Any]:
    return dict(row) if not isinstance(row, dict) else row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/enterpriserag")
    parser.add_argument("--limit-docs", type=int, default=10_000)
    parser.add_argument("--limit-questions", type=int)
    parser.add_argument(
        "--no-include-question-docs",
        action="store_true",
        help="Do not force the selected questions' gold documents into the corpus.",
    )
    args = parser.parse_args()
    if args.limit_docs is not None and args.limit_docs < 0:
        raise SystemExit("--limit-docs must not be negative")

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install tools/requirements-data.txt first") from exc

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    question_rows = load_dataset(
        "onyx-dot-app/EnterpriseRAG-Bench", "questions", split="test"
    )
    questions: list[dict[str, Any]] = []
    for index, row in enumerate(question_rows):
        record = row_to_dict(row)
        expected = record.get("expected_doc_ids") or []
        if isinstance(expected, str):
            expected = [expected]
        question = {
            "id": str(record.get("question_id") or record.get("id") or f"q_{index:04d}"),
            "question": str(record.get("question") or ""),
            "question_type": record.get("question_type") or "unknown",
            "category": record.get("question_type") or "enterprise_rag",
            "source_dataset": "onyx-dot-app/EnterpriseRAG-Bench",
            "source_types": record.get("source_types") or [],
            "expected_doc_ids": [str(value) for value in expected],
            "gold_answer": record.get("gold_answer"),
            "answer_facts": record.get("answer_facts") or [],
        }
        if question["question"].strip():
            questions.append(question)
        if args.limit_questions is not None and len(questions) >= args.limit_questions:
            break

    required_doc_ids = {
        doc_id for question in questions for doc_id in question["expected_doc_ids"] if doc_id
    }
    include_gold = not args.no_include_question_docs
    emitted: set[str] = set()
    required_found: set[str] = set()
    filler_count = 0
    documents_count = 0
    document_rows = load_dataset(
        "onyx-dot-app/EnterpriseRAG-Bench", "documents", split="test"
    )
    with (out_dir / "docs.jsonl").open("w", encoding="utf-8") as docs, (
        out_dir / "acl_docs.jsonl"
    ).open("w", encoding="utf-8") as acls:
        for row in document_rows:
            record = row_to_dict(row)
            doc_id = str(record.get("doc_id") or record.get("id") or "")
            if not doc_id or doc_id in emitted:
                continue
            required = doc_id in required_doc_ids
            filler_full = args.limit_docs is not None and filler_count >= args.limit_docs
            if not required and filler_full:
                if not include_gold or required_found >= required_doc_ids:
                    break
                continue
            source_type = str(record.get("source_type") or "unknown")
            content = str(record.get("content") or record.get("text") or "")
            if not content.strip():
                continue
            document = {
                "doc_id": doc_id,
                "title": str(record.get("title") or ""),
                "text": content,
                "source_path": f"enterpriserag:{source_type}:{doc_id}",
                "source_dataset": "onyx-dot-app/EnterpriseRAG-Bench",
                "source_type": source_type,
            }
            acl = {
                "doc_id": doc_id,
                "tenant_id": "tenant_redwood",
                "source_type": source_type,
                "classification": "internal",
                "allowed_group_ids": [
                    "role:admin",
                    "role:employee",
                    f"source:{source_type}",
                ],
                "denied_group_ids": [],
            }
            docs.write(json.dumps(document, ensure_ascii=False) + "\n")
            acls.write(json.dumps(acl, ensure_ascii=False) + "\n")
            emitted.add(doc_id)
            documents_count += 1
            if required:
                required_found.add(doc_id)
            else:
                filler_count += 1

    payload = {
        "source_dataset": "onyx-dot-app/EnterpriseRAG-Bench",
        "questions": questions,
    }
    (out_dir / "questions.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    missing = sorted(required_doc_ids - required_found)
    source = f"""# Source

- Dataset: `onyx-dot-app/EnterpriseRAG-Bench`
- Documents exported: {documents_count}
- Filler documents exported: {filler_count}
- Questions exported: {len(questions)}
- Required gold documents found: {len(required_found)}/{len(required_doc_ids)}
- Missing required document IDs: {len(missing)}

The corpus is external data and is not redistributed by this repository. Review the
current dataset card and license before redistribution.
"""
    (out_dir / "SOURCE.md").write_text(source, encoding="utf-8")
    if missing:
        (out_dir / "missing_gold_doc_ids.json").write_text(
            json.dumps(missing, indent=2) + "\n", encoding="utf-8"
        )
        raise SystemExit(f"Missing {len(missing)} gold documents; see missing_gold_doc_ids.json")
    print(f"Wrote {documents_count} documents and {len(questions)} questions to {out_dir}")


if __name__ == "__main__":
    main()
