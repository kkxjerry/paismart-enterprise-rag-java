#!/usr/bin/env python3
"""Attribute each answer fact to retrieval, evidence, window, generation, or citation failure."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.adaptive_rag.attribution import attribute_row, summarize_attributions
from tools.qwen_plus_rag_pipeline import atomic_write_jsonl, load_jsonl, validate_output_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts", required=True, type=Path, help="Java full evidence contexts JSONL")
    parser.add_argument("--answers", required=True, type=Path, help="Adaptive generation JSONL")
    parser.add_argument("--details", type=Path, help="Optional Java retrieval details JSONL")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary-output", required=True, type=Path)
    parser.add_argument("--coverage-threshold", type=float, default=0.60)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Require an answer row for every context row instead of analyzing their qid intersection.",
    )
    args = parser.parse_args()
    if not 0.0 <= args.coverage_threshold <= 1.0:
        parser.error("--coverage-threshold must be between 0 and 1")
    return args


def keyed(rows: list[dict[str, Any]], *fields: str) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = ""
        for field in fields:
            if row.get(field):
                key = str(row[field])
                break
        if key:
            output[key] = row
    return output


def main() -> int:
    args = parse_args()
    validate_output_paths(
        [value for value in (args.contexts, args.answers, args.details) if value is not None],
        [args.output, args.summary_output],
    )
    source_rows = load_jsonl(args.contexts)
    answers = keyed(load_jsonl(args.answers), "qid", "id")
    details = keyed(load_jsonl(args.details), "question_id", "qid", "id") if args.details else {}
    missing_answers = [
        str(row.get("qid") or row.get("id"))
        for row in source_rows
        if str(row.get("qid") or row.get("id")) not in answers
    ]
    if missing_answers and args.strict:
        raise ValueError(f"answers are missing qids: {missing_answers[:10]}")
    selected_sources = [
        source
        for source in source_rows
        if str(source.get("qid") or source.get("id")) in answers
    ]
    if not selected_sources:
        raise ValueError("contexts and answers have no qids in common")
    output = []
    for source in selected_sources:
        qid = str(source.get("qid") or source.get("id"))
        output.append(
            attribute_row(
                source,
                answers[qid],
                details.get(qid),
                coverage_threshold=args.coverage_threshold,
            )
        )
    atomic_write_jsonl(args.output, output)
    summary = {
        "contexts": str(args.contexts),
        "answers": str(args.answers),
        "details": str(args.details) if args.details else None,
        "coverage_threshold": args.coverage_threshold,
        "context_rows_total": len(source_rows),
        "answer_rows_total": len(answers),
        "missing_answer_rows": len(missing_answers),
        "strict": args.strict,
        **summarize_attributions(output),
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
