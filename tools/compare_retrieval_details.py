#!/usr/bin/env python3
"""Compare two retrieval/evidence details JSONL files question by question."""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
            qid = str(row.get("question_id") or row.get("qid") or row.get("id") or "").strip()
            if not qid:
                raise ValueError(f"missing question id at {path}:{line_number}")
            if qid in rows:
                raise ValueError(f"duplicate question id {qid!r} in {path}")
            rows[qid] = row
    return rows


def first_expected_rank(row: dict[str, Any]) -> int | None:
    expected = {str(value) for value in row.get("expected_doc_ids") or []}
    if not expected:
        return None
    for rank, document in enumerate(row.get("ranked_documents") or [], start=1):
        if str(document.get("doc_id") or "") in expected:
            return rank
    return None


def numeric(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def group_value(row: dict[str, Any], field: str) -> str:
    value = row.get(field)
    if isinstance(value, list):
        return "+".join(sorted(str(item) for item in value if str(item).strip())) or "unknown"
    return str(value or "unknown")


def compare_group(items: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(items)

    def hit(side: str, k: int) -> float:
        key = f"{side}_rank"
        return sum(item[key] is not None and item[key] <= k for item in items) / total

    def mrr(side: str) -> float:
        key = f"{side}_rank"
        return statistics.fmean(
            1.0 / item[key] if item[key] is not None and item[key] <= 10 else 0.0
            for item in items
        )

    evidence_deltas = [
        item["candidate_evidence_recall"] - item["baseline_evidence_recall"]
        for item in items
        if item["baseline_evidence_recall"] is not None
        and item["candidate_evidence_recall"] is not None
    ]
    return {
        "questions": total,
        "baseline": {
            "hit@1": hit("baseline", 1),
            "hit@5": hit("baseline", 5),
            "hit@10": hit("baseline", 10),
            "mrr@10": mrr("baseline"),
        },
        "candidate": {
            "hit@1": hit("candidate", 1),
            "hit@5": hit("candidate", 5),
            "hit@10": hit("candidate", 10),
            "mrr@10": mrr("candidate"),
        },
        "delta": {
            "hit@1": hit("candidate", 1) - hit("baseline", 1),
            "hit@5": hit("candidate", 5) - hit("baseline", 5),
            "hit@10": hit("candidate", 10) - hit("baseline", 10),
            "mrr@10": mrr("candidate") - mrr("baseline"),
            "evidence_fact_token_recall": statistics.fmean(evidence_deltas)
            if evidence_deltas
            else None,
        },
    }


def build_comparison(
    baseline: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    missing_candidate = sorted(set(baseline) - set(candidate))
    missing_baseline = sorted(set(candidate) - set(baseline))
    comparable: list[dict[str, Any]] = []
    for qid in sorted(set(baseline) & set(candidate)):
        base = baseline[qid]
        cand = candidate[qid]
        expected = base.get("expected_doc_ids") or []
        if not expected:
            continue
        comparable.append(
            {
                "qid": qid,
                "question": base.get("question"),
                "question_type": group_value(base, "question_type"),
                "source_types": group_value(base, "source_types"),
                "baseline_rank": first_expected_rank(base),
                "candidate_rank": first_expected_rank(cand),
                "baseline_evidence_recall": numeric(base, "evidence_fact_token_recall"),
                "candidate_evidence_recall": numeric(cand, "evidence_fact_token_recall"),
            }
        )

    hit_flips: dict[str, Any] = {}
    for k in (1, 5, 10):
        miss_to_hit = [
            item["qid"]
            for item in comparable
            if (item["baseline_rank"] is None or item["baseline_rank"] > k)
            and item["candidate_rank"] is not None
            and item["candidate_rank"] <= k
        ]
        hit_to_miss = [
            item["qid"]
            for item in comparable
            if item["baseline_rank"] is not None
            and item["baseline_rank"] <= k
            and (item["candidate_rank"] is None or item["candidate_rank"] > k)
        ]
        hit_flips[f"hit@{k}"] = {
            "miss_to_hit": miss_to_hit,
            "hit_to_miss": hit_to_miss,
            "net": len(miss_to_hit) - len(hit_to_miss),
        }

    grouped: dict[str, Any] = {}
    for field in ("source_types", "question_type"):
        values: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in comparable:
            values[item[field]].append(item)
        grouped[field] = {
            name: compare_group(items)
            for name, items in sorted(values.items())
        }

    def rank_value(value: int | None) -> int:
        return value if value is not None else 1_000_000

    rank_changes = sorted(
        comparable,
        key=lambda item: (
            rank_value(item["baseline_rank"]) - rank_value(item["candidate_rank"]),
            item["qid"],
        ),
        reverse=True,
    )
    return {
        "baseline_rows": len(baseline),
        "candidate_rows": len(candidate),
        "missing_in_candidate": missing_candidate,
        "missing_in_baseline": missing_baseline,
        "evaluable_questions": len(comparable),
        "overall": compare_group(comparable),
        "hit_flips": hit_flips,
        "by": grouped,
        "largest_rank_improvements": rank_changes[:20],
        "largest_rank_regressions": list(reversed(rank_changes[-20:])),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build_comparison(load_jsonl(args.baseline), load_jsonl(args.candidate))
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
