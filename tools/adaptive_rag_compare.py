#!/usr/bin/env python3
"""Paired comparison for adaptive RAG answer JSONL artifacts.

The tool aligns rows by qid, excludes errored rows, reports paired deltas with a
deterministic bootstrap interval, and keeps latency/token cost next to quality.
It never uses benchmark labels to change a system result; labels are only used
for post-hoc slices.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

METRICS = (
    "answer_fact_token_recall",
    "answer_fact_coverage_proxy",
    "context_fact_token_recall_upper_bound",
    "context_fact_coverage_proxy",
    "gold_answer_token_f1",
    "gold_answer_token_recall",
    "citation_coverage",
    "citation_precision",
    "gold_doc_cited",
    "grounded_answer_proxy",
    "unanswerable_abstain_correct",
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def successful_by_qid(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        qid = str(row.get("qid") or "").strip()
        if not qid or row.get("error"):
            continue
        if qid in result:
            raise ValueError(f"duplicate successful qid: {qid}")
        result[qid] = row
    return result


def metric_value(row: dict[str, Any], name: str) -> float | None:
    value = (row.get("metrics") or {}).get(name)
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def bootstrap_interval(
    deltas: list[float],
    *,
    samples: int = 2_000,
    seed: int = 20260903,
) -> tuple[float | None, float | None]:
    if not deltas:
        return None, None
    if len(deltas) == 1:
        return deltas[0], deltas[0]
    rng = random.Random(seed)
    means = [
        statistics.fmean(deltas[rng.randrange(len(deltas))] for _ in deltas)
        for _ in range(samples)
    ]
    means.sort()
    low = means[max(0, math.floor(samples * 0.025) - 1)]
    high = means[min(samples - 1, math.ceil(samples * 0.975) - 1)]
    return low, high


def two_sided_sign_p(improved: int, regressed: int) -> float | None:
    trials = improved + regressed
    if not trials:
        return None
    smaller = min(improved, regressed)
    tail = sum(math.comb(trials, value) for value in range(smaller + 1)) / (2**trials)
    return min(1.0, 2.0 * tail)


def compare_metric(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    name: str,
) -> dict[str, Any]:
    values: list[tuple[float, float]] = []
    for baseline, candidate in pairs:
        left = metric_value(baseline, name)
        right = metric_value(candidate, name)
        if left is not None and right is not None:
            values.append((left, right))
    deltas = [right - left for left, right in values]
    low, high = bootstrap_interval(deltas)
    epsilon = 1e-12
    improved = sum(delta > epsilon for delta in deltas)
    regressed = sum(delta < -epsilon for delta in deltas)
    return {
        "questions": len(values),
        "baseline_avg": statistics.fmean(left for left, _ in values) if values else None,
        "candidate_avg": statistics.fmean(right for _, right in values) if values else None,
        "mean_delta": statistics.fmean(deltas) if deltas else None,
        "bootstrap_95_ci": [low, high],
        "improved": improved,
        "regressed": regressed,
        "unchanged": len(deltas) - improved - regressed,
        "sign_test_two_sided_p": two_sided_sign_p(improved, regressed),
    }


def nested_number(row: dict[str, Any], *path: str) -> float | None:
    value: Any = row
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def average_path(rows: Iterable[dict[str, Any]], *path: str) -> float | None:
    values = [value for row in rows if (value := nested_number(row, *path)) is not None]
    return statistics.fmean(values) if values else None


def usage(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    totals = Counter()
    for row in rows:
        for stage in ("requirements", "generation", "verification", "secondary_retrieval"):
            node = row.get(stage) or {}
            stage_usage = node.get("usage") or {}
            for key in ("prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens"):
                totals[key] += int(stage_usage.get(key) or 0)
    return {key: totals[key] for key in ("prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens")}


def estimated_cost(
    token_usage: dict[str, int],
    *,
    input_per_million: float | None,
    cached_input_per_million: float | None,
    output_per_million: float | None,
) -> float | None:
    if input_per_million is None or output_per_million is None:
        return None
    cached_price = cached_input_per_million if cached_input_per_million is not None else input_per_million
    cached = token_usage.get("cached_tokens", 0)
    uncached = max(0, token_usage.get("prompt_tokens", 0) - cached)
    output = token_usage.get("completion_tokens", 0)
    return (
        uncached * input_per_million
        + cached * cached_price
        + output * output_per_million
    ) / 1_000_000.0


def summarize(
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    baseline = successful_by_qid(baseline_rows)
    candidate = successful_by_qid(candidate_rows)
    qids = sorted(set(baseline) & set(candidate))
    pairs = [(baseline[qid], candidate[qid]) for qid in qids]
    baseline_paired = [left for left, _ in pairs]
    candidate_paired = [right for _, right in pairs]

    by_type: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for pair in pairs:
        by_type[str(pair[0].get("question_type") or "unknown")].append(pair)

    baseline_usage = usage(baseline_paired)
    candidate_usage = usage(candidate_paired)
    return {
        "baseline": str(args.baseline),
        "candidate": str(args.candidate),
        "baseline_rows": len(baseline_rows),
        "candidate_rows": len(candidate_rows),
        "paired_successful_questions": len(pairs),
        "baseline_only_successes": len(set(baseline) - set(candidate)),
        "candidate_only_successes": len(set(candidate) - set(baseline)),
        "metrics": {name: compare_metric(pairs, name) for name in METRICS},
        "by_question_type": {
            name: {
                "questions": len(group),
                "answer_fact_token_recall": compare_metric(group, "answer_fact_token_recall"),
                "gold_answer_token_f1": compare_metric(group, "gold_answer_token_f1"),
                "grounded_answer_proxy": compare_metric(group, "grounded_answer_proxy"),
            }
            for name, group in sorted(by_type.items())
        },
        "router_mode_counts": {
            "baseline": dict(Counter(str(row.get("router", {}).get("mode") or "unknown") for row in baseline_paired)),
            "candidate": dict(Counter(str(row.get("router", {}).get("mode") or "unknown") for row in candidate_paired)),
        },
        "latency_ms": {
            "baseline_avg": average_path(baseline_paired, "total_latency_ms"),
            "candidate_avg": average_path(candidate_paired, "total_latency_ms"),
        },
        "usage": {
            "baseline": baseline_usage,
            "candidate": candidate_usage,
            "candidate_minus_baseline": {
                key: candidate_usage[key] - baseline_usage[key] for key in baseline_usage
            },
        },
        "estimated_cost": {
            "currency": args.currency,
            "baseline": estimated_cost(
                baseline_usage,
                input_per_million=args.input_price,
                cached_input_per_million=args.cached_input_price,
                output_per_million=args.output_price,
            ),
            "candidate": estimated_cost(
                candidate_usage,
                input_per_million=args.input_price,
                cached_input_per_million=args.cached_input_price,
                output_per_million=args.output_price,
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--input-price", type=float)
    parser.add_argument("--cached-input-price", type=float)
    parser.add_argument("--output-price", type=float)
    parser.add_argument("--currency", default="CNY")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.baseline.resolve() == args.output.resolve() or args.candidate.resolve() == args.output.resolve():
        raise SystemExit("output must not overwrite an input")
    result = summarize(load_jsonl(args.baseline), load_jsonl(args.candidate), args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
