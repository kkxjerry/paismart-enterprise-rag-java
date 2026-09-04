#!/usr/bin/env python3
"""Reference-based blind A/B judge for two RAG answer JSONL files."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import statistics
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.qwen_plus_rag_pipeline import (  # noqa: E402
    DEFAULT_API_BASE,
    DEFAULT_MODEL,
    PipelineError,
    QwenClient,
    atomic_write_jsonl,
    load_jsonl,
    validate_output_paths,
)

SYSTEM = """Blindly evaluate Answer A and Answer B against the reference answer and required facts.
Do not prefer the first or longer answer. Score each independently from 0 to 10 for correctness,
completeness, and directness. Exact names, dates, numbers, conditions, exceptions, and conflicts matter.
A refusal is correct only when the reference says the information is unavailable.
Return JSON only:
{"A":{"correctness":0,"completeness":0,"directness":0},
 "B":{"correctness":0,"completeness":0,"directness":0},
 "winner":"A"}
Winner must be A, B, or tie; correctness and completeness dominate directness."""


def validate_judgement(payload: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for label in ("A", "B"):
        node = payload.get(label)
        if not isinstance(node, dict):
            raise PipelineError(f"missing {label} scores")
        scores: dict[str, float] = {}
        for metric in ("correctness", "completeness", "directness"):
            value = node.get(metric)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise PipelineError(f"invalid {label}.{metric}")
            value = float(value)
            if not math.isfinite(value) or not 0 <= value <= 10:
                raise PipelineError(f"out-of-range {label}.{metric}")
            scores[metric] = value
        result[label] = scores
    winner = str(payload.get("winner") or "")
    if winner not in {"A", "B", "tie"}:
        raise PipelineError("winner must be A, B, or tie")
    result["winner"] = winner
    return result


def load_pairs(baseline: Path, candidate: Path, include_unanswerable: bool) -> list[dict[str, Any]]:
    left = {str(row.get("qid")): row for row in load_jsonl(baseline)}
    right = {str(row.get("qid")): row for row in load_jsonl(candidate)}
    if set(left) != set(right):
        raise ValueError("baseline and candidate qids differ")
    pairs = []
    for qid in sorted(left):
        a, b = left[qid], right[qid]
        if a.get("error") or b.get("error"):
            continue
        metrics = a.get("metrics") or {}
        eligible = bool(metrics.get("is_answer_evaluable"))
        eligible = eligible or (include_unanswerable and bool(metrics.get("is_unanswerable")))
        if eligible:
            pairs.append({"qid": qid, "baseline": a, "candidate": b})
    return pairs


def stratified_select(pairs: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    if not limit or limit >= len(pairs):
        return pairs
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        groups[str(pair["baseline"].get("question_type") or "unknown")].append(pair)
    names = sorted(groups)
    offsets = {name: 0 for name in names}
    selected = []
    while len(selected) < limit:
        changed = False
        for name in names:
            offset = offsets[name]
            if offset < len(groups[name]):
                selected.append(groups[name][offset])
                offsets[name] += 1
                changed = True
                if len(selected) == limit:
                    break
        if not changed:
            break
    return selected


def blinded_order(qid: str) -> tuple[str, str]:
    return ("baseline", "candidate") if hashlib.sha256(qid.encode()).digest()[0] % 2 == 0 else ("candidate", "baseline")


def answer(row: dict[str, Any]) -> str:
    node = row.get("generation") or {}
    return str(node.get("answer") or "") + "\nCitations: " + json.dumps(node.get("citations") or [])


def build_messages(pair: dict[str, Any]) -> tuple[list[dict[str, str]], dict[str, str]]:
    first, second = blinded_order(str(pair["qid"]))
    labels = {"A": first, "B": second}
    reference = pair["baseline"]
    prompt = (
        f"Question:\n{reference.get('question') or ''}\n\n"
        f"Reference answer:\n{reference.get('gold_answer') or ''}\n\n"
        f"Required facts:\n{json.dumps(reference.get('answer_facts') or [], ensure_ascii=False)}\n\n"
        f"Answer A:\n{answer(pair[first])}\n\nAnswer B:\n{answer(pair[second])}"
    )
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": prompt},
    ], labels


def judge(pair: dict[str, Any], client: QwenClient, max_tokens: int) -> dict[str, Any]:
    messages, labels = build_messages(pair)
    reference = pair["baseline"]
    started = time.perf_counter()
    try:
        api = client.complete_json(
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.0,
            validator=validate_judgement,
        )
        mapped_scores = {pipeline: api.value[label] for label, pipeline in labels.items()}
        winner = api.value["winner"]
        return {
            "qid": pair["qid"],
            "question_type": reference.get("question_type"),
            "labels": labels,
            "winner": "tie" if winner == "tie" else labels[winner],
            "scores": mapped_scores,
            "latency_ms": api.latency_ms,
            "usage": api.usage,
            "attempts": api.attempts,
            "error": None,
        }
    except Exception as exc:
        return {
            "qid": pair["qid"],
            "question_type": reference.get("question_type"),
            "labels": labels,
            "winner": None,
            "scores": {},
            "latency_ms": float(
                getattr(exc, "latency_ms", (time.perf_counter() - started) * 1000)
            ),
            "usage": dict(getattr(exc, "usage", {})),
            "attempts": int(getattr(exc, "attempts", 0)),
            "error": str(exc),
        }


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def summary(rows: list[dict[str, Any]], args: argparse.Namespace, eligible: int) -> dict[str, Any]:
    ok = [row for row in rows if not row.get("error")]
    scores: dict[str, Any] = {}
    for side in ("baseline", "candidate"):
        scores[side] = {
            metric: mean([float(row["scores"][side][metric]) for row in ok])
            for metric in ("correctness", "completeness", "directness")
        }
        scores[side]["primary_mean"] = mean([
            (row["scores"][side]["correctness"] + row["scores"][side]["completeness"]) / 2
            for row in ok
        ])
    by_type = {}
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ok:
        groups[str(row.get("question_type") or "unknown")].append(row)
    for name, group in sorted(groups.items()):
        by_type[name] = {
            "questions": len(group),
            "baseline_wins": sum(row["winner"] == "baseline" for row in group),
            "candidate_wins": sum(row["winner"] == "candidate" for row in group),
            "ties": sum(row["winner"] == "tie" for row in group),
        }
    return {
        "baseline": str(args.baseline),
        "candidate": str(args.candidate),
        "judge_model": args.model,
        "eligible_pairs": eligible,
        "selected_pairs": len(rows),
        "successful_pairs": len(ok),
        "errored_pairs": len(rows) - len(ok),
        "wins": {name: sum(row["winner"] == name for row in ok) for name in ("baseline", "candidate", "tie")},
        "scores": scores,
        "usage": {
            key: sum(int((row.get("usage") or {}).get(key) or 0) for row in rows)
            for key in ("prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens")
        },
        "failed_attempt_usage": {
            key: sum(
                int((row.get("usage") or {}).get(key) or 0)
                for row in rows
                if row.get("error")
            )
            for key in ("prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens")
        },
        "by_question_type": by_type,
        "note": "Blind reference-based Qwen Plus judge; same-model-family bias is possible.",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary-output", required=True, type=Path)
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--stratified", action="store_true")
    parser.add_argument("--include-unanswerable", action="store_true")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timeout-seconds", type=float, default=120)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--progress-every", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validate_output_paths(
        [args.baseline, args.candidate],
        [args.output, args.summary_output],
    )
    api_key = os.getenv(args.api_key_env, "")
    if not api_key:
        raise SystemExit(f"missing environment variable {args.api_key_env}")
    eligible = load_pairs(args.baseline, args.candidate, args.include_unanswerable)
    selected = stratified_select(eligible, args.limit)
    client = QwenClient(
        api_base=args.api_base,
        api_key=api_key,
        model=args.model,
        timeout_seconds=args.timeout_seconds,
        retries=args.retries,
    )
    ordering = {pair["qid"]: index for index, pair in enumerate(selected)}
    rows: list[dict[str, Any]] = []
    lock = threading.Lock()
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(judge, pair, client, args.max_tokens) for pair in selected]
        for completed, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            with lock:
                rows.append(future.result())
                if args.progress_every and (completed % args.progress_every == 0 or completed == len(selected)):
                    print(json.dumps({
                        "event": "qwen_plus_pair_judge_progress",
                        "completed": completed,
                        "total": len(selected),
                        "qps": round(completed / max(time.perf_counter() - started, 0.001), 3),
                        "errors": sum(bool(row.get("error")) for row in rows),
                    }), flush=True)
    rows.sort(key=lambda row: ordering[str(row["qid"])])
    atomic_write_jsonl(args.output, rows)
    result = summary(rows, args, len(eligible))
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
