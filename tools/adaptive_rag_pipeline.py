#!/usr/bin/env python3
"""Run the adaptive Fast/Quality/Deep RAG controller.

Optimization profile uses qwen-flash for requirement mapping, generation and
conditional claim verification. Validation profile uses qwen-plus. Model names
can be overridden per role without changing the retrieval/evidence input.
"""
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
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.adaptive_rag.controller import (
    GENERATION_SYSTEM_PROMPT,
    AdaptiveRagConfig,
    AdaptiveRagController,
)
from tools.adaptive_rag.requirements import REQUIREMENT_SYSTEM_PROMPT
from tools.adaptive_rag.retrieval import SearchPrincipal, SecondaryRetrievalClient
from tools.adaptive_rag.verifier import VERIFY_SYSTEM_PROMPT
from tools.qwen_plus_rag_pipeline import QwenClient, atomic_write_jsonl, load_jsonl, validate_output_paths

DEFAULT_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts", required=True, type=Path)
    parser.add_argument("--details", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary-output", required=True, type=Path)
    parser.add_argument("--profile", choices=("optimize", "validate"), default="optimize")
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--mapper-model")
    parser.add_argument("--generator-model")
    parser.add_argument("--verifier-model")
    parser.add_argument("--force-mode", choices=("fast", "quality", "deep"))
    parser.add_argument("--map-fast-mode", action="store_true")
    parser.add_argument("--requirements-max-input-chars", type=int, default=48_000)
    parser.add_argument("--requirements-max-count", type=int, default=12)
    parser.add_argument("--requirements-max-selected", type=int, default=16)
    parser.add_argument("--requirements-max-tokens", type=int, default=1_024)
    parser.add_argument("--generation-max-tokens", type=int, default=1_024)
    parser.add_argument("--verifier-mode", choices=("off", "conditional", "always"), default="conditional")
    parser.add_argument("--verifier-max-input-chars", type=int, default=24_000)
    parser.add_argument("--verifier-max-tokens", type=int, default=1_024)
    parser.add_argument("--search-api-url")
    parser.add_argument("--search-api-key-env", default="RAG_SEARCH_API_KEY")
    parser.add_argument("--tenant-id")
    parser.add_argument("--group-id", action="append", default=[])
    parser.add_argument("--classification", action="append", default=[])
    parser.add_argument("--source-type", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--stratified", action="store_true")
    parser.add_argument("--qid-file", type=Path)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.concurrency <= 0:
        parser.error("--concurrency must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    positive = {
        "requirements-max-input-chars": args.requirements_max_input_chars,
        "requirements-max-count": args.requirements_max_count,
        "requirements-max-selected": args.requirements_max_selected,
        "requirements-max-tokens": args.requirements_max_tokens,
        "generation-max-tokens": args.generation_max_tokens,
        "verifier-max-input-chars": args.verifier_max_input_chars,
        "verifier-max-tokens": args.verifier_max_tokens,
        "timeout-seconds": args.timeout_seconds,
    }
    for name, value in positive.items():
        if value <= 0:
            parser.error(f"--{name} must be positive")
    if args.retries < 0:
        parser.error("--retries must not be negative")
    if not 0.0 <= args.temperature <= 2.0:
        parser.error("--temperature must be between 0 and 2")
    if args.search_api_url and not args.tenant_id:
        parser.error("--tenant-id is required when --search-api-url is set")
    return args


def profile_models(args: argparse.Namespace) -> tuple[str, str, str]:
    default = "qwen-flash" if args.profile == "optimize" else "qwen-plus"
    return (
        args.mapper_model or default,
        args.generator_model or default,
        args.verifier_model or default,
    )


def run_signature(args: argparse.Namespace, models: tuple[str, str, str]) -> str:
    payload = {
        "schema_version": 3,
        "contexts_sha256": sha256_file(args.contexts),
        "details_sha256": sha256_file(args.details) if args.details else None,
        "qid_file_sha256": sha256_file(args.qid_file) if args.qid_file else None,
        "profile": args.profile,
        "models": models,
        "force_mode": args.force_mode,
        "stratified": args.stratified,
        "map_fast_mode": args.map_fast_mode,
        "verifier_mode": args.verifier_mode,
        "requirements": {
            "max_input_chars": args.requirements_max_input_chars,
            "max_count": args.requirements_max_count,
            "max_selected": args.requirements_max_selected,
            "max_tokens": args.requirements_max_tokens,
        },
        "generation_max_tokens": args.generation_max_tokens,
        "verifier": {
            "max_input_chars": args.verifier_max_input_chars,
            "max_tokens": args.verifier_max_tokens,
        },
        "search_api_url": args.search_api_url,
        "tenant_id": args.tenant_id,
        "group_ids": sorted(args.group_id),
        "classifications": sorted(args.classification),
        "source_types": sorted(args.source_type),
        "temperature": args.temperature,
        "prompts": {
            "requirements_sha256": sha256_text(REQUIREMENT_SYSTEM_PROMPT),
            "generation_sha256": sha256_text(GENERATION_SYSTEM_PROMPT),
            "verifier_sha256": sha256_text(VERIFY_SYSTEM_PROMPT),
        },
        "adaptive_implementation_sha256": adaptive_implementation_sha256(),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def adaptive_implementation_sha256() -> str:
    digest = hashlib.sha256()
    paths = [
        Path(__file__),
        PROJECT_ROOT / "tools" / "qwen_plus_rag_pipeline.py",
    ] + sorted((PROJECT_ROOT / "tools" / "adaptive_rag").glob("*.py"))
    for path in paths:
        digest.update(str(path.relative_to(PROJECT_ROOT)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def sha256_file(path: Path | None) -> str | None:
    if path is None:
        return None
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_qids(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload if isinstance(payload, list) else payload.get("qids")
    if not isinstance(values, list):
        raise ValueError("qid file must contain a JSON array or {\"qids\": [...]} object")
    return {str(value) for value in values}


def select_rows(
    rows: list[dict[str, Any]],
    qids: set[str] | None,
    limit: int | None,
    *,
    stratified: bool,
) -> list[dict[str, Any]]:
    selected = [row for row in rows if qids is None or str(row.get("qid") or row.get("id")) in qids]
    if limit is None or limit >= len(selected):
        return selected
    if not stratified:
        return selected[:limit]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        grouped[str(row.get("question_type") or "unknown")].append(row)
    output: list[dict[str, Any]] = []
    offsets = {name: 0 for name in grouped}
    names = sorted(grouped)
    while len(output) < limit:
        progressed = False
        for name in names:
            offset = offsets[name]
            if offset >= len(grouped[name]):
                continue
            output.append(grouped[name][offset])
            offsets[name] += 1
            progressed = True
            if len(output) >= limit:
                break
        if not progressed:
            break
    return output


def load_details(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    return {
        str(row.get("question_id") or row.get("qid") or row.get("id")): row
        for row in load_jsonl(path)
    }


def load_resume(path: Path, signature: str) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = load_jsonl(path)
    mismatched = [row.get("qid") for row in rows if row.get("run_signature") != signature]
    if mismatched:
        raise ValueError(f"resume output belongs to another input/configuration: {mismatched[:5]}")
    return {str(row.get("qid")): row for row in rows if row.get("qid") and not row.get("error")}


def row_principal(row: dict[str, Any], args: argparse.Namespace) -> SearchPrincipal | None:
    principal = row.get("principal") or {}
    tenant = str(principal.get("tenant_id") or args.tenant_id or "")
    groups = principal.get("group_ids") or args.group_id
    classifications = principal.get("classifications") or args.classification
    source_types = principal.get("source_types") or args.source_type
    if not tenant:
        return None
    return SearchPrincipal(
        tenant_id=tenant,
        group_ids=tuple(str(value) for value in groups),
        classifications=tuple(str(value) for value in classifications),
        source_types=tuple(str(value) for value in source_types),
    )


def summarize(
    rows: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
    models: tuple[str, str, str],
    signature: str,
    wall_time: float,
) -> dict[str, Any]:
    successful = [row for row in rows if not row.get("error")]
    evaluable = [row for row in successful if (row.get("metrics") or {}).get("is_answer_evaluable")]
    unanswerable = [row for row in successful if (row.get("metrics") or {}).get("is_unanswerable")]

    def avg(path: tuple[str, ...], selected: list[dict[str, Any]] = successful) -> float | None:
        values: list[float] = []
        for row in selected:
            value: Any = row
            for key in path:
                if not isinstance(value, dict):
                    value = None
                    break
                value = value.get(key)
            if isinstance(value, bool):
                values.append(float(value))
            elif isinstance(value, (int, float)) and math.isfinite(float(value)):
                values.append(float(value))
        return statistics.fmean(values) if values else None

    # Billing includes failed attempts too. Error rows preserve QwenRequestError
    # usage, so aggregate over every row rather than silently undercounting.
    usage = {
        key: sum(int((row.get("usage") or {}).get(key) or 0) for row in rows)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens")
    }
    failed_usage = {
        key: sum(
            int((row.get("usage") or {}).get(key) or 0)
            for row in rows
            if row.get("error")
        )
        for key in ("prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens")
    }
    route_counts = Counter((row.get("router") or {}).get("mode") or "unknown" for row in successful)
    verifier_counts = Counter((row.get("verification") or {}).get("status") or "unknown" for row in successful)
    secondary_attempted = [row for row in successful if (row.get("secondary_retrieval") or {}).get("attempted")]
    summary = {
        "input": str(args.contexts),
        "details": str(args.details) if args.details else None,
        "output": str(args.output),
        "run_signature": signature,
        "profile": args.profile,
        "models": {"mapper": models[0], "generator": models[1], "verifier": models[2]},
        "questions_total": len(rows),
        "questions_successful": len(successful),
        "questions_errored": len(rows) - len(successful),
        "router_mode_counts": dict(sorted(route_counts.items())),
        "secondary_retrieval_attempted": len(secondary_attempted),
        "secondary_retrieval_added_contexts": sum(
            int((row.get("secondary_retrieval") or {}).get("added_contexts") or 0)
            for row in successful
        ),
        "verifier_status_counts": dict(sorted(verifier_counts.items())),
        "generation": {
            "answer_fact_token_recall_avg": avg(("metrics", "answer_fact_token_recall"), evaluable),
            "answer_fact_coverage_proxy_avg": avg(("metrics", "answer_fact_coverage_proxy"), evaluable),
            "context_fact_token_recall_upper_bound_avg": avg(
                ("metrics", "context_fact_token_recall_upper_bound"), evaluable
            ),
            "context_fact_coverage_proxy_avg": avg(
                ("metrics", "context_fact_coverage_proxy"), evaluable
            ),
            "gold_answer_token_f1_avg": avg(("metrics", "gold_answer_token_f1"), evaluable),
            "gold_answer_token_recall_avg": avg(("metrics", "gold_answer_token_recall"), evaluable),
            "citation_coverage_avg": avg(("metrics", "citation_coverage")),
            "citation_precision_avg": avg(("metrics", "citation_precision")),
            "gold_doc_citation_rate": avg(
                ("metrics", "gold_doc_cited"),
                [row for row in successful if row.get("expected_doc_ids")],
            ),
            "grounded_answer_proxy_rate": avg(("metrics", "grounded_answer_proxy"), evaluable),
            "unanswerable_abstain_accuracy": avg(
                ("metrics", "unanswerable_abstain_correct"), unanswerable
            ),
        },
        "requirements": {
            "coverage_avg": avg(("requirements", "coverage")),
            "missing_avg": avg(("requirements", "missing_count")),
        },
        "budget": {
            "prompt_chars_avg": avg(("budget", "rendered_chars")),
            "context_count_avg": avg(("budget", "context_count")),
            "expanded_rate": avg(("budget", "expanded_to_maximum")),
        },
        "usage": usage,
        "failed_attempt_usage": failed_usage,
        "avg_total_latency_ms": avg(("total_latency_ms",)),
        "wall_time_seconds": wall_time,
        "configuration": {
            "force_mode": args.force_mode,
            "stratified": args.stratified,
            "map_fast_mode": args.map_fast_mode,
            "verifier_mode": args.verifier_mode,
            "requirements_max_input_chars": args.requirements_max_input_chars,
            "requirements_max_count": args.requirements_max_count,
            "requirements_max_selected": args.requirements_max_selected,
            "requirements_max_tokens": args.requirements_max_tokens,
            "generation_max_tokens": args.generation_max_tokens,
            "verifier_max_input_chars": args.verifier_max_input_chars,
            "verifier_max_tokens": args.verifier_max_tokens,
            "concurrency": args.concurrency,
            "temperature": args.temperature,
            "search_api_enabled": bool(args.search_api_url),
        },
    }
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in successful:
        grouped[(row.get("router") or {}).get("mode") or "unknown"].append(row)
    summary["by_router_mode"] = {
        mode: {
            "questions": len(group),
            "answer_fact_token_recall_avg": avg(("metrics", "answer_fact_token_recall"), [
                row for row in group if (row.get("metrics") or {}).get("is_answer_evaluable")
            ]),
            "grounded_answer_proxy_rate": avg(("metrics", "grounded_answer_proxy"), [
                row for row in group if (row.get("metrics") or {}).get("is_answer_evaluable")
            ]),
            "avg_total_latency_ms": avg(("total_latency_ms",), group),
        }
        for mode, group in sorted(grouped.items())
    }
    return summary


def main() -> int:
    args = parse_args()
    validate_output_paths(
        [value for value in (args.contexts, args.details, args.qid_file) if value is not None],
        [args.output, args.summary_output],
    )
    api_key = os.getenv(args.api_key_env, "")
    if not api_key:
        raise SystemExit(f"missing API key environment variable: {args.api_key_env}")
    models = profile_models(args)
    signature = run_signature(args, models)
    contexts = select_rows(
        load_jsonl(args.contexts),
        load_qids(args.qid_file),
        args.limit,
        stratified=args.stratified,
    )
    details = load_details(args.details)
    resumed = load_resume(args.output, signature) if args.resume else {}
    secondary = None
    if args.search_api_url:
        secondary = SecondaryRetrievalClient(
            api_url=args.search_api_url,
            api_key=os.getenv(args.search_api_key_env, ""),
            timeout_seconds=args.timeout_seconds,
        )
    controller = AdaptiveRagController(
        mapper_client=QwenClient(
            api_base=args.api_base,
            api_key=api_key,
            model=models[0],
            timeout_seconds=args.timeout_seconds,
            retries=args.retries,
        ),
        generator_client=QwenClient(
            api_base=args.api_base,
            api_key=api_key,
            model=models[1],
            timeout_seconds=args.timeout_seconds,
            retries=args.retries,
        ),
        verifier_client=QwenClient(
            api_base=args.api_base,
            api_key=api_key,
            model=models[2],
            timeout_seconds=args.timeout_seconds,
            retries=args.retries,
        ),
        secondary_retrieval=secondary,
        config=AdaptiveRagConfig(
            map_fast_mode=args.map_fast_mode,
            requirements_max_chars=args.requirements_max_input_chars,
            requirements_max_count=args.requirements_max_count,
            requirements_max_selected=args.requirements_max_selected,
            requirements_max_tokens=args.requirements_max_tokens,
            generation_max_tokens=args.generation_max_tokens,
            verifier_mode=args.verifier_mode,
            verifier_max_chars=args.verifier_max_input_chars,
            verifier_max_tokens=args.verifier_max_tokens,
            temperature=args.temperature,
        ),
    )
    started = time.perf_counter()
    lock = threading.Lock()
    completed = 0
    order = {str(row.get("qid") or row.get("id")): index for index, row in enumerate(contexts)}
    results: list[dict[str, Any]] = []

    def evaluate(row: dict[str, Any]) -> dict[str, Any]:
        qid = str(row.get("qid") or row.get("id"))
        if qid in resumed:
            return resumed[qid]
        result = controller.process(
            row,
            details=details.get(qid),
            principal=row_principal(row, args),
            forced_mode=args.force_mode,
        )
        result["run_signature"] = signature
        return result

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(evaluate, row) for row in contexts]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            with lock:
                results.append(result)
                completed += 1
                if args.progress_every > 0 and (
                    completed % args.progress_every == 0 or completed == len(contexts)
                ):
                    print(json.dumps({
                        "event": "adaptive_rag_progress",
                        "completed": completed,
                        "total": len(contexts),
                        "errors": sum(bool(value.get("error")) for value in results),
                    }), flush=True)
                    atomic_write_jsonl(
                        args.output,
                        sorted(results, key=lambda value: order.get(str(value.get("qid")), 10**9)),
                    )
    results.sort(key=lambda value: order.get(str(value.get("qid")), 10**9))
    atomic_write_jsonl(args.output, results)
    wall_time = time.perf_counter() - started
    summary = summarize(results, args=args, models=models, signature=signature, wall_time=wall_time)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
