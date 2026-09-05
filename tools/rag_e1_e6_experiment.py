#!/usr/bin/env python3
"""Unified E1-E6 RAG experiment over fixed authorized evidence.

Offline arms isolate evidence changes. Live arms use the same Qwen model, budget,
question set and zero-temperature setting. Gold labels enter only post-selection
metrics. E3 index export and E5 late-chunking capability are executed by their
own tools and linked through the final run registry.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.adaptive_rag.budget import BudgetDecision, DynamicEvidenceBudget
from tools.adaptive_rag.canonical_generation import generate_canonical_answer
from tools.adaptive_rag.claims import split_cited_segments
from tools.adaptive_rag.controller import build_generation_messages, validate_adaptive_generation
from tools.adaptive_rag.hierarchical_evidence import (
    DEFAULT_HIERARCHY_CONFIG,
    HierarchyConfig,
    inferred_route,
    pack_hierarchical_evidence,
    requires_canonical_document,
    summary_tree,
)
from tools.adaptive_rag.requirements import deterministic_requirement_plan
from tools.qwen_plus_rag_pipeline import QwenClient, fact_scores, load_jsonl, score_result, token_f1, token_recall
from tools.rag_evidence_experiment import choose
from tools.rag_packing_loop10 import CONDITION_RE, NEGATION_RE, exact_recall, fact_coverage, mean, subset_coverage

OFFLINE_ARMS = (
    "legacy",
    "query-spans-v3",
    "e1-leaf-parent",
    "e4-context-prefix",
    "e5-proposition",
    "e6-routed-global",
)
LOCAL_LIVE_ARMS = ("query-spans-flat", "e5-flat", "e2-canonical")
GLOBAL_LIVE_ARMS = ("query-spans-flat", "e6-global-structured")
_CITATION_RE = re.compile(r"\[?(S[1-9][0-9]*)\]?")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_metadata() -> dict[str, Any]:
    return {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        "dirty": subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True),
    }


def hierarchy_config(arm: str, row: dict[str, Any]) -> HierarchyConfig:
    if arm == "e1-leaf-parent":
        return replace(DEFAULT_HIERARCHY_CONFIG, leaf_mode="sentence", contextual_prefix=False, route_mode="local")
    if arm == "e4-context-prefix":
        return replace(DEFAULT_HIERARCHY_CONFIG, leaf_mode="sentence", contextual_prefix=True, route_mode="local")
    if arm in {"e5-proposition", "e5-flat", "e2-canonical"}:
        return replace(DEFAULT_HIERARCHY_CONFIG, leaf_mode="proposition", contextual_prefix=True, route_mode="local")
    if arm in {"e6-routed-global", "e6-global-structured"}:
        return replace(
            DEFAULT_HIERARCHY_CONFIG,
            leaf_mode="proposition",
            contextual_prefix=True,
            route_mode=inferred_route(str(row.get("question") or ""), str(row.get("question_type") or "")),
        )
    raise ValueError(f"unsupported hierarchy arm: {arm}")


def pack_arm(row: dict[str, Any], arm: str, *, max_chars: int, max_contexts: int) -> dict[str, Any]:
    question = str(row.get("question") or "")
    contexts = list(row.get("contexts") or [])
    if arm in {"legacy", "query-spans-v3", "query-spans-flat"}:
        strategy = "legacy" if arm == "legacy" else "query-spans"
        plan = deterministic_requirement_plan(question)
        evidence = DynamicEvidenceBudget(strategy).build(
            contexts,
            plan=plan,
            decision=BudgetDecision("fast", max_chars, max_chars, max_contexts, 2, ("e1-e6-fixed",)),
            question=question,
        )
        return {
            "rendered": evidence.rendered,
            "contexts": list(evidence.contexts),
            "metadata": evidence.to_dict(),
            "canonical_documents": [],
            "summary_tree": [],
        }
    config = hierarchy_config(arm, row)
    evidence = pack_hierarchical_evidence(
        contexts,
        question=question,
        max_chars=max_chars,
        max_contexts=max_contexts,
        config=config,
    )
    return {
        "rendered": evidence.rendered,
        "contexts": list(evidence.contexts),
        "metadata": evidence.to_dict(),
        "canonical_documents": list(evidence.canonical_documents),
        "summary_tree": summary_tree(contexts) if config.route_mode == "global" else [],
        "hierarchical_evidence": evidence,
    }


def score_prompt(row: dict[str, Any], packed: dict[str, Any]) -> dict[str, Any]:
    contexts = packed["contexts"]
    text = "\n".join(str(value.get("text") or "") for value in contexts)
    facts = [str(value) for value in row.get("answer_facts") or [] if str(value).strip()]
    expected = {
        str(value)
        for value in (row.get("expected_accessible_doc_ids") or row.get("expected_doc_ids") or [])
        if str(value)
    }
    selected_docs = {str(value.get("doc_id") or "") for value in contexts if value.get("doc_id")}
    canonical_docs = packed.get("canonical_documents") or []
    canonical_doc = str(canonical_docs[0].get("doc_id") or "") if canonical_docs else ""
    lexical = mean(token_recall(text, fact) for fact in facts)
    return {
        "qid": str(row.get("qid") or row.get("id") or ""),
        "question_type": str(row.get("question_type") or "unknown"),
        "source_types": list(row.get("source_types") or []),
        "evaluable": bool(facts) and row.get("question_type") != "info_not_found",
        "prompt_lexical_recall": lexical,
        "prompt_requirement_evidence_coverage": fact_coverage(text, facts),
        "prompt_exact_value_recall": exact_recall(text, facts),
        "prompt_list_item_recall": fact_coverage(text, facts) if len(facts) >= 5 else None,
        "prompt_condition_exception_recall": subset_coverage(text, facts, lambda fact: bool(CONDITION_RE.search(fact))),
        "prompt_negation_recall": subset_coverage(text, facts, lambda fact: bool(NEGATION_RE.search(fact))),
        "gold_doc_retained": float(bool(expected & selected_docs)) if expected else None,
        "canonical_eligible": requires_canonical_document(str(row.get("question") or "")),
        "canonical_doc_expected": float(bool(expected and canonical_doc in expected)) if canonical_doc else None,
        "canonical_near_ties": sum(value.get("role") == "near_tie" for value in canonical_docs),
        "rendered_chars": len(packed["rendered"]),
        "context_count": len(contexts),
        "selected_doc_count": len(selected_docs),
        "selected_citations": [str(value.get("citation_id") or "") for value in contexts],
        "selected_doc_ids": sorted(selected_docs),
        "canonical_doc_id": canonical_doc or None,
    }


def aggregate_prompt(rows: Sequence[dict[str, Any]], baseline: dict[str, dict[str, Any]]) -> dict[str, Any]:
    evaluable = [row for row in rows if row["evaluable"]]
    canonical = [row for row in rows if row["canonical_eligible"] and row.get("canonical_doc_expected") is not None]
    deltas = []
    for row in evaluable:
        before = baseline[row["qid"]].get("prompt_lexical_recall")
        after = row.get("prompt_lexical_recall")
        if isinstance(before, (int, float)) and isinstance(after, (int, float)):
            deltas.append(after - before)
    return {
        "questions": len(rows),
        "evaluable": len(evaluable),
        "prompt_lexical_recall": mean(row.get("prompt_lexical_recall") for row in evaluable),
        "prompt_requirement_evidence_coverage": mean(row.get("prompt_requirement_evidence_coverage") for row in evaluable),
        "prompt_exact_value_recall": mean(row.get("prompt_exact_value_recall") for row in evaluable),
        "prompt_list_item_recall": mean(row.get("prompt_list_item_recall") for row in evaluable),
        "prompt_condition_exception_recall": mean(row.get("prompt_condition_exception_recall") for row in evaluable),
        "prompt_negation_recall": mean(row.get("prompt_negation_recall") for row in evaluable),
        "gold_doc_retained_rate": mean(row.get("gold_doc_retained") for row in rows),
        "canonical_source_selection_accuracy": mean(row.get("canonical_doc_expected") for row in canonical),
        "canonical_eligible_questions": len(canonical),
        "mean_canonical_near_ties": mean(row.get("canonical_near_ties") for row in canonical),
        "mean_rendered_chars": mean(row.get("rendered_chars") for row in rows),
        "mean_context_count": mean(row.get("context_count") for row in rows),
        "mean_selected_doc_count": mean(row.get("selected_doc_count") for row in rows),
        "paired_wins": sum(value > 1e-12 for value in deltas),
        "paired_regressions": sum(value < -1e-12 for value in deltas),
        "paired_ties": sum(abs(value) <= 1e-12 for value in deltas),
        "packing_regression_rate": sum(value < -1e-12 for value in deltas) / len(deltas) if deltas else None,
        "packing_severe_regression_rate": sum(value <= -0.10 for value in deltas) / len(deltas) if deltas else None,
    }


def offline(args: argparse.Namespace) -> dict[str, Any]:
    rows = load_jsonl(args.contexts)
    arms = tuple(args.arm or OFFLINE_ARMS)
    records: dict[str, list[dict[str, Any]]] = {arm: [] for arm in arms}
    baseline: dict[str, dict[str, Any]] = {}
    output_path = args.output_dir / "offline-records.jsonl"
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with output_path.open("x", encoding="utf-8") as stream:
        for index, row in enumerate(rows, start=1):
            for arm in arms:
                packed = pack_arm(row, arm, max_chars=args.max_chars, max_contexts=args.max_contexts)
                scored = {"arm": arm, **score_prompt(row, packed)}
                records[arm].append(scored)
                if arm == "legacy":
                    baseline[scored["qid"]] = scored
                stream.write(json.dumps(scored, ensure_ascii=False) + "\n")
            if index % 100 == 0:
                print(f"OFFLINE {index}/{len(rows)}", flush=True)
    if "legacy" not in records:
        legacy = []
        for row in rows:
            value = score_prompt(row, pack_arm(row, "legacy", max_chars=args.max_chars, max_contexts=args.max_contexts))
            legacy.append(value)
            baseline[value["qid"]] = value
    aggregate = {arm: aggregate_prompt(values, baseline) for arm, values in records.items()}
    by_type: dict[str, dict[str, Any]] = {}
    question_types = sorted({value["question_type"] for values in records.values() for value in values})
    for question_type in question_types:
        by_type[question_type] = {
            arm: aggregate_prompt([value for value in values if value["question_type"] == question_type], baseline)
            for arm, values in records.items()
        }
    by_source: dict[str, dict[str, Any]] = {}
    sources = sorted({source for values in records.values() for value in values for source in value["source_types"]})
    for source in sources:
        by_source[source] = {
            arm: aggregate_prompt([value for value in values if source in value["source_types"]], baseline)
            for arm, values in records.items()
        }
    result = {
        "schema_version": 1,
        "phase": "offline",
        "metadata": {
            **git_metadata(),
            "input": str(args.contexts),
            "input_sha256": sha256_file(args.contexts),
            "runner_sha256": sha256_file(Path(__file__)),
            "questions": len(rows),
            "max_chars": args.max_chars,
            "max_contexts": args.max_contexts,
            "arms": list(arms),
            "gold_used_for_selection": False,
        },
        "aggregates": aggregate,
        "by_question_type": by_type,
        "by_source_type": by_source,
    }
    (args.output_dir / "offline-summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def generate_flat(client: QwenClient, row: dict[str, Any], packed: dict[str, Any], *, max_tokens: int) -> tuple[dict[str, Any], ApiResultLike]:
    question = str(row.get("question") or "")
    plan = deterministic_requirement_plan(question)
    citations = {str(value.get("citation_id") or "") for value in packed["contexts"]}
    api = client.complete_json(
        messages=build_generation_messages(question=question, plan=plan, rendered_contexts=packed["rendered"]),
        max_tokens=max_tokens,
        temperature=0.0,
        validator=lambda payload: validate_adaptive_generation(payload, valid_citations=citations, requirement_ids={"R1"}),
    )
    return dict(api.value), api


class ApiResultLike:
    value: dict[str, Any]
    latency_ms: float
    usage: dict[str, int]
    request_id: str
    returned_model: str
    attempts: int


def score_live_answer(row: dict[str, Any], packed: dict[str, Any], generation: dict[str, Any], api: ApiResultLike) -> dict[str, Any]:
    answer = str(generation.get("answer") or "")
    citations = [str(value) for value in generation.get("citations") or []]
    contexts = packed["contexts"]
    base = score_result(
        row,
        selected_contexts=contexts,
        answerable=bool(generation.get("answerable")),
        answer=answer,
        citations=citations,
    )
    facts = [str(value) for value in row.get("answer_facts") or [] if str(value).strip()]
    expected = {
        str(value)
        for value in (row.get("expected_accessible_doc_ids") or row.get("expected_doc_ids") or [])
        if str(value)
    }
    citation_map = {str(value.get("citation_id") or ""): value for value in contexts}
    cited_docs = {
        str(citation_map[value].get("doc_id") or "") for value in citations if value in citation_map
    }
    canonical_eligible = requires_canonical_document(str(row.get("question") or ""))
    lexical_precision, lexical_recall, unsupported = citation_support_proxy(answer, citations, citation_map)
    return {
        "answerable": bool(generation.get("answerable")),
        "answer": answer,
        "citations": citations,
        "answer_lexical_recall": base.get("answer_fact_token_recall"),
        "answer_fact_coverage": base.get("answer_fact_coverage_proxy"),
        "gold_answer_f1": base.get("gold_answer_token_f1"),
        "answer_exact_value_accuracy": exact_recall(answer, facts),
        "requirement_completion": fact_coverage(answer, facts),
        "answer_list_completeness": fact_coverage(answer, facts) if len(facts) >= 5 else None,
        "answer_condition_accuracy": subset_coverage(answer, facts, lambda fact: bool(CONDITION_RE.search(fact))),
        "answer_negation_accuracy": subset_coverage(answer, facts, lambda fact: bool(NEGATION_RE.search(fact))),
        "citation_id_validity": base.get("citation_precision"),
        "gold_doc_cited": float(bool(expected & cited_docs)) if expected else None,
        "source_contamination_proxy": float(bool(canonical_eligible and expected and cited_docs - expected)),
        "cited_doc_ids": sorted(cited_docs),
        "citation_lexical_support_precision_proxy": lexical_precision,
        "citation_lexical_support_recall_proxy": lexical_recall,
        "unsupported_claim_rate_proxy": unsupported,
        "latency_ms": api.latency_ms,
        "usage": dict(api.usage),
        "request_id": api.request_id,
        "returned_model": api.returned_model,
        "attempts": api.attempts,
    }


def citation_support_proxy(
    answer: str,
    citations: Sequence[str],
    contexts: dict[str, dict[str, Any]],
) -> tuple[float | None, float | None, float | None]:
    del citations
    claims = []
    supported = 0
    valid_claims = 0
    for segment in split_cited_segments(answer):
        ids = _CITATION_RE.findall(segment)
        claim = _CITATION_RE.sub("", segment).strip()
        if not re.search(r"[A-Za-z0-9\u3400-\u9fff]", claim):
            continue
        claims.append((claim, ids))
        if not ids:
            continue
        evidence = "\n".join(str(contexts[value].get("text") or "") for value in ids if value in contexts)
        if evidence:
            valid_claims += 1
            supported += token_recall(evidence, claim) >= 0.35
    if not claims:
        return None, None, None
    precision = supported / valid_claims if valid_claims else 0.0
    recall = supported / len(claims)
    return precision, recall, 1.0 - recall


def aggregate_live(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if not row.get("error")]
    evaluable = [row for row in valid if row.get("evaluable")]
    metrics = (
        "prompt_lexical_recall",
        "prompt_requirement_evidence_coverage",
        "prompt_exact_value_recall",
        "prompt_list_item_recall",
        "prompt_condition_exception_recall",
        "prompt_negation_recall",
        "answer_lexical_recall",
        "answer_fact_coverage",
        "gold_answer_f1",
        "answer_exact_value_accuracy",
        "requirement_completion",
        "answer_list_completeness",
        "answer_condition_accuracy",
        "answer_negation_accuracy",
        "citation_id_validity",
        "gold_doc_cited",
        "source_contamination_proxy",
        "citation_lexical_support_precision_proxy",
        "citation_lexical_support_recall_proxy",
        "unsupported_claim_rate_proxy",
    )
    result = {name: mean(row.get(name) for row in evaluable) for name in metrics}
    latencies = [float(row["latency_ms"]) for row in valid if isinstance(row.get("latency_ms"), (int, float))]
    result.update(
        {
            "questions": len(rows),
            "successful": len(valid),
            "errors": len(rows) - len(valid),
            "evaluable": len(evaluable),
            "answerable": sum(bool(row.get("answerable")) for row in valid),
            "abstentions": sum(not bool(row.get("answerable")) for row in valid),
            "model_calls": sum(int(row.get("attempts") or 0) for row in rows),
            "usage": {
                key: sum(int((row.get("usage") or {}).get(key) or 0) for row in rows)
                for key in ("prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens")
            },
            "mean_latency_ms": statistics.fmean(latencies) if latencies else None,
            "p50_latency_ms": percentile(latencies, 0.50),
            "p95_latency_ms": percentile(latencies, 0.95),
            "failed_qids": [row["qid"] for row in rows if row.get("error")],
        }
    )
    return result


def live(args: argparse.Namespace, *, global_only: bool) -> dict[str, Any]:
    rows = load_jsonl(args.contexts)
    if global_only:
        selected_rows = [row for row in rows if row.get("question_type") == "high_level"]
        arms = tuple(args.arm or GLOBAL_LIVE_ARMS)
    else:
        selected_rows = choose(rows, args.limit, args.qid, args.seed)
        arms = tuple(args.arm or LOCAL_LIVE_ARMS)
    if not selected_rows:
        raise ValueError("live experiment selected no questions")
    if len(selected_rows) > args.max_live_questions:
        raise ValueError(f"live experiment exceeds hard cap {args.max_live_questions}")
    api_key = os.getenv(args.api_key_env, "")
    if not api_key:
        raise ValueError(f"missing API key environment variable: {args.api_key_env}")
    client = QwenClient(
        api_base=args.api_base,
        api_key=api_key,
        model=args.model,
        timeout_seconds=args.timeout_seconds,
        retries=args.retries,
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output_path = args.output_dir / "live-records.jsonl"
    records: dict[str, list[dict[str, Any]]] = {arm: [] for arm in arms}
    with output_path.open("x", encoding="utf-8") as output:
        for row_index, row in enumerate(selected_rows):
            ordered = arms if row_index % 2 == 0 else tuple(reversed(arms))
            for arm in ordered:
                qid = str(row.get("qid") or row.get("id") or "")
                record: dict[str, Any] = {
                    "qid": qid,
                    "question": row.get("question"),
                    "question_type": row.get("question_type"),
                    "source_types": row.get("source_types") or [],
                    "arm": arm,
                    "evaluable": bool(row.get("answer_facts")) and row.get("question_type") != "info_not_found",
                    "error": None,
                }
                started = time.perf_counter()
                try:
                    pack_name = (
                        "query-spans-flat"
                        if arm == "query-spans-flat"
                        else "e6-global-structured"
                        if arm == "e6-global-structured"
                        else "e5-flat"
                        if arm == "e5-flat"
                        else "e2-canonical"
                    )
                    packed = pack_arm(row, pack_name, max_chars=args.max_chars, max_contexts=args.max_contexts)
                    record.update(score_prompt(row, packed))
                    record["arm"] = arm
                    record["packing"] = packed["metadata"]
                    if arm in {"e2-canonical", "e6-global-structured"}:
                        hierarchical = packed.get("hierarchical_evidence")
                        if hierarchical is None:
                            raise ValueError("structured generation requires hierarchical evidence")
                        generated = generate_canonical_answer(
                            client,
                            question=str(row.get("question") or ""),
                            evidence=hierarchical,
                            max_tokens=args.max_tokens,
                            temperature=0.0,
                        )
                        generation = {
                            "answerable": generated.answerable,
                            "answer": generated.answer,
                            "citations": list(generated.citations),
                        }
                        api = generated.api
                        record["canonical_generation"] = generated.to_dict()
                    else:
                        generation, api = generate_flat(client, row, packed, max_tokens=args.max_tokens)
                    record.update(score_live_answer(row, packed, generation, api))
                    record["generation"] = generation
                    record["wall_ms"] = (time.perf_counter() - started) * 1000.0
                    print(
                        f"LIVE {qid} {arm} answerable={record['answerable']} "
                        f"tokens={(record.get('usage') or {}).get('total_tokens', 0)}",
                        flush=True,
                    )
                except Exception as exc:
                    record.update(
                        {
                            "error": f"{type(exc).__name__}: {exc}",
                            "usage": dict(getattr(exc, "usage", {})),
                            "attempts": int(getattr(exc, "attempts", 0)),
                            "wall_ms": (time.perf_counter() - started) * 1000.0,
                        }
                    )
                    print(f"ERROR {qid} {arm}: {record['error']}", flush=True)
                records[arm].append(record)
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                output.flush()
    aggregates = {arm: aggregate_live(values) for arm, values in records.items()}
    result = {
        "schema_version": 1,
        "phase": "live-global" if global_only else "live-local",
        "metadata": {
            **git_metadata(),
            "input": str(args.contexts),
            "input_sha256": sha256_file(args.contexts),
            "runner_sha256": sha256_file(Path(__file__)),
            "qids": [str(row.get("qid") or row.get("id")) for row in selected_rows],
            "arms": list(arms),
            "model": args.model,
            "temperature": 0.0,
            "max_tokens": args.max_tokens,
            "retries": args.retries,
            "max_chars": args.max_chars,
            "max_contexts": args.max_contexts,
            "execution": "paired sequential alternating arm order",
        },
        "aggregates": aggregates,
        "paired": paired_live(records),
    }
    (args.output_dir / "live-summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def paired_live(records: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    by_qid: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for arm, values in records.items():
        for value in values:
            by_qid[value["qid"]][arm] = value
    arms = list(records)
    baseline_arm = arms[0]
    comparisons = {}
    for arm in arms[1:]:
        metric_rows = {}
        for metric in (
            "answer_lexical_recall",
            "answer_exact_value_accuracy",
            "requirement_completion",
            "answer_list_completeness",
            "answer_condition_accuracy",
            "answer_negation_accuracy",
            "source_contamination_proxy",
        ):
            deltas = []
            for qid, values in by_qid.items():
                before, after = values.get(baseline_arm), values.get(arm)
                if not before or not after or before.get("error") or after.get("error"):
                    continue
                left, right = before.get(metric), after.get(metric)
                if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                    deltas.append((qid, right - left))
            metric_rows[metric] = {
                "n": len(deltas),
                "mean_delta": statistics.fmean(value for _, value in deltas) if deltas else None,
                "wins": sum(value > 1e-12 for _, value in deltas),
                "regressions": sum(value < -1e-12 for _, value in deltas),
                "ties": sum(abs(value) <= 1e-12 for _, value in deltas),
                "regression_qids": [qid for qid, value in sorted(deltas, key=lambda item: item[1]) if value < -1e-12],
            }
        comparisons[f"{arm}_vs_{baseline_arm}"] = metric_rows
    return comparisons


def percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("offline", "live-local", "live-global"), required=True)
    parser.add_argument("--arm", action="append")
    parser.add_argument("--max-chars", type=int, default=10_000)
    parser.add_argument("--max-contexts", type=int, default=12)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--qid", action="append", default=[])
    parser.add_argument("--seed", default="rag-evidence-cycle-v1")
    parser.add_argument("--model", default="qwen-flash")
    parser.add_argument("--api-base", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--retries", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=float, default=90.0)
    parser.add_argument("--max-live-questions", type=int, default=30)
    args = parser.parse_args(argv)
    if not args.contexts.is_file():
        parser.error(f"contexts are missing: {args.contexts}")
    if args.max_chars <= 0 or args.max_contexts <= 0 or args.max_tokens <= 0:
        parser.error("budgets must be positive")
    if args.retries < 0:
        parser.error("retries must be non-negative")
    valid = set(OFFLINE_ARMS) if args.phase == "offline" else set(LOCAL_LIVE_ARMS if args.phase == "live-local" else GLOBAL_LIVE_ARMS)
    if args.arm and not set(args.arm) <= valid:
        parser.error(f"invalid arms for {args.phase}: {sorted(set(args.arm) - valid)}")
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    result = offline(args) if args.phase == "offline" else live(args, global_only=args.phase == "live-global")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    errors = sum(value.get("errors", 0) for value in result.get("aggregates", {}).values())
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
