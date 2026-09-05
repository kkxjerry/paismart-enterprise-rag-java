#!/usr/bin/env python3
"""Evaluate E1/E2/E4/E5/E6 on fixed EnterpriseRAG evidence.

replay: query-spans-v3 vs leaf-parent vs contextual prefix vs proposition keys.
canonical-live: same packed evidence, standard generation vs E2 canonical-source schema.
global-live: high-level rows, flat generation vs E6 extractive document hierarchy.

No selector receives gold labels. Gold fields are used after selection/generation for
metrics only. Every output directory is immutable and created exactly once.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.adaptive_rag.budget import BudgetDecision, DynamicEvidenceBudget
from tools.adaptive_rag.canonical_generation import (
    build_canonical_generation_messages,
    build_global_generation_messages,
    build_source_binding_messages,
    contexts_for_bindings,
    validate_canonical_generation,
    validate_global_generation,
    validate_source_bindings,
)
from tools.adaptive_rag.controller import build_generation_messages, validate_adaptive_generation
from tools.adaptive_rag.hierarchical import HierarchyConfig, pack_hierarchical
from tools.adaptive_rag.requirements import deterministic_requirement_plan
from tools.qwen_plus_rag_pipeline import QwenClient, fact_scores, load_jsonl, score_result
from tools.rag_evidence_experiment import anchor_checks, choose
from tools.rag_packing_loop10 import CONDITION_RE, NEGATION_RE, exact_recall, fact_coverage, subset_coverage

PACKING_ARMS = ("query-spans", "leaf-parent", "contextual-leaf-parent", "proposition-parent")


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def mean(values: list[float | None]) -> float | None:
    selected = [float(value) for value in values if isinstance(value, (int, float)) and math.isfinite(float(value))]
    return statistics.fmean(selected) if selected else None


def pack(row: dict[str, Any], arm: str, max_chars: int, max_contexts: int) -> dict[str, Any]:
    question = str(row.get("question") or "")
    plan = deterministic_requirement_plan(question)
    if arm == "query-spans":
        decision = BudgetDecision("fast", max_chars, max_chars, max_contexts, 2, ("hierarchy-ablation",))
        value = DynamicEvidenceBudget("query-spans").build(
            list(row.get("contexts") or []), plan=plan, decision=decision, question=question
        )
        return {
            "rendered": value.rendered,
            "contexts": list(value.contexts),
            "trace": list(value.selection_trace),
            "strategy": arm,
        }
    if arm not in PACKING_ARMS:
        raise ValueError(f"unknown packing arm: {arm}")
    value = pack_hierarchical(
        row.get("contexts") or [],
        question=question,
        max_chars=max_chars,
        max_contexts=max_contexts,
        config=HierarchyConfig(strategy=arm),
    )
    return {
        "rendered": value.rendered,
        "contexts": list(value.contexts),
        "trace": list(value.trace),
        "selected_leaves": list(value.selected_leaves),
        "strategy": arm,
    }


def prompt_metrics(row: dict[str, Any], packed: dict[str, Any]) -> dict[str, Any]:
    contexts = packed["contexts"]
    text = "\n".join(str(value.get("text") or "") for value in contexts)
    facts = [str(value) for value in row.get("answer_facts") or [] if str(value).strip()]
    expected = {str(value) for value in row.get("expected_doc_ids") or [] if str(value)}
    selected = {str(value.get("doc_id") or "") for value in contexts if value.get("doc_id")}
    lexical = fact_scores(text, facts)[0] if facts else None
    return {
        "qid": str(row.get("qid") or row.get("id")),
        "question_type": row.get("question_type"),
        "source_types": row.get("source_types") or [],
        "strategy": packed["strategy"],
        "evaluable": bool(facts) and row.get("question_type") != "info_not_found",
        "prompt_lexical_recall": lexical,
        "prompt_exact_value_recall": exact_recall(text, facts),
        "prompt_requirement_evidence_coverage": fact_coverage(text, facts),
        "prompt_list_item_recall": fact_coverage(text, facts) if len(facts) >= 5 else None,
        "prompt_condition_exception_recall": subset_coverage(text, facts, lambda fact: bool(CONDITION_RE.search(fact))),
        "gold_doc_retained": float(bool(expected & selected)) if expected else None,
        "rendered_chars": len(packed["rendered"]),
        "context_count": len(contexts),
        "selected_citations": [str(value.get("citation_id") or "") for value in contexts],
        "selected_doc_ids": sorted(selected),
        "known_case": anchor_checks(str(row.get("qid") or row.get("id")), packed["rendered"]),
    }


def aggregate_prompt(records: list[dict[str, Any]], baseline_arm: str = "query-spans") -> dict[str, Any]:
    by_arm: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_arm.setdefault(record["strategy"], []).append(record)
    metrics = (
        "prompt_lexical_recall",
        "prompt_exact_value_recall",
        "prompt_requirement_evidence_coverage",
        "prompt_list_item_recall",
        "prompt_condition_exception_recall",
        "gold_doc_retained",
        "rendered_chars",
        "context_count",
    )
    aggregates = {}
    for arm, values in sorted(by_arm.items()):
        evaluable = [value for value in values if value.get("evaluable")]
        aggregates[arm] = {
            "rows": len(values),
            "fact_evaluable_rows": len(evaluable),
            **{key: mean([value.get(key) for value in evaluable]) for key in metrics},
            "metric_denominators": {
                key: sum(isinstance(value.get(key), (int, float)) for value in evaluable)
                for key in metrics
            },
        }
    paired: dict[str, dict[str, dict[str, Any]]] = {}
    for value in records:
        paired.setdefault(value["qid"], {})[value["strategy"]] = value
    comparisons = {}
    for arm in sorted(by_arm):
        if arm == baseline_arm:
            continue
        deltas = []
        severe = []
        for qid, values in paired.items():
            if not values.get(baseline_arm, {}).get("evaluable") or not values.get(arm, {}).get("evaluable"):
                continue
            before = values.get(baseline_arm, {}).get("prompt_lexical_recall")
            after = values.get(arm, {}).get("prompt_lexical_recall")
            if not isinstance(before, (int, float)) or not isinstance(after, (int, float)):
                continue
            delta = float(after) - float(before)
            deltas.append({"qid": qid, "delta": delta, "question_type": values[arm].get("question_type")})
            if delta <= -0.10:
                severe.append(qid)
        comparisons[arm] = {
            "pairs": len(deltas),
            "wins": sum(value["delta"] > 1e-12 for value in deltas),
            "regressions": sum(value["delta"] < -1e-12 for value in deltas),
            "ties": sum(abs(value["delta"]) <= 1e-12 for value in deltas),
            "regression_rate": sum(value["delta"] < -1e-12 for value in deltas) / len(deltas) if deltas else None,
            "severe_regression_rate": len(severe) / len(deltas) if deltas else None,
            "severe_qids": severe,
            "largest_regressions": sorted(deltas, key=lambda value: value["delta"])[:20],
            "largest_gains": sorted(deltas, key=lambda value: -value["delta"])[:20],
        }
    return {"aggregates": aggregates, "comparisons_vs_query_spans": comparisons}


def answer_metrics(
    row: dict[str, Any],
    packed: dict[str, Any],
    generation: dict[str, Any],
) -> dict[str, Any]:
    facts = [str(value) for value in row.get("answer_facts") or [] if str(value).strip()]
    answer = str(generation.get("answer") or "")
    citations = [str(value) for value in generation.get("citations") or []]
    contexts = packed["contexts"]
    result = score_result(
        row,
        selected_contexts=contexts,
        answerable=bool(generation.get("answerable")),
        answer=answer,
        citations=citations,
    )
    citation_to_doc = {
        str(value.get("citation_id") or ""): str(value.get("doc_id") or "")
        for value in contexts
    }
    cited_docs = {citation_to_doc[value] for value in citations if value in citation_to_doc}
    expected_docs = {str(value) for value in row.get("expected_doc_ids") or [] if str(value)}
    wrong_docs = cited_docs - expected_docs if expected_docs else set()
    canonical_docs = set(str(value) for value in generation.get("canonical_doc_ids") or [] if str(value))
    return {
        "answer_lexical_recall": result["answer_fact_token_recall"],
        "answer_exact_value_accuracy": exact_recall(answer, facts),
        "requirement_completion": fact_coverage(answer, facts),
        "answer_list_completeness": fact_coverage(answer, facts) if len(facts) >= 5 else None,
        "answer_condition_accuracy": subset_coverage(answer, facts, lambda fact: bool(CONDITION_RE.search(fact))),
        "answer_negation_accuracy": subset_coverage(answer, facts, lambda fact: bool(NEGATION_RE.search(fact))),
        "answer_fact_coverage": result["answer_fact_coverage_proxy"],
        "gold_answer_f1": result["gold_answer_token_f1"],
        "citation_id_validity": result["citation_precision"],
        "abstention_accuracy": result["unanswerable_abstain_correct"],
        "cited_doc_ids": sorted(cited_docs),
        "wrong_cited_doc_ids": sorted(wrong_docs),
        "source_contaminated": float(bool(wrong_docs)) if expected_docs else None,
        "canonical_source_accuracy": (
            float(bool(canonical_docs & expected_docs)) if canonical_docs and expected_docs else None
        ),
    }


def aggregate_live(records: list[dict[str, Any]]) -> dict[str, Any]:
    arms = sorted({str(value["arm"]) for value in records})
    fields = (
        "prompt_lexical_recall",
        "prompt_exact_value_recall",
        "prompt_requirement_evidence_coverage",
        "answer_lexical_recall",
        "answer_exact_value_accuracy",
        "requirement_completion",
        "answer_list_completeness",
        "answer_condition_accuracy",
        "answer_negation_accuracy",
        "answer_fact_coverage",
        "gold_answer_f1",
        "citation_id_validity",
        "source_contaminated",
        "canonical_source_accuracy",
        "latency_ms",
        "rendered_chars",
    )
    output = {}
    for arm in arms:
        values = [value for value in records if value["arm"] == arm]
        valid = [value for value in values if not value.get("error")]
        output[arm] = {
            "rows": len(values),
            "errors": len(values) - len(valid),
            **{field: mean([value.get(field) for value in valid]) for field in fields},
            "metric_denominators": {
                field: sum(isinstance(value.get(field), (int, float)) for value in valid)
                for field in fields
            },
            "model_calls": sum(int(value.get("model_calls") or 1) for value in valid),
            "usage": {
                key: sum(int((value.get("usage") or {}).get(key) or 0) for value in values)
                for key in ("prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens")
            },
            "abstentions": sum((value.get("generation") or {}).get("answerable") is False for value in valid),
            "failed_qids": [value["qid"] for value in values if value.get("error")],
        }
    return output


def call(
    client: QwenClient,
    *,
    messages: list[dict[str, str]],
    validator: Callable[[dict[str, Any]], dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw: dict[str, Any] = {}
    def wrapped(payload: dict[str, Any]) -> dict[str, Any]:
        raw.update(payload)
        return validator(payload)
    result = client.complete_json(messages=messages, max_tokens=1200, temperature=0.0, validator=wrapped)
    return result.value, {
        "usage": result.usage,
        "latency_ms": result.latency_ms,
        "request_id": result.request_id,
        "returned_model": result.returned_model,
        "attempts": result.attempts,
        "raw_generation": raw,
        "model_calls": 1,
    }


def run_replay(args: argparse.Namespace, rows: list[dict[str, Any]], output: Path) -> int:
    arms = args.arm or list(PACKING_ARMS)
    records = []
    with (output / "records.jsonl").open("x", encoding="utf-8") as stream:
        for index, row in enumerate(rows, start=1):
            for arm in arms:
                value = prompt_metrics(row, pack(row, arm, args.max_chars, args.max_contexts))
                stream.write(json.dumps(value, ensure_ascii=False) + "\n")
                records.append(value)
            if index % 100 == 0:
                print(f"REPLAY {index}/{len(rows)}", flush=True)
    summary = aggregate_prompt(records)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def fixed_live_rows(rows: list[dict[str, Any]], limit: int, qids: list[str], seed: str) -> list[dict[str, Any]]:
    return choose(rows, limit, qids, seed)


def run_canonical_live(args: argparse.Namespace, rows: list[dict[str, Any]], output: Path) -> int:
    selected = fixed_live_rows(rows, args.limit, args.qid, args.seed)
    client = QwenClient(
        api_base=args.api_base,
        api_key=os.getenv(args.api_key_env, ""),
        model=args.model,
        timeout_seconds=90,
        retries=0,
    )
    records = []
    with (output / "records.jsonl").open("x", encoding="utf-8") as stream:
        for index, row in enumerate(selected):
            question = str(row.get("question") or "")
            plan = deterministic_requirement_plan(question)
            packed = pack(row, args.packing_arm, args.max_chars, args.max_contexts)
            base_prompt = prompt_metrics(row, packed)
            arms = ("standard", "canonical") if index % 2 == 0 else ("canonical", "standard")
            for arm in arms:
                record = {
                    "qid": str(row.get("qid") or row.get("id")),
                    "question": question,
                    "question_type": row.get("question_type"),
                    "arm": arm,
                    "packing_arm": args.packing_arm,
                    "rendered_chars": len(packed["rendered"]),
                    **{key: base_prompt.get(key) for key in (
                        "prompt_lexical_recall",
                        "prompt_exact_value_recall",
                        "prompt_requirement_evidence_coverage",
                    )},
                    "selected_contexts": packed["contexts"],
                    "error": None,
                }
                try:
                    valid_citations = {str(value.get("citation_id") or "") for value in packed["contexts"]}
                    if arm == "standard":
                        messages = build_generation_messages(
                            question=question,
                            plan=plan,
                            rendered_contexts=packed["rendered"],
                        )
                        validator = lambda payload: validate_adaptive_generation(
                            payload,
                            valid_citations=valid_citations,
                            requirement_ids={"R1"},
                        )
                    else:
                        binding_messages, binding_citation_to_doc, single_source = build_source_binding_messages(
                            question=question,
                            plan=plan,
                            contexts=packed["contexts"],
                        )
                        bindings, binding_api = call(
                            client,
                            messages=binding_messages,
                            validator=lambda payload: validate_source_bindings(
                                payload,
                                citation_to_doc=binding_citation_to_doc,
                                requirement_ids={"R1"},
                                single_source=single_source,
                            ),
                        )
                        bound_contexts = contexts_for_bindings(packed["contexts"], bindings)
                        if not bound_contexts:
                            generation = {
                                "answerable": False,
                                "answer": "INSUFFICIENT_EVIDENCE",
                                "citations": [],
                                "covered_requirements": [],
                                "missing_requirements": ["R1"],
                                "requirements": [],
                                "canonical_doc_ids": [],
                            }
                            api = binding_api
                            messages = binding_messages
                            record["generation_skipped_after_missing_binding"] = True
                        else:
                            messages, citation_to_doc, _ = build_canonical_generation_messages(
                                question=question,
                                plan=plan,
                                contexts=bound_contexts,
                            )
                            bound_citations = {
                                str(value.get("citation_id") or "") for value in bound_contexts
                            }
                            generation, generation_api = call(
                                client,
                                messages=messages,
                                validator=lambda payload: validate_canonical_generation(
                                    payload,
                                    valid_citations=bound_citations,
                                    citation_to_doc=citation_to_doc,
                                    requirement_ids={"R1"},
                                    single_source=single_source,
                                ),
                            )
                            api = {
                                **generation_api,
                                "usage": {
                                    key: int((binding_api.get("usage") or {}).get(key) or 0)
                                    + int((generation_api.get("usage") or {}).get(key) or 0)
                                    for key in ("prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens")
                                },
                                "latency_ms": float(binding_api.get("latency_ms") or 0.0)
                                + float(generation_api.get("latency_ms") or 0.0),
                                "attempts": int(binding_api.get("attempts") or 0)
                                + int(generation_api.get("attempts") or 0),
                                "model_calls": 2,
                            }
                        record["source_binding"] = bindings
                        record["source_binding_messages"] = binding_messages
                        record["source_binding_api"] = binding_api
                        record["bound_contexts"] = bound_contexts
                    if arm == "standard":
                        generation, api = call(client, messages=messages, validator=validator)
                    record.update(api)
                    record["generation"] = generation
                    record["messages"] = messages
                    record.update(answer_metrics(row, packed, generation))
                    print(
                        f"CANONICAL {record['qid']} {arm} answerable={generation['answerable']} "
                        f"tokens={api['usage']['total_tokens']}",
                        flush=True,
                    )
                except Exception as exc:
                    record["error"] = str(exc)
                    record["usage"] = dict(getattr(exc, "usage", {}))
                    record["attempts"] = int(getattr(exc, "attempts", 0))
                    print(f"ERROR {record['qid']} {arm}: {type(exc).__name__}", flush=True)
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                records.append(record)
    summary = aggregate_live(records)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if any(value.get("error") for value in records) else 0


def run_global_live(args: argparse.Namespace, rows: list[dict[str, Any]], output: Path) -> int:
    high = [row for row in rows if row.get("question_type") == "high_level"]
    if args.limit:
        high = high[:args.limit]
    client = QwenClient(
        api_base=args.api_base,
        api_key=os.getenv(args.api_key_env, ""),
        model=args.model,
        timeout_seconds=90,
        retries=0,
    )
    records = []
    with (output / "records.jsonl").open("x", encoding="utf-8") as stream:
        for index, row in enumerate(high):
            question = str(row.get("question") or "")
            plan = deterministic_requirement_plan(question)
            flat = pack(row, args.packing_arm, min(args.max_chars, 14000), args.max_contexts)
            base_prompt = prompt_metrics(row, flat)
            for arm in (("flat", "global") if index % 2 == 0 else ("global", "flat")):
                record = {
                    "qid": str(row.get("qid") or row.get("id")),
                    "question": question,
                    "question_type": row.get("question_type"),
                    "arm": arm,
                    "packing_arm": args.packing_arm,
                    "rendered_chars": len(flat["rendered"]),
                    **{key: base_prompt.get(key) for key in (
                        "prompt_lexical_recall",
                        "prompt_exact_value_recall",
                        "prompt_requirement_evidence_coverage",
                    )},
                    "selected_contexts": flat["contexts"],
                    "error": None,
                }
                try:
                    if arm == "flat":
                        valid_citations = {str(value.get("citation_id") or "") for value in flat["contexts"]}
                        messages = build_generation_messages(
                            question=question, plan=plan, rendered_contexts=flat["rendered"]
                        )
                        validator = lambda payload: validate_adaptive_generation(
                            payload, valid_citations=valid_citations, requirement_ids={"R1"}
                        )
                        packed_for_metrics = flat
                    else:
                        messages, hierarchy = build_global_generation_messages(
                            question=question,
                            contexts=row.get("contexts") or [],
                            max_documents=8,
                            leaves_per_document=3,
                            max_chars=min(args.max_chars, 14000),
                        )
                        valid_citations = set(hierarchy["source_citations"])
                        valid_documents = {value["doc_id"] for value in hierarchy["documents"]}
                        validator = lambda payload: validate_global_generation(
                            payload,
                            valid_citations=valid_citations,
                            valid_documents=valid_documents,
                        )
                        contexts_by_citation = {
                            str(value.get("citation_id") or ""): dict(value)
                            for value in row.get("contexts") or []
                        }
                        selected_contexts = [
                            contexts_by_citation[value]
                            for value in hierarchy["source_citations"]
                            if value in contexts_by_citation
                        ]
                        packed_for_metrics = {
                            "rendered": hierarchy["rendered"],
                            "contexts": selected_contexts,
                        }
                        record["rendered_chars"] = hierarchy["rendered_chars"]
                        record["global_hierarchy"] = hierarchy
                        prompt = prompt_metrics(row, {
                            "rendered": hierarchy["rendered"],
                            "contexts": selected_contexts,
                            "strategy": "global",
                        })
                        record.update({
                            "prompt_lexical_recall": prompt["prompt_lexical_recall"],
                            "prompt_exact_value_recall": prompt["prompt_exact_value_recall"],
                            "prompt_requirement_evidence_coverage": prompt["prompt_requirement_evidence_coverage"],
                        })
                    generation, api = call(client, messages=messages, validator=validator)
                    record.update(api)
                    record["generation"] = generation
                    record["messages"] = messages
                    record["selected_contexts"] = packed_for_metrics["contexts"]
                    record.update(answer_metrics(row, packed_for_metrics, generation))
                    print(
                        f"GLOBAL {record['qid']} {arm} answerable={generation['answerable']} "
                        f"tokens={api['usage']['total_tokens']}",
                        flush=True,
                    )
                except Exception as exc:
                    record["error"] = str(exc)
                    record["usage"] = dict(getattr(exc, "usage", {}))
                    print(f"ERROR {record['qid']} {arm}: {type(exc).__name__}", flush=True)
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                records.append(record)
    summary = aggregate_live(records)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if any(value.get("error") for value in records) else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("replay", "canonical-live", "global-live"), required=True)
    parser.add_argument("--arm", action="append", choices=PACKING_ARMS)
    parser.add_argument("--packing-arm", choices=PACKING_ARMS, default="contextual-leaf-parent")
    parser.add_argument("--max-chars", type=int, default=10000)
    parser.add_argument("--max-contexts", type=int, default=12)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--qid", action="append", default=[])
    parser.add_argument("--seed", default="rag-hierarchy-e1-e6-v1")
    parser.add_argument("--model", default="qwen-flash")
    parser.add_argument("--api-base", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--api-key-env", default="DASHSCOPE_API_KEY")
    args = parser.parse_args()
    if args.max_chars <= 0 or args.max_contexts <= 0 or args.limit <= 0:
        parser.error("budgets and limit must be positive")
    if args.phase in {"canonical-live", "global-live"} and not os.getenv(args.api_key_env):
        parser.error(f"missing {args.api_key_env}")
    rows = load_jsonl(args.contexts)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        "phase": args.phase,
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "dirty_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True),
        "input": str(args.contexts),
        "input_sha256": sha(args.contexts),
        "runner_sha256": sha(Path(__file__)),
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "label_boundary": "selection and generation inputs exclude gold_answer, answer_facts and expected_doc_ids",
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    if args.phase == "replay":
        return run_replay(args, rows, args.output_dir)
    if args.phase == "canonical-live":
        return run_canonical_live(args, rows, args.output_dir)
    return run_global_live(args, rows, args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
