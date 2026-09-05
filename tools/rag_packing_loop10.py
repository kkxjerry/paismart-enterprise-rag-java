#!/usr/bin/env python3
"""Run ten sequential evidence-packing optimization rounds on fixed real RAG evidence.

Selection never receives gold answers, answer_facts or expected doc ids. Those fields
are used only after packing for evaluation. A deterministic 75/25 split of the 480
fact-evaluable rows is used every round. The guard split is monitored but never used
to generate new mutations inside this script; mutations are declared up front.

This is a P0 Evidence->Prompt loop. Generation/citation/secondary/verifier metrics are
reported as not_measured here and must come from the bounded live follow-up.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.adaptive_rag.budget import BudgetDecision, DynamicEvidenceBudget
from tools.adaptive_rag.evidence_spans import DEFAULT_PACKING_CONFIG, PackingConfig, pack_evidence
from tools.adaptive_rag.requirements import deterministic_requirement_plan
from tools.qwen_plus_rag_pipeline import token_recall

VALUE_RE = re.compile(
    r"(?:\b\d{1,2}:\d{2}(?:\s*(?i:utc|gmt|[ap]m))?\b|"
    r"\b\d+(?:\.\d+)?\s*%|"
    r"\b\d+(?:\.\d+)?\s*(?i:ms|seconds?|minutes?|hours?|days?|gb|mb|kb|mib|gib)\b|"
    r"\b[vV]?\d+(?:\.\d+){1,3}\b|(?i:\btens? of minutes\b)|"
    r"\b[A-Za-z][A-Za-z0-9]+(?:_[A-Za-z0-9]+)+\b|"
    r"\b[A-Z][a-z0-9]+(?:[A-Z][A-Za-z0-9]+)+\b|"
    r"/[A-Za-z0-9._~!$&'()*+,;=:@%/-]+|"
    r"\"[^\"\n]{3,100}\")"
)
CONDITION_RE = re.compile(
    r"(?i)\b(?:if|when|unless|only if|provided|assuming|depends? on|except|otherwise|"
    r"must not|cannot|can't|should not|not supported|requires?|at least|at most|before|after)\b|"
    r"如果|仅当|除非|取决于|不能|不得|至少|至多|之前|之后"
)
NEGATION_RE = re.compile(
    r"(?i)\b(?:no|not|never|without|cannot|can't|must not|should not|unsupported|not supported)\b|"
    r"不|无|未|不能|不得|禁止"
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_name(qid: str) -> str:
    # 75/25 stable split independent of file order and labels.
    value = int(hashlib.sha256(f"packing-loop10:{qid}".encode()).hexdigest()[:8], 16) % 4
    return "guard" if value == 0 else "tune"


def exact_values(facts: Iterable[str]) -> list[str]:
    output: list[str] = []
    for fact in facts:
        for value in VALUE_RE.findall(str(fact)):
            normalized = re.sub(r"\s+", " ", value.casefold()).strip().strip('"').rstrip(".,;:!?")
            if normalized not in output:
                output.append(normalized)
    return output


def exact_recall(candidate: str, facts: list[str]) -> float | None:
    values = exact_values(facts)
    if not values:
        return None
    normalized = re.sub(r"\s+", " ", candidate.casefold())
    return sum(value in normalized for value in values) / len(values)


def fact_coverage(candidate: str, facts: list[str], threshold: float = 0.6) -> float | None:
    if not facts:
        return None
    return sum(token_recall(candidate, fact) >= threshold for fact in facts) / len(facts)


def subset_coverage(candidate: str, facts: list[str], predicate) -> float | None:
    selected = [fact for fact in facts if predicate(fact)]
    if not selected:
        return None
    return sum(token_recall(candidate, fact) >= 0.6 for fact in selected) / len(selected)


def mean(values: Iterable[float | None]) -> float | None:
    selected = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]
    return statistics.fmean(selected) if selected else None


def pack_row(row: dict[str, Any], config: PackingConfig, max_chars: int, max_contexts: int) -> dict[str, Any]:
    packed = pack_evidence(
        row.get("contexts") or [], question=str(row.get("question") or ""),
        max_chars=max_chars, max_contexts=max_contexts, config=config,
    )
    text = "\n".join(str(context.get("text") or "") for context in packed.contexts)
    facts = [str(v) for v in row.get("answer_facts") or [] if str(v).strip()]
    expected_docs = {str(v) for v in row.get("expected_doc_ids") or [] if str(v)}
    selected_docs = {str(c.get("doc_id") or "") for c in packed.contexts}
    return {
        "qid": str(row.get("qid") or row.get("id")),
        "split": split_name(str(row.get("qid") or row.get("id"))),
        "lexical_recall": mean(token_recall(text, fact) for fact in facts),
        "requirement_evidence_coverage": fact_coverage(text, facts),
        "exact_value_recall": exact_recall(text, facts),
        "list_item_recall": fact_coverage(text, facts) if len(facts) >= 5 else None,
        "condition_exception_recall": subset_coverage(text, facts, lambda fact: bool(CONDITION_RE.search(fact))),
        "gold_doc_retained": float(bool(expected_docs & selected_docs)) if expected_docs else None,
        "rendered_chars": len(packed.rendered),
        "contexts": len(packed.contexts),
    }


def legacy_row(row: dict[str, Any], max_chars: int, max_contexts: int) -> dict[str, Any]:
    plan = deterministic_requirement_plan(str(row.get("question") or ""))
    evidence = DynamicEvidenceBudget("legacy").build(
        list(row.get("contexts") or []), plan=plan,
        decision=BudgetDecision("fast", max_chars, max_chars, max_contexts, 2, ("loop10-fixed",)),
        question=str(row.get("question") or ""),
    )
    copied = dict(row)
    copied["contexts"] = list(evidence.contexts)
    text = "\n".join(str(context.get("text") or "") for context in evidence.contexts)
    facts = [str(v) for v in row.get("answer_facts") or [] if str(v).strip()]
    expected_docs = {str(v) for v in row.get("expected_doc_ids") or [] if str(v)}
    selected_docs = {str(c.get("doc_id") or "") for c in evidence.contexts}
    return {
        "qid": str(row.get("qid") or row.get("id")),
        "split": split_name(str(row.get("qid") or row.get("id"))),
        "lexical_recall": mean(token_recall(text, fact) for fact in facts),
        "requirement_evidence_coverage": fact_coverage(text, facts),
        "exact_value_recall": exact_recall(text, facts),
        "list_item_recall": fact_coverage(text, facts) if len(facts) >= 5 else None,
        "condition_exception_recall": subset_coverage(text, facts, lambda fact: bool(CONDITION_RE.search(fact))),
        "gold_doc_retained": float(bool(expected_docs & selected_docs)) if expected_docs else None,
        "rendered_chars": len(evidence.rendered),
        "contexts": len(evidence.contexts),
    }


def aggregate(rows: list[dict[str, Any]], baseline: dict[str, dict[str, Any]]) -> dict[str, Any]:
    def metric(name: str) -> float | None:
        return mean(row.get(name) for row in rows)
    deltas = []
    severe = 0
    for row in rows:
        before = baseline[row["qid"]].get("lexical_recall")
        after = row.get("lexical_recall")
        if before is None or after is None:
            continue
        delta = after - before
        deltas.append(delta)
        severe += delta <= -0.10
    return {
        "questions": len(rows),
        "prompt_lexical_recall": metric("lexical_recall"),
        "requirement_evidence_coverage": metric("requirement_evidence_coverage"),
        "prompt_exact_value_recall": metric("exact_value_recall"),
        "prompt_list_item_recall": metric("list_item_recall"),
        "prompt_condition_exception_recall": metric("condition_exception_recall"),
        "gold_doc_retained_rate": metric("gold_doc_retained"),
        "mean_rendered_chars": metric("rendered_chars"),
        "mean_contexts": metric("contexts"),
        "packing_regression_rate": sum(delta < -1e-12 for delta in deltas) / len(deltas) if deltas else None,
        "packing_severe_regression_rate": severe / len(deltas) if deltas else None,
        "paired_wins": sum(delta > 1e-12 for delta in deltas),
        "paired_regressions": sum(delta < -1e-12 for delta in deltas),
        "paired_ties": sum(abs(delta) <= 1e-12 for delta in deltas),
    }


def quality_score(metrics: dict[str, Any]) -> float:
    # Research-only scalar for acceptance. The report still exposes every metric.
    weights = {
        "requirement_evidence_coverage": 3.0,
        "prompt_exact_value_recall": 2.0,
        "prompt_list_item_recall": 2.0,
        "prompt_condition_exception_recall": 2.0,
        "prompt_lexical_recall": 1.0,
        "gold_doc_retained_rate": 1.0,
    }
    score = sum(weights[k] * float(metrics.get(k) or 0.0) for k in weights)
    score -= 2.5 * float(metrics.get("packing_regression_rate") or 0.0)
    score -= 1.0 * float(metrics.get("packing_severe_regression_rate") or 0.0)
    score -= 0.15 * float(metrics.get("mean_rendered_chars") or 0.0) / 10000.0
    return score


def acceptable(candidate: dict[str, Any], incumbent: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons = []
    # No typed metric may fall more than 0.5pp on the tune split.
    for key in ("requirement_evidence_coverage", "prompt_exact_value_recall",
                "prompt_list_item_recall", "prompt_condition_exception_recall",
                "gold_doc_retained_rate"):
        c, i = candidate.get(key), incumbent.get(key)
        if c is not None and i is not None and c < i - 0.005:
            reasons.append(f"{key} dropped {(i-c)*100:.2f}pp")
    if candidate.get("packing_regression_rate") is not None and incumbent.get("packing_regression_rate") is not None:
        if candidate["packing_regression_rate"] > incumbent["packing_regression_rate"] + 0.01:
            reasons.append("regression rate rose >1pp")
    if quality_score(candidate) <= quality_score(incumbent) + 1e-6:
        reasons.append("composite did not improve")
    return not reasons, reasons


def mutation(round_no: int, config: PackingConfig) -> tuple[str, PackingConfig]:
    # Declared before results: each round changes one mechanism, not a hidden search over labels.
    if round_no == 1:
        return "reduce_same_document_penalty", replace(config, diversity_penalty=0.08)
    if round_no == 2:
        return "stronger_retrieval_rank_prior", replace(config, rank_exponent=config.rank_exponent + 0.15)
    if round_no == 3:
        return "add_java_evidence_score_prior", replace(config, evidence_score_weight=0.20)
    if round_no == 4:
        return "add_query_coverage_prior", replace(config, query_coverage_weight=0.20)
    if round_no == 5:
        return "weaken_length_penalty", replace(config, density_exponent=max(0.25, config.density_exponent - 0.10))
    if round_no == 6:
        return "smaller_whole_chunk_threshold", replace(config, whole_chunk_chars=2000)
    if round_no == 7:
        return "preserve_wider_prose_conditions", replace(config, neighbor_paragraphs=2)
    if round_no == 8:
        return "stronger_list_shape_hint", replace(config, list_boost=config.list_boost + 0.75)
    if round_no == 9:
        return "stronger_duration_shape_hint", replace(config, duration_boost=config.duration_boost + 0.75)
    if round_no == 10:
        return "reduce_requirement_saturation", replace(config, saturation_power=0.75)
    raise ValueError(round_no)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-chars", type=int, default=10000)
    parser.add_argument("--max-contexts", type=int, default=12)
    parser.add_argument("--round-sample", type=int, default=160)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.contexts.read_text(encoding="utf-8").splitlines() if line.strip()]
    evaluable = [row for row in rows if row.get("answer_facts") and row.get("question_type") != "info_not_found"]
    if args.round_sample <= 0 or args.round_sample > len(evaluable):
        parser.error("--round-sample must be within the fact-evaluable set")
    round_rows = sorted(
        evaluable,
        key=lambda row: hashlib.sha256(
            f"packing-loop10-round-sample:{row.get('qid') or row.get('id')}".encode()
        ).hexdigest(),
    )[:args.round_sample]
    baseline_rows = [legacy_row(row, args.max_chars, args.max_contexts) for row in round_rows]
    baseline = {row["qid"]: row for row in baseline_rows}
    tune_ids = {row["qid"] for row in baseline_rows if row["split"] == "tune"}
    guard_ids = set(baseline) - tune_ids
    baseline_tune = aggregate([baseline[qid] for qid in tune_ids], baseline)
    baseline_guard = aggregate([baseline[qid] for qid in guard_ids], baseline)

    incumbent = DEFAULT_PACKING_CONFIG
    incumbent_rows = [pack_row(row, incumbent, args.max_chars, args.max_contexts) for row in round_rows]
    incumbent_map = {row["qid"]: row for row in incumbent_rows}
    incumbent_tune = aggregate([incumbent_map[qid] for qid in tune_ids], baseline)
    incumbent_guard = aggregate([incumbent_map[qid] for qid in guard_ids], baseline)
    rounds = []
    for round_no in range(1, 11):
        name, candidate = mutation(round_no, incumbent)
        values = [pack_row(row, candidate, args.max_chars, args.max_contexts) for row in round_rows]
        mapped = {row["qid"]: row for row in values}
        tune = aggregate([mapped[qid] for qid in tune_ids], baseline)
        guard = aggregate([mapped[qid] for qid in guard_ids], baseline)
        tune_ok, tune_reasons = acceptable(tune, incumbent_tune)
        # Guard is a safety gate: no >1pp typed regression and no >2pp regression-rate increase.
        guard_reasons = []
        for key in ("requirement_evidence_coverage", "prompt_exact_value_recall",
                    "prompt_list_item_recall", "prompt_condition_exception_recall",
                    "gold_doc_retained_rate"):
            c, i = guard.get(key), incumbent_guard.get(key)
            if c is not None and i is not None and c < i - 0.01:
                guard_reasons.append(f"guard {key} dropped {(i-c)*100:.2f}pp")
        if (guard.get("packing_regression_rate") is not None
                and incumbent_guard.get("packing_regression_rate") is not None
                and guard["packing_regression_rate"] > incumbent_guard["packing_regression_rate"] + 0.02):
            guard_reasons.append("guard regression rate rose >2pp")
        accepted = tune_ok and not guard_reasons
        rounds.append({
            "round": round_no, "mutation": name, "candidate_config": asdict(candidate),
            "accepted": accepted, "rejection_reasons": tune_reasons + guard_reasons,
            "tune": tune, "guard": guard,
            "tune_composite": quality_score(tune), "guard_composite": quality_score(guard),
        })
        if accepted:
            incumbent, incumbent_rows, incumbent_map = candidate, values, mapped
            incumbent_tune, incumbent_guard = tune, guard
        print(
            f"ROUND {round_no:02d} {name} accepted={accepted} "
            f"tune_recall={tune['prompt_lexical_recall']:.6f} "
            f"tune_reg={tune['packing_regression_rate']:.4f} "
            f"guard_recall={guard['prompt_lexical_recall']:.6f} "
            f"guard_reg={guard['packing_regression_rate']:.4f}", flush=True,
        )

    # One full 480-row evaluation after all ten mutations; it is not used to choose
    # intermediate mutations, so the per-round loop remains bounded and reproducible.
    full_baseline_rows = [legacy_row(row, args.max_chars, args.max_contexts) for row in evaluable]
    full_baseline = {row["qid"]: row for row in full_baseline_rows}
    full_final_rows = [pack_row(row, incumbent, args.max_chars, args.max_contexts) for row in evaluable]
    final_all = aggregate(full_final_rows, full_baseline)
    initial_v2_round = [pack_row(row, DEFAULT_PACKING_CONFIG, args.max_chars, args.max_contexts) for row in round_rows]
    result = {
        "schema_version": 1,
        "input": str(args.contexts), "input_sha256": sha(args.contexts),
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "dirty_state": subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True),
        "sample_counts": {"total": len(rows), "fact_evaluable": len(evaluable),
                          "round_sample": len(round_rows), "tune": len(tune_ids), "guard": len(guard_ids),
                          "final_full_evaluation": len(evaluable)},
        "fixed_budget": {"max_chars": args.max_chars, "max_contexts": args.max_contexts},
        "retrieval_regression_guard": {"hit_at_1": 0.8915, "hit_at_5": 0.9596,
            "hit_at_10": 0.9809, "mrr_at_10": 0.9217, "retrieval_misses": 9,
            "acl_source_violations": 0, "status": "not_rerun_unchanged_input"},
        "legacy": {"tune": baseline_tune, "guard": baseline_guard},
        "initial_v2": {"config": asdict(DEFAULT_PACKING_CONFIG),
                       "tune": aggregate([r for r in initial_v2_round if r['split']=='tune'], baseline),
                       "guard": aggregate([r for r in initial_v2_round if r['split']=='guard'], baseline)},
        "rounds": rounds,
        "final_config": asdict(incumbent), "final_all": final_all,
        "promotion_gate": {
            "packing_regression_rate_target": 0.15,
            "packing_regression_rate_met": (final_all.get("packing_regression_rate") or 1.0) <= 0.15,
            "typed_metric_max_regression_pp_per_type": 1.0,
            "generation_metrics": "not_measured_in_offline_loop",
            "citation_metrics": "not_measured_in_offline_loop",
            "secondary_retrieval_metrics": "not_measured_in_offline_loop",
            "verifier_metrics": "not_measured_in_offline_loop",
        },
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"final_config": result["final_config"], "final_all": final_all,
                      "accepted_rounds": [r["round"] for r in rounds if r["accepted"]]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
