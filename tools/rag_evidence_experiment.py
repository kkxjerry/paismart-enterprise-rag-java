#!/usr/bin/env python3
"""Paired evidence-packing replay and optional bounded live generation.

Gold fields are read only by post-selection scoring. The two arms receive the
same question, deterministic plan, authorized input, prompt and character cap.
This isolates the selector, NOT the full adaptive controller. Lexical scores are
proxies, not correctness. No model judge, mapper, verifier or secondary search.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.adaptive_rag.budget import BudgetDecision, DynamicEvidenceBudget
from tools.adaptive_rag.controller import build_generation_messages, validate_adaptive_generation
from tools.adaptive_rag.requirements import deterministic_requirement_plan
from tools.adaptive_rag_pipeline import adaptive_implementation_sha256
from tools.qwen_plus_rag_pipeline import QwenClient, fact_scores, load_jsonl, score_result
from tools.rag_packing_loop10 import (
    CONDITION_RE,
    NEGATION_RE,
    exact_recall,
    fact_coverage,
    subset_coverage,
)

SEED = "rag-evidence-cycle-v1"
FIELDS = ("timestamp", "service_a", "service_a_version", "service_b", "service_b_version",
          "endpoint", "dcts_version", "test_case_id", "result", "failure_reason", "run_id")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def choose(rows: list[dict[str, Any]], limit: int | None, qids: list[str], seed: str) -> list[dict[str, Any]]:
    by_id = {str(row.get("qid") or row.get("id") or ""): row for row in rows}
    if "" in by_id or len(by_id) != len(rows):
        raise ValueError("missing or duplicate input qid")
    missing = set(qids) - set(by_id)
    if missing:
        raise ValueError(f"unknown requested qids: {sorted(missing)}")
    explicit = list(dict.fromkeys(qids))
    if limit is not None and (limit <= 0 or limit < len(explicit)):
        raise ValueError("limit must be positive and cover explicitly requested qids")
    rest = sorted(set(by_id) - set(explicit), key=lambda q: hashlib.sha256(f"{seed}:{q}".encode()).hexdigest())
    selected = (explicit + rest)[:limit] if limit is not None else explicit + rest
    return [by_id[qid] for qid in selected]


def anchor_checks(qid: str, text: str) -> dict[str, Any] | None:
    """Known-case exact-value probes only; never used by packing or generation."""
    if qid == "qst_0026":
        return {"typical_duration_present": bool(re.search(r"tens of minutes", text, re.I)),
                "database_size_condition_present": bool(re.search(r"size.{0,30}database|database.{0,30}size", text, re.I)),
                "snapshot_condition_present": "snapshot" in text.casefold()}
    if qid == "qst_0174":
        found = [field for field in FIELDS if re.search(r"(?<!\w)" + re.escape(field) + r"(?!\w)", text)]
        return {"field_count": len(found), "complete_11_fields": len(found) == len(FIELDS),
                "missing_fields": sorted(set(FIELDS) - set(found))}
    return None


def summarize(records: list[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any]:
    arms = sorted({r["strategy"] for r in records})
    aggregates: dict[str, Any] = {}
    for arm in arms:
        values = [r for r in records if r["strategy"] == arm]
        valid = [r for r in values if not r.get("error")]
        evaluable = [r for r in valid if r.get("evaluable")]
        def avg(name: str, group: list[dict[str, Any]] = evaluable):
            data = [r[name] for r in group if isinstance(r.get(name), (int, float))]
            return statistics.fmean(data) if data else None
        aggregates[arm] = {
            "rows": len(values), "errors": len(values) - len(valid), "evaluable": len(evaluable),
            "prompt_lexical_recall": avg("prompt_lexical_recall"),
            "prompt_exact_value_recall": avg("prompt_exact_value_recall"),
            "prompt_requirement_evidence_coverage": avg("prompt_requirement_evidence_coverage"),
            "prompt_list_item_recall": avg("prompt_list_item_recall"),
            "prompt_condition_exception_recall": avg("prompt_condition_exception_recall"),
            "answer_lexical_recall": avg("answer_lexical_recall"),
            "answer_exact_value_accuracy": avg("answer_exact_value_accuracy"),
            "requirement_completion": avg("requirement_completion"),
            "answer_list_completeness": avg("answer_list_completeness"),
            "answer_condition_accuracy": avg("answer_condition_accuracy"),
            "answer_negation_accuracy": avg("answer_negation_accuracy"),
            "answer_fact_coverage": avg("answer_fact_coverage"),
            "gold_answer_f1": avg("gold_answer_f1"),
            "citation_id_validity": avg("citation_id_validity", valid),
            "abstention_accuracy": avg("abstention_accuracy", valid),
            "claim_citation_support_precision": None,
            "claim_citation_support_recall": None,
            "unsupported_claim_rate": None,
            "mean_rendered_chars": avg("rendered_chars", valid),
            "mean_latency_ms": avg("latency_ms", valid),
            "abstentions": sum(r.get("generation", {}).get("answerable") is False for r in valid),
            "usage": {key: sum(int(r.get("usage", {}).get(key, 0)) for r in values)
                      for key in ("prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens")},
            "attempts": sum(r.get("attempts", 0) for r in values),
        }
    paired = {}
    for record in records:
        paired.setdefault(record["qid"], {})[record["strategy"]] = record
    deltas = []
    for qid, pair in paired.items():
        if len(pair) != len(arms):
            continue
        baseline = pair.get("legacy")
        if not baseline:
            continue
        for arm in arms:
            candidate = pair[arm]
            if (arm == "legacy" or baseline.get("error") or candidate.get("error")
                    or not baseline.get("evaluable") or not candidate.get("evaluable")):
                continue
            key = "answer_lexical_recall" if metadata["phase"] == "live" else "prompt_lexical_recall"
            if baseline.get(key) is not None and candidate.get(key) is not None:
                deltas.append({"qid": qid, "strategy": arm, "metric": key,
                               "delta": candidate[key] - baseline[key]})
    return {"metadata": metadata, "aggregates": aggregates,
            "evaluable_pair_count": len(deltas),
            "proxy_gains": sum(r["delta"] > 1e-8 for r in deltas),
            "proxy_regressions": sum(r["delta"] < -1e-8 for r in deltas),
            "proxy_ties": sum(abs(r["delta"]) <= 1e-8 for r in deltas),
            "largest_proxy_regressions": sorted(deltas, key=lambda r: r["delta"])[:10],
            "largest_proxy_gains": sorted(deltas, key=lambda r: -r["delta"])[:10],
            "known_case_checks": [{k: r.get(k) for k in ("qid", "strategy", "prompt_checks", "answer_checks", "error")}
                                  for r in records if r.get("prompt_checks") is not None]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("replay", "live", "show"), default="replay")
    parser.add_argument("--strategy", action="append", choices=("legacy", "query-spans"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--qid", action="append", default=[])
    parser.add_argument("--seed", default=SEED)
    parser.add_argument("--model", default="qwen-flash")
    parser.add_argument("--api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--max-chars", type=int, default=10000)
    parser.add_argument("--max-contexts", type=int, default=12)
    args = parser.parse_args()
    if args.phase == "show":
        for row in load_jsonl(args.output_dir / "pairs.jsonl"):
            if not args.qid or row["qid"] in args.qid:
                print(json.dumps({k: row.get(k) for k in ("qid", "strategy", "question", "gold_answer", "generation",
                      "raw_generation", "prompt_checks", "answer_checks", "selected_citations", "error")}, ensure_ascii=False))
        return 0
    if args.contexts is None or args.max_chars <= 0 or args.max_contexts <= 0:
        parser.error("contexts and positive budgets are required")
    if args.phase == "live" and (args.limit is None or args.limit > 40):
        parser.error("live experiments require an explicit limit of at most 40 questions")
    rows = choose(load_jsonl(args.contexts), args.limit, args.qid, args.seed)
    arms = list(dict.fromkeys(args.strategy or ["legacy", "query-spans"]))
    client = None
    if args.phase == "live":
        client = QwenClient(api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
                            api_key=os.getenv(args.api_key_env, ""), model=args.model,
                            timeout_seconds=60, retries=0)
    decision = BudgetDecision("fast", args.max_chars, args.max_chars, args.max_contexts, 2, ("fixed-selector-ablation",))
    metadata = {"phase": args.phase, "input_sha256": sha(args.contexts),
                "implementation_sha256": adaptive_implementation_sha256(), "runner_sha256": sha(Path(__file__)),
                "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "dirty_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True),
                "qids": [r.get("qid") or r.get("id") for r in rows], "seed": args.seed,
                "strategies": arms, "budget": asdict(decision), "model": args.model if client else None,
                "max_output_tokens": 1024, "retries": 0, "temperature": 0.0,
                "mapper": "deterministic", "verifier": "off", "secondary_retrieval": False,
                "ordering": "alternating-paired-sequential", "accuracy_metric": "none; lexical proxies and known-case probes only"}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    records = []
    # Append each completed arm immediately; crashes never invent successful rows.
    with (args.output_dir / "pairs.jsonl").open("x", encoding="utf-8") as stream:
        for index, row in enumerate(rows):
            qid = str(row.get("qid") or row.get("id"))
            question = str(row.get("question") or "")
            plan = deterministic_requirement_plan(question)
            for arm in (arms if index % 2 == 0 else list(reversed(arms))):
                facts = [str(value) for value in row.get("answer_facts") or [] if str(value).strip()]
                record: dict[str, Any] = {"qid": qid, "question": question, "strategy": arm,
                                          "gold_answer": row.get("gold_answer"), "error": None,
                                          "question_type": row.get("question_type"),
                                          "evaluable": bool(facts) and row.get("question_type") != "info_not_found"}
                raw_generation: dict[str, Any] = {}
                try:
                    evidence = DynamicEvidenceBudget(arm).build(row.get("contexts") or [], plan=plan,
                                                                decision=decision, question=question)
                    prompt_text = "\n".join(c["text"] for c in evidence.contexts)
                    record.update({"rendered_chars": evidence.rendered_chars, "budget": evidence.to_dict(),
                                   "selected_citations": [c["citation_id"] for c in evidence.contexts],
                                   "selected_contexts": list(evidence.contexts),
                                   "prompt_checks": anchor_checks(qid, evidence.rendered),
                                   "prompt_lexical_recall": fact_scores(prompt_text, facts)[0] if facts else None,
                                   "prompt_exact_value_recall": exact_recall(prompt_text, facts),
                                   "prompt_requirement_evidence_coverage": fact_coverage(prompt_text, facts),
                                   "prompt_list_item_recall": fact_coverage(prompt_text, facts) if len(facts) >= 5 else None,
                                   "prompt_condition_exception_recall": subset_coverage(
                                       prompt_text, facts, lambda fact: bool(CONDITION_RE.search(fact))
                                   )})
                    if client:
                        if not evidence.contexts:
                            raise ValueError("no evidence fits the fixed budget; generation not called")
                        messages = build_generation_messages(question=question, plan=plan, rendered_contexts=evidence.rendered)
                        record["messages"] = messages
                        def validate(payload):
                            raw_generation.update(payload)
                            return validate_adaptive_generation(payload, valid_citations=set(record["selected_citations"]), requirement_ids={"R1"})
                        result = client.complete_json(messages=messages, max_tokens=1024, temperature=0.0, validator=validate)
                        answer_text = str(result.value["answer"])
                        record.update({"generation": result.value, "raw_generation": raw_generation,
                                       "usage": result.usage, "latency_ms": result.latency_ms, "attempts": result.attempts,
                                       "request_id": result.request_id, "returned_model": result.returned_model,
                                       "answer_checks": anchor_checks(qid, answer_text),
                                       "answer_exact_value_accuracy": exact_recall(answer_text, facts),
                                       "requirement_completion": fact_coverage(answer_text, facts),
                                       "answer_list_completeness": fact_coverage(answer_text, facts) if len(facts) >= 5 else None,
                                       "answer_condition_accuracy": subset_coverage(
                                           answer_text, facts, lambda fact: bool(CONDITION_RE.search(fact))
                                       ),
                                       "answer_negation_accuracy": subset_coverage(
                                           answer_text, facts, lambda fact: bool(NEGATION_RE.search(fact))
                                       )})
                        metrics = score_result(row, selected_contexts=list(evidence.contexts),
                                               answerable=result.value["answerable"], answer=result.value["answer"], citations=result.value["citations"])
                        record["answer_lexical_recall"] = metrics["answer_fact_token_recall"]
                        record["answer_fact_coverage"] = metrics["answer_fact_coverage_proxy"]
                        record["gold_answer_f1"] = metrics["gold_answer_token_f1"]
                        record["citation_id_validity"] = metrics["citation_precision"]
                        record["abstention_accuracy"] = metrics["unanswerable_abstain_correct"]
                        print(f"LIVE {qid} {arm} answerable={result.value['answerable']} tokens={result.usage['total_tokens']}", flush=True)
                except Exception as exc:
                    record.update({"error": str(exc), "raw_generation": raw_generation,
                                   "usage": dict(getattr(exc, "usage", {})), "attempts": int(getattr(exc, "attempts", 0))})
                    print(f"ERROR {qid} {arm}: {type(exc).__name__}", flush=True)
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                records.append(record)
            if args.phase == "replay" and (index + 1) % 100 == 0:
                print(f"REPLAY {index + 1}/{len(rows)}", flush=True)
    summary = summarize(records, metadata)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({**summary, "metadata": {k: v for k, v in metadata.items() if k != "qids"}}, ensure_ascii=False, indent=2))
    return 1 if any(r.get("error") for r in records) else 0


if __name__ == "__main__":
    raise SystemExit(main())
