#!/usr/bin/env python3
"""Generate a concise E1-E6 report and machine registry from completed runs."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def pct(value: Any) -> str:
    return "—" if not isinstance(value, (int, float)) or not math.isfinite(float(value)) else f"{100 * float(value):.2f}%"


def number(value: Any, digits: int = 1) -> str:
    return "—" if not isinstance(value, (int, float)) or not math.isfinite(float(value)) else f"{float(value):.{digits}f}"


def delta(left: Any, right: Any) -> float | None:
    if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
        return None
    return float(right) - float(left)


def pp(left: Any, right: Any) -> str:
    value = delta(left, right)
    return "—" if value is None else f"{value * 100:+.2f}pp"


def ratio(left: Any, right: Any) -> str:
    if not isinstance(left, (int, float)) or not isinstance(right, (int, float)) or not left:
        return "—"
    return f"{right / left:.3f}×"


def aggregate_errors(summary: dict[str, Any]) -> int:
    return sum(int(value.get("errors") or 0) for value in summary.get("aggregates", {}).values())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL row must be an object: {path}")
        rows.append(value)
    return rows


def retrieval_baseline(path: Path, source_types: set[str]) -> dict[str, Any]:
    rows = []
    for row in load_jsonl(path):
        row_sources = {str(value).casefold() for value in row.get("source_types") or [] if str(value)}
        if not row_sources or not row_sources <= source_types:
            continue
        expected = [
            str(value)
            for value in (row.get("expected_accessible_doc_ids") or row.get("expected_doc_ids") or [])
            if str(value)
        ]
        if not expected:
            continue
        ranked = [str(value) for value in row.get("ranked_doc_ids") or []]
        ranks = [ranked.index(doc_id) + 1 for doc_id in expected if doc_id in ranked]
        rows.append(
            {
                "hit_at_1": bool(ranks and min(ranks) <= 1),
                "hit_at_5": bool(ranks and min(ranks) <= 5),
                "hit_at_10": bool(ranks and min(ranks) <= 10),
                "mrr": 1.0 / min(ranks) if ranks else 0.0,
                "evidence_fact_token_recall": row.get("evidence_fact_token_recall"),
            }
        )
    def average(name: str) -> float | None:
        values = [float(row[name]) for row in rows if isinstance(row.get(name), (int, float, bool))]
        return sum(values) / len(values) if values else None
    return {
        "questions": len(rows),
        "hit_at_1": average("hit_at_1"),
        "hit_at_5": average("hit_at_5"),
        "hit_at_10": average("hit_at_10"),
        "mrr": average("mrr"),
        "evidence_fact_token_recall": average("evidence_fact_token_recall"),
    }


def decision(name: str, passed: bool, evidence: str, limitation: str = "") -> dict[str, Any]:
    return {"experiment": name, "decision": "accept" if passed else "hold", "evidence": evidence, "limitation": limitation}


def generate(args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    offline = load(args.offline)
    local = load(args.live_local)
    global_live = load(args.live_global)
    e3_sentence = load(args.e3_sentence)
    e3_proposition = load(args.e3_proposition)
    late_native = load(args.late_native)
    late_adapter = load(args.late_adapter)
    sentence_build = load(args.sentence_build)
    sentence_retrieval = load(args.sentence_retrieval)
    proposition_build = load(args.proposition_build)
    proposition_retrieval = load(args.proposition_retrieval)
    e1_topdoc = load(args.e1_topdoc)
    source_subset = {"fireflies", "confluence", "gmail"}
    current_retrieval = retrieval_baseline(args.contexts, source_subset)

    offline_agg = offline["aggregates"]
    local_agg = local["aggregates"]
    global_agg = global_live["aggregates"]
    high_level = offline.get("by_question_type", {}).get("high_level", {})
    legacy = offline_agg["legacy"]
    query = offline_agg["query-spans-v3"]
    e1 = offline_agg["e1-leaf-parent"]
    e4 = offline_agg["e4-context-prefix"]
    e5 = offline_agg["e5-proposition"]
    e6 = offline_agg["e6-routed-global"]

    local_base = local_agg["query-spans-flat"]
    local_leaf = local_agg["e5-flat"]
    local_canonical = local_agg["e2-canonical"]
    global_base = global_agg["query-spans-flat"]
    global_candidate = global_agg["e6-global-structured"]

    e1_runtime_baseline = e1_topdoc.get("baseline") or {}
    e1_runtime_candidate = e1_topdoc.get("candidate") or {}
    e1_pass = (
        (e1_runtime_candidate.get("prompt_requirement_evidence_coverage") or 0)
            >= (e1_runtime_baseline.get("prompt_requirement_evidence_coverage") or 0) + 0.03
        and (e1_runtime_candidate.get("packing_regression_rate") or 1) <= 0.15
        and (e1_runtime_candidate.get("gold_doc_retained_rate") or 0)
            >= (e1_runtime_baseline.get("gold_doc_retained_rate") or 0) - 0.005
        and not e1_topdoc.get("empty_fetch_qids")
    )
    e4_pass = (
        (e4.get("prompt_requirement_evidence_coverage") or 0) >= (e1.get("prompt_requirement_evidence_coverage") or 0)
        and (e4.get("canonical_source_selection_accuracy") or 0) >= (e1.get("canonical_source_selection_accuracy") or 0)
    )
    e5_pass = (
        (e5.get("prompt_exact_value_recall") or 0) >= (e4.get("prompt_exact_value_recall") or 0) - 0.005
        and (e5.get("prompt_condition_exception_recall") or 0) >= (e4.get("prompt_condition_exception_recall") or 0) - 0.005
    )
    e2_pass = (
        not aggregate_errors(local)
        and (local_canonical.get("source_contamination_proxy") or 0) <= (local_base.get("source_contamination_proxy") or 0)
        and (local_canonical.get("requirement_completion") or 0) >= (local_base.get("requirement_completion") or 0)
    )
    e3_export_pass = (
        int(e3_sentence.get("offset_violations") or 0) == 0
        and int(e3_proposition.get("offset_violations") or 0) == 0
        and int(e3_sentence.get("missing_acl_documents") or 0) == 0
        and int(e3_proposition.get("missing_acl_documents") or 0) == 0
        and int(sentence_build.get("index_documents") or 0)
            == int(sentence_build.get("parents") or 0) + int(sentence_build.get("leaves") or 0)
        and int(proposition_build.get("index_documents") or 0)
            == int(proposition_build.get("parents") or 0) + int(proposition_build.get("leaves") or 0)
    )
    e3_retrieval_pass = (
        int(sentence_retrieval.get("errors") or 0) == 0
        and (sentence_retrieval.get("hit_at_10") or 0) >= (current_retrieval.get("hit_at_10") or 0) - 0.005
        and (sentence_retrieval.get("parent_fact_token_recall") or 0)
            >= (current_retrieval.get("evidence_fact_token_recall") or 0)
    )
    e3_pass = e3_export_pass and e3_retrieval_pass
    e6_pass = (
        not aggregate_errors(global_live)
        and (global_candidate.get("requirement_completion") or 0) >= (global_base.get("requirement_completion") or 0)
        and (global_candidate.get("source_contamination_proxy") or 0) <= (global_base.get("source_contamination_proxy") or 0)
    )
    late_ready = bool(late_native.get("late_chunking_ready")) or bool(late_adapter.get("late_chunking_ready"))

    decisions = [
        decision("E1 Top-document Leaf→Parent", e1_pass,
                 f"Existing-index query-time expansion: Requirement coverage {pct(e1_runtime_baseline.get('prompt_requirement_evidence_coverage'))} → {pct(e1_runtime_candidate.get('prompt_requirement_evidence_coverage'))}; exact {pct(e1_runtime_baseline.get('prompt_exact_value_recall'))} → {pct(e1_runtime_candidate.get('prompt_exact_value_recall'))}; regression {pct(e1_runtime_candidate.get('packing_regression_rate'))}; fetch mean/p95={number(e1_topdoc.get('mean_fetch_ms'))}/{number(e1_topdoc.get('p95_fetch_ms'))}ms.",
                 "文档allow-list来自冻结ACL检索的ranked_doc_ids；不等同于新查询的完整在线ACL验收。"),
        decision("E2 Canonical Requirement Generation", e2_pass,
                 f"Completion {pct(local_base.get('requirement_completion'))} → {pct(local_canonical.get('requirement_completion'))}; contamination {pct(local_base.get('source_contamination_proxy'))} → {pct(local_canonical.get('source_contamination_proxy'))}.",
                 "20题Flash样本；semantic citation尚未人工校准。"),
        decision("E3 Source-specific Parent/Leaf Index", e3_pass,
                 f"Build sentence={sentence_build.get('index_documents')} docs, proposition={proposition_build.get('index_documents')} docs; Hit@10 current={pct(current_retrieval.get('hit_at_10'))}, sentence={pct(sentence_retrieval.get('hit_at_10'))}, proposition={pct(proposition_retrieval.get('hit_at_10'))}; parent recall current={pct(current_retrieval.get('evidence_fact_token_recall'))}, sentence={pct(sentence_retrieval.get('parent_fact_token_recall'))}.",
                 "独立索引已真实构建和检索，但未切换在线alias；ACL评测只覆盖tenant/source过滤。"),
        decision("E4 Deterministic Context Prefix", e4_pass,
                 f"Requirement coverage {pct(e1.get('prompt_requirement_evidence_coverage'))} → {pct(e4.get('prompt_requirement_evidence_coverage'))}; canonical selection {pct(e1.get('canonical_source_selection_accuracy'))} → {pct(e4.get('canonical_source_selection_accuracy'))}.",
                 "Prefix仅用于搜索，不进入引用原文。"),
        decision("E5 Proposition Index", e5_pass,
                 f"Exact {pct(e4.get('prompt_exact_value_recall'))} → {pct(e5.get('prompt_exact_value_recall'))}; condition {pct(e4.get('prompt_condition_exception_recall'))} → {pct(e5.get('prompt_condition_exception_recall'))}.",
                 "Late Chunking capability单独判定。"),
        decision("E5 Late Chunking", late_ready,
                 f"Native: {late_native.get('reason')}; adapter: {late_adapter.get('reason')}.",
                 "当前OpenAI-compatible端点若仅返回pooled vector，不能假装支持token级late chunking。"),
        decision("E6 Global Route", e6_pass,
                 f"High-level completion {pct(global_base.get('requirement_completion'))} → {pct(global_candidate.get('requirement_completion'))}; answer recall {pct(global_base.get('answer_lexical_recall'))} → {pct(global_candidate.get('answer_lexical_recall'))}.",
                 "仅10道high_level题，不替换local主链。"),
    ]

    lines = [
        "# RAG E1–E6 实施与效果",
        "",
        "## 结论",
        "",
        "| 实验 | 决策 | 关键依据 |",
        "|---|---|---|",
    ]
    for value in decisions:
        lines.append(f"| {value['experiment']} | **{value['decision']}** | {value['evidence']} |")

    lines.extend([
        "",
        "## E1：现有索引内 Top-document Leaf→Parent（500题）",
        "",
        "| Arm | Req Evidence | Exact | List | Condition | Negation | Gold doc | Chars | Regression | Fetch mean/p95 | Pack mean/p95 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| query-spans-v3 | {pct(e1_runtime_baseline.get('prompt_requirement_evidence_coverage'))} | {pct(e1_runtime_baseline.get('prompt_exact_value_recall'))} | {pct(e1_runtime_baseline.get('prompt_list_item_recall'))} | {pct(e1_runtime_baseline.get('prompt_condition_exception_recall'))} | {pct(e1_runtime_baseline.get('prompt_negation_recall'))} | {pct(e1_runtime_baseline.get('gold_doc_retained_rate'))} | {number(e1_runtime_baseline.get('mean_rendered_chars'))} | {pct(e1_runtime_baseline.get('packing_regression_rate'))} | — | — |",
        f"| topdoc leaf→parent | {pct(e1_runtime_candidate.get('prompt_requirement_evidence_coverage'))} | {pct(e1_runtime_candidate.get('prompt_exact_value_recall'))} | {pct(e1_runtime_candidate.get('prompt_list_item_recall'))} | {pct(e1_runtime_candidate.get('prompt_condition_exception_recall'))} | {pct(e1_runtime_candidate.get('prompt_negation_recall'))} | {pct(e1_runtime_candidate.get('gold_doc_retained_rate'))} | {number(e1_runtime_candidate.get('mean_rendered_chars'))} | {pct(e1_runtime_candidate.get('packing_regression_rate'))} | {number(e1_topdoc.get('mean_fetch_ms'))}/{number(e1_topdoc.get('p95_fetch_ms'))} | {number(e1_topdoc.get('mean_pack_ms'))}/{number(e1_topdoc.get('p95_pack_ms'))} |",
        "",
        f"每题平均取回 {number(e1_topdoc.get('mean_fetched_chunks'))} 个Chunk、覆盖 {number(e1_topdoc.get('mean_fetched_documents'))} 个授权Top文档；empty fetch={len(e1_topdoc.get('empty_fetch_qids') or [])}。",
        "",
        "## 500题离线证据指标",
        "",
        "| Arm | Req Evidence | Exact | List | Condition | Negation | Gold doc | Canonical doc | Chars | Regression |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for arm in ("legacy", "query-spans-v3", "e1-leaf-parent", "e4-context-prefix", "e5-proposition", "e6-routed-global"):
        value = offline_agg[arm]
        lines.append(
            f"| {arm} | {pct(value.get('prompt_requirement_evidence_coverage'))} | "
            f"{pct(value.get('prompt_exact_value_recall'))} | {pct(value.get('prompt_list_item_recall'))} | "
            f"{pct(value.get('prompt_condition_exception_recall'))} | {pct(value.get('prompt_negation_recall'))} | "
            f"{pct(value.get('gold_doc_retained_rate'))} | {pct(value.get('canonical_source_selection_accuracy'))} | "
            f"{number(value.get('mean_rendered_chars'))} | {pct(value.get('packing_regression_rate'))} |"
        )

    lines.extend([
        "",
        "## 20题真实 Qwen Flash：E2",
        "",
        "| Arm | Answer Recall | Exact | Completion | List | Condition | Negation | Contamination | Citation lexical support* | Tokens | Mean / P95 ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for arm in ("query-spans-flat", "e5-flat", "e2-canonical"):
        value = local_agg[arm]
        lines.append(
            f"| {arm} | {pct(value.get('answer_lexical_recall'))} | {pct(value.get('answer_exact_value_accuracy'))} | "
            f"{pct(value.get('requirement_completion'))} | {pct(value.get('answer_list_completeness'))} | "
            f"{pct(value.get('answer_condition_accuracy'))} | {pct(value.get('answer_negation_accuracy'))} | "
            f"{pct(value.get('source_contamination_proxy'))} | {pct(value.get('citation_lexical_support_recall_proxy'))} | "
            f"{value.get('usage', {}).get('total_tokens', 0)} | {number(value.get('mean_latency_ms'))} / {number(value.get('p95_latency_ms'))} |"
        )
    lines.extend([
        "",
        "\* Citation lexical support是诊断代理，不是semantic entailment结论。",
        "",
        "## 10题 high-level：E6",
        "",
        "| Arm | Answer Recall | Exact | Completion | Condition | Contamination | Tokens | Mean / P95 ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for arm in ("query-spans-flat", "e6-global-structured"):
        value = global_agg[arm]
        lines.append(
            f"| {arm} | {pct(value.get('answer_lexical_recall'))} | {pct(value.get('answer_exact_value_accuracy'))} | "
            f"{pct(value.get('requirement_completion'))} | {pct(value.get('answer_condition_accuracy'))} | "
            f"{pct(value.get('source_contamination_proxy'))} | {value.get('usage', {}).get('total_tokens', 0)} | "
            f"{number(value.get('mean_latency_ms'))} / {number(value.get('p95_latency_ms'))} |"
        )

    lines.extend([
        "",
        "## E3独立 Elasticsearch 索引与检索",
        "",
        "| System | Questions | Hit@1 | Hit@5 | Hit@10 | MRR | Evidence/Parent Recall | Index docs | Build s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| Current Java four-route subset | {current_retrieval.get('questions')} | {pct(current_retrieval.get('hit_at_1'))} | {pct(current_retrieval.get('hit_at_5'))} | {pct(current_retrieval.get('hit_at_10'))} | {pct(current_retrieval.get('mrr'))} | {pct(current_retrieval.get('evidence_fact_token_recall'))} | — | — |",
        f"| Sentence leaf → parent | {sentence_retrieval.get('retrieval_evaluable')} | {pct(sentence_retrieval.get('hit_at_1'))} | {pct(sentence_retrieval.get('hit_at_5'))} | {pct(sentence_retrieval.get('hit_at_10'))} | {pct(sentence_retrieval.get('mrr'))} | {pct(sentence_retrieval.get('parent_fact_token_recall'))} | {sentence_build.get('index_documents')} | {number(sentence_build.get('elapsed_seconds'))} |",
        f"| Proposition leaf → parent | {proposition_retrieval.get('retrieval_evaluable')} | {pct(proposition_retrieval.get('hit_at_1'))} | {pct(proposition_retrieval.get('hit_at_5'))} | {pct(proposition_retrieval.get('hit_at_10'))} | {pct(proposition_retrieval.get('mrr'))} | {pct(proposition_retrieval.get('parent_fact_token_recall'))} | {proposition_build.get('index_documents')} | {number(proposition_build.get('elapsed_seconds'))} |",
        "",
        "ACL说明：新索引保留完整ACL字段；本次查询只执行tenant/source过滤，因为问题文件没有可直接注入的用户→组解析结果，因此不能把本次检索称为完整ACL安全验收。两个索引均未修改alias。",
        "",
        "## E3完整语料导出",
        "",
        "| Leaf mode | Documents | Parents | Leaves | Parent p50/p95 | Leaf p50/p95 | Missing ACL | Offset violations |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for label, value in (("sentence", e3_sentence), ("proposition", e3_proposition)):
        lines.append(
            f"| {label} | {value.get('documents_exported')} | {value.get('parents')} | {value.get('leaves')} | "
            f"{number(value.get('parent_characters', {}).get('p50'))}/{number(value.get('parent_characters', {}).get('p95'))} | "
            f"{number(value.get('leaf_characters', {}).get('p50'))}/{number(value.get('leaf_characters', {}).get('p95'))} | "
            f"{value.get('missing_acl_documents')} | {value.get('offset_violations')} |"
        )

    lines.extend([
        "",
        "## 逐层解释",
        "",
        f"- E1现有索引真实扩展相对query-spans：Req Evidence {pp(e1_runtime_baseline.get('prompt_requirement_evidence_coverage'), e1_runtime_candidate.get('prompt_requirement_evidence_coverage'))}；Exact {pp(e1_runtime_baseline.get('prompt_exact_value_recall'), e1_runtime_candidate.get('prompt_exact_value_recall'))}；回退率 {pct(e1_runtime_candidate.get('packing_regression_rate'))}。已有Context内拆Leaf的结果只作为机制消融，不再冒充完整E1。",
        f"- E4相对E1：Context Prefix只影响检索排序。Req Evidence {pp(e1.get('prompt_requirement_evidence_coverage'), e4.get('prompt_requirement_evidence_coverage'))}；Canonical Source {pp(e1.get('canonical_source_selection_accuracy'), e4.get('canonical_source_selection_accuracy'))}。",
        f"- E5相对E4：Proposition粒度。Exact {pp(e4.get('prompt_exact_value_recall'), e5.get('prompt_exact_value_recall'))}；Condition {pp(e4.get('prompt_condition_exception_recall'), e5.get('prompt_condition_exception_recall'))}。",
        f"- E2相对flat：Completion {pp(local_base.get('requirement_completion'), local_canonical.get('requirement_completion'))}；Contamination {pp(local_base.get('source_contamination_proxy'), local_canonical.get('source_contamination_proxy'))}；Token {ratio(local_base.get('usage', {}).get('total_tokens'), local_canonical.get('usage', {}).get('total_tokens'))}。",
        f"- E6只看high-level：Completion {pp(global_base.get('requirement_completion'), global_candidate.get('requirement_completion'))}；Answer Recall {pp(global_base.get('answer_lexical_recall'), global_candidate.get('answer_lexical_recall'))}。",
        f"- Late Chunking：native ready={late_native.get('late_chunking_ready')}；adapter ready={late_adapter.get('late_chunking_ready')}。",
        "",
        "## 边界",
        "",
        "- 500题是开发诊断集，不是未见holdout；词法/typed指标不能冒充人工正确率。",
        "- E3仅导出Parent/Leaf索引输入，没有重建Embedding索引、切alias或改在线服务。",
        "- E2/E6使用真实Qwen Flash；模型调用错误、Token和延迟均计入机器结果。",
        "- Canonical source与source contamination依赖expected doc做事后评分，expected doc不进入选择或Prompt。",
        "- Late Chunking只有token级上下文向量和offset同时存在才算ready。",
        "",
    ])

    machine = {
        "schema_version": 1,
        "offline": offline,
        "live_local": local,
        "live_global": global_live,
        "e3_sentence": e3_sentence,
        "e3_proposition": e3_proposition,
        "e3_current_retrieval_subset": current_retrieval,
        "e3_sentence_index_build": sentence_build,
        "e3_sentence_retrieval": sentence_retrieval,
        "e3_proposition_index_build": proposition_build,
        "e3_proposition_retrieval": proposition_retrieval,
        "e1_top_document_leaf_retrieval": e1_topdoc,
        "late_chunking_native": late_native,
        "late_chunking_adapter": late_adapter,
        "decisions": decisions,
        "overall_errors": {
            "offline": aggregate_errors(offline),
            "live_local": aggregate_errors(local),
            "live_global": aggregate_errors(global_live),
        },
    }
    return "\n".join(lines), machine


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", type=Path, required=True)
    parser.add_argument("--live-local", type=Path, required=True)
    parser.add_argument("--live-global", type=Path, required=True)
    parser.add_argument("--e3-sentence", type=Path, required=True)
    parser.add_argument("--e3-proposition", type=Path, required=True)
    parser.add_argument("--late-native", type=Path, required=True)
    parser.add_argument("--late-adapter", type=Path, required=True)
    parser.add_argument("--contexts", type=Path, required=True)
    parser.add_argument("--sentence-build", type=Path, required=True)
    parser.add_argument("--sentence-retrieval", type=Path, required=True)
    parser.add_argument("--proposition-build", type=Path, required=True)
    parser.add_argument("--proposition-retrieval", type=Path, required=True)
    parser.add_argument("--e1-topdoc", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    args = parser.parse_args(argv)
    for name in (
        "offline", "live_local", "live_global", "e3_sentence", "e3_proposition",
        "late_native", "late_adapter", "contexts", "sentence_build", "sentence_retrieval",
        "proposition_build", "proposition_retrieval", "e1_topdoc",
    ):
        path = getattr(args, name)
        if not path.is_file():
            parser.error(f"missing {name}: {path}")
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    report, registry = generate(args)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.registry.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report + "\n", encoding="utf-8")
    args.registry.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(report)
    return 1 if any(registry["overall_errors"].values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
