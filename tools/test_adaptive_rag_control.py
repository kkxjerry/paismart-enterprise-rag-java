from __future__ import annotations

import unittest

from tools.adaptive_rag.attribution import attribute_row, summarize_attributions
from tools.adaptive_rag.budget import DynamicEvidenceBudget, prioritize_contexts
from tools.adaptive_rag.controller import AdaptiveRagConfig, AdaptiveRagController, validate_adaptive_generation
from tools.adaptive_rag.features import AdaptiveRouter, extract_features
from tools.adaptive_rag.requirements import (
    Requirement,
    RequirementPlan,
    validate_requirement_plan,
)
from tools.adaptive_rag.retrieval import merge_contexts, secondary_queries
from tools.adaptive_rag.verifier import validate_verification, verification_trigger_reasons
from tools.qwen_plus_rag_pipeline import ApiResult, PipelineError


class FakeClient:
    def __init__(self, model: str, payloads: list[dict]) -> None:
        self.model = model
        self.payloads = list(payloads)
        self.calls = 0

    def complete_json(self, *, validator, **_kwargs):
        self.calls += 1
        payload = self.payloads.pop(0)
        value = validator(payload)
        return ApiResult(
            value=value,
            latency_ms=10.0,
            usage={"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120, "cached_tokens": 0},
            request_id=f"req-{self.calls}",
            returned_model=self.model,
            attempts=1,
            finish_reason="stop",
            max_tokens_used=512,
        )


def context(citation: str, doc: str, rank: int, text: str, *, source: str = "jira", routes=None):
    return {
        "citation_id": citation,
        "doc_id": doc,
        "chunk_es_id": f"{doc}:{citation}",
        "document_rank": rank,
        "rank": rank,
        "source_type": source,
        "title": doc,
        "text": text,
        "query_coverage": 0.5,
        "evidence_score": 1.0 / rank,
        "route_signals": routes or [],
    }


def plan(*, status="answerable", requirements=None, selected=("S1",)):
    reqs = requirements or (
        Requirement("R1", "state the limit", "supported", ("S1",), "limit"),
    )
    return RequirementPlan(
        answerability=status,
        requirements=tuple(reqs),
        selected_citations=tuple(selected),
        conflict_citations=tuple(),
        model="qwen-flash",
        latency_ms=1.0,
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2, "cached_tokens": 0},
        request_id="r",
        attempts=1,
    )


class AdaptiveControlTest(unittest.TestCase):
    def test_router_uses_retrieval_signals_not_benchmark_labels(self):
        row = {
            "question": "What is the limit?",
            "question_type": "constrained",
            "expected_doc_ids": ["gold"],
            "answer_facts": ["secret"],
            "contexts": [
                context("S1", "d1", 1, "limit 10", routes=[
                    {"route": "dense", "rank": 1},
                    {"route": "bm25_original", "rank": 1},
                    {"route": "bm25_english", "rank": 1},
                ])
            ],
        }
        decision = AdaptiveRouter().decide(row)
        self.assertEqual(decision.mode, "fast")
        self.assertNotIn("question_type", decision.features.to_dict())

    def test_router_sends_conflicting_multi_part_query_to_deep(self):
        row = {
            "question": "What caused the incident, what was the workaround, and which version is correct?",
            "contexts": [context("S1", "d1", 1, "old value"), context("S2", "d2", 2, "new value")],
            "evidence_conflicts": [{"id": "c1"}],
        }
        decision = AdaptiveRouter().decide(row)
        self.assertEqual(decision.mode, "deep")

    def test_feature_margin_comes_from_ranked_documents(self):
        features = extract_features(
            {"question": "limit", "contexts": [context("S1", "d1", 1, "limit")]},
            {"ranked_documents": [{"score": 1.0}, {"score": 0.995}]},
        )
        self.assertAlmostEqual(features.top1_margin or 0.0, 0.005)

    def test_requirement_plan_validation(self):
        value = validate_requirement_plan(
            {
                "answerability": "partial",
                "requirements": [
                    {"id": "R1", "requirement": "cause", "status": "supported", "citations": ["S1"], "search_query": ""},
                    {"id": "R2", "requirement": "ship date", "status": "missing", "citations": [], "search_query": "ship date"},
                ],
                "selected_citations": ["S1"],
                "conflict_citations": [],
            },
            valid_citations={"S1", "S2"},
            max_requirements=4,
            max_selected=4,
        )
        self.assertEqual(value["answerability"], "partial")
        self.assertEqual(value["requirements"][1]["search_query"], "ship date")

    def test_requirement_plan_drops_selected_citations_not_used_by_requirements(self):
        value = validate_requirement_plan(
            {
                "answerability": "answerable",
                "requirements": [
                    {"id": "R1", "requirement": "state the limit", "status": "supported", "citations": ["S2"], "search_query": ""},
                ],
                "selected_citations": ["S1", "S2", "S3"],
                "conflict_citations": [],
            },
            valid_citations={"S1", "S2", "S3"},
            max_requirements=4,
            max_selected=4,
        )
        self.assertEqual(value["selected_citations"], ["S2"])

    def test_requirement_plan_compacts_optional_evidence_without_losing_requirement_coverage(self):
        value = validate_requirement_plan(
            {
                "answerability": "answerable",
                "requirements": [
                    {"id": "R1", "requirement": "first", "status": "supported", "citations": ["S1", "S4"], "search_query": ""},
                    {"id": "R2", "requirement": "second", "status": "supported", "citations": ["S2", "S5"], "search_query": ""},
                    {"id": "R3", "requirement": "third", "status": "supported", "citations": ["S3", "S6"], "search_query": ""},
                ],
                "selected_citations": ["S4", "S5", "S6", "S1", "S2", "S3"],
                "conflict_citations": [],
            },
            valid_citations={"S1", "S2", "S3", "S4", "S5", "S6"},
            max_requirements=4,
            max_selected=3,
        )
        self.assertTrue(value["selection_compacted"])
        self.assertEqual(value["selected_citations_before_compaction"], 6)
        self.assertEqual(value["selected_citations"], ["S1", "S2", "S3"])
        self.assertEqual(
            [requirement["citations"] for requirement in value["requirements"]],
            [["S1"], ["S2"], ["S3"]],
        )

    def test_supported_requirement_requires_citation(self):
        with self.assertRaises(PipelineError):
            validate_requirement_plan(
                {
                    "answerability": "answerable",
                    "requirements": [{"id": "R1", "requirement": "cause", "status": "supported", "citations": []}],
                    "selected_citations": [],
                    "conflict_citations": [],
                },
                valid_citations={"S1"},
                max_requirements=2,
                max_selected=2,
            )

    def test_dynamic_budget_expands_when_selected_citation_is_late(self):
        contexts = [context(f"S{i}", f"d{i}", i, "x" * 1200) for i in range(1, 8)]
        late_plan = plan(selected=("S7",))
        route = AdaptiveRouter().decide({"question": "What is x?", "contexts": contexts}, forced_mode="fast")
        policy = DynamicEvidenceBudget()
        decision = policy.decide(route, late_plan)
        result = policy.build(contexts, plan=late_plan, decision=decision)
        self.assertIn("S7", result.selected_citations_present)

    def test_dynamic_budget_reserves_capacity_for_many_selected_citations(self):
        contexts = [context(f"S{i}", f"d{i}", i, "x" * 1_200) for i in range(1, 17)]
        requirements = tuple(
            Requirement(f"R{i}", f"requirement {i}", "supported", (f"S{i}",), "")
            for i in range(1, 17)
        )
        many = plan(requirements=requirements, selected=tuple(f"S{i}" for i in range(1, 17)))
        route = AdaptiveRouter().decide(
            {"question": "Provide the complete procedure", "contexts": contexts},
            forced_mode="quality",
        )
        policy = DynamicEvidenceBudget()
        decision = policy.decide(route, many)
        result = policy.build(contexts, plan=many, decision=decision)
        self.assertGreaterEqual(decision.maximum_chars, 24_000)
        self.assertEqual(result.selected_citations_missing, tuple())
        self.assertEqual(len(result.selected_citations_present), 16)

    def test_priority_preserves_selected_evidence_over_document_cap(self):
        contexts = [context("S1", "d1", 1, "a"), context("S2", "d1", 1, "b"), context("S3", "d1", 1, "c")]
        ordered = prioritize_contexts(contexts, selected_citations=["S1", "S2", "S3"], conflict_citations=[], max_per_document=1)
        self.assertEqual([value["citation_id"] for value in ordered], ["S1", "S2", "S3"])

    def test_secondary_query_targets_missing_requirement(self):
        missing = Requirement("R2", "ship date", "missing", tuple(), "ticket ship date")
        p = plan(status="partial", requirements=(Requirement("R1", "cause", "supported", ("S1",), ""), missing))
        route = AdaptiveRouter().decide({"question": "cause and date", "contexts": [context("S1", "d1", 1, "cause")]}, forced_mode="quality")
        self.assertEqual(secondary_queries("cause and date", route, p, max_queries=2), ["ticket ship date"])

    def test_secondary_contexts_are_deduplicated_and_recited(self):
        original = [context("S1", "d1", 1, "a")]
        duplicate = dict(original[0])
        new = context("S1", "d2", 2, "b")
        merged = merge_contexts(original, [{"contexts": [duplicate, new]}], ["subquery"])
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[1]["citation_id"], "S2")
        self.assertTrue(merged[1]["secondary_retrieval"])

    def test_generation_requirement_coverage_validation(self):
        result = validate_adaptive_generation(
            {
                "answerable": True,
                "answer": "Limit is 10. [S1]",
                "citations": ["S1"],
                "covered_requirements": ["R1"],
                "missing_requirements": [],
            },
            valid_citations={"S1"},
            requirement_ids={"R1"},
        )
        self.assertEqual(result["covered_requirements"], ["R1"])

    def test_generation_normalizes_uncited_factual_sentence_for_verification(self):
        result = validate_adaptive_generation(
            {
                "answerable": True,
                "answer": "The limit is 10 MiB. The rollout starts Monday. [S1]",
                "citations": ["S1"],
                "covered_requirements": ["R1"],
                "missing_requirements": [],
            },
            valid_citations={"S1"},
            requirement_ids={"R1"},
        )
        self.assertTrue(result["sentence_citation_normalized"])
        self.assertIn("10 MiB. [S1]", result["answer"])

    def test_verifier_accepts_none_placeholder_only_for_unsupported_claim(self):
        result = validate_verification(
            {
                "status": "repair",
                "claims": [
                    {"id": "C1", "citations": ["NONE"], "status": "unsupported"},
                ],
                "answerable": True,
                "revised_answer": "The available evidence does not establish the missing value.",
                "citations": ["S1"],
            },
            valid_citations={"S1"},
            claim_text_by_id={"C1": "The missing value is 10."},
            original_answer="The missing value is 10. [S1]",
            original_citations=["S1"],
        )
        self.assertEqual(result["claims"][0]["citations"], [])
        self.assertEqual(result["status"], "reject")

    def test_verifier_normalizes_pass_with_unsupported_claims_to_repair(self):
        result = validate_verification(
            {
                "status": "pass",
                "claims": [
                    {"id": "C1", "citations": ["S1"], "status": "supported"},
                    {"id": "C2", "citations": ["S1"], "status": "unsupported"},
                ],
                "answerable": True,
                "revised_answer": "",
                "citations": [],
            },
            valid_citations={"S1"},
            claim_text_by_id={
                "C1": "The documented limit is 10. [S1]",
                "C2": "The undocumented limit is 20. [S1]",
            },
            original_answer="The documented limit is 10. [S1]\nThe undocumented limit is 20. [S1]",
            original_citations=["S1"],
        )
        self.assertEqual(result["status"], "repair")
        self.assertTrue(result["status_normalized"])
        self.assertIn("documented limit is 10", result["revised_answer"])
        self.assertNotIn("undocumented limit is 20", result["revised_answer"])

    def test_verifier_trigger_for_exact_quality_answer(self):
        reasons = verification_trigger_reasons(
            answer="Version v1.2 ships on 2026-09-03. [S1][S2][S3]",
            citations=["S1", "S2", "S3"],
            contexts=[
                context("S1", "d1", 1, "v1.2", source="jira"),
                context("S2", "d2", 2, "2026-09-03", source="confluence"),
                context("S3", "d3", 3, "ship date", source="slack"),
            ],
            plan=plan(),
            route=AdaptiveRouter().decide({"question": "version?", "contexts": [context("S1", "d1", 1, "v1.2")]}, forced_mode="quality"),
        )
        self.assertIn("cross_source_exact_values", reasons)

    def test_controller_returns_grounded_refusal_when_acl_yields_no_contexts(self):
        unused = FakeClient("qwen-flash", [])
        controller = AdaptiveRagController(
            mapper_client=unused,
            generator_client=unused,
            verifier_client=unused,
        )

        result = controller.process({"qid": "q-empty", "question": "private value", "contexts": []})

        self.assertIsNone(result["error"])
        self.assertFalse(result["generation"]["answerable"])
        self.assertEqual(result["generation"]["answer"], "INSUFFICIENT_EVIDENCE")
        self.assertEqual(result["requirements"]["answerability"], "insufficient")
        self.assertEqual(unused.calls, 0)

    def test_controller_quality_path_maps_generates_and_verifies(self):
        mapper = FakeClient("qwen-flash", [{
            "answerability": "answerable",
            "requirements": [{"id": "R1", "requirement": "state the limit", "status": "supported", "citations": ["S1"], "search_query": ""}],
            "selected_citations": ["S1"],
            "conflict_citations": [],
        }])
        generator = FakeClient("qwen-flash", [{
            "answerable": True,
            "answer": "The limit is 10 MiB. [S1]",
            "citations": ["S1"],
            "covered_requirements": ["R1"],
            "missing_requirements": [],
        }])
        verifier = FakeClient("qwen-flash", [{
            "status": "pass",
            "claims": [{"claim": "The limit is 10 MiB.", "citations": ["S1"], "status": "supported"}],
            "answerable": True,
            "revised_answer": "The limit is 10 MiB. [S1]",
            "citations": ["S1"],
        }])
        controller = AdaptiveRagController(
            mapper_client=mapper,
            generator_client=generator,
            verifier_client=verifier,
            config=AdaptiveRagConfig(verifier_mode="always"),
        )
        row = {
            "qid": "q1",
            "question": "What is the upload limit and exact unit?",
            "contexts": [context("S1", "d1", 1, "The limit is 10 MiB.")],
            "expected_doc_ids": ["d1"],
            "answer_facts": ["The limit is 10 MiB."],
            "is_evaluable": True,
            "retrieval_hit_at_10": True,
        }
        result = controller.process(row, forced_mode="quality")
        self.assertIsNone(result["error"])
        self.assertEqual(result["generation"]["answer"], "The limit is 10 MiB. [S1]")
        self.assertEqual(result["verification"]["status"], "pass")

    def test_failure_attribution_identifies_window_miss(self):
        source = {
            "qid": "q1",
            "question": "limit",
            "expected_doc_ids": ["d1"],
            "ranked_doc_ids": ["d1"],
            "answer_facts": ["limit is 10 MiB"],
            "contexts": [context("S1", "d1", 1, "limit is 10 MiB")],
        }
        result = {
            "selected_contexts": [context("S2", "d2", 2, "unrelated")],
            "generation": {"answer": "unknown", "citations": []},
            "metrics": {"invalid_citations": []},
            "router": {"mode": "fast"},
            "verification": {"status": "skipped", "claims": []},
        }
        attribution = attribute_row(source, result)
        self.assertEqual(attribution["dominant_failure_stage"], "R3_WINDOW_MISS")

    def test_failure_summary_counts_stages(self):
        summary = summarize_attributions([
            {"dominant_failure_stage": "R1_RETRIEVAL_MISS", "fact_attributions": [{"stage": "R1_RETRIEVAL_MISS"}]},
            {"dominant_failure_stage": "OK", "fact_attributions": [{"stage": "OK"}]},
        ])
        self.assertEqual(summary["questions_total"], 2)
        self.assertEqual(summary["dominant_failure_stage_counts"]["OK"], 1)


if __name__ == "__main__":
    unittest.main()
