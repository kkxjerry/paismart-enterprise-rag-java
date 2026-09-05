"""Integration and regression gates for the evidence selector experiment."""
from __future__ import annotations

import unittest
from unittest.mock import Mock

from tools.adaptive_rag.budget import BudgetDecision, DynamicEvidenceBudget
from tools.adaptive_rag.controller import AdaptiveRagConfig, AdaptiveRagController
from tools.adaptive_rag.evidence_spans import pack_evidence
from tools.adaptive_rag.requirements import Requirement, RequirementPlan
from tools.qwen_plus_rag_pipeline import ApiResult
from tools.rag_evidence_experiment import anchor_checks, choose, summarize
from tools.rag_packing_loop10 import exact_recall, exact_values


def context(citation, text, rank=1, doc="primary"):
    return {"citation_id": citation, "text": text, "doc_id": doc,
            "document_rank": rank, "title": "Runbook", "source_type": "fixture"}


def mapped_plan(selected):
    return RequirementPlan(
        answerability="conflicting", requirements=(Requirement("R1", "restore time", "conflicting", selected, ""),),
        selected_citations=selected, conflict_citations=selected, model="fixture", latency_ms=0.0,
        usage={"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120, "cached_tokens": 0},
        request_id="fixture", attempts=1)


class SelectorRankRegressionTest(unittest.TestCase):
    def test_retrieval_prior_breaks_lexical_tie_not_input_order(self):
        sources = [context("S1", "restore time 20 minutes", 9, "other"),
                   context("S2", "restore time 20 minutes", 1)]
        packed = pack_evidence(sources, question="restore time", max_chars=1000, max_contexts=1)
        self.assertEqual(packed.contexts[0]["citation_id"], "S2")

    def test_required_source_overrides_document_prior(self):
        sources = [context("S1", "restore time 20 minutes", 1),
                   context("S2", "Contradictory archived timing", 9, "other")]
        packed = pack_evidence(sources, question="restore time", required_citations=("S2",),
                               max_chars=1000, max_contexts=1)
        self.assertEqual(packed.contexts[0]["citation_id"], "S2")

    def test_nonfinite_rank_does_not_break_budget_or_determinism(self):
        for rank in (float("nan"), float("inf"), -1, 0, None):
            sources = [context("S1", "restore time 20 minutes", rank)]
            first = pack_evidence(sources, question="restore time", max_chars=1000, max_contexts=1)
            second = pack_evidence(sources, question="restore time", max_chars=1000, max_contexts=1)
            self.assertEqual(first.rendered, second.rendered)
            self.assertLessEqual(len(first.rendered), 1000)

    def test_conflict_budget_expands_without_dropping_either_side(self):
        sources = [context("S1", "a" * 160), context("S2", "b" * 160, 2, "other")]
        result = DynamicEvidenceBudget("query-spans").build(
            sources, plan=mapped_plan(("S1", "S2")),
            decision=BudgetDecision("quality", 300, 1000, 2, 1, ()), question="restore time")
        self.assertTrue(result.expanded_to_maximum)
        self.assertEqual(result.selected_citations_present, ("S1", "S2"))
        self.assertEqual(result.selected_citations_missing, ())


class ControllerIntegrationTest(unittest.TestCase):
    def test_candidate_reaches_real_generation_messages_without_extra_stage_calls(self):
        mapper, verifier = Mock(model="mapper"), Mock(model="verifier")
        generator = Mock(model="generator")
        captured = []
        def complete(**kwargs):
            captured.append(kwargs["messages"])
            value = kwargs["validator"]({"answerable": True, "answer": "Restore takes tens of minutes [S21].",
                "citations": ["S21"], "covered_requirements": ["R1"], "missing_requirements": []})
            return ApiResult(value, 1.0, {"total_tokens": 1}, "synthetic", "synthetic", 1, "stop", 1024)
        generator.complete_json.side_effect = complete
        controller = AdaptiveRagController(mapper_client=mapper, generator_client=generator,
            verifier_client=verifier, config=AdaptiveRagConfig(evidence_strategy="query-spans", verifier_mode="off"))
        row = {"qid": "fixture", "question": "How long does restore take?", "contexts": [
            context("S1", "Upgrade overview"), context("S11", "Backup preparation"),
            context("S21", "Restore takes tens of minutes depending on snapshot mechanism and database size")],
            "gold_answer": "DO_NOT_LEAK_GOLD", "answer_facts": ["DO_NOT_LEAK_FACT"]}
        result = controller.process(row, forced_mode="fast")
        self.assertIsNone(result["error"])
        self.assertIn("tens of minutes", captured[0][1]["content"])
        self.assertNotIn("DO_NOT_LEAK", str(captured))
        self.assertEqual(generator.complete_json.call_count, 1)
        mapper.complete_json.assert_not_called()
        verifier.complete_json.assert_not_called()
        self.assertEqual(result["budget"]["selection_strategy"], "query-spans")
        self.assertFalse(result["budget"]["per_document_limit_applied"])

    def test_missing_conflict_side_stops_generation_and_preserves_mapper_cost(self):
        unused = Mock(model="must-not-be-called")
        budget = DynamicEvidenceBudget("query-spans")
        budget.decide = Mock(return_value=BudgetDecision("quality", 300, 300, 2, 2, ()))
        controller = AdaptiveRagController(mapper_client=unused, generator_client=unused,
                                          verifier_client=unused, budget=budget)
        plan = mapped_plan(("S1", "S2"))
        controller._plan = Mock(return_value=(plan, [], 0))
        row = {"question": "restore time", "contexts": [context("S1", "Short evidence"),
                                                         context("S2", "x" * 3000)]}
        result = controller.process(row, forced_mode="fast")
        self.assertEqual(result["error"]["stage"], "budget")
        self.assertIn("S2", result["budget"]["selected_citations_missing"])
        self.assertEqual(result["usage"]["total_tokens"], 120)
        unused.complete_json.assert_not_called()


class ExperimentRegistryTest(unittest.TestCase):
    def test_selection_is_stable_with_input_reordering_and_changed_gold(self):
        rows = [{"qid": f"q{i}", "gold_answer": "before"} for i in range(20)]
        ids = lambda items: [r["qid"] for r in items]
        left = choose(rows, 8, ["q4"], "seed")
        right = choose([dict(r, gold_answer="after") for r in reversed(rows)], 8, ["q4"], "seed")
        self.assertEqual(ids(left), ids(right))
        self.assertEqual(ids(left)[0], "q4")

    def test_duplicate_or_missing_ids_rejected(self):
        for rows in ([{"qid": "q"}, {"qid": "q"}], [{"question": "no id"}]):
            with self.assertRaises(ValueError):
                choose(rows, None, [], "seed")

    def test_unknown_requested_qid_and_too_small_limit_rejected(self):
        for qids, limit in ((["absent"], 1), (["q1", "q2"], 1)):
            with self.assertRaises(ValueError):
                choose([{"qid": "q1"}, {"qid": "q2"}], limit, qids, "seed")

    def test_nonevaluable_rows_do_not_pad_proxy_ties(self):
        records = [{"qid": "q", "strategy": strategy, "evaluable": False,
                    "prompt_lexical_recall": 0.0} for strategy in ("legacy", "query-spans")]
        summary = summarize(records, {"phase": "replay"})
        self.assertEqual(summary["evaluable_pair_count"], 0)
        self.assertEqual(summary["proxy_ties"], 0)

    def test_error_usage_is_included_but_error_is_not_quality_success(self):
        records = [{"qid": "q", "strategy": "query-spans", "error": "fixture", "evaluable": True,
                    "usage": {"total_tokens": 99}, "attempts": 1}]
        summary = summarize(records, {"phase": "live"})
        self.assertEqual(summary["aggregates"]["query-spans"]["usage"]["total_tokens"], 99)
        self.assertEqual(summary["aggregates"]["query-spans"]["evaluable"], 0)
        self.assertEqual(summary["aggregates"]["query-spans"]["errors"], 1)

    def test_ledger_probe_does_not_count_identifier_prefixes(self):
        self.assertEqual(anchor_checks("qst_0174", "service_a_version service_b_version")["field_count"], 2)
        self.assertIsNone(anchor_checks("other", "service_a"))

    def test_exact_value_scorer_includes_names_ids_titles_and_paths(self):
        facts = [
            "The mechanism is named TrafficEscrow and uses the traffic_escrow service.",
            'The page is titled "Operational Flows and Policy Gallery".',
            "Use /confluence/templates/access-request-template.",
        ]
        values = exact_values(facts)
        self.assertIn("trafficescrow", values)
        self.assertIn("traffic_escrow", values)
        self.assertIn("operational flows and policy gallery", values)
        self.assertIn("/confluence/templates/access-request-template", values)
        self.assertEqual(
            exact_recall(
                "TrafficEscrow uses traffic_escrow. Operational Flows and Policy Gallery. "
                "/confluence/templates/access-request-template",
                facts,
            ),
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
