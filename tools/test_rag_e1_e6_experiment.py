"""Tests for the unified E1-E6 experiment and metric isolation."""
from __future__ import annotations

import unittest

from tools.rag_e1_e6_experiment import (
    aggregate_prompt,
    citation_support_proxy,
    hierarchy_config,
    pack_arm,
    paired_live,
    score_prompt,
)


def row():
    return {
        "qid": "q1",
        "question": "Where can I find the Operational Flows and Policy Gallery page?",
        "question_type": "basic",
        "source_types": ["confluence"],
        "answer_facts": ["The page is Operational Flows and Policy Gallery."],
        "expected_doc_ids": ["right"],
        "contexts": [
            {
                "citation_id": "S1",
                "doc_id": "wrong",
                "title": "Process Canvas",
                "text": "Similar approvals overview.",
                "source_type": "confluence",
                "document_rank": 1,
                "evidence_score": 0.4,
                "query_coverage": 0.2,
            },
            {
                "citation_id": "S2",
                "doc_id": "right",
                "title": "Operational Flows and Policy Gallery",
                "text": "The canonical page is Operational Flows and Policy Gallery.",
                "source_type": "confluence",
                "document_rank": 2,
                "evidence_score": 0.6,
                "query_coverage": 0.8,
            },
        ],
    }


class UnifiedExperimentTest(unittest.TestCase):
    def test_all_offline_arms_pack_under_budget(self) -> None:
        for arm in (
            "legacy", "query-spans-v3", "e1-leaf-parent", "e4-context-prefix",
            "e5-proposition", "e6-routed-global",
        ):
            packed = pack_arm(row(), arm, max_chars=1000, max_contexts=3)
            self.assertLessEqual(len(packed["rendered"]), 1000)
            self.assertLessEqual(len(packed["contexts"]), 3)

    def test_e4_prefix_and_e5_proposition_configs_are_distinct(self) -> None:
        e1 = hierarchy_config("e1-leaf-parent", row())
        e4 = hierarchy_config("e4-context-prefix", row())
        e5 = hierarchy_config("e5-proposition", row())
        self.assertFalse(e1.contextual_prefix)
        self.assertTrue(e4.contextual_prefix)
        self.assertEqual(e4.leaf_mode, "sentence")
        self.assertEqual(e5.leaf_mode, "proposition")

    def test_score_uses_gold_only_after_selection(self) -> None:
        selected = pack_arm(row(), "e4-context-prefix", max_chars=1000, max_contexts=2)
        before = [value["citation_id"] for value in selected["contexts"]]
        changed = dict(row(), answer_facts=["unrelated gold mutation"], expected_doc_ids=["wrong"])
        selected_again = pack_arm(changed, "e4-context-prefix", max_chars=1000, max_contexts=2)
        self.assertEqual(before, [value["citation_id"] for value in selected_again["contexts"]])
        score = score_prompt(row(), selected)
        self.assertEqual(score["canonical_doc_expected"], 1.0)

    def test_aggregate_reports_regressions_against_legacy(self) -> None:
        baseline_row = {
            "qid": "q1", "evaluable": True, "canonical_eligible": False,
            "prompt_lexical_recall": 1.0, "prompt_requirement_evidence_coverage": 1.0,
            "prompt_exact_value_recall": None, "prompt_list_item_recall": None,
            "prompt_condition_exception_recall": None, "prompt_negation_recall": None,
            "gold_doc_retained": 1.0, "canonical_doc_expected": None,
            "canonical_near_ties": 0, "rendered_chars": 100, "context_count": 1,
            "selected_doc_count": 1,
        }
        candidate = dict(baseline_row, prompt_lexical_recall=0.5)
        aggregate = aggregate_prompt([candidate], {"q1": baseline_row})
        self.assertEqual(aggregate["paired_regressions"], 1)
        self.assertEqual(aggregate["packing_regression_rate"], 1.0)

    def test_citation_support_proxy_penalizes_uncited_claims(self) -> None:
        contexts = {"S1": {"text": "The threshold is 95 percent."}}
        precision, recall, unsupported = citation_support_proxy(
            "The threshold is 95 percent [S1]. Another claim.", ["S1"], contexts
        )
        self.assertEqual(precision, 1.0)
        self.assertEqual(recall, 0.5)
        self.assertEqual(unsupported, 0.5)

    def test_paired_live_keeps_typed_metric_denominators(self) -> None:
        records = {
            "base": [{"qid": "q", "error": None, "answer_exact_value_accuracy": 0.0}],
            "candidate": [{"qid": "q", "error": None, "answer_exact_value_accuracy": 1.0}],
        }
        result = paired_live(records)["candidate_vs_base"]["answer_exact_value_accuracy"]
        self.assertEqual(result["n"], 1)
        self.assertEqual(result["wins"], 1)
        self.assertEqual(result["regressions"], 0)


if __name__ == "__main__":
    unittest.main()
