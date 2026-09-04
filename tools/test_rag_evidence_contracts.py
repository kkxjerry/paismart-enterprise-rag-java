"""Regression tests for bugs reproduced from the 2026-09-04 trace audit."""
from __future__ import annotations

import unittest
from dataclasses import replace

from tools.adaptive_rag.budget import BudgetDecision, DynamicEvidenceBudget
from tools.adaptive_rag.claims import split_cited_segments
from tools.adaptive_rag.controller import normalize_sentence_citations, uncited_factual_segments
from tools.adaptive_rag.requirements import Requirement, RequirementPlan
from tools.adaptive_rag.verifier import extract_numbered_claims
from tools.qwen_plus_rag_pipeline import render_contexts
from tools.rag_trace_audit import audit


def make_plan(selected: tuple[str, ...]) -> RequirementPlan:
    return RequirementPlan(
        answerability="answerable",
        requirements=(Requirement("R1", "Read the value", "supported", selected, ""),),
        selected_citations=selected, conflict_citations=(), model="fixture",
        latency_ms=0.0, usage={}, request_id="fixture", attempts=0,
    )


class CitationBoundaryRegressionTest(unittest.TestCase):
    def test_post_period_citation_belongs_to_previous_claim(self) -> None:
        claims = extract_numbered_claims("Fintech was first. [S3][S4] There were three stories [S5].")
        self.assertEqual(len(claims), 2)
        self.assertEqual(claims[0]["citations"], ["S3", "S4"])
        self.assertEqual(claims[1]["citations"], ["S5"])

    def test_existing_citations_are_not_replaced_by_all_answer_citations(self) -> None:
        text = "First fact. [S1] Second fact [S2]."
        self.assertEqual(normalize_sentence_citations(text, ["S1", "S2"]), (text, 0))
        self.assertEqual(uncited_factual_segments(text), [])

    def test_pre_period_citations_keep_their_claim(self) -> None:
        self.assertEqual(
            [c["citations"] for c in extract_numbered_claims("First [S1]. Second [S2].")],
            [["S1"], ["S2"]],
        )

    def test_newline_after_post_period_citation(self) -> None:
        claims = extract_numbered_claims("First. [S1]\nSecond. [S2]")
        self.assertEqual([c["citations"] for c in claims], [["S1"], ["S2"]])

    def test_abbreviation_vs_is_not_a_sentence_boundary(self) -> None:
        text = "Snapshot restore vs. logical restore depends on storage [S1]."
        self.assertEqual(split_cited_segments(text), [text])

    def test_list_ordinal_is_not_a_claim(self) -> None:
        self.assertEqual(split_cited_segments("1. First [S1].\n2. Second [S2]."),
                         ["1. First [S1].", "2. Second [S2]."])

    def test_uncited_sentence_remains_visible_to_diagnostics(self) -> None:
        self.assertEqual(uncited_factual_segments("First [S1]. Second fact."), ["Second fact."])

    def test_decimal_and_empty_answer(self) -> None:
        text = "The limit is 3.14 MiB [S1]."
        self.assertEqual(split_cited_segments(text), [text])
        self.assertEqual(split_cited_segments(""), [])


class BudgetContentRegressionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.contexts = [{"citation_id": "S1", "doc_id": "d1", "title": "Test",
                          "source_type": "fixture", "text": "a" * 160 + " critical value"}]
        _, _, size = render_contexts(self.contexts, max_contexts=1, max_chars=1000)
        self.decision = BudgetDecision("quality", size - 20, size, 1, 1, ())

    def test_required_id_present_but_truncated_triggers_expansion(self) -> None:
        result = DynamicEvidenceBudget().build(self.contexts, plan=make_plan(("S1",)), decision=self.decision)
        self.assertTrue(result.expanded_to_maximum)
        self.assertEqual(result.contexts[0]["text"], self.contexts[0]["text"])
        self.assertEqual(result.selected_citations_truncated, ())
        self.assertEqual(result.selected_citations_missing, ())

    def test_hard_cap_reports_truncation_instead_of_claiming_complete(self) -> None:
        decision = replace(self.decision, maximum_chars=self.decision.initial_chars)
        result = DynamicEvidenceBudget().build(self.contexts, plan=make_plan(("S1",)), decision=decision)
        self.assertFalse(result.expanded_to_maximum)
        self.assertEqual(result.selected_citations_truncated, ("S1",))
        self.assertEqual(result.selected_citations_missing, ())
        self.assertLessEqual(result.rendered_chars, decision.maximum_chars)

    def test_fallback_chunk_does_not_trigger_expansion(self) -> None:
        result = DynamicEvidenceBudget().build(self.contexts, plan=make_plan(()), decision=self.decision)
        self.assertFalse(result.expanded_to_maximum)
        self.assertEqual(result.selected_citations_truncated, ())


class AuditIntegrityTest(unittest.TestCase):
    def test_duplicate_sources_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            audit([{'qid': 'q1'}, {'qid': 'q1'}], [])

    def test_mixed_signatures_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, 'Mixed run'):
            audit([{'qid': 'q1'}, {'qid': 'q2'}],
                  [{'qid': 'q1', 'run_signature': 'a'}, {'qid': 'q2', 'run_signature': 'b'}])

    def test_question_mismatch_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, 'Question mismatch'):
            audit([{'qid': 'q1', 'question': 'original'}],
                  [{'qid': 'q1', 'question': 'different'}])


if __name__ == "__main__":
    unittest.main()
