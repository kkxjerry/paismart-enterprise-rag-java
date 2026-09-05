"""Tests for E2 canonical-document requirement generation."""
from __future__ import annotations

import unittest
from unittest.mock import Mock

from tools.adaptive_rag.canonical_generation import (
    build_canonical_messages,
    build_canonical_plan,
    decompose_question,
    generate_canonical_answer,
    validate_canonical_payload,
)
from tools.adaptive_rag.hierarchical_evidence import pack_hierarchical_evidence
from tools.qwen_plus_rag_pipeline import ApiResult, PipelineError


def context(citation: str, doc: str, title: str, text: str, rank: int):
    return {
        "citation_id": citation,
        "doc_id": doc,
        "title": title,
        "text": text,
        "source_type": "confluence",
        "document_rank": rank,
        "query_coverage": 0.5,
        "evidence_score": 0.5,
    }


def evidence(question: str):
    return pack_hierarchical_evidence(
        [
            context("S1", "wrong", "Process Canvas", "A similar approvals page.", 1),
            context(
                "S2",
                "canonical",
                "Operational Flows and Policy Gallery",
                "The canonical one stop page is Operational Flows and Policy Gallery. Templates are in Templates and artifacts (canonical).",
                3,
            ),
        ],
        question=question,
        max_chars=2_000,
        max_contexts=4,
    )


class CanonicalGenerationTest(unittest.TestCase):
    def test_question_decomposition_is_conservative(self) -> None:
        self.assertEqual(
            decompose_question("What is the start time and when did mitigation finish?"),
            ("What is the start time", "when did mitigation finish"),
        )
        self.assertEqual(
            decompose_question("What is the page title and template section?"),
            ("What is the page title and template section?",),
        )

    def test_single_artifact_plan_allows_only_canonical_document(self) -> None:
        question = "Where can I find the Operational Flows and Policy Gallery page?"
        plan = build_canonical_plan(question, evidence(question))
        self.assertEqual(plan.canonical_doc_ids, ("canonical",))
        self.assertTrue(plan.requirements[0].canonical_only)
        self.assertEqual(plan.requirements[0].allowed_doc_ids, ("canonical",))

    def test_comparative_question_keeps_documents_separate_but_allowed(self) -> None:
        question = "Compare the Process Canvas versus Operational Flows and Policy Gallery."
        plan = build_canonical_plan(question, evidence(question))
        self.assertFalse(plan.requirements[0].canonical_only)
        self.assertEqual(set(plan.requirements[0].allowed_doc_ids), {"wrong", "canonical"})

    def test_prompt_labels_canonical_and_does_not_merge_supplements(self) -> None:
        question = "Where can I find the Operational Flows and Policy Gallery page?"
        value = evidence(question)
        plan = build_canonical_plan(question, value)
        prompt = build_canonical_messages(plan, value)[1]["content"]
        self.assertIn("CANONICAL DOCUMENT: doc_id=canonical", prompt)
        self.assertNotIn("SUPPLEMENTAL/COMPARATIVE", prompt)
        self.assertNotIn("A similar approvals page", prompt)

    def test_validator_rejects_source_contamination(self) -> None:
        question = "Where can I find the Operational Flows and Policy Gallery page?"
        plan = build_canonical_plan(question, evidence(question))
        with self.assertRaisesRegex(PipelineError, "outside allowed|canonical"):
            validate_canonical_payload(
                {
                    "requirements": [
                        {
                            "id": "R1",
                            "status": "supported",
                            "answer": "Process Canvas",
                            "citations": ["S1"],
                            "source_doc_ids": ["wrong"],
                        }
                    ]
                },
                plan=plan,
            )

    def test_validator_checks_citation_provenance(self) -> None:
        question = "Where can I find the Operational Flows and Policy Gallery page?"
        plan = build_canonical_plan(question, evidence(question))
        with self.assertRaisesRegex(PipelineError, "must match"):
            validate_canonical_payload(
                {
                    "requirements": [
                        {
                            "id": "R1",
                            "status": "supported",
                            "answer": "Operational Flows and Policy Gallery",
                            "citations": ["S2"],
                            "source_doc_ids": ["wrong"],
                        }
                    ]
                },
                plan=plan,
            )

    def test_generation_composes_requirement_answers_deterministically(self) -> None:
        question = "Where can I find the Operational Flows and Policy Gallery page?"
        value = evidence(question)
        client = Mock(model="fixture")

        def complete(**kwargs):
            payload = {
                "requirements": [
                    {
                        "id": "R1",
                        "status": "supported",
                        "answer": "Use Operational Flows and Policy Gallery [S2].",
                        "citations": ["S2"],
                        "source_doc_ids": ["canonical"],
                    }
                ]
            }
            validated = kwargs["validator"](payload)
            return ApiResult(validated, 1.0, {"total_tokens": 10}, "req", "fixture", 1, "stop", 256)

        client.complete_json.side_effect = complete
        result = generate_canonical_answer(client, question=question, evidence=value)
        self.assertEqual(result.answer, "Use Operational Flows and Policy Gallery [S2].")
        self.assertEqual(result.citations, ("S2",))
        self.assertEqual(result.source_contamination_rate, 0.0)
        self.assertEqual(result.canonical_source_accuracy_proxy, 1.0)


if __name__ == "__main__":
    unittest.main()
