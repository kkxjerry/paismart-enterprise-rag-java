from __future__ import annotations

import copy
import unittest

from tools.adaptive_rag.canonical_generation import (
    build_canonical_generation_messages,
    build_global_generation_messages,
    build_source_binding_messages,
    contexts_for_bindings,
    validate_canonical_generation,
    validate_global_generation,
    validate_source_bindings,
)
from tools.adaptive_rag.hierarchical import (
    HierarchyConfig,
    build_global_hierarchy,
    contextual_prefix,
    pack_hierarchical,
)
from tools.adaptive_rag.requirements import Requirement, RequirementPlan
from tools.qwen_plus_rag_pipeline import PipelineError


def context(citation: str, text: str, *, doc: str, rank: int = 1, title: str = "Doc", source: str = "confluence", section: str = ""):
    return {
        "citation_id": citation,
        "text": text,
        "doc_id": doc,
        "document_rank": rank,
        "title": title,
        "source_type": source,
        "section_path": section,
        "query_coverage": 0.5,
        "evidence_score": 0.7,
    }


def plan() -> RequirementPlan:
    return RequirementPlan(
        answerability="answerable",
        requirements=(Requirement("R1", "name the canonical page", "supported", (), ""),),
        selected_citations=(),
        conflict_citations=(),
        model="fixture",
        latency_ms=0.0,
        usage={},
        request_id="",
        attempts=0,
    )


class HierarchicalPackingTest(unittest.TestCase):
    def test_leaf_match_returns_contiguous_parent(self) -> None:
        values = [
            context("S1", "Overview of upgrades.\n\nRollback preconditions are documented.", doc="d1"),
            context("S2", "Support coverage details.\n\nRestore is typically tens of minutes when snapshots exist.", doc="d1"),
        ]
        before = copy.deepcopy(values)
        result = pack_hierarchical(
            values,
            question="How long is restore with snapshots?",
            max_chars=900,
            max_contexts=1,
        )
        self.assertEqual(result.contexts[0]["citation_id"], "S2")
        self.assertIn("tens of minutes", result.contexts[0]["text"])
        span = result.contexts[0]["hierarchy"]
        self.assertEqual(result.contexts[0]["text"], values[1]["text"][span["start_char"]:span["end_char"]])
        self.assertEqual(values, before)

    def test_contextual_prefix_can_distinguish_similar_pages(self) -> None:
        values = [
            context("S1", "This is the canonical reference.", doc="wrong", rank=1, title="Process Canvas"),
            context("S2", "This is the canonical reference.", doc="right", rank=2, title="Operational Flows and Policy Gallery"),
        ]
        result = pack_hierarchical(
            values,
            question="Where is the Operational Flows and Policy Gallery page?",
            max_chars=900,
            max_contexts=1,
            config=HierarchyConfig(strategy="contextual-leaf-parent"),
        )
        self.assertEqual(result.contexts[0]["citation_id"], "S2")
        self.assertIn("title=Operational Flows", contextual_prefix(values[1]))

    def test_proposition_key_does_not_replace_raw_citation_text(self) -> None:
        original = context("S1", "- max_file_size: 10MiB", doc="d1", title="Upload Policy")
        result = pack_hierarchical(
            [original],
            question="What is max_file_size?",
            max_chars=800,
            max_contexts=1,
            config=HierarchyConfig(strategy="proposition-parent"),
        )
        self.assertIn("max_file_size", result.selected_leaves[0]["leaf_text"])
        self.assertIn("- max_file_size: 10MiB", result.contexts[0]["text"])

    def test_exact_character_budget_includes_headers(self) -> None:
        values = [context("S1", "A" * 400, doc="d1", title="T" * 200)]
        result = pack_hierarchical(values, question="A", max_chars=100, max_contexts=2)
        self.assertEqual(result.rendered, "")
        self.assertEqual(result.contexts, ())

    def test_required_citation_is_reserved(self) -> None:
        values = [
            context("S1", "high lexical restore duration", doc="d1"),
            context("S2", "conflicting archival statement", doc="d2", rank=9),
        ]
        result = pack_hierarchical(
            values,
            question="restore duration",
            required_citations=("S2",),
            max_chars=1000,
            max_contexts=1,
        )
        self.assertEqual(result.contexts[0]["citation_id"], "S2")

    def test_global_hierarchy_is_extractive_and_keeps_child_citations(self) -> None:
        values = [
            context("S1", "Mission emphasizes safe inference.", doc="d1", title="Mission"),
            context("S2", "Strategy emphasizes efficient inference.", doc="d2", rank=2, title="Strategy"),
        ]
        result = build_global_hierarchy(values, question="What are the overall mission and strategy themes?")
        self.assertEqual({d["doc_id"] for d in result["documents"]}, {"d1", "d2"})
        self.assertIn("[S1]", result["rendered"])
        self.assertIn("[S2]", result["rendered"])


class CanonicalGenerationTest(unittest.TestCase):
    def test_messages_group_sources_and_do_not_receive_gold(self) -> None:
        values = [
            context("S1", "Process canvas.", doc="wrong", title="Process Canvas"),
            context("S2", "Operational gallery.", doc="right", title="Operational Flows and Policy Gallery"),
        ]
        messages, citation_to_doc, single = build_canonical_generation_messages(
            question="Where is the Operational Flows and Policy Gallery page?",
            plan=plan(),
            contexts=values,
        )
        self.assertTrue(single)
        self.assertEqual(citation_to_doc, {"S1": "wrong", "S2": "right"})
        self.assertIn("===== NEXT DOCUMENT =====", messages[1]["content"])

    def test_source_binding_rejects_evidence_from_another_document(self) -> None:
        with self.assertRaisesRegex(PipelineError, "do not belong"):
            validate_source_bindings(
                {
                    "bindings": [{
                        "id": "R1",
                        "canonical_doc_id": "right",
                        "evidence_citations": ["S1"],
                        "missing": False,
                        "conflicting": False,
                    }]
                },
                citation_to_doc={"S1": "wrong", "S2": "right"},
                requirement_ids={"R1"},
                single_source=True,
            )

    def test_source_binding_restricts_generation_contexts(self) -> None:
        values = [
            context("S1", "Wrong page.", doc="wrong"),
            context("S2", "Canonical page.", doc="right"),
        ]
        messages, citation_to_doc, single = build_source_binding_messages(
            question="Where is the canonical page?",
            plan=plan(),
            contexts=values,
        )
        self.assertIn("Candidate document cards", messages[1]["content"])
        bindings = validate_source_bindings(
            {
                "bindings": [{
                    "id": "R1",
                    "canonical_doc_id": "right",
                    "evidence_citations": ["S2"],
                    "missing": False,
                    "conflicting": False,
                }]
            },
            citation_to_doc=citation_to_doc,
            requirement_ids={"R1"},
            single_source=single,
        )
        self.assertEqual([value["citation_id"] for value in contexts_for_bindings(values, bindings)], ["S2"])

    def test_validator_normalizes_doc_id_to_one_concrete_citation_document(self) -> None:
        payload = {
            "requirements": [{
                "id": "R1",
                "canonical_doc_id": "copied-wrong-id",
                "answer": "Use the page [S1].",
                "citations": ["S1"],
                "missing": False,
            }]
        }
        result = validate_canonical_generation(
            payload,
            valid_citations={"S1", "S2"},
            citation_to_doc={"S1": "actual", "S2": "other"},
            requirement_ids={"R1"},
            single_source=True,
        )
        self.assertEqual(result["canonical_doc_ids"], ["actual"])
        self.assertTrue(result["requirements"][0]["normalized"])

    def test_validator_rejects_true_cross_document_citations(self) -> None:
        payload = {
            "requirements": [{
                "id": "R1",
                "canonical_doc_id": "d1",
                "answer": "Mixed answer [S1][S2].",
                "citations": ["S1", "S2"],
                "missing": False,
            }]
        }
        with self.assertRaisesRegex(PipelineError, "outside canonical"):
            validate_canonical_generation(
                payload,
                valid_citations={"S1", "S2"},
                citation_to_doc={"S1": "d1", "S2": "d2"},
                requirement_ids={"R1"},
                single_source=True,
            )

    def test_validator_reconstructs_answer_from_bound_requirement(self) -> None:
        payload = {
            "answer": "untrusted top level",
            "requirements": [{
                "id": "R1",
                "canonical_doc_id": "right",
                "answer": "Use Operational Flows [S2].",
                "citations": ["S2"],
                "missing": False,
            }],
        }
        result = validate_canonical_generation(
            payload,
            valid_citations={"S1", "S2"},
            citation_to_doc={"S1": "wrong", "S2": "right"},
            requirement_ids={"R1"},
            single_source=True,
        )
        self.assertEqual(result["answer"], "Use Operational Flows [S2].")
        self.assertEqual(result["canonical_doc_ids"], ["right"])
        self.assertTrue(result["top_level_answer_ignored"])

    def test_single_source_contract_rejects_two_canonical_documents(self) -> None:
        payload = {
            "requirements": [
                {"id": "R1", "canonical_doc_id": "d1", "answer": "A [S1].", "citations": ["S1"], "missing": False},
                {"id": "R2", "canonical_doc_id": "d2", "answer": "B [S2].", "citations": ["S2"], "missing": False},
            ]
        }
        with self.assertRaisesRegex(PipelineError, "more than one"):
            validate_canonical_generation(
                payload,
                valid_citations={"S1", "S2"},
                citation_to_doc={"S1": "d1", "S2": "d2"},
                requirement_ids={"R1", "R2"},
                single_source=True,
            )

    def test_global_validator_requires_multiple_documents(self) -> None:
        result = validate_global_generation(
            {
                "answerable": True,
                "answer": "Global claim [S1].",
                "citations": ["S1"],
                "documents_used": ["d1"],
            },
            valid_citations={"S1"},
            valid_documents={"d1", "d2"},
        )
        self.assertFalse(result["answerable"])
        self.assertTrue(result["global_support_insufficient"])

    def test_global_messages_use_extractive_hierarchy(self) -> None:
        messages, hierarchy = build_global_generation_messages(
            question="What themes appear across the company?",
            contexts=[
                context("S1", "Safety is a theme.", doc="d1"),
                context("S2", "Efficiency is a theme.", doc="d2", rank=2),
            ],
        )
        self.assertEqual(len(hierarchy["documents"]), 2)
        self.assertIn("[S1]", messages[1]["content"])


if __name__ == "__main__":
    unittest.main()
