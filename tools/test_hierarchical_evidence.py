"""Contract tests for E1/E3/E4/E5/E6 hierarchical evidence experiments."""
from __future__ import annotations

import inspect
import unittest
from dataclasses import replace

from tools.adaptive_rag.hierarchical_evidence import (
    DEFAULT_HIERARCHY_CONFIG,
    HierarchyConfig,
    build_hierarchy,
    contextual_prefix,
    inferred_route,
    pack_hierarchical_evidence,
    requires_canonical_document,
    summary_tree,
)


def context(citation: str, text: str, *, doc: str = "d1", source: str = "confluence", rank: int = 1, **extra):
    return {
        "citation_id": citation,
        "doc_id": doc,
        "text": text,
        "title": extra.pop("title", "Runbook"),
        "source_type": source,
        "document_rank": rank,
        "evidence_score": extra.pop("evidence_score", 0.5),
        "query_coverage": extra.pop("query_coverage", 0.5),
        **extra,
    }


class LeafParentTest(unittest.TestCase):
    def test_third_same_document_chunk_can_supply_answer(self) -> None:
        values = [
            context("S1", "Upgrade overview."),
            context("S11", "Take a snapshot before upgrade."),
            context("S21", "Service restoration is typically tens of minutes. Exact timing depends on database size."),
        ]
        packed = pack_hierarchical_evidence(
            values,
            question="How long does restoration take and what does it depend on?",
            max_chars=1200,
            max_contexts=2,
        )
        self.assertIn("tens of minutes", packed.rendered)
        self.assertIn("database size", packed.rendered)
        self.assertIn("S21", [value["citation_id"] for value in packed.contexts])

    def test_returned_evidence_is_contiguous_original_text(self) -> None:
        original = "Preamble.\n\nThe value is 42 ms.\n\nOnly when the cache is warm.\n\nTail."
        packed = pack_hierarchical_evidence(
            [context("S1", original)], question="What is the value and condition?",
            max_chars=1000, max_contexts=2,
        )
        for value in packed.contexts:
            span = value["hierarchy"]
            self.assertEqual(value["text"], original[span["start_char"]:span["end_char"]])

    def test_serialized_headers_count_against_budget(self) -> None:
        values = [context("S1", "42 ms", title="x" * 1000)]
        packed = pack_hierarchical_evidence(values, question="latency", max_chars=100, max_contexts=1)
        self.assertEqual(packed.contexts, ())
        self.assertLessEqual(len(packed.rendered), 100)

    def test_duplicate_citations_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate"):
            build_hierarchy([context("S1", "one"), context("S1", "two", doc="d2")])


class SourceShapeTest(unittest.TestCase):
    def test_fireflies_sections_become_distinct_parents(self) -> None:
        text = "summary:\nA meeting.\n\ntopics:\n- latency\n- cost\n\nnext_steps:\n- Alice sends logs"
        parents, leaves = build_hierarchy([context("S1", text, source="fireflies")])
        self.assertEqual([parent.kind for parent in parents], [
            "meeting_summary", "meeting_topics", "meeting_next_steps"
        ])
        self.assertTrue(any(leaf.kind == "list_item" for leaf in leaves))

    def test_confluence_heading_hierarchy_is_retained(self) -> None:
        text = "# Policy\nIntro.\n\n## Exceptions\nOnly admins may bypass.\n\n### Audit\nRecord every bypass."
        parents, _ = build_hierarchy([context("S1", text)])
        self.assertEqual(parents[-1].section_path, "Policy / Exceptions / Audit")

    def test_gmail_quoted_messages_are_separate_parents(self) -> None:
        text = (
            "From: Alice <a@example.com>\nSubject: Decision\nUse option B.\n\n"
            "On Tue, Bob wrote:\nFrom: Bob <b@example.com>\nSubject: Re: Decision\nUse option A."
        )
        parents, _ = build_hierarchy([context("S1", text, source="gmail")])
        self.assertGreaterEqual(len(parents), 2)
        self.assertTrue(all(parent.kind == "email_message" for parent in parents))

    def test_slack_metadata_is_carried_to_search_prefix(self) -> None:
        value = context("S1", "Deploy at 16:00.", source="slack", speaker="Lee", event_time="16:00", thread_id="t1")
        prefix = contextual_prefix(value, section_path="thread/t1", kind="conversation_turn")
        self.assertIn("speaker=Lee", prefix)
        self.assertIn("time=16:00", prefix)


class ContextualPrefixTest(unittest.TestCase):
    def test_prefix_is_search_only(self) -> None:
        value = context("S1", "The threshold is 95%.", title="Release Guardrails")
        parents, leaves = build_hierarchy([value])
        self.assertIn("Release Guardrails", leaves[0].search_text)
        packed = pack_hierarchical_evidence([value], question="release threshold", max_chars=1000, max_contexts=1)
        self.assertNotIn("[source=", packed.contexts[0]["text"])
        self.assertFalse(packed.contexts[0]["hierarchy"]["search_context_prefix_returned_to_generator"])
        self.assertEqual(parents[0].text, value["text"])

    def test_prefix_can_resolve_title_only_query(self) -> None:
        values = [
            context("S1", "Generic overview.", doc="wrong", rank=1, title="Other Page"),
            context("S2", "Generic overview.", doc="right", rank=2, title="Operational Flows and Policy Gallery"),
        ]
        with_prefix = pack_hierarchical_evidence(
            values, question="Where is the Operational Flows and Policy Gallery?",
            max_chars=300, max_contexts=1,
        )
        without_prefix = pack_hierarchical_evidence(
            values, question="Where is the Operational Flows and Policy Gallery?",
            max_chars=300, max_contexts=1,
            config=replace(DEFAULT_HIERARCHY_CONFIG, contextual_prefix=False),
        )
        self.assertEqual(with_prefix.contexts[0]["doc_id"], "right")
        self.assertEqual(without_prefix.contexts[0]["doc_id"], "wrong")


class PropositionTest(unittest.TestCase):
    def test_proposition_mode_splits_semicolon_clauses(self) -> None:
        text = "The gateway signed compressed bytes; retries regenerated the gzip envelope; the fix signs canonical JSON."
        _, sentence = build_hierarchy([context("S1", text)], HierarchyConfig(leaf_mode="sentence"))
        _, propositions = build_hierarchy([context("S1", text)], HierarchyConfig(leaf_mode="proposition"))
        self.assertEqual(len(sentence), 1)
        self.assertGreaterEqual(len(propositions), 3)
        self.assertTrue(all(leaf.kind == "proposition" for leaf in propositions))

    def test_selection_api_has_no_gold_parameters(self) -> None:
        signature = inspect.signature(pack_hierarchical_evidence)
        forbidden = {"gold_answer", "answer_facts", "expected_doc_ids"}
        self.assertFalse(forbidden & set(signature.parameters))


class CanonicalAndGlobalTest(unittest.TestCase):
    def test_canonical_document_uses_leaf_support_not_only_rank(self) -> None:
        values = [
            context("S1", "Generic operations page.", doc="ranked", rank=1, title="Process Canvas"),
            context("S2", "The canonical page is Operational Flows and Policy Gallery.", doc="answer", rank=3,
                    title="Operational Flows and Policy Gallery"),
        ]
        packed = pack_hierarchical_evidence(
            values, question="Where is the Operational Flows and Policy Gallery page?",
            max_chars=1000, max_contexts=2,
        )
        self.assertEqual(packed.canonical_documents[0]["doc_id"], "answer")
        self.assertEqual(packed.canonical_documents[0]["role"], "canonical")

    def test_route_detection_is_label_independent_but_accepts_known_type(self) -> None:
        self.assertEqual(inferred_route("What is the company mission?"), "global")
        self.assertEqual(inferred_route("List timeout values"), "local")
        self.assertEqual(inferred_route("anything", "high_level"), "global")
        self.assertTrue(requires_canonical_document("Where can I find the runbook?"))

    def test_global_route_preserves_document_diversity(self) -> None:
        values = [
            context("S1", "Security strategy uses least privilege.", doc="d1", rank=1),
            context("S2", "Security details repeat least privilege.", doc="d1", rank=1),
            context("S3", "Cost strategy uses reserved capacity.", doc="d2", rank=2),
        ]
        packed = pack_hierarchical_evidence(
            values, question="Summarize the overall strategy across security and cost.",
            max_chars=1000, max_contexts=2,
            config=replace(DEFAULT_HIERARCHY_CONFIG, route_mode="global"),
        )
        self.assertEqual({value["doc_id"] for value in packed.contexts}, {"d1", "d2"})

    def test_summary_tree_is_deterministic_and_per_document(self) -> None:
        values = [
            context("S1", "Alpha goal. Alpha constraint. Alpha detail.", doc="d1"),
            context("S2", "Beta goal. Beta constraint.", doc="d2", rank=2),
        ]
        first = summary_tree(values)
        second = summary_tree(list(reversed(values)))
        self.assertEqual(first, second)
        self.assertEqual({value["doc_id"] for value in first}, {"d1", "d2"})


if __name__ == "__main__":
    unittest.main()
