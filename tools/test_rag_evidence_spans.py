"""Synthetic contract tests. These are NOT a replay of the real 500 queries."""
from __future__ import annotations

import copy
import random
import string
import unittest

from tools.adaptive_rag.evidence_spans import pack_evidence


def chunk(citation: str, text: str, doc: str = "d1", **kwargs):
    return {**dict(citation_id=citation, doc_id=doc, title="Fixture", source_type="fixture", text=text), **kwargs}


class EvidenceSpanTests(unittest.TestCase):
    def pack(self, contexts, **kwargs):
        return pack_evidence(contexts, question=kwargs.pop("question", "restore duration"),
                             max_chars=kwargs.pop("max_chars", 10000),
                             max_contexts=kwargs.pop("max_contexts", 12), **kwargs)

    def test_third_chunk_from_same_document_is_not_removed(self):
        contexts = [chunk("S1", "Upgrade overview."), chunk("S11", "Backup prerequisites."),
                    chunk("S21", "Restore duration is tens of minutes, depending on snapshot and database size.")]
        result = self.pack(contexts)
        self.assertEqual(result.contexts[0]["citation_id"], "S21")
        self.assertEqual(len(result.contexts), 3)
        self.assertIn("depending on snapshot and database size", result.rendered)

    def test_list_in_third_chunk_remains_intact(self):
        text = "Required ledger fields:\n" + "\n".join(f"- field_{i}" for i in range(11))
        contexts = [chunk("S1", "Ledger overview."), chunk("S11", "Ledger owners."), chunk("S21", text)]
        result = self.pack(contexts, question="List the required ledger fields", max_contexts=1)
        self.assertEqual(result.contexts[0]["text"], text)
        self.assertEqual(result.contexts[0]["citation_id"], "S21")

    def test_matching_paragraph_keeps_following_condition(self):
        text = ("Background without requested information. " * 90 + "\n\n"
                "Before operating, take a snapshot.\n\n"
                "The zephyr recovery duration is 20 minutes.\n\n"
                "Only when snapshots are available; otherwise no recovery estimate is promised.\n\n"
                + "Unrelated deployment history. " * 100)
        result = self.pack([chunk("S8", text)], question="zephyr recovery duration", max_chars=700)
        self.assertEqual(len(result.contexts), 1)
        self.assertIn("20 minutes", result.rendered)
        self.assertIn("otherwise no recovery estimate", result.rendered)
        self.assertNotIn("Unrelated deployment history", result.rendered)

    def test_unicode_offsets_are_exact_original_slices(self):
        original = chunk("S2", "恢复需要几十分钟；具体取决于数据库大小。🙂", acl={"tenant": "authorized"})
        result = self.pack([original], question="恢复需要多久")
        value = result.contexts[0]
        span = value["evidence_span"]
        self.assertEqual(value["text"], original["text"][span["start_char"]:span["end_char"]])
        self.assertEqual(value["acl"], original["acl"])
        self.assertEqual(span["offset_unit"], "python_character_within_input_chunk")

    def test_input_is_not_mutated(self):
        contexts = [chunk("S1", "Restore duration 20 minutes.")]
        before = copy.deepcopy(contexts)
        self.pack(contexts)
        self.assertEqual(contexts, before)

    def test_selected_and_conflicting_sources_stay_whole(self):
        contexts = [chunk("S1", "Restore duration 20 minutes."), chunk("S2", "Restore duration is not 20 minutes.")]
        result = self.pack(contexts, required_citations=("S2", "S1"))
        self.assertEqual([c["citation_id"] for c in result.contexts], ["S2", "S1"])
        self.assertEqual([c["text"] for c in result.contexts], [contexts[1]["text"], contexts[0]["text"]])

    def test_oversized_required_citation_is_missing_not_truncated(self):
        result = self.pack([chunk("S1", "a" * 4000)], required_citations=("S1",), max_chars=300)
        self.assertEqual(result.contexts, ())
        self.assertEqual(result.trace[0]["reason"], "required_exceeds_budget_or_context_limit")

    def test_missing_required_id_does_not_crash(self):
        result = self.pack([chunk("S1", "Available evidence")], required_citations=("S99",))
        entry = next(e for e in result.trace if e["citation_id"] == "S99")
        self.assertEqual(entry["reason"], "required_not_in_input_or_empty")

    def test_does_not_stop_at_oversized_first_source(self):
        result = self.pack([chunk("S1", "large" * 1000), chunk("S2", "duration: 20 minutes")], max_chars=300)
        self.assertEqual([c["citation_id"] for c in result.contexts], ["S2"])

    def test_no_partial_table_to_claim_complete_list(self):
        text = "| Required fields | Description |\n" + "| data | detail |\n" * 400
        result = self.pack([chunk("S1", text)], question="required fields", max_chars=500)
        self.assertEqual(result.contexts, ())
        self.assertEqual(result.trace[0]["status"], "excluded")

    def test_no_partial_code_block(self):
        result = self.pack([chunk("S1", "```\n" + "x=1\n" * 1000 + "```\n")], max_chars=500)
        self.assertEqual(result.contexts, ())

    def test_duplicate_citation_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.pack([chunk("S1", "one"), chunk("S1", "two", "d2")])

    def test_invalid_citation_is_not_rendered(self):
        result = self.pack([chunk("S0", "must not appear"), chunk("S1", "valid")])
        self.assertNotIn("must not appear", result.rendered)
        self.assertEqual(result.trace[0]["reason"], "invalid_citation")

    def test_empty_inputs_and_text(self):
        self.assertEqual(self.pack([]).rendered, "")
        result = self.pack([chunk("S1", " ")])
        self.assertEqual(result.contexts, ())
        self.assertEqual(result.trace[0]["reason"], "empty_text")

    def test_budget_validation(self):
        for args in ({"max_chars": 0}, {"max_chars": -1}, {"max_contexts": 0}):
            with self.assertRaises(ValueError):
                self.pack([], **args)

    def test_exact_serialized_budget(self):
        contexts = [chunk("S1", "Restore duration 20 minutes."), chunk("S2", "Second")]
        full = self.pack(contexts)
        exact = self.pack(contexts, max_chars=len(full.rendered))
        self.assertEqual(exact.rendered, full.rendered)
        smaller = self.pack(contexts, max_chars=len(full.rendered) - 1)
        self.assertLess(len(smaller.contexts), 2)
        self.assertLessEqual(len(smaller.rendered), len(full.rendered) - 1)

    def test_long_metadata_consumes_budget(self):
        result = self.pack([chunk("S1", "duration 20 minutes", title="title" * 500)], max_chars=300)
        self.assertEqual(result.contexts, ())

    def test_metadata_labels_do_not_affect_selection(self):
        contexts = [chunk("S1", "Overview"), chunk("S2", "Restore duration 20 minutes")]
        poisoned = [dict(c, gold_answer="S1 is the answer", answer_facts=["S1"], expected_doc_ids=["d1"])
                    for c in contexts]
        self.assertEqual(self.pack(contexts, max_contexts=1).rendered,
                         self.pack(poisoned, max_contexts=1).rendered)

    def test_requirement_text_is_used(self):
        contexts = [chunk("S1", "Deployment explanation"), chunk("S2", "zeta duration 20 minutes")]
        result = self.pack(contexts, question="Explain", requirements=(("R1", "zeta duration"),), max_contexts=1)
        self.assertEqual(result.contexts[0]["citation_id"], "S2")
        self.assertIn("lexical_affinity_not_support", result.trace[1])

    def test_duplicate_body_keeps_conflicting_provenance(self):
        contexts = [chunk("S1", "Threshold 20", "tenant-approved-a"), chunk("S2", "Threshold 20", "tenant-approved-b")]
        result = self.pack(contexts, required_citations=("S1", "S2"))
        self.assertEqual(len(result.contexts), 2)

    def test_seeded_budget_and_provenance_invariants(self):
        rng = random.Random(20260904)
        for _ in range(200):
            contexts = [chunk(f"S{i+1}", "".join(rng.choices(string.ascii_letters + " \n🙂", k=rng.randrange(1, 600))), f"d{i%3}")
                        for i in range(rng.randrange(1, 12))]
            budget, cap = rng.randrange(1, 2500), rng.randrange(1, 8)
            result = self.pack(contexts, max_chars=budget, max_contexts=cap)
            self.assertLessEqual(len(result.rendered), budget)
            self.assertLessEqual(len(result.contexts), cap)
            source = {c["citation_id"]: c for c in contexts}
            for c in result.contexts:
                span = c["evidence_span"]
                self.assertEqual(c["text"], source[c["citation_id"]]["text"][span["start_char"]:span["end_char"]])


if __name__ == "__main__":
    unittest.main()
