"""Tests for hierarchical leaf retrieval and parent expansion evaluation."""
from __future__ import annotations

import unittest

from tools.hierarchical_retrieval_eval import (
    collapse_documents,
    query_text,
    search_filter,
    select_questions,
    weighted_rrf,
)


class HierarchicalRetrievalEvalTest(unittest.TestCase):
    def test_query_instruction_format(self) -> None:
        self.assertEqual(query_text("latency", "retrieve passages"), "Instruct: retrieve passages\nQuery: latency")
        self.assertEqual(query_text("latency", ""), "latency")

    def test_filter_is_label_blind_and_tenant_scoped(self) -> None:
        filters = search_filter({"tenant_id": "t1", "expected_doc_ids": ["secret"]}, ["gmail"])
        rendered = str(filters)
        self.assertIn("tenantId", rendered)
        self.assertIn("sourceType", rendered)
        self.assertNotIn("secret", rendered)

    def test_weighted_rrf_and_document_collapse(self) -> None:
        dense = [
            {"nodeId": "l1", "docId": "d1", "parentNodeId": "p1"},
            {"nodeId": "l2", "docId": "d2", "parentNodeId": "p2"},
        ]
        bm25 = [
            {"nodeId": "l2", "docId": "d2", "parentNodeId": "p2"},
            {"nodeId": "l3", "docId": "d1", "parentNodeId": "p3"},
        ]
        fused = weighted_rrf([("dense", dense, 1.0), ("bm25", bm25, 1.0)], rrf_k=10)
        self.assertEqual(fused[0]["nodeId"], "l2")
        docs = collapse_documents(fused, 2)
        self.assertEqual({value["doc_id"] for value in docs}, {"d1", "d2"})
        d1 = next(value for value in docs if value["doc_id"] == "d1")
        self.assertEqual(set(d1["parent_ids"]), {"p1", "p3"})

    def test_source_question_filter(self) -> None:
        rows = [
            {"qid": "a", "source_types": ["gmail"]},
            {"qid": "b", "source_types": ["slack"]},
            {"qid": "c", "source_types": ["gmail", "jira"]},
        ]
        self.assertEqual([row["qid"] for row in select_questions(rows, ["gmail"], None)], ["a"])
        self.assertEqual([row["qid"] for row in select_questions(rows, ["gmail", "jira"], None)], ["a", "c"])
        self.assertEqual([row["qid"] for row in select_questions(rows, [], 2)], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
