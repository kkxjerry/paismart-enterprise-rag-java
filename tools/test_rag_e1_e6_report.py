"""Tests for E1-E6 report aggregation and fair retrieval comparison."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.rag_e1_e6_report import retrieval_baseline


class E1E6ReportTest(unittest.TestCase):
    def test_retrieval_baseline_uses_only_complete_source_subset(self) -> None:
        rows = [
            {
                "qid": "gmail",
                "source_types": ["gmail"],
                "expected_accessible_doc_ids": ["d1"],
                "ranked_doc_ids": ["d1", "d2"],
                "evidence_fact_token_recall": 0.8,
            },
            {
                "qid": "cross-source",
                "source_types": ["gmail", "slack"],
                "expected_accessible_doc_ids": ["d3"],
                "ranked_doc_ids": ["d3"],
                "evidence_fact_token_recall": 1.0,
            },
            {
                "qid": "confluence-miss",
                "source_types": ["confluence"],
                "expected_doc_ids": ["d4"],
                "ranked_doc_ids": ["other"],
                "evidence_fact_token_recall": 0.2,
            },
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "contexts.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            result = retrieval_baseline(path, {"gmail", "confluence", "fireflies"})
        self.assertEqual(result["questions"], 2)
        self.assertEqual(result["hit_at_1"], 0.5)
        self.assertEqual(result["hit_at_10"], 0.5)
        self.assertEqual(result["mrr"], 0.5)
        self.assertAlmostEqual(result["evidence_fact_token_recall"], 0.5)

    def test_empty_subset_is_explicit_not_perfect(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "contexts.jsonl"
            path.write_text(json.dumps({"qid": "q", "source_types": ["slack"]}) + "\n")
            result = retrieval_baseline(path, {"gmail"})
        self.assertEqual(result["questions"], 0)
        self.assertIsNone(result["hit_at_10"])
        self.assertIsNone(result["evidence_fact_token_recall"])


if __name__ == "__main__":
    unittest.main()
