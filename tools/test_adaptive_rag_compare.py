from __future__ import annotations

import argparse
import unittest

from tools.adaptive_rag_compare import (
    bootstrap_interval,
    compare_metric,
    estimated_cost,
    successful_by_qid,
    two_sided_sign_p,
    usage,
)


class AdaptiveRagCompareTest(unittest.TestCase):
    def test_aligns_only_successful_unique_rows(self) -> None:
        values = successful_by_qid([
            {"qid": "q1", "error": None},
            {"qid": "q2", "error": {"stage": "generation"}},
        ])
        self.assertEqual(set(values), {"q1"})

    def test_paired_metric_reports_direction_and_interval(self) -> None:
        pairs = [
            ({"metrics": {"answer_fact_token_recall": 0.2}}, {"metrics": {"answer_fact_token_recall": 0.5}}),
            ({"metrics": {"answer_fact_token_recall": 0.4}}, {"metrics": {"answer_fact_token_recall": 0.4}}),
            ({"metrics": {"answer_fact_token_recall": 0.8}}, {"metrics": {"answer_fact_token_recall": 0.7}}),
        ]
        result = compare_metric(pairs, "answer_fact_token_recall")
        self.assertAlmostEqual(result["mean_delta"], (0.3 + 0.0 - 0.1) / 3)
        self.assertEqual(result["improved"], 1)
        self.assertEqual(result["regressed"], 1)
        self.assertEqual(result["unchanged"], 1)
        self.assertEqual(len(result["bootstrap_95_ci"]), 2)

    def test_bootstrap_is_deterministic(self) -> None:
        first = bootstrap_interval([0.1, 0.2, -0.1], samples=200)
        second = bootstrap_interval([0.1, 0.2, -0.1], samples=200)
        self.assertEqual(first, second)

    def test_sign_test_handles_ties_outside_trial_count(self) -> None:
        self.assertEqual(two_sided_sign_p(1, 1), 1.0)
        self.assertIsNone(two_sided_sign_p(0, 0))

    def test_usage_includes_all_adaptive_stages(self) -> None:
        row = {
            "requirements": {"usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11}},
            "generation": {"usage": {"prompt_tokens": 20, "completion_tokens": 2, "total_tokens": 22}},
            "verification": {"usage": {"prompt_tokens": 30, "completion_tokens": 3, "total_tokens": 33}},
            "secondary_retrieval": {"usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}},
        }
        totals = usage([row])
        self.assertEqual(totals["prompt_tokens"], 60)
        self.assertEqual(totals["completion_tokens"], 6)
        self.assertEqual(totals["total_tokens"], 66)

    def test_cost_separates_cached_input(self) -> None:
        cost = estimated_cost(
            {"prompt_tokens": 1_000_000, "cached_tokens": 200_000, "completion_tokens": 100_000},
            input_per_million=0.15,
            cached_input_per_million=0.03,
            output_per_million=1.5,
        )
        self.assertAlmostEqual(cost, 0.8 * 0.15 + 0.2 * 0.03 + 0.1 * 1.5)


if __name__ == "__main__":
    unittest.main()
