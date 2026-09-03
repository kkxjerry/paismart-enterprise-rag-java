from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.qwen_plus_pair_judge import (
    blinded_order,
    build_messages,
    load_pairs,
    stratified_select,
    validate_judgement,
)
from tools.qwen_plus_rag_pipeline import PipelineError


class QwenPlusPairJudgeTest(unittest.TestCase):
    def test_validates_scores_and_winner(self) -> None:
        result = validate_judgement(
            {
                "A": {"correctness": 8, "completeness": 7.5, "directness": 9},
                "B": {"correctness": 6, "completeness": 5, "directness": 8},
                "winner": "A",
            }
        )
        self.assertEqual(result["winner"], "A")
        self.assertEqual(result["A"]["completeness"], 7.5)

        with self.assertRaises(PipelineError):
            validate_judgement(
                {
                    "A": {"correctness": 11, "completeness": 7, "directness": 9},
                    "B": {"correctness": 6, "completeness": 5, "directness": 8},
                    "winner": "A",
                }
            )

    def test_blinding_is_deterministic_and_prompt_hides_pipeline_names(self) -> None:
        pair = {
            "qid": "q1",
            "baseline": row("q1", "first factual answer"),
            "candidate": row("q1", "second factual answer"),
        }
        self.assertEqual(blinded_order("q1"), blinded_order("q1"))
        messages, labels = build_messages(pair)
        prompt = messages[1]["content"]
        self.assertNotIn("baseline", prompt)
        self.assertNotIn("candidate", prompt)
        self.assertEqual(set(labels.values()), {"baseline", "candidate"})

    def test_load_pairs_excludes_non_evaluable_rows_unless_unanswerable_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline.jsonl"
            candidate = root / "candidate.jsonl"
            rows = [
                row("q1", "answer", answer_evaluable=True),
                row("q2", "refusal", answer_evaluable=False, unanswerable=True),
                row("q3", "high level", answer_evaluable=False, unanswerable=False),
            ]
            text = "".join(json.dumps(value) + "\n" for value in rows)
            baseline.write_text(text, encoding="utf-8")
            candidate.write_text(text, encoding="utf-8")

            without = load_pairs(baseline, candidate, include_unanswerable=False)
            with_unanswerable = load_pairs(baseline, candidate, include_unanswerable=True)
            self.assertEqual([pair["qid"] for pair in without], ["q1"])
            self.assertEqual([pair["qid"] for pair in with_unanswerable], ["q1", "q2"])

    def test_stratified_selection_round_robins_types(self) -> None:
        pairs = [
            pair("a1", "a"),
            pair("a2", "a"),
            pair("b1", "b"),
            pair("b2", "b"),
        ]
        selected = stratified_select(pairs, 3)
        self.assertEqual([value["qid"] for value in selected], ["a1", "b1", "a2"])


def row(
    qid: str,
    answer: str,
    *,
    answer_evaluable: bool = True,
    unanswerable: bool = False,
) -> dict[str, object]:
    return {
        "qid": qid,
        "question": "Question?",
        "question_type": "basic",
        "gold_answer": "Reference",
        "answer_facts": ["Required fact"],
        "generation": {"answer": answer, "citations": ["S1"]},
        "metrics": {
            "is_answer_evaluable": answer_evaluable,
            "is_unanswerable": unanswerable,
        },
        "error": None,
    }


def pair(qid: str, question_type: str) -> dict[str, object]:
    baseline = row(qid, "baseline")
    candidate = row(qid, "candidate")
    baseline["question_type"] = question_type
    candidate["question_type"] = question_type
    return {"qid": qid, "baseline": baseline, "candidate": candidate}


if __name__ == "__main__":
    unittest.main()
