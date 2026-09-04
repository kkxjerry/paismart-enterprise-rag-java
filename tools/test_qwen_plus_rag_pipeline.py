from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.qwen_plus_rag_pipeline import (
    ApiResult,
    PipelineError,
    QwenClient,
    QwenRequestError,
    expand_selected_contexts,
    load_resume,
    process_row,
    render_contexts,
    score_result,
    select_rows,
    validate_enhancement,
    validate_generation,
    validate_output_paths,
)


class FakeClient:
    model = "qwen-plus"

    def __init__(self, payloads: list[dict[str, object]]) -> None:
        self.payloads = list(payloads)
        self.calls = 0

    def complete_json(self, *, validator, **_kwargs) -> ApiResult:
        self.calls += 1
        payload = self.payloads.pop(0)
        return ApiResult(
            value=validator(payload),
            latency_ms=12.5,
            usage={
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120,
                "cached_tokens": 0,
            },
            request_id=f"request-{self.calls}",
            returned_model="qwen-plus",
            attempts=1,
            finish_reason="stop",
            max_tokens_used=256,
        )


class QwenPlusRagPipelineTest(unittest.TestCase):
    def test_resume_rejects_rows_from_another_run_signature(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "rows.jsonl"
            output.write_text(
                json.dumps({"qid": "q1", "run_signature": "old", "error": None}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "another input/configuration"):
                load_resume(output, "new")
            self.assertEqual(list(load_resume(output, "old")), ["q1"])

    def test_output_paths_cannot_overwrite_inputs_or_each_other(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "contexts.jsonl"
            source.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "overwrite inputs"):
                validate_output_paths([source], [source, root / "summary.json"])
            with self.assertRaisesRegex(ValueError, "must be distinct"):
                validate_output_paths([source], [root / "same.json", root / "same.json"])

    def test_client_expands_output_budget_after_length_finish(self) -> None:
        requests = []

        class Response:
            def __init__(self, payload):
                self.payload = payload
                self.headers = {"x-request-id": "request-id"}

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps(self.payload).encode("utf-8")

        responses = [
            Response(
                {
                    "model": "qwen-plus",
                    "choices": [
                        {
                            "finish_reason": "length",
                            "message": {"content": '{"answerable":true'},
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 768,
                        "total_tokens": 868,
                    },
                }
            ),
            Response(
                {
                    "model": "qwen-plus",
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"content": '{"ok":true}'},
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 10,
                        "total_tokens": 110,
                    },
                }
            ),
        ]

        def fake_urlopen(request, timeout):
            del timeout
            requests.append(json.loads(request.data))
            return responses.pop(0)

        client = QwenClient(
            api_base="https://example.test/v1",
            api_key="secret",
            model="qwen-plus",
            timeout_seconds=10,
            retries=2,
        )
        with patch("tools.qwen_plus_rag_pipeline.urllib.request.urlopen", side_effect=fake_urlopen), patch(
            "tools.qwen_plus_rag_pipeline.time.sleep", return_value=None
        ):
            result = client.complete_json(
                messages=[{"role": "user", "content": "test"}],
                max_tokens=768,
                temperature=0.0,
                validator=lambda payload: payload,
            )

        self.assertEqual([request["max_tokens"] for request in requests], [768, 1536])
        self.assertEqual(result.attempts, 2)
        self.assertEqual(result.max_tokens_used, 1536)
        self.assertEqual(result.usage["total_tokens"], 978)

    def test_terminal_client_error_preserves_failed_attempt_usage(self) -> None:
        class Response:
            headers = {"x-request-id": "request-id"}

            def __init__(self, total_tokens: int) -> None:
                self.total_tokens = total_tokens

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps(
                    {
                        "model": "qwen-flash",
                        "choices": [{"finish_reason": "stop", "message": {"content": '{"ok":true}'}}],
                        "usage": {
                            "prompt_tokens": self.total_tokens - 10,
                            "completion_tokens": 10,
                            "total_tokens": self.total_tokens,
                        },
                    }
                ).encode("utf-8")

        responses = [Response(110), Response(120)]
        client = QwenClient(
            api_base="https://example.test/v1",
            api_key="secret",
            model="qwen-flash",
            timeout_seconds=10,
            retries=1,
        )
        with patch(
            "tools.qwen_plus_rag_pipeline.urllib.request.urlopen",
            side_effect=lambda *_args, **_kwargs: responses.pop(0),
        ), patch("tools.qwen_plus_rag_pipeline.time.sleep", return_value=None):
            with self.assertRaises(QwenRequestError) as raised:
                client.complete_json(
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=128,
                    temperature=0.0,
                    validator=lambda _payload: (_ for _ in ()).throw(PipelineError("invalid")),
                )

        self.assertEqual(raised.exception.usage["total_tokens"], 230)
        self.assertEqual(raised.exception.attempts, 2)
        self.assertEqual(raised.exception.max_tokens_used, 512)

    def test_enhancement_rejects_unknown_or_excessive_citations(self) -> None:
        with self.assertRaisesRegex(PipelineError, "unknown citations"):
            validate_enhancement(
                {
                    "answerability": "answerable",
                    "selected_citations": ["S9"],
                    "conflict_citations": [],
                },
                valid_citations={"S1", "S2"},
                max_selected=2,
            )

        with self.assertRaisesRegex(PipelineError, "exceeding limit"):
            validate_enhancement(
                {
                    "answerability": "answerable",
                    "selected_citations": ["S1", "S2"],
                    "conflict_citations": [],
                },
                valid_citations={"S1", "S2"},
                max_selected=1,
            )

    def test_generation_normalizes_missing_inline_marker_and_rejects_unknown(self) -> None:
        result = validate_generation(
            {
                "answerable": True,
                "answer": "The limit is 50 MiB.",
                "citations": ["S2"],
            },
            valid_citations={"S1", "S2"},
        )
        self.assertEqual(result["citations"], ["S2"])
        self.assertTrue(result["answer"].endswith("[S2]"))
        self.assertTrue(result["citation_normalized"])

        with self.assertRaisesRegex(PipelineError, "unknown citations"):
            validate_generation(
                {
                    "answerable": True,
                    "answer": "Unsupported claim [S8]",
                    "citations": ["S8"],
                },
                valid_citations={"S1"},
            )

    def test_enhance_generate_uses_untouched_selected_source_text(self) -> None:
        row = sample_row()
        client = FakeClient(
            [
                {
                    "answerability": "answerable",
                    "selected_citations": ["S2"],
                    "conflict_citations": [],
                },
                {
                    "answerable": True,
                    "answer": "The request limit is 50 MiB [S2]",
                    "citations": ["S2"],
                },
            ]
        )

        result = process_row(
            row,
            client=client,
            pipeline="enhance-generate",
            enhance_max_contexts=10,
            enhance_max_input_chars=10_000,
            enhance_max_selected=4,
            enhance_max_tokens=128,
            selection_expansion="none",
            expansion_max_per_document=3,
            generation_max_contexts=4,
            generation_max_input_chars=10_000,
            generation_max_tokens=256,
            temperature=0.0,
        )

        self.assertIsNone(result["error"])
        self.assertEqual(client.calls, 2)
        self.assertEqual(result["enhancement"]["selected_citations"], ["S2"])
        self.assertEqual(len(result["selected_contexts"]), 1)
        self.assertEqual(result["selected_contexts"][0]["citation_id"], "S2")
        self.assertEqual(
            result["selected_contexts"][0]["text"],
            row["contexts"][1]["text"],
        )
        self.assertEqual(result["generation"]["citations"], ["S2"])
        self.assertEqual(result["metrics"]["citation_precision"], 1.0)

    def test_document_expansion_adds_only_bounded_sibling_chunks(self) -> None:
        contexts = [
            {"citation_id": "S1", "doc_id": "doc-a", "text": "a1"},
            {"citation_id": "S2", "doc_id": "doc-b", "text": "b1"},
            {"citation_id": "S3", "doc_id": "doc-a", "text": "a2"},
            {"citation_id": "S4", "doc_id": "doc-a", "text": "a3"},
            {"citation_id": "S5", "doc_id": "doc-a", "text": "a4"},
        ]
        expanded = expand_selected_contexts(
            contexts,
            ["S3"],
            mode="document",
            max_per_document=3,
        )
        self.assertEqual([row["citation_id"] for row in expanded], ["S3", "S1", "S4"])
        self.assertNotIn("S2", [row["citation_id"] for row in expanded])
        self.assertNotIn("S5", [row["citation_id"] for row in expanded])

        reranked = expand_selected_contexts(
            contexts,
            ["S3"],
            mode="rerank",
            max_per_document=3,
        )
        self.assertEqual(
            [row["citation_id"] for row in reranked],
            ["S3", "S1", "S2", "S4", "S5"],
        )

    def test_render_contexts_respects_limits_and_skips_invalid_ids(self) -> None:
        contexts = [
            {"citation_id": "bad", "text": "ignored"},
            {"citation_id": "S1", "text": "a" * 50, "title": "one"},
            {"citation_id": "S2", "text": "b" * 50, "title": "two"},
        ]
        rendered, included, count = render_contexts(
            contexts,
            max_contexts=3,
            max_chars=80,
        )
        self.assertEqual(count, len(rendered))
        self.assertEqual([row["citation_id"] for row in included], ["S1"])
        self.assertLessEqual(len(rendered), 80)

    def test_score_result_includes_high_level_facts_but_separates_unanswerable(self) -> None:
        high_level = sample_row()
        high_level["is_evaluable"] = False
        high_level["expected_doc_ids"] = []
        high_level["question_type"] = "high_level"
        high_level_scores = score_result(
            high_level,
            selected_contexts=[high_level["contexts"][1]],
            answerable=True,
            answer="The request limit is 50 MiB [S2]",
            citations=["S2"],
        )
        self.assertTrue(high_level_scores["is_answer_evaluable"])

        missing = sample_row()
        missing["question_type"] = "info_not_found"
        missing_scores = score_result(
            missing,
            selected_contexts=[],
            answerable=False,
            answer="INSUFFICIENT_EVIDENCE",
            citations=[],
        )
        self.assertFalse(missing_scores["is_answer_evaluable"])
        self.assertTrue(missing_scores["unanswerable_abstain_correct"])

    def test_stratified_selection_round_robins_question_types(self) -> None:
        rows = [
            {"qid": "a1", "question_type": "a"},
            {"qid": "a2", "question_type": "a"},
            {"qid": "b1", "question_type": "b"},
            {"qid": "b2", "question_type": "b"},
        ]
        selected = select_rows(rows, limit=3, stratified=True, qids=None)
        self.assertEqual([row["qid"] for row in selected], ["a1", "b1", "a2"])


def sample_row() -> dict[str, object]:
    return {
        "qid": "q1",
        "question": "What is the total request limit?",
        "question_type": "basic",
        "source_types": ["github"],
        "expected_doc_ids": ["doc-1"],
        "expected_accessible_doc_ids": ["doc-1"],
        "gold_answer": "50 MiB",
        "answer_facts": ["The total request limit is 50 MiB."],
        "is_evaluable": True,
        "retrieval_hit_at_10": True,
        "contexts": [
            {
                "citation_id": "S1",
                "doc_id": "doc-noise",
                "title": "Background",
                "source_type": "github",
                "text": "Uploads use multipart requests.",
            },
            {
                "citation_id": "S2",
                "doc_id": "doc-1",
                "title": "Limits",
                "source_type": "github",
                "text": "The total request limit is 50 MiB.",
            },
        ],
    }


if __name__ == "__main__":
    unittest.main()
