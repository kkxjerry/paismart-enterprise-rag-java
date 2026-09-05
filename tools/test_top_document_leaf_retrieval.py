"""Tests for E1 query-time leaf retrieval inside authorized top documents."""
from __future__ import annotations

import unittest
from unittest.mock import patch

from tools.top_document_leaf_retrieval import contexts_from_chunks, fetch_document_chunks


class TopDocumentLeafRetrievalTest(unittest.TestCase):
    def test_fetch_uses_only_ranked_allowlist_and_tenant(self) -> None:
        observed = {}

        def fake_request(method, url, **kwargs):
            observed.update(kwargs["payload"])
            return {
                "hits": {
                    "hits": [
                        {
                            "_id": "chunk-1",
                            "_source": {
                                "benchmarkDocId": "d1",
                                "chunkId": 1,
                                "textContent": "first",
                            },
                        },
                        {
                            "_id": "chunk-2",
                            "_source": {
                                "benchmarkDocId": "d1",
                                "chunkId": 2,
                                "textContent": "second",
                            },
                        },
                    ]
                }
            }

        with patch("tools.top_document_leaf_retrieval.request_json", side_effect=fake_request):
            chunks = fetch_document_chunks(
                es_url="http://example.invalid",
                index="evidence",
                doc_ids=["d1", "d2"],
                tenant_id="tenant-a",
                chunks_per_document=1,
                timeout=1.0,
            )
        filters = observed["query"]["bool"]["filter"]
        self.assertIn({"terms": {"benchmarkDocId": ["d1", "d2"]}}, filters)
        self.assertIn({"term": {"tenantId": "tenant-a"}}, filters)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["chunk_es_id"], "chunk-1")
        self.assertNotIn("expected_doc_ids", str(observed))

    def test_contexts_preserve_document_rank_and_source_metadata(self) -> None:
        chunks = [
            {
                "chunk_es_id": "es-2",
                "benchmarkDocId": "d2",
                "chunkId": 4,
                "chunkKind": "issue_resolution",
                "sectionPath": "Resolution",
                "sourceType": "jira",
                "sourcePath": "PROJ-1",
                "title": "Incident",
                "textContent": "The fix signs canonical JSON.",
            },
            {
                "chunk_es_id": "es-1",
                "benchmarkDocId": "d1",
                "chunkId": 1,
                "sourceType": "confluence",
                "title": "Overview",
                "textContent": "Overview text.",
            },
        ]
        values = contexts_from_chunks(chunks, ["d1", "d2"])
        self.assertEqual(values[0]["citation_id"], "S1")
        self.assertEqual(values[0]["document_rank"], 2)
        self.assertEqual(values[0]["chunk_kind"], "issue_resolution")
        self.assertEqual(values[0]["section_path"], "Resolution")
        self.assertEqual(values[1]["document_rank"], 1)

    def test_chunks_outside_authorized_ranked_documents_are_dropped(self) -> None:
        values = contexts_from_chunks(
            [{"benchmarkDocId": "outside", "chunkId": 1, "textContent": "secret"}],
            ["d1"],
        )
        self.assertEqual(values, [])


if __name__ == "__main__":
    unittest.main()
