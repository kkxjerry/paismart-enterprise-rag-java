"""Tests for the isolated E3/E4/E5 Elasticsearch index builder."""
from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.hierarchical_index_build import build, elastic_document, index_mapping


class HierarchicalIndexBuildTest(unittest.TestCase):
    def test_mapping_separates_search_and_raw_text(self) -> None:
        properties = index_mapping(2048)["mappings"]["properties"]
        self.assertEqual(properties["vector"]["dims"], 2048)
        self.assertTrue(properties["vector"]["index"])
        self.assertFalse(properties["rawText"]["index"])
        self.assertEqual(properties["searchText"]["type"], "text")
        self.assertEqual(index_mapping(2048)["mappings"]["dynamic"], "strict")

    def test_mapping_rejects_invalid_dimension(self) -> None:
        with self.assertRaises(ValueError):
            index_mapping(0)

    def test_elastic_document_normalizes_acl_and_vector(self) -> None:
        value = elastic_document(
            {
                "node_id": "n1",
                "node_type": "leaf",
                "parent_node_id": "p1",
                "doc_id": "d1",
                "source_type": "gmail",
                "raw_text": "raw",
                "search_text": "prefix raw",
                "acl": {
                    "tenant_id": "t1",
                    "allowed_group_ids": ["ops"],
                    "denied_user_ids": ["u2"],
                },
            },
            [0.1, 0.2],
        )
        self.assertEqual(value["tenantId"], "t1")
        self.assertEqual(value["allowedGroupIds"], ["ops"])
        self.assertEqual(value["deniedUserIds"], ["u2"])
        self.assertEqual(value["vector"], [0.1, 0.2])

    def test_build_refuses_existing_index(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parents = root / "parents.jsonl"
            leaves = root / "leaves.jsonl"
            parents.write_text("")
            leaves.write_text("")
            args = self.args(parents, leaves)
            with patch("tools.hierarchical_index_build.request_status", return_value=200):
                with self.assertRaisesRegex(ValueError, "overwrite"):
                    build(args)

    def test_build_indexes_parent_and_embedded_leaf_without_alias(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parents = root / "parents.jsonl"
            leaves = root / "leaves.jsonl"
            parents.write_text(json.dumps({
                "node_id": "p1", "node_type": "parent", "doc_id": "d1",
                "source_type": "confluence", "raw_text": "parent", "search_text": "parent",
            }) + "\n")
            leaves.write_text(json.dumps({
                "node_id": "l1", "node_type": "leaf", "parent_node_id": "p1", "doc_id": "d1",
                "source_type": "confluence", "raw_text": "leaf", "search_text": "prefix leaf",
            }) + "\n")
            indexed = []
            def fake_request(method, url, **kwargs):
                if url.endswith("/_count"):
                    return {"count": 2}
                return {}
            with (
                patch("tools.hierarchical_index_build.request_status", return_value=404),
                patch("tools.hierarchical_index_build.request_json", side_effect=fake_request),
                patch("tools.hierarchical_index_build.embed", return_value=[[0.1, 0.2]]),
                patch("tools.hierarchical_index_build.bulk_index", side_effect=lambda es, index, docs, timeout: indexed.extend(docs)),
            ):
                result = build(self.args(parents, leaves))
            self.assertEqual(result["parents"], 1)
            self.assertEqual(result["leaves"], 1)
            self.assertEqual(result["index_documents"], 2)
            self.assertFalse(result["alias_changed"])
            self.assertEqual([value[0] for value in indexed], ["p1", "l1"])
            self.assertNotIn("vector", indexed[0][1])
            self.assertEqual(indexed[1][1]["vector"], [0.1, 0.2])

    @staticmethod
    def args(parents: Path, leaves: Path) -> argparse.Namespace:
        return argparse.Namespace(
            parents=parents,
            leaves=leaves,
            es_url="http://example.invalid",
            index="new-index",
            embedding_url="http://example.invalid/embeddings",
            embedding_model="model",
            embedding_dimension=2,
            embedding_batch_size=4,
            bulk_size=4,
            progress_every=0,
            timeout_seconds=1.0,
            delete_failed_index=False,
            output=None,
        )


if __name__ == "__main__":
    unittest.main()
