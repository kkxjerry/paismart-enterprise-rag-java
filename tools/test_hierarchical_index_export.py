"""Tests for E3/E4/E5 hierarchy index export."""
from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from tools.hierarchical_index_export import export, stable_id


class HierarchicalIndexExportTest(unittest.TestCase):
    def test_export_preserves_offsets_acl_and_deterministic_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            docs = root / "docs.jsonl"
            acl = root / "acl.jsonl"
            output = root / "output"
            document = {
                "doc_id": "doc-1",
                "title": "Meeting",
                "source_type": "fireflies",
                "text": "summary:\nLatency was high.\n\nnext_steps:\n- Alice sends logs.\n- Bob retries.",
                "source_path": "meetings/1",
            }
            docs.write_text(json.dumps(document) + "\n", encoding="utf-8")
            acl.write_text(json.dumps({"doc_id": "doc-1", "tenant_id": "tenant-a", "allowed_group_ids": ["ops"]}) + "\n", encoding="utf-8")
            args = argparse.Namespace(
                docs=docs,
                acl_docs=acl,
                output_dir=output,
                source_type=[],
                leaf_mode="proposition",
                no_contextual_prefix=False,
                fail_on_missing_acl=True,
                limit=None,
                progress_every=0,
            )
            result = export(args)
            parents = [json.loads(line) for line in (output / "parents.jsonl").read_text().splitlines()]
            leaves = [json.loads(line) for line in (output / "leaves.jsonl").read_text().splitlines()]
            self.assertEqual(result["summary"]["documents_exported"], 1)
            self.assertEqual(result["summary"]["offset_violations"], 0)
            self.assertGreaterEqual(len(parents), 2)
            self.assertGreaterEqual(len(leaves), 3)
            parent_ids = {value["node_id"] for value in parents}
            self.assertTrue(all(value["parent_node_id"] in parent_ids for value in leaves))
            self.assertTrue(all(value["acl"]["tenant_id"] == "tenant-a" for value in leaves))
            self.assertTrue(all(value["raw_text"] in document["text"] for value in parents + leaves))
            self.assertTrue(all(value["search_text"] != value["raw_text"] for value in leaves))
            self.assertEqual(result["manifest"]["parents_sha256"], result["manifest"]["parents_sha256"])

    def test_missing_acl_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            docs = root / "docs.jsonl"
            acl = root / "acl.jsonl"
            docs.write_text(json.dumps({"doc_id": "d", "source_type": "gmail", "text": "From: A\nDecision."}) + "\n")
            acl.write_text("")
            with self.assertRaisesRegex(ValueError, "missing ACL"):
                export(argparse.Namespace(
                    docs=docs, acl_docs=acl, output_dir=root / "out", source_type=[],
                    leaf_mode="sentence", no_contextual_prefix=False, fail_on_missing_acl=True,
                    limit=None, progress_every=0,
                ))

    def test_source_filter_and_prefix_toggle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            docs = root / "docs.jsonl"
            docs.write_text("\n".join([
                json.dumps({"doc_id": "a", "source_type": "confluence", "title": "A", "text": "# H\nFact."}),
                json.dumps({"doc_id": "b", "source_type": "gmail", "title": "B", "text": "From: B\nFact."}),
            ]) + "\n")
            result = export(argparse.Namespace(
                docs=docs, acl_docs=None, output_dir=root / "out", source_type=["gmail"],
                leaf_mode="sentence", no_contextual_prefix=True, fail_on_missing_acl=False,
                limit=None, progress_every=0,
            ))
            self.assertEqual(result["summary"]["documents_exported"], 1)
            self.assertEqual(result["summary"]["source_documents"], {"gmail": 1})
            leaf = json.loads((root / "out" / "leaves.jsonl").read_text().splitlines()[0])
            self.assertEqual(leaf["search_text"], leaf["raw_text"])

    def test_stable_id_changes_with_document_and_offsets(self) -> None:
        self.assertEqual(stable_id("a", 1, 2), stable_id("a", 1, 2))
        self.assertNotEqual(stable_id("a", 1, 2), stable_id("b", 1, 2))
        self.assertNotEqual(stable_id("a", 1, 2), stable_id("a", 1, 3))


if __name__ == "__main__":
    unittest.main()
