"""Tests for the E5 late-chunking capability boundary."""
from __future__ import annotations

import math
import unittest

from tools.late_chunking_capability import inspect_embedding_response, pool_token_embeddings


class LateChunkingCapabilityTest(unittest.TestCase):
    def test_flat_openai_embedding_is_not_late_chunking_ready(self) -> None:
        result = inspect_embedding_response({"data": [{"embedding": [0.1, 0.2, 0.3]}]})
        self.assertTrue(result["pooled_embedding_available"])
        self.assertFalse(result["token_embeddings_available"])
        self.assertFalse(result["late_chunking_ready"])
        self.assertIn("pooled", result["reason"])

    def test_nested_vectors_without_offsets_are_not_ready(self) -> None:
        result = inspect_embedding_response({"data": [{"embedding": [[1.0, 0.0], [0.0, 1.0]]}]})
        self.assertTrue(result["token_embeddings_available"])
        self.assertFalse(result["offset_mapping_available"])
        self.assertFalse(result["late_chunking_ready"])

    def test_token_vectors_and_offsets_are_ready(self) -> None:
        result = inspect_embedding_response({
            "data": [{
                "token_embeddings": [[1.0, 0.0], [0.0, 1.0]],
                "offset_mapping": [[0, 5], [6, 10]],
            }]
        })
        self.assertTrue(result["late_chunking_ready"])
        self.assertEqual(result["dimension"], 2)
        self.assertEqual(result["token_count"], 2)

    def test_pooling_uses_overlapping_tokens_and_normalizes(self) -> None:
        vectors = pool_token_embeddings(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            [(0, 5), (6, 10), (11, 15)],
            [(0, 10), (11, 15)],
        )
        self.assertEqual(len(vectors), 2)
        self.assertAlmostEqual(math.sqrt(sum(value * value for value in vectors[0])), 1.0)
        self.assertAlmostEqual(vectors[1][0], 2 ** -0.5)
        self.assertAlmostEqual(vectors[1][1], 2 ** -0.5)

    def test_invalid_alignment_and_empty_span_fail(self) -> None:
        with self.assertRaisesRegex(ValueError, "lengths differ"):
            pool_token_embeddings([[1.0]], [], [(0, 1)])
        with self.assertRaisesRegex(ValueError, "no tokens"):
            pool_token_embeddings([[1.0]], [(0, 1)], [(2, 3)])


if __name__ == "__main__":
    unittest.main()
