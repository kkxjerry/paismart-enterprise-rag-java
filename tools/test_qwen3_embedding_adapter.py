from __future__ import annotations

import math
import unittest

from tools.qwen3_embedding_adapter import AdapterError, adapt_response, prepare_upstream_request


class Qwen3EmbeddingAdapterTest(unittest.TestCase):
    def test_truncates_and_l2_normalizes_each_vector(self) -> None:
        payload = {
            "object": "list",
            "data": [
                {"object": "embedding", "index": 0, "embedding": [3.0, 4.0, 12.0]},
                {"object": "embedding", "index": 1, "embedding": [0.0, 5.0, 8.0]},
            ],
            "model": "Qwen/Qwen3-Embedding-4B",
            "usage": {"prompt_tokens": 2, "total_tokens": 2},
        }

        result = adapt_response(payload, 2)

        self.assertEqual(len(result["data"][0]["embedding"]), 2)
        self.assertAlmostEqual(result["data"][0]["embedding"][0], 0.6)
        self.assertAlmostEqual(result["data"][0]["embedding"][1], 0.8)
        for row in result["data"]:
            norm = math.sqrt(sum(value * value for value in row["embedding"]))
            self.assertAlmostEqual(norm, 1.0)
        self.assertEqual(result["usage"], payload["usage"])

    def test_strips_dimension_hints_before_forwarding(self) -> None:
        request = prepare_upstream_request(
            {
                "model": "Qwen/Qwen3-Embedding-4B",
                "input": ["hello"],
                "dimensions": 2048,
                "dimension": 2048,
                "input_type": "query",
                "encoding_format": "float",
            }
        )

        self.assertNotIn("dimensions", request)
        self.assertNotIn("dimension", request)
        self.assertNotIn("input_type", request)
        self.assertEqual(request["encoding_format"], "float")

    def test_rejects_invalid_vectors_and_non_float_encoding(self) -> None:
        with self.assertRaises(AdapterError):
            adapt_response({"data": [{"embedding": [1.0]}]}, 2)
        with self.assertRaises(AdapterError):
            adapt_response({"data": [{"embedding": [0.0, 0.0]}]}, 2)
        with self.assertRaises(AdapterError):
            prepare_upstream_request({"encoding_format": "base64"})


if __name__ == "__main__":
    unittest.main()
