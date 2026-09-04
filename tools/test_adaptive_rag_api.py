from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from tools.adaptive_rag.retrieval import SearchPrincipal, SecondaryRetrievalClient
from tools.adaptive_rag_api import TtlCache, answer_cache_key, parse_request, validate_service_auth


class AdaptiveRagApiTest(unittest.TestCase):
    def test_parse_request_normalizes_acl_arrays(self) -> None:
        query, principal = parse_request(
            {
                "query": "  What is the limit?  ",
                "principal": {
                    "tenant_id": " tenant-a ",
                    "group_ids": [" group-b ", "group-a", "group-a"],
                    "classifications": ["internal"],
                },
                "source_types": ["jira", "jira"],
            },
            allow_body_principal=True,
        )

        self.assertEqual(query, "What is the limit?")
        self.assertEqual(principal.tenant_id, "tenant-a")
        self.assertEqual(principal.group_ids, ("group-a", "group-b"))
        self.assertEqual(principal.source_types, ("jira",))

    def test_parse_request_rejects_string_acl_list(self) -> None:
        with self.assertRaisesRegex(ValueError, "group_ids must be an array"):
            parse_request(
                {
                    "query": "value",
                    "principal": {"tenant_id": "tenant-a", "group_ids": "group-a"},
                },
                allow_body_principal=True,
            )

    def test_parse_request_uses_trusted_headers_and_ignores_forged_body(self) -> None:
        query, principal = parse_request(
            {
                "query": "value",
                "principal": {
                    "tenant_id": "tenant-forged",
                    "group_ids": ["admin"],
                    "classifications": ["secret"],
                },
            },
            headers={
                "X-RAG-Tenant-Id": "tenant-header",
                "X-RAG-Group-Ids": "group-b,group-a",
                "X-RAG-Classifications": "internal,confidential",
                "X-RAG-Source-Types": "jira,slack",
            },
        )
        self.assertEqual(query, "value")
        self.assertEqual(principal.tenant_id, "tenant-header")
        self.assertEqual(principal.group_ids, ("group-a", "group-b"))
        self.assertEqual(principal.classifications, ("confidential", "internal"))
        self.assertEqual(principal.source_types, ("jira", "slack"))

    def test_principal_requires_tenant(self) -> None:
        with self.assertRaisesRegex(ValueError, "tenant_id is required"):
            SearchPrincipal("", tuple())

    def test_principal_requires_explicit_classification_scope(self) -> None:
        with self.assertRaisesRegex(ValueError, "classification"):
            SearchPrincipal("tenant-a", ("group-a",))

    def test_service_auth_is_fail_closed_and_loopback_only(self) -> None:
        secured = SimpleNamespace(
            host="127.0.0.1",
            search_api_url="http://127.0.0.1:18090",
            allow_unauthenticated_loopback=False,
            allow_body_principal=False,
        )
        with self.assertRaisesRegex(ValueError, "Answer API requires"):
            validate_service_auth(secured, "", "")

        local = SimpleNamespace(
            host="127.0.0.1",
            search_api_url="http://localhost:18090",
            allow_unauthenticated_loopback=True,
            allow_body_principal=False,
        )
        validate_service_auth(local, "", "")

        exposed = SimpleNamespace(
            host="0.0.0.0",
            search_api_url="http://localhost:18090",
            allow_unauthenticated_loopback=True,
            allow_body_principal=False,
        )
        with self.assertRaisesRegex(ValueError, "Answer API requires"):
            validate_service_auth(exposed, "", "")

        remote_search = SimpleNamespace(
            host="127.0.0.1",
            search_api_url="https://search.example.com",
            allow_unauthenticated_loopback=True,
            allow_body_principal=False,
        )
        with self.assertRaisesRegex(ValueError, "Search API requires"):
            validate_service_auth(remote_search, "answer-key", "")

        body_principal_exposed = SimpleNamespace(
            host="0.0.0.0",
            search_api_url="http://127.0.0.1:18090",
            allow_unauthenticated_loopback=False,
            allow_body_principal=True,
        )
        with self.assertRaisesRegex(ValueError, "body principal"):
            validate_service_auth(body_principal_exposed, "answer-key", "search-key")

    def test_cache_key_is_acl_and_model_scoped(self) -> None:
        models = {"mapper": "qwen-flash", "generator": "qwen-flash", "verifier": "qwen-flash"}
        first = answer_cache_key(
            "question",
            SearchPrincipal("tenant-a", ("group-a",), ("jira",), ("internal",)),
            "fast",
            "optimize",
            models,
        )
        second = answer_cache_key(
            "question",
            SearchPrincipal("tenant-a", ("group-b",), ("jira",), ("internal",)),
            "fast",
            "optimize",
            models,
        )
        third = answer_cache_key(
            "question",
            SearchPrincipal("tenant-a", ("group-a",), ("jira",), ("internal",)),
            "fast",
            "validate",
            {**models, "generator": "qwen-plus"},
        )

        self.assertNotEqual(first, second)
        self.assertNotEqual(first, third)

    def test_ttl_cache_returns_copy(self) -> None:
        cache = TtlCache(max_entries=2, ttl_seconds=10)
        cache.put("key", {"answer": {"text": "original"}})

        result = cache.get("key")
        self.assertIsNotNone(result)
        result["answer"]["text"] = "mutated"

        self.assertEqual(cache.get("key")["answer"]["text"], "original")

    def test_secondary_client_sends_principal_in_trusted_headers_not_body(self) -> None:
        captured = {}

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return b'{"contexts":[]}'

        def fake_urlopen(request, timeout):
            captured["payload"] = json.loads(request.data)
            captured["headers"] = {key.lower(): value for key, value in request.header_items()}
            captured["timeout"] = timeout
            return Response()

        client = SecondaryRetrievalClient(
            api_url="http://localhost:18090",
            api_key="search-key",
            timeout_seconds=5,
        )
        principal = SearchPrincipal(
            "tenant-a",
            ("group-b", "group-a"),
            ("jira",),
            ("internal",),
        )
        with patch("tools.adaptive_rag.retrieval.urllib.request.urlopen", side_effect=fake_urlopen):
            result = client.search("limit", principal)

        self.assertEqual(result, {"contexts": []})
        self.assertNotIn("principal", captured["payload"])
        self.assertNotIn("source_types", captured["payload"])
        self.assertEqual(captured["headers"]["x-rag-tenant-id"], "tenant-a")
        self.assertEqual(captured["headers"]["x-rag-group-ids"], "group-a,group-b")
        self.assertEqual(captured["headers"]["x-rag-classifications"], "internal")
        self.assertEqual(captured["headers"]["x-rag-source-types"], "jira")
        self.assertEqual(captured["headers"]["authorization"], "Bearer search-key")

    def test_secondary_client_accepts_base_or_endpoint_url(self) -> None:
        self.assertEqual(
            SecondaryRetrievalClient(api_url="http://localhost:18090").api_url,
            "http://localhost:18090/v1/search",
        )
        self.assertEqual(
            SecondaryRetrievalClient(api_url="http://localhost:18090/v1/search").api_url,
            "http://localhost:18090/v1/search",
        )


if __name__ == "__main__":
    unittest.main()
