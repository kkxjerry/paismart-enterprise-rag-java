#!/usr/bin/env python3
"""Online Answer API for the adaptive RAG controller.

The service delegates tenant/ACL-aware retrieval to the Java search API and
uses Model Studio Qwen models for planning, generation, and conditional claim
verification. There is no browser/UI endpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import sys
import threading
import time
import uuid
import urllib.parse
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.adaptive_rag.controller import AdaptiveRagConfig, AdaptiveRagController
from tools.adaptive_rag.retrieval import SearchPrincipal, SecondaryRetrievalClient
from tools.qwen_plus_rag_pipeline import QwenClient

DEFAULT_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"


class TtlCache:
    def __init__(self, max_entries: int, ttl_seconds: float) -> None:
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self.values: OrderedDict[str, tuple[float, dict[str, Any]]] = OrderedDict()
        self.lock = threading.Lock()

    def get(self, key: str) -> dict[str, Any] | None:
        with self.lock:
            value = self.values.get(key)
            if value is None:
                return None
            expires, payload = value
            if expires < time.time():
                self.values.pop(key, None)
                return None
            self.values.move_to_end(key)
            return json.loads(json.dumps(payload))

    def put(self, key: str, value: dict[str, Any]) -> None:
        if self.ttl_seconds <= 0:
            return
        with self.lock:
            self.values[key] = (time.time() + self.ttl_seconds, json.loads(json.dumps(value)))
            self.values.move_to_end(key)
            while len(self.values) > self.max_entries:
                self.values.popitem(last=False)


class Application:
    def __init__(self, args: argparse.Namespace) -> None:
        model_default = "qwen-flash" if args.profile == "optimize" else "qwen-plus"
        api_key = os.getenv(args.api_key_env, "")
        if not api_key:
            raise ValueError(f"missing API key environment variable: {args.api_key_env}")
        search_key = os.getenv(args.search_api_key_env, "")
        self.answer_api_key = os.getenv(args.server_api_key_env, "")
        validate_service_auth(args, self.answer_api_key, search_key)
        self.search = SecondaryRetrievalClient(
            api_url=args.search_api_url,
            api_key=search_key,
            timeout_seconds=args.timeout_seconds,
            max_contexts_per_query=args.search_max_contexts,
        )
        self.controller = AdaptiveRagController(
            mapper_client=QwenClient(
                api_base=args.api_base,
                api_key=api_key,
                model=args.mapper_model or model_default,
                timeout_seconds=args.timeout_seconds,
                retries=args.retries,
            ),
            generator_client=QwenClient(
                api_base=args.api_base,
                api_key=api_key,
                model=args.generator_model or model_default,
                timeout_seconds=args.timeout_seconds,
                retries=args.retries,
            ),
            verifier_client=QwenClient(
                api_base=args.api_base,
                api_key=api_key,
                model=args.verifier_model or model_default,
                timeout_seconds=args.timeout_seconds,
                retries=args.retries,
            ),
            secondary_retrieval=self.search,
            config=AdaptiveRagConfig(
                map_fast_mode=args.map_fast_mode,
                requirements_max_chars=args.requirements_max_input_chars,
                requirements_max_count=args.requirements_max_count,
                requirements_max_selected=args.requirements_max_selected,
                requirements_max_tokens=args.requirements_max_tokens,
                generation_max_tokens=args.generation_max_tokens,
                verifier_mode=args.verifier_mode,
                verifier_max_chars=args.verifier_max_input_chars,
                verifier_max_tokens=args.verifier_max_tokens,
                temperature=args.temperature,
                secondary_max_queries=args.secondary_max_queries,
            ),
        )
        self.allow_body_principal = args.allow_body_principal
        self.max_request_bytes = args.max_request_bytes
        self.cache = TtlCache(args.cache_max_entries, args.cache_ttl_seconds)
        self.semaphore = threading.BoundedSemaphore(args.max_concurrent_requests)
        self.metrics_lock = threading.Lock()
        self.metrics = {
            "requests": 0,
            "errors": 0,
            "cache_hits": 0,
            "search_requests": 0,
            "answer_requests": 0,
            "total_latency_ms": 0.0,
        }
        self.profile = args.profile
        self.models = {
            "mapper": args.mapper_model or model_default,
            "generator": args.generator_model or model_default,
            "verifier": args.verifier_model or model_default,
        }

    def authorized(self, headers) -> bool:
        if not self.answer_api_key:
            return True
        return hmac.compare_digest(
            str(headers.get("Authorization") or ""),
            f"Bearer {self.answer_api_key}",
        )

    def handle_search(self, payload: dict[str, Any], headers) -> dict[str, Any]:
        query, principal = parse_request(
            payload,
            headers=headers,
            allow_body_principal=self.allow_body_principal,
        )
        response = self.search.search(query, principal)
        with self.metrics_lock:
            self.metrics["search_requests"] += 1
        return response

    def handle_answer(self, payload: dict[str, Any], headers) -> dict[str, Any]:
        query, principal = parse_request(
            payload,
            headers=headers,
            allow_body_principal=self.allow_body_principal,
        )
        forced_mode = payload.get("mode")
        if forced_mode is not None and forced_mode not in {"fast", "quality", "deep"}:
            raise ValueError("mode must be fast, quality, or deep")
        cache_key = answer_cache_key(query, principal, forced_mode, self.profile, self.models)
        cached = self.cache.get(cache_key)
        if cached is not None:
            with self.metrics_lock:
                self.metrics["cache_hits"] += 1
                self.metrics["answer_requests"] += 1
            cached["trace_id"] = str(uuid.uuid4())
            cached["cached"] = True
            return cached

        search_response = self.search.search(query, principal)
        row = {
            "qid": search_response.get("trace_id") or str(uuid.uuid4()),
            "question": query,
            "contexts": search_response.get("contexts") or [],
            "ranked_doc_ids": search_response.get("ranked_doc_ids") or [],
            "ranked_documents": search_response.get("ranked_documents") or [],
            "evidence_conflicts": search_response.get("evidence_conflicts") or [],
            "principal": principal.to_dict(),
            "is_evaluable": False,
            "retrieval_hit_at_10": False,
        }
        result = self.controller.process(
            row,
            details={"ranked_documents": search_response.get("ranked_documents") or []},
            principal=principal,
            forced_mode=forced_mode,
        )
        response = {
            "trace_id": str(uuid.uuid4()),
            "cached": False,
            "query": query,
            "router": result.get("router"),
            "requirements": result.get("requirements"),
            "secondary_retrieval": result.get("secondary_retrieval"),
            "budget": result.get("budget"),
            "answer": (result.get("generation") or {}).get("answer"),
            "answerable": (result.get("generation") or {}).get("answerable"),
            "citations": (result.get("generation") or {}).get("citations") or [],
            "sources": result.get("selected_contexts") or [],
            "verification": result.get("verification"),
            "usage": result.get("usage"),
            "latency_ms": result.get("total_latency_ms"),
            "error": result.get("error"),
        }
        if result.get("error"):
            raise RuntimeError(json.dumps(result["error"], ensure_ascii=False))
        stored = dict(response)
        stored.pop("trace_id", None)
        stored.pop("cached", None)
        self.cache.put(cache_key, stored)
        with self.metrics_lock:
            self.metrics["answer_requests"] += 1
        return response


def validate_service_auth(
    args: argparse.Namespace,
    answer_api_key: str,
    search_api_key: str,
) -> None:
    local_hosts = {"127.0.0.1", "localhost", "::1"}
    if not answer_api_key:
        if not args.allow_unauthenticated_loopback or args.host not in local_hosts:
            raise ValueError(
                "Answer API requires a non-empty server API key; unauthenticated mode is loopback-only and must be explicit"
            )
    if args.allow_body_principal and args.host not in local_hosts:
        raise ValueError("body principal mode is allowed only on a loopback host")
    search_host = urllib.parse.urlparse(args.search_api_url).hostname or ""
    if not search_api_key:
        if not args.allow_unauthenticated_loopback or search_host not in local_hosts:
            raise ValueError(
                "Search API requires a non-empty API key; unauthenticated upstream access is loopback-only and must be explicit"
            )


def parse_request(
    payload: dict[str, Any],
    *,
    headers: Any | None = None,
    allow_body_principal: bool = False,
) -> tuple[str, SearchPrincipal]:
    query = str(payload.get("query") or "").strip()
    if not query or len(query) > 8_000:
        raise ValueError("query must contain 1..8000 characters")
    if allow_body_principal:
        raw = payload.get("principal") or {}
        if not isinstance(raw, dict):
            raise ValueError("principal must be an object")
        principal = SearchPrincipal(
            tenant_id=str(raw.get("tenant_id") or ""),
            group_ids=_string_array(raw.get("group_ids"), "principal.group_ids"),
            classifications=_string_array(raw.get("classifications"), "principal.classifications"),
            source_types=_string_array(
                payload.get("source_types") if "source_types" in payload else raw.get("source_types"),
                "source_types",
            ),
        )
        return query, principal
    if headers is None:
        raise ValueError("trusted principal headers are required")
    return query, SearchPrincipal(
        tenant_id=str(headers.get("X-RAG-Tenant-Id") or ""),
        group_ids=_header_array(headers, "X-RAG-Group-Ids"),
        classifications=_header_array(headers, "X-RAG-Classifications"),
        source_types=_header_array(headers, "X-RAG-Source-Types"),
    )


def _header_array(headers: Any, name: str) -> tuple[str, ...]:
    raw_values = headers.get_all(name) if hasattr(headers, "get_all") else None
    if raw_values is None:
        value = headers.get(name) if hasattr(headers, "get") else None
        raw_values = [] if value is None else [value]
    values: list[str] = []
    for raw in raw_values:
        values.extend(part.strip() for part in str(raw).split(",") if part.strip())
    return tuple(values)


def _string_array(value: Any, field: str) -> tuple[str, ...]:
    if value is None:
        return tuple()
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an array")
    return tuple(str(item).strip() for item in value if str(item).strip())


def answer_cache_key(
    query: str,
    principal: SearchPrincipal,
    mode: str | None,
    profile: str,
    models: dict[str, str],
) -> str:
    payload = {
        "query": query,
        "principal": principal.to_dict(),
        "mode": mode,
        "profile": profile,
        "models": models,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def handler(application: Application):
    class Handler(BaseHTTPRequestHandler):
        server_version = "AdaptiveRagApi/1.0"

        def log_message(self, fmt: str, *args: Any) -> None:
            return

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self.reply(200, {"status": "ok"})
                return
            if self.path == "/metrics":
                if not application.authorized(self.headers):
                    self.reply(401, {"error": "unauthorized"})
                    return
                with application.metrics_lock:
                    metrics = dict(application.metrics)
                    metrics["cache_entries"] = len(application.cache.values)
                self.reply(200, metrics)
                return
            self.reply(404, {"error": "not_found"})

        def do_POST(self) -> None:  # noqa: N802
            started = time.perf_counter()
            with application.metrics_lock:
                application.metrics["requests"] += 1
            if not application.authorized(self.headers):
                self.reply(401, {"error": "unauthorized"})
                return
            if not application.semaphore.acquire(blocking=False):
                self.reply(429, {"error": "too_many_requests"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > application.max_request_bytes:
                    self.reply(413, {"error": "invalid_request_size"})
                    return
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("request body must be a JSON object")
                if self.path == "/v1/search":
                    result = application.handle_search(payload, self.headers)
                elif self.path == "/v1/answer":
                    result = application.handle_answer(payload, self.headers)
                else:
                    self.reply(404, {"error": "not_found"})
                    return
                self.reply(200, result)
            except ValueError as exc:
                with application.metrics_lock:
                    application.metrics["errors"] += 1
                self.reply(400, {"error": "invalid_request", "message": str(exc)})
            except Exception as exc:
                with application.metrics_lock:
                    application.metrics["errors"] += 1
                self.reply(500, {"error": "request_failed", "message": str(exc)})
            finally:
                application.semaphore.release()
                with application.metrics_lock:
                    application.metrics["total_latency_ms"] += (time.perf_counter() - started) * 1000.0

        def reply(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18091)
    parser.add_argument("--search-api-url", default="http://127.0.0.1:18090")
    parser.add_argument("--search-api-key-env", default="RAG_SEARCH_API_KEY")
    parser.add_argument("--server-api-key-env", default="RAG_ANSWER_API_KEY")
    parser.add_argument("--allow-unauthenticated-loopback", action="store_true")
    parser.add_argument("--allow-body-principal", action="store_true")
    parser.add_argument("--profile", choices=("optimize", "validate"), default="optimize")
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--mapper-model")
    parser.add_argument("--generator-model")
    parser.add_argument("--verifier-model")
    parser.add_argument("--map-fast-mode", action="store_true")
    parser.add_argument("--requirements-max-input-chars", type=int, default=48_000)
    parser.add_argument("--requirements-max-count", type=int, default=12)
    parser.add_argument("--requirements-max-selected", type=int, default=16)
    parser.add_argument("--requirements-max-tokens", type=int, default=1_024)
    parser.add_argument("--generation-max-tokens", type=int, default=1_024)
    parser.add_argument("--verifier-mode", choices=("off", "conditional", "always"), default="conditional")
    parser.add_argument("--verifier-max-input-chars", type=int, default=24_000)
    parser.add_argument("--verifier-max-tokens", type=int, default=1_024)
    parser.add_argument("--secondary-max-queries", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--search-max-contexts", type=int, default=30)
    parser.add_argument("--max-request-bytes", type=int, default=1_048_576)
    parser.add_argument("--max-concurrent-requests", type=int, default=16)
    parser.add_argument("--cache-ttl-seconds", type=float, default=0.0)
    parser.add_argument("--cache-max-entries", type=int, default=2_000)
    args = parser.parse_args()
    for name in (
        "port",
        "timeout_seconds",
        "search_max_contexts",
        "max_request_bytes",
        "max_concurrent_requests",
        "cache_max_entries",
        "requirements_max_input_chars",
        "requirements_max_count",
        "requirements_max_selected",
        "requirements_max_tokens",
        "generation_max_tokens",
        "verifier_max_input_chars",
        "verifier_max_tokens",
        "secondary_max_queries",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.cache_ttl_seconds < 0:
        parser.error("--cache-ttl-seconds must not be negative")
    return args


def main() -> int:
    args = parse_args()
    application = Application(args)
    server = ThreadingHTTPServer((args.host, args.port), handler(application))
    print(json.dumps({
        "event": "adaptive_rag_api_started",
        "host": args.host,
        "port": args.port,
        "profile": args.profile,
        "models": application.models,
        "search_api_url": args.search_api_url,
        "auth_enabled": bool(application.answer_api_key),
    }, ensure_ascii=False), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
