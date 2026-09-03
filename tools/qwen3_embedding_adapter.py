#!/usr/bin/env python3
"""OpenAI-compatible embedding adapter for a fixed lower output dimension.

The upstream service returns the model's native vector. This adapter removes
provider-specific dimension hints, truncates each vector to ``--target-dimension``,
L2-normalizes the truncated vector, and returns the original response envelope.

It is intentionally small and dependency-light so the historical 2048-dimensional
Qwen3 index can be reproduced without pretending that the model natively emits
2048 dimensions.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.error
import urllib.request
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


class AdapterError(RuntimeError):
    pass


def adapt_response(payload: dict[str, Any], target_dimension: int) -> dict[str, Any]:
    rows = payload.get("data")
    if not isinstance(rows, list):
        raise AdapterError("upstream response does not contain a data array")

    adapted_rows: list[dict[str, Any]] = []
    for position, raw_row in enumerate(rows):
        if not isinstance(raw_row, dict):
            raise AdapterError(f"upstream data[{position}] is not an object")
        vector = raw_row.get("embedding")
        if not isinstance(vector, list):
            raise AdapterError(f"upstream data[{position}].embedding is not an array")
        if len(vector) < target_dimension:
            raise AdapterError(
                f"upstream vector dimension {len(vector)} is smaller than target {target_dimension}"
            )
        truncated = [float(value) for value in vector[:target_dimension]]
        norm = math.sqrt(sum(value * value for value in truncated))
        if not math.isfinite(norm) or norm <= 0.0:
            raise AdapterError(f"upstream data[{position}] has an invalid L2 norm")
        normalized = [value / norm for value in truncated]
        row = dict(raw_row)
        row["embedding"] = normalized
        adapted_rows.append(row)

    result = dict(payload)
    result["data"] = adapted_rows
    return result


def prepare_upstream_request(payload: dict[str, Any]) -> dict[str, Any]:
    request = dict(payload)
    request.pop("dimensions", None)
    request.pop("dimension", None)
    request.pop("input_type", None)
    if request.get("encoding_format") not in (None, "float"):
        raise AdapterError("only encoding_format=float is supported")
    request["encoding_format"] = "float"
    return request


def build_handler(upstream_url: str, target_dimension: int, timeout_seconds: float):
    class Handler(BaseHTTPRequestHandler):
        server_version = "Qwen3EmbeddingAdapter/1.0"

        def log_message(self, fmt: str, *args: Any) -> None:
            sys.stderr.write(
                "%s - - [%s] %s\n"
                % (self.client_address[0], self.log_date_time_string(), fmt % args)
            )

        def _json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self._json(
                    HTTPStatus.OK,
                    {
                        "status": "ok",
                        "upstream_url": upstream_url,
                        "target_dimension": target_dimension,
                        "transform": "first_n_then_l2_normalize",
                    },
                )
                return
            if self.path == "/v1/models":
                try:
                    request = urllib.request.Request(upstream_url.rsplit("/", 1)[0] + "/models")
                    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                        body = response.read()
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                except Exception as exc:  # pragma: no cover - network path
                    self._json(HTTPStatus.BAD_GATEWAY, error_payload(str(exc), "upstream_error"))
                return
            self._json(HTTPStatus.NOT_FOUND, error_payload("not found", "not_found"))

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/v1/embeddings":
                self._json(HTTPStatus.NOT_FOUND, error_payload("not found", "not_found"))
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0:
                    raise AdapterError("request body is empty")
                raw = self.rfile.read(length)
                payload = json.loads(raw)
                if not isinstance(payload, dict):
                    raise AdapterError("request body must be a JSON object")
                upstream_payload = prepare_upstream_request(payload)
                upstream_request = urllib.request.Request(
                    upstream_url,
                    data=json.dumps(upstream_payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                authorization = self.headers.get("Authorization")
                if authorization:
                    upstream_request.add_header("Authorization", authorization)
                with urllib.request.urlopen(
                    upstream_request, timeout=timeout_seconds
                ) as response:
                    upstream_response = json.loads(response.read())
                if not isinstance(upstream_response, dict):
                    raise AdapterError("upstream response must be a JSON object")
                result = adapt_response(upstream_response, target_dimension)
                self._json(HTTPStatus.OK, result)
            except urllib.error.HTTPError as exc:
                message = exc.read().decode("utf-8", errors="replace")
                self._json(
                    HTTPStatus.BAD_GATEWAY,
                    error_payload(
                        f"upstream HTTP {exc.code}: {message[:1000]}", "upstream_http_error"
                    ),
                )
            except (json.JSONDecodeError, ValueError, AdapterError) as exc:
                self._json(HTTPStatus.BAD_REQUEST, error_payload(str(exc), "invalid_request"))
            except Exception as exc:  # pragma: no cover - network path
                self._json(HTTPStatus.BAD_GATEWAY, error_payload(str(exc), "upstream_error"))

    return Handler


def error_payload(message: str, error_type: str) -> dict[str, Any]:
    return {
        "error": {
            "message": message,
            "type": error_type,
            "param": None,
            "code": None,
        }
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18084)
    parser.add_argument(
        "--upstream-url", default="http://127.0.0.1:18085/v1/embeddings"
    )
    parser.add_argument("--target-dimension", type=int, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    args = parser.parse_args()
    if args.target_dimension <= 0:
        parser.error("--target-dimension must be positive")
    if args.port <= 0 or args.port > 65535:
        parser.error("--port must be between 1 and 65535")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    return args


def main() -> int:
    args = parse_args()
    server = ThreadingHTTPServer(
        (args.host, args.port),
        build_handler(args.upstream_url, args.target_dimension, args.timeout_seconds),
    )
    print(
        json.dumps(
            {
                "event": "embedding_adapter_started",
                "host": args.host,
                "port": args.port,
                "upstream_url": args.upstream_url,
                "target_dimension": args.target_dimension,
                "transform": "first_n_then_l2_normalize",
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
