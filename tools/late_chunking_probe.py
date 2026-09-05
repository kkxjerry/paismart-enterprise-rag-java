#!/usr/bin/env python3
"""Probe whether an OpenAI-compatible embedding endpoint exposes token states.

True late chunking needs token-level hidden states for one long-document forward
pass. A final pooled vector per input is insufficient. This probe records endpoint
schema and response shape without assuming that an accepted unknown parameter means
support. It never logs credentials or vector values.
"""
from __future__ import annotations

import argparse
import json
import math
import urllib.request
from pathlib import Path
from typing import Any


def request_json(url: str, *, body: dict[str, Any] | None = None, timeout: float = 30.0) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8") if body is not None else None,
        headers={"Content-Type": "application/json"},
        method="POST" if body is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise ValueError("endpoint response must be a JSON object")
    return value


def vector_shape(response: dict[str, Any]) -> dict[str, Any]:
    data = response.get("data")
    if not isinstance(data, list) or not data or not isinstance(data[0], dict):
        return {"data_rows": 0, "embedding_dimension": None, "row_keys": []}
    embedding = data[0].get("embedding")
    return {
        "data_rows": len(data),
        "embedding_dimension": len(embedding) if isinstance(embedding, list) else None,
        "row_keys": sorted(str(key) for key in data[0]),
        "token_state_fields": sorted(
            str(key) for key in data[0]
            if "token" in str(key).casefold() or "hidden" in str(key).casefold()
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-4B")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--openapi-url")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")

    sample = (
        "Document title: Upgrade rollback guidance. "
        "With pre-upgrade snapshots in place, restoration is typically tens of minutes. "
        "Exact timing depends on the snapshot mechanism and database size."
    )
    response = request_json(
        args.endpoint,
        body={"model": args.model, "input": [sample]},
        timeout=args.timeout,
    )
    shape = vector_shape(response)
    openapi = None
    openapi_mentions = []
    if args.openapi_url:
        try:
            openapi = request_json(args.openapi_url, timeout=args.timeout)
            encoded = json.dumps(openapi, ensure_ascii=False).casefold()
            openapi_mentions = [
                name for name in ("token_embeddings", "hidden_states", "last_hidden_state", "pooling")
                if name in encoded
            ]
        except Exception as exc:  # Visible diagnostic, not silent capability success.
            openapi_mentions = [f"openapi_error:{type(exc).__name__}:{exc}"]

    token_fields = shape.get("token_state_fields") or []
    supported = bool(token_fields or any(
        value in openapi_mentions for value in ("token_embeddings", "hidden_states", "last_hidden_state")
    ))
    result = {
        "endpoint": args.endpoint,
        "model": args.model,
        "response_top_level_keys": sorted(str(key) for key in response),
        **shape,
        "openapi_url": args.openapi_url,
        "openapi_token_state_mentions": openapi_mentions,
        "true_late_chunking_supported_by_observed_api": supported,
        "decision": (
            "eligible_for_hidden-state implementation POC"
            if supported
            else "blocked: observed API exposes pooled embeddings only; use parent-child/context prefix instead"
        ),
        "vectors_logged": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
