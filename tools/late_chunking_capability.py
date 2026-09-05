#!/usr/bin/env python3
"""E5 late-chunking capability probe and token-vector pooling utilities.

Late chunking needs token-level contextual vectors plus token/character alignment.
A normal OpenAI-compatible embedding response containing one flat vector per input
is insufficient. This probe records that distinction instead of treating endpoint
availability as late-chunking support.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import urllib.request
from pathlib import Path
from typing import Any, Iterable, Sequence

PROBE_VERSION = "late-chunking-capability-v1"


def inspect_embedding_response(payload: dict[str, Any]) -> dict[str, Any]:
    data = payload.get("data")
    if not isinstance(data, list) or not data or not isinstance(data[0], dict):
        return {
            "response_valid": False,
            "pooled_embedding_available": False,
            "token_embeddings_available": False,
            "offset_mapping_available": False,
            "late_chunking_ready": False,
            "reason": "response has no OpenAI-compatible data[0] object",
        }
    first = data[0]
    embedding = first.get("embedding")
    pooled = (
        isinstance(embedding, list)
        and bool(embedding)
        and all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in embedding)
    )
    nested = (
        isinstance(embedding, list)
        and bool(embedding)
        and all(
            isinstance(row, list)
            and row
            and all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in row)
            for row in embedding
        )
    )
    token_embeddings = first.get("token_embeddings") or payload.get("token_embeddings")
    if isinstance(token_embeddings, list) and token_embeddings and isinstance(token_embeddings[0], list):
        nested = True
    offsets = first.get("offset_mapping") or first.get("offsets") or payload.get("offset_mapping")
    offset_available = (
        isinstance(offsets, list)
        and bool(offsets)
        and all(
            isinstance(value, (list, tuple))
            and len(value) == 2
            and all(isinstance(item, int) and not isinstance(item, bool) for item in value)
            for value in offsets
        )
    )
    ready = nested and offset_available
    if ready:
        reason = "token embeddings and character offsets are available"
    elif pooled:
        reason = "endpoint returns one pooled vector per input, not token-level contextual vectors"
    elif nested:
        reason = "token-level vectors exist but no character offset mapping is exposed"
    else:
        reason = "no usable embedding vector was found"
    dimension = None
    token_count = None
    if pooled:
        dimension = len(embedding)
    elif nested:
        vectors = token_embeddings if isinstance(token_embeddings, list) else embedding
        token_count = len(vectors)
        dimension = len(vectors[0]) if vectors else None
    return {
        "response_valid": pooled or nested,
        "pooled_embedding_available": pooled,
        "token_embeddings_available": nested,
        "offset_mapping_available": offset_available,
        "late_chunking_ready": ready,
        "dimension": dimension,
        "token_count": token_count,
        "reason": reason,
    }


def pool_token_embeddings(
    token_embeddings: Sequence[Sequence[float]],
    offset_mapping: Sequence[tuple[int, int]],
    spans: Sequence[tuple[int, int]],
) -> list[list[float]]:
    """Mean-pool contextual token vectors for character spans and L2-normalize."""
    if len(token_embeddings) != len(offset_mapping):
        raise ValueError("token embeddings and offset mapping lengths differ")
    if not token_embeddings:
        raise ValueError("token embeddings must not be empty")
    dimension = len(token_embeddings[0])
    if dimension <= 0 or any(len(vector) != dimension for vector in token_embeddings):
        raise ValueError("token embeddings must have one positive, stable dimension")
    vectors: list[list[float]] = []
    for start, end in spans:
        if start < 0 or end <= start:
            raise ValueError(f"invalid character span: {(start, end)}")
        selected = [
            vector
            for vector, (token_start, token_end) in zip(token_embeddings, offset_mapping)
            if token_end > start and token_start < end and token_end > token_start
        ]
        if not selected:
            raise ValueError(f"no tokens overlap character span {(start, end)}")
        pooled = [sum(float(vector[index]) for vector in selected) / len(selected) for index in range(dimension)]
        norm = math.sqrt(sum(value * value for value in pooled))
        if norm <= 0 or not math.isfinite(norm):
            raise ValueError("pooled token vector has an invalid norm")
        vectors.append([value / norm for value in pooled])
    return vectors


def probe_endpoint(
    *,
    endpoint: str,
    model: str,
    api_key: str = "",
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    payload = {"model": model, "input": ["Alpha context. Beta value is 42 ms."]}
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        result = json.load(response)
        status = int(getattr(response, "status", 200))
    if not isinstance(result, dict):
        raise ValueError("embedding response must be an object")
    return {
        "probe_version": PROBE_VERSION,
        "endpoint": endpoint,
        "model_requested": model,
        "http_status": status,
        "model_returned": result.get("model"),
        **inspect_embedding_response(result),
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--endpoint")
    source.add_argument("--response-json", type=Path)
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-4B")
    parser.add_argument("--api-key-env", default="MODELSCOPE_API_KEY")
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.response_json is not None and not args.response_json.is_file():
        parser.error(f"response JSON is missing: {args.response_json}")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.response_json:
        payload = json.loads(args.response_json.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise SystemExit("response JSON must contain an object")
        result = {"probe_version": PROBE_VERSION, "source": str(args.response_json), **inspect_embedding_response(payload)}
    else:
        result = probe_endpoint(
            endpoint=str(args.endpoint),
            model=args.model,
            api_key=os.getenv(args.api_key_env, ""),
            timeout_seconds=args.timeout_seconds,
        )
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result.get("response_valid") else 2


if __name__ == "__main__":
    raise SystemExit(main())
