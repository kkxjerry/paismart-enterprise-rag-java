"""Sentence boundaries that keep trailing source markers with their claim.

This is syntactic segmentation only; attaching a marker never proves support.
"""
from __future__ import annotations

import re

_BOUNDARY = re.compile(r"(?<=[.!?。！？])(?:\s*\[S[1-9][0-9]*\])*(?=\s|$)|\n+")
_ABBREVIATION = re.compile(r"\b(?:vs|e\.g|i\.e|Mr|Mrs|Ms|Dr)\.$", re.IGNORECASE)


def split_cited_segments(answer: str) -> list[str]:
    """Keep 'First. [S1] Second [S2].' as two correctly cited segments."""
    segments: list[str] = []
    start = 0
    for boundary in _BOUNDARY.finditer(answer):
        end = boundary.end()
        candidate = answer[start:end].strip()
        # A zero-width punctuation boundary must not turn 'vs.' or a list
        # ordinal into a separate factual claim. Newlines still delimit items.
        if boundary.start() == end and (
            _ABBREVIATION.search(candidate) or re.fullmatch(r"\d+\.", candidate)
        ):
            continue
        if candidate:
            segments.append(candidate)
        start = end
    tail = answer[start:].strip()
    if tail:
        segments.append(tail)
    return segments
