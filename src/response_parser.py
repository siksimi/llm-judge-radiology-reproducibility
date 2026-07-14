"""
response_parser.py — parse & validate an LLM raw output into the structured 5-item score.
  1) strip code fences, then parse the first {...} JSON block.
  2) on failure, regex fallback: extract the 5 keys "key": N directly, even from truncated output.
Raw text is preserved; only the parsed result is produced.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional

REQUIRED_ITEMS = [
    "finding_completeness",
    "diagnostic_accuracy",
    "clarity_structure",
    "clinical_appropriateness",
    "overall_global_quality",
]

_JSON_BLOCK_RE = re.compile(r"\{.*\}", flags=re.DOTALL)


@dataclass
class ParseResult:
    ok: bool
    scores: dict = field(default_factory=dict)
    reason: Optional[str] = None
    recovered: bool = False  # whether recovered via the regex fallback


def _extract_json(raw: str):
    raw = raw.strip()
    try:
        return json.loads(raw)
    except Exception:
        pass
    m = _JSON_BLOCK_RE.search(raw)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _validate(obj) -> ParseResult:
    if not isinstance(obj, dict):
        return ParseResult(False, reason="not_an_object")
    scores = {}
    for item in REQUIRED_ITEMS:
        if item not in obj:
            return ParseResult(False, reason=f"missing_item:{item}")
        val = obj[item]
        if isinstance(val, str) and val.strip().lstrip("-").isdigit():
            val = int(val.strip())
        if isinstance(val, bool) or not isinstance(val, int):
            return ParseResult(False, reason=f"non_integer:{item}={obj[item]!r}")
        if not (1 <= val <= 5):
            return ParseResult(False, reason=f"out_of_range:{item}={val}")
        scores[item] = val
    return ParseResult(True, scores=scores)


def _regex_fallback(raw: str) -> ParseResult:
    """Extract "key": N patterns directly from truncated/irregular output (missing key or out-of-range fails)."""
    scores = {}
    for item in REQUIRED_ITEMS:
        m = re.search(rf'"{item}"\s*:\s*(-?\d+)', raw)
        if not m:
            return ParseResult(False, reason=f"missing_item:{item}")
        v = int(m.group(1))
        if not (1 <= v <= 5):
            return ParseResult(False, reason=f"out_of_range:{item}={v}")
        scores[item] = v
    return ParseResult(True, scores=scores, recovered=True)


def parse_scores(raw: str) -> ParseResult:
    obj = _extract_json(raw)
    if obj is not None:
        res = _validate(obj)
        if res.ok:
            return res
    # JSON parse/validation failed -> regex fallback
    fb = _regex_fallback(raw)
    return fb if fb.ok else ParseResult(False, reason=(fb.reason or "no_json_found"))


if __name__ == "__main__":
    tests = [
        '{"finding_completeness":5,"diagnostic_accuracy":4,"clarity_structure":3,"clinical_appropriateness":4,"overall_global_quality":4}',
        '```json\n{"finding_completeness":"3","diagnostic_accuracy":2,"clarity_structure":2,"clinical_appropriateness":3,"overall_global_quality":2}\n```',
        # truncated output (no closing brace) — expect fallback recovery
        '{\n "finding_completeness": 4,\n "diagnostic_accuracy": 5,\n "clarity_structure": 5,\n "clinical_appropriateness": 5,\n "overall_global_quality": 4',
        'sorry I cannot comply',
        '{"finding_completeness":5,"diagnostic_accuracy":6,"clarity_structure":3,"clinical_appropriateness":4,"overall_global_quality":4}',
    ]
    for t in tests:
        r = parse_scores(t)
        print(r.ok, "recovered" if r.recovered else "", r.reason or r.scores)
