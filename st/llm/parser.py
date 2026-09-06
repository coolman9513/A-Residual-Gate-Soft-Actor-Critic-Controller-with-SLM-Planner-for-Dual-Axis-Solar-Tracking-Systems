"""Parsing and validation helpers for SLM guidance responses."""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence


def _strip_common_wrappers(text: str) -> str:
    """Remove common LLM wrappers such as markdown fences and think blocks."""

    cleaned = str(text).strip()
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL | re.IGNORECASE).strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    return cleaned


def _candidate_json_objects(text: str) -> list[str]:
    """Return balanced JSON-object candidates from text."""

    candidates: list[str] = []
    start = None
    depth = 0

    for index, char in enumerate(text):
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    candidates.append(text[start:index + 1])
                    start = None

    return candidates


def _repair_common_json_typos(text: str) -> str:
    """Repair small JSON typos commonly produced by compact local LLMs.

    The repair is intentionally conservative: it only removes a stray quote
    immediately after a numeric literal before a comma or closing brace, e.g.
    ``"target_az":120.0",`` -> ``"target_az":120.0,``.
    """

    repaired = str(text)
    repaired = re.sub(r'(?<=\d)"(?=\s*[,}])', '', repaired)
    repaired = re.sub(r",\s*}", "}", repaired)
    return repaired


_FIELD_PATTERNS: dict[str, tuple[str, str]] = {
    "target_tilt":       (r'"target_tilt"\s*:\s*([\d.]+)',       "float"),
    "target_az":         (r'"target_az"\s*:\s*([\d.]+)',          "float"),
    "motion_budget_deg": (r'"motion_budget_deg"\s*:\s*([\d.]+)',  "float"),
    "pre_position_tilt": (r'"pre_position_tilt"\s*:\s*([\d.]+)',  "float"),
    "pre_position_az":   (r'"pre_position_az"\s*:\s*([\d.]+)',    "float"),
    "hold":              (r'"hold"\s*:\s*(true|false)',            "bool"),
    "horizon_min":       (r'"horizon_min"\s*:\s*(\d+)',           "int"),
    "confidence":        (r'"confidence"\s*:\s*([\d.]+)',         "float"),
}
_REQUIRED_FIELDS = {"target_tilt", "target_az", "motion_budget_deg"}


def _extract_fields(text: str) -> Mapping[str, Any] | None:
    """Extract GoalOutput fields individually from malformed/truncated JSON."""

    fields: dict[str, Any] = {}
    for key, (pattern, kind) in _FIELD_PATTERNS.items():
        m = re.search(pattern, text)
        if not m:
            continue
        raw = m.group(1)
        if kind == "bool":
            fields[key] = raw == "true"
        elif kind == "int":
            fields[key] = int(raw)
        else:
            fields[key] = float(raw)

    return fields if _REQUIRED_FIELDS.issubset(fields) else None


def extract_json_object(text: str) -> Mapping[str, Any]:
    """Extract the first valid JSON object from a model response."""

    if not text:
        raise ValueError("empty model response")

    cleaned = _strip_common_wrappers(text)

    for candidate in [cleaned, _repair_common_json_typos(cleaned)]:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    for candidate in _candidate_json_objects(cleaned):
        for attempt in [candidate, _repair_common_json_typos(candidate)]:
            try:
                parsed = json.loads(attempt)
            except json.JSONDecodeError:
                continue

            if isinstance(parsed, dict):
                return parsed

    # Fallback: extract individual fields for models that produce truncated JSON
    fields = _extract_fields(cleaned)
    if fields is not None:
        return fields

    raise ValueError(f"no valid JSON object found in response: {text!r}")


def extract_choice_from_text(text: str, allowed: Sequence[str], fallback: str) -> str:
    """Recover an allowed categorical label from non-JSON text if possible."""

    cleaned = _strip_common_wrappers(text).lower()

    for value in allowed:
        if re.search(rf"\b{re.escape(value.lower())}\b", cleaned):
            return value

    return fallback


def validate_choice(value: Any, allowed: Sequence[str], fallback: str) -> str:
    """Return a valid categorical choice or fallback."""

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in allowed:
            return normalized
    return fallback
