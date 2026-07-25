"""Resilient JSON extraction, field normalization, and structured KIE metrics."""

from __future__ import annotations

import ast
import json
import re
import statistics
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from rapidfuzz.fuzz import ratio

from receipt_kie.prompts import CANONICAL_FIELDS, canonical_target

CODE_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", flags=re.IGNORECASE)
TRAILING_COMMA = re.compile(r",\s*([}\]])")


@dataclass(frozen=True)
class ParsedOutput:
    """A parsed prediction and the repair method used."""

    valid: bool
    value: dict[str, str] | None
    method: str
    error: str | None = None


def extract_json(raw_output: str) -> ParsedOutput:
    """Parse direct/fenced/embedded JSON and narrowly repair harmless syntax."""
    raw = raw_output.strip()
    attempts: list[tuple[str, str]] = [("direct", raw)]
    unfenced = CODE_FENCE.sub("", raw).strip()
    if unfenced != raw:
        attempts.append(("code_fence_removed", unfenced))
    embedded = _first_balanced_object(unfenced)
    if embedded and embedded not in {text for _, text in attempts}:
        attempts.append(("first_object", embedded))
    for method, text in attempts:
        parsed = _json_dict(text)
        if parsed is not None:
            return ParsedOutput(True, canonical_target(parsed), method)
        repaired = TRAILING_COMMA.sub(r"\1", text)
        if repaired != text:
            parsed = _json_dict(repaired)
            if parsed is not None:
                return ParsedOutput(True, canonical_target(parsed), f"{method}+trailing_comma")
        try:
            literal = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            literal = None
        if isinstance(literal, dict):
            return ParsedOutput(True, canonical_target(literal), f"{method}+literal_dict")
    return ParsedOutput(False, None, "invalid", "No safely parseable JSON object found")


def _json_dict(text: str) -> dict[str, Any] | None:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _first_balanced_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if escaped:
            escaped = False
            continue
        if char == "\\" and in_string:
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
        elif not in_string and char == "{":
            depth += 1
        elif not in_string and char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def normalize_text(value: str) -> str:
    """Normalize case, Unicode, punctuation, and whitespace conservatively."""
    normalized = unicodedata.normalize("NFKC", value).casefold().strip()
    normalized = re.sub(r"[^\w\s]", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def normalize_date(value: str) -> str:
    """Normalize common receipt date formats without guessing ambiguous text."""
    stripped = value.strip()
    formats = (
        "%d/%m/%Y",
        "%d-%m-%Y",
        "%Y-%m-%d",
        "%d.%m.%Y",
        "%d/%m/%y",
        "%d-%m-%y",
        "%d %b %Y",
        "%d %B %Y",
    )
    for date_format in formats:
        try:
            return datetime.strptime(stripped, date_format).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return normalize_text(stripped)


def normalize_total(value: str) -> str:
    """Normalize currency symbols, grouping separators, and decimal formatting."""
    cleaned = re.sub(r"[^\d,.\-]", "", value.strip())
    if "," in cleaned and "." in cleaned:
        cleaned = cleaned.replace(",", "")
    elif cleaned.count(",") == 1 and "." not in cleaned:
        left, right = cleaned.split(",")
        cleaned = f"{left}.{right}" if len(right) <= 2 else f"{left}{right}"
    else:
        cleaned = cleaned.replace(",", "")
    try:
        return format(Decimal(cleaned).quantize(Decimal("0.01")), "f")
    except (InvalidOperation, ValueError):
        return normalize_text(value)


def normalize_field(field: str, value: str) -> str:
    if field == "date":
        return normalize_date(value)
    if field == "total":
        return normalize_total(value)
    return normalize_text(value)


def evaluate_predictions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate raw/normalized exact match, similarity, validity, and latency."""
    if not rows:
        raise ValueError("Cannot calculate metrics from zero predictions")
    field_raw: dict[str, list[float]] = {field: [] for field in CANONICAL_FIELDS}
    field_normalized: dict[str, list[float]] = {field: [] for field in CANONICAL_FIELDS}
    field_similarity: dict[str, list[float]] = {field: [] for field in CANONICAL_FIELDS}
    complete_raw: list[float] = []
    complete_normalized: list[float] = []
    latencies: list[float] = []
    valid_count = 0
    peak_memory: list[float] = []
    for row in rows:
        truth = canonical_target(row["ground_truth"])
        parsed = row.get("parsed_prediction")
        valid = bool(row.get("valid_json") and isinstance(parsed, dict))
        if valid:
            valid_count += 1
            prediction = canonical_target(parsed)
        else:
            prediction = {field: "" for field in CANONICAL_FIELDS}
        raw_matches: list[bool] = []
        normalized_matches: list[bool] = []
        for field in CANONICAL_FIELDS:
            raw_match = prediction[field] == truth[field]
            normalized_prediction = normalize_field(field, prediction[field])
            normalized_truth = normalize_field(field, truth[field])
            normalized_match = normalized_prediction == normalized_truth
            similarity = ratio(normalized_prediction, normalized_truth) / 100.0
            field_raw[field].append(float(raw_match))
            field_normalized[field].append(float(normalized_match))
            field_similarity[field].append(similarity)
            raw_matches.append(raw_match)
            normalized_matches.append(normalized_match)
        complete_raw.append(float(all(raw_matches)))
        complete_normalized.append(float(all(normalized_matches)))
        if row.get("latency_seconds") is not None:
            latencies.append(float(row["latency_seconds"]))
        if row.get("peak_gpu_memory_mib") is not None:
            peak_memory.append(float(row["peak_gpu_memory_mib"]))
    per_field = {
        field: {
            "raw_exact_match": statistics.mean(field_raw[field]),
            "normalized_exact_match": statistics.mean(field_normalized[field]),
            "normalized_similarity": statistics.mean(field_similarity[field]),
        }
        for field in CANONICAL_FIELDS
    }
    return {
        "sample_count": len(rows),
        "valid_json_rate": valid_count / len(rows),
        "per_field": per_field,
        "company_accuracy": per_field["company"]["normalized_exact_match"],
        "address_similarity": per_field["address"]["normalized_similarity"],
        "date_accuracy": per_field["date"]["normalized_exact_match"],
        "total_accuracy": per_field["total"]["normalized_exact_match"],
        "complete_record_raw_exact_match": statistics.mean(complete_raw),
        "complete_record_normalized_exact_match": statistics.mean(complete_normalized),
        "average_inference_latency_seconds": statistics.mean(latencies) if latencies else None,
        "median_inference_latency_seconds": statistics.median(latencies) if latencies else None,
        "peak_gpu_memory_mib": max(peak_memory) if peak_memory else None,
    }
