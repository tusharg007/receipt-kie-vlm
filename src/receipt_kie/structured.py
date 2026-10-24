"""Strict receipt contracts and conservative review signals, without model imports."""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any

from receipt_kie.prompts import CANONICAL_FIELDS


def receipt_schema() -> dict[str, Any]:
    """Keep the four string fields used by the trained adapters and SROIE metrics."""
    return {
        "type": "object",
        "properties": {field: {"type": "string"} for field in CANONICAL_FIELDS},
        "required": list(CANONICAL_FIELDS),
        "additionalProperties": False,
    }


def validate_receipt(value: Any) -> list[str]:
    """Reject missing/extra keys and wrong types rather than coercing model errors."""
    if not isinstance(value, dict):
        return ["Receipt must be a JSON object"]
    errors = []
    for field in CANONICAL_FIELDS:
        if field not in value:
            errors.append(f"Missing field: {field}")
        elif not isinstance(value[field], str):
            errors.append(f"Field must be a string: {field}")
    errors.extend(f"Unexpected field: {key}" for key in value if key not in CANONICAL_FIELDS)
    return errors


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON number: {value}")


def strict_receipt(raw_output: str) -> tuple[dict[str, str] | None, list[str]]:
    """Accept only a complete, direct, schema-valid JSON object; never repair it."""
    try:
        value = json.loads(
            raw_output, object_pairs_hook=_unique_object, parse_constant=_reject_constant
        )
    except (ValueError, TypeError) as exc:
        return None, [str(exc)]
    errors = validate_receipt(value)
    if errors:
        return None, errors
    return {field: value[field] for field in CANONICAL_FIELDS}, []


def review_issues(value: dict[str, str]) -> list[str]:
    """Flag missing fields and suspicious values without inventing replacements.

    These checks do not establish that text is supported by the image. Dates are
    tested in both day-first and month-first formats; the original text is kept.
    """
    issues = [f"missing:{field}" for field in CANONICAL_FIELDS if not value[field].strip()]
    date = value["date"].strip()
    if date:
        formats = (
            "%d/%m/%Y",
            "%m/%d/%Y",
            "%d/%m/%y",
            "%m/%d/%y",
            "%d-%m-%Y",
            "%m-%d-%Y",
            "%d-%m-%y",
            "%m-%d-%y",
            "%Y-%m-%d",
            "%Y/%m/%d",
            "%d.%m.%Y",
            "%d %b %Y",
            "%d %B %Y",
        )
        for pattern in formats:
            try:
                datetime.strptime(date, pattern)
                break
            except ValueError:
                continue
        else:
            issues.append("date:unrecognized_or_invalid")
    total = value["total"].strip()
    if total:
        # Allow currency prefixes/suffixes and common grouping/decimal separators.
        amount = re.sub(r"^[A-Za-z$€£₹¥]+\s*|\s*[A-Za-z$€£₹¥]+$", "", total).strip()
        if not re.fullmatch(
            r"[+-]?(?:\d+|\d+[.,]\d{1,2}|\d{1,3}(?:,\d{3})+(?:\.\d{1,2})?"
            r"|\d{1,3}(?:\.\d{3})+(?:,\d{1,2})?)",
            amount,
        ):
            issues.append("total:unrecognized_amount")
    return issues


def structured_metadata(raw_output: str, *, limit_hit: bool = False) -> dict[str, Any]:
    """Expose schema compliance separately from the historical resilient parser."""
    value, errors = strict_receipt(raw_output)
    issues = review_issues(value) if value is not None else ["schema:invalid"]
    if limit_hit:
        issues.append("generation:token_limit")
    return {
        "schema_valid": value is not None,
        "schema_errors": errors,
        "review_required": bool(issues),
        "review_issues": issues,
    }
