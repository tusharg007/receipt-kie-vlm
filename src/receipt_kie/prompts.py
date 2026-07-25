"""Prompt construction and deterministic structured targets."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

CANONICAL_FIELDS = ("company", "address", "date", "total")
FIELD_ALIASES = {
    "company_name": "company",
    "merchant": "company",
    "merchant_name": "company",
    "addr": "address",
    "invoice_date": "date",
    "amount": "total",
    "grand_total": "total",
}
SYSTEM_PROMPT = (
    "You extract structured key information from financial receipt images. "
    "Follow the requested schema exactly."
)
USER_PROMPT = (
    "Extract company, address, date and total from this receipt. "
    "Return valid JSON only with exactly these keys in this order: "
    "company, address, date, total. Do not include Markdown or explanations. "
    "Use an empty string when a field is not visible."
)


def canonical_target(payload: Mapping[str, Any]) -> dict[str, str]:
    """Normalize known annotation aliases into the four-field canonical schema."""
    normalized: dict[str, Any] = {}
    for key, value in payload.items():
        normalized_key = FIELD_ALIASES.get(str(key).strip().lower(), str(key).strip().lower())
        normalized[normalized_key] = value
    return {
        field: "" if normalized.get(field) is None else str(normalized.get(field, "")).strip()
        for field in CANONICAL_FIELDS
    }


def canonical_json(payload: Mapping[str, Any]) -> str:
    """Serialize a target with stable key ordering and whitespace."""
    return json.dumps(
        canonical_target(payload),
        ensure_ascii=False,
        separators=(", ", ": "),
    )


def build_messages(
    image: str,
    target: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build the SmolVLM multimodal chat conversation."""
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": USER_PROMPT},
            ],
        },
    ]
    if target is not None:
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": canonical_json(target)}],
            }
        )
    return messages
