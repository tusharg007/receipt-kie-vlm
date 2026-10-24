from __future__ import annotations

import json

import pytest

from receipt_kie.structured import (
    receipt_schema,
    review_issues,
    strict_receipt,
    structured_metadata,
)

RECEIPT = {"company": "ACME", "address": "1 Road", "date": "06/05/13", "total": "285.80"}


def test_schema_matches_the_training_contract() -> None:
    schema = receipt_schema()
    assert schema["required"] == list(RECEIPT)
    assert schema["additionalProperties"] is False
    assert strict_receipt(json.dumps(RECEIPT)) == (RECEIPT, [])


@pytest.mark.parametrize(
    "raw",
    [
        "{}",
        "[]",
        '{"company":"ACME"',
        json.dumps({**RECEIPT, "total": 285.8}),
        json.dumps({**RECEIPT, "total": None}),
        json.dumps({**RECEIPT, "extra": "x"}),
        json.dumps(RECEIPT)[:-1] + ',"total":"1"}',
        json.dumps(RECEIPT).replace('"285.80"', "NaN"),
        "```json\n" + json.dumps(RECEIPT) + "\n```",
        json.dumps(RECEIPT) + " trailing explanation",
    ],
)
def test_strict_validation_never_repairs_or_coerces(raw: str) -> None:
    value, errors = strict_receipt(raw)
    assert value is None
    assert errors
    assert structured_metadata(raw)["review_required"]


def test_review_flags_do_not_rewrite_values() -> None:
    receipt = {**RECEIPT, "date": "31/02/2024", "total": "12O.5O", "address": ""}
    issues = review_issues(receipt)
    assert set(issues) == {
        "missing:address",
        "date:unrecognized_or_invalid",
        "total:unrecognized_amount",
    }
    assert receipt["total"] == "12O.5O"
    assert structured_metadata(json.dumps(receipt))["schema_valid"]


@pytest.mark.parametrize("date", ["12/31/2024", "31/12/2024", "2024-12-31", "29/02/2024"])
def test_review_accepts_calendar_dates_without_changing_locale(date: str) -> None:
    assert not review_issues({**RECEIPT, "date": date})


@pytest.mark.parametrize("total", ["RM 1,234.50", "$285.80", "1.234,50", "-12.50", "0"])
def test_review_accepts_common_monetary_notation(total: str) -> None:
    assert not review_issues({**RECEIPT, "total": total})


def test_token_limit_is_visible_even_if_object_is_complete() -> None:
    metadata = structured_metadata(json.dumps(RECEIPT), limit_hit=True)
    assert metadata["schema_valid"]
    assert metadata["review_issues"] == ["generation:token_limit"]
