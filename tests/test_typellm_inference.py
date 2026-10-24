from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from receipt_kie.typellm_inference import (
    TOTAL_SOURCES,
    TypeLLMReceiptPredictor,
    receipt_questions,
)


@dataclass
class FakeUsage:
    completion_tokens: int = 8
    cached_tokens: int = 10


class FakeClient:
    def __init__(self, **overrides):
        self.result = {
            "company": "ACME",
            "address": "1 Road",
            "date": "06/05/13",
            "total": 285.8,
            "total_source": {"value": "printed_total", "probabilities": {}},
            **overrides,
        }
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(result=self.result, usage=FakeUsage())


def test_schema_batches_extraction_and_checks_total_afterwards() -> None:
    questions = receipt_questions(256)
    assert questions["total"]["type"] == ["number", "null"]
    assert questions["total_source"]["depends_on"] == ["total"]
    assert questions["total_source"]["enum"] == list(TOTAL_SOURCES)
    assert questions["total_source"]["permutations"] == "auto"
    assert questions["total"]["thinking_budget"] == 256
    assert "thinking" not in receipt_questions()["total"]


def test_native_client_passes_the_image_and_returns_compatible_fields(tmp_path) -> None:
    image = tmp_path / "receipt.png"
    image.write_bytes(b"fake client does not decode image")
    client = FakeClient()
    row = TypeLLMReceiptPredictor(client=client).predict(image)
    assert client.calls[0]["images"] == [str(image.resolve())]
    assert row["parsed_prediction"]["total"] == "285.80"
    assert row["usage"]["cached_tokens"] == 10
    assert row["schema_valid"] and not row["review_required"]
    assert row["typed_result"]["total"] == 285.8


def test_null_abstention_and_unsupported_total_are_reviewed(tmp_path) -> None:
    image = tmp_path / "receipt.png"
    image.touch()
    client = FakeClient(
        total=None,
        address=None,
        total_source={"value": "unclear", "probabilities": {}},
    )
    row = TypeLLMReceiptPredictor(client=client).predict(image)
    assert row["parsed_prediction"]["total"] == ""
    assert row["review_required"]
    assert "total:source_unclear" in row["review_issues"]


@pytest.mark.parametrize("total", [True, "285.80", float("nan"), float("inf")])
def test_invalid_native_numbers_fail_closed(tmp_path, total) -> None:
    image = tmp_path / "receipt.png"
    image.touch()
    with pytest.raises(ValueError, match="finite number"):
        TypeLLMReceiptPredictor(client=FakeClient(total=total)).predict(image)


def test_native_response_cannot_silently_omit_a_field(tmp_path) -> None:
    image = tmp_path / "receipt.png"
    image.touch()
    client = FakeClient()
    del client.result["address"]
    with pytest.raises(ValueError, match="field set"):
        TypeLLMReceiptPredictor(client=client).predict(image)


def test_invalid_settings_fail_before_loading_a_client() -> None:
    with pytest.raises(ValueError):
        TypeLLMReceiptPredictor(thinking_budget=-1)
    with pytest.raises(ValueError):
        TypeLLMReceiptPredictor(timeout=0)
    with pytest.raises(ValueError):
        TypeLLMReceiptPredictor(base_url="")
