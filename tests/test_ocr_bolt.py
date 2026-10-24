from __future__ import annotations

import json

import httpx

from receipt_kie.ocr_bolt import BoltOCRPredictor


def test_ocr_text_is_the_only_model_evidence_and_strict_schema_is_requested(tmp_path) -> None:
    image = tmp_path / "receipt.png"
    image.touch()
    calls = []
    expected = {"company": "ACME", "address": "1 Road", "date": "01/02/2024", "total": "12.50"}

    def respond(request):
        payload = json.loads(request.content)
        calls.append(payload)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": json.dumps(expected)}, "finish_reason": "stop"}]
            },
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        predictor = BoltOCRPredictor(
            client=client, ocr=lambda _: [{"text": "ACME TOTAL 12.50", "confidence": 0.9}]
        )
        result = predictor.predict(image)
    assert result["parsed_prediction"] == expected
    assert result["review_required"] and not result["deployment_allowed"]
    assert json.loads(calls[0]["messages"][1]["content"])["ocr_lines"] == ["ACME TOTAL 12.50"]
    assert calls[0]["response_format"]["json_schema"]["schema"]["additionalProperties"] is False
    assert calls[0]["reasoning_effort"] == "none"
    assert "ground_truth" not in json.dumps(calls)


def test_no_ocr_evidence_abstains_without_querying_the_model(tmp_path) -> None:
    image = tmp_path / "receipt.png"
    image.touch()

    def never_call(request):
        raise AssertionError("Model must not be asked to invent a receipt")

    with httpx.Client(transport=httpx.MockTransport(never_call)) as client:
        result = BoltOCRPredictor(client=client, ocr=lambda _: []).predict(image)
    assert all(value == "" for value in result["parsed_prediction"].values())
    assert result["review_required"]


def test_malformed_model_fields_are_not_coerced_into_a_success(tmp_path) -> None:
    image = tmp_path / "receipt.png"
    image.touch()

    def respond(request):
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"total":12.50}'}}]})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        result = BoltOCRPredictor(client=client, ocr=lambda _: [{"text": "TOTAL 12.50"}]).predict(
            image
        )
    assert not result["valid_json"] and result["parsed_prediction"] is None


def test_upstream_error_remains_an_invalid_scored_record(tmp_path) -> None:
    image = tmp_path / "receipt.png"
    image.touch()
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(503))) as client:
        result = BoltOCRPredictor(client=client, ocr=lambda _: [{"text": "ACME"}]).predict(image)
    assert not result["valid_json"] and result["inference_error"] == "HTTPStatusError"


def test_token_budget_exhaustion_is_not_scored_as_success_even_with_json(tmp_path) -> None:
    image = tmp_path / "receipt.png"
    image.touch()
    complete = json.dumps({"company": "ACME", "address": "1 Rd", "date": "", "total": "1"})
    payload = {"choices": [{"message": {"content": complete}, "finish_reason": "length"}]}
    transport = httpx.MockTransport(lambda _: httpx.Response(200, json=payload))
    with httpx.Client(transport=transport) as client:
        result = BoltOCRPredictor(client=client, ocr=lambda _: [{"text": "ACME"}]).predict(image)
    assert not result["valid_json"] and result["review_required"]
