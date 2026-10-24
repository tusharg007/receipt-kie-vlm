from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
import torch
from PIL import Image

from receipt_kie.inference import ReceiptKIEPredictor
from receipt_kie.structured import receipt_schema

RECEIPT = {"company": "ACME", "address": "1 Road", "date": "06/05/13", "total": "285.80"}


def _predictor(raw: str, *, structured: bool) -> ReceiptKIEPredictor:
    predictor = ReceiptKIEPredictor.__new__(ReceiptKIEPredictor)
    predictor.structured_output = structured
    predictor.device = "cpu"
    predictor.dtype = torch.float32
    predictor.image_longest_edge_override = None
    predictor.processor_configuration = {}
    predictor.processor = MagicMock()
    predictor.processor.batch_decode.return_value = [raw]
    inputs = MagicMock()
    inputs.to.return_value = inputs
    tensors = {
        "pixel_values": torch.zeros(1, 1, 3, 2, 2),
        "input_ids": torch.tensor([[1, 2]]),
    }
    inputs.__getitem__.side_effect = tensors.__getitem__
    inputs.keys.return_value = list(tensors)
    predictor.processor.return_value = inputs
    predictor.model = MagicMock()
    predictor.model.generate.return_value = torch.tensor([[1, 2, 3]])
    predictor._token_constraint = MagicMock() if structured else None
    return predictor


@pytest.mark.parametrize("structured", [True, False])
def test_generation_applies_filter_only_when_requested(monkeypatch, structured) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    predictor = _predictor(json.dumps(RECEIPT), structured=structured)
    result = predictor.predict_images([Image.new("RGB", (2, 2))], "Extract receipt")
    kwargs = predictor.model.generate.call_args.kwargs
    assert ("prefix_allowed_tokens_fn" in kwargs) == structured
    assert result["schema_valid"]
    assert result["parsed_prediction"] == RECEIPT


def test_constrained_generation_does_not_accept_repaired_or_partial_output(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    raw = json.dumps({**RECEIPT, "total": 285.8})
    baseline = _predictor(raw, structured=False).predict_images(
        [Image.new("RGB", (2, 2))],
        "Extract receipt",
    )
    constrained = _predictor(raw, structured=True).predict_images(
        [Image.new("RGB", (2, 2))],
        "Extract receipt",
    )
    assert baseline["valid_json"] and not baseline["schema_valid"]
    assert not constrained["valid_json"]
    assert constrained["parsed_prediction"] is None
    assert constrained["review_required"]


def test_real_grammar_accepts_receipt_and_blocks_extra_keys_and_wrong_types() -> None:
    lmfe = pytest.importorskip("lmformatenforcer")

    def accepts(text: str) -> bool:
        parser = lmfe.JsonSchemaParser(receipt_schema())
        for character in text:
            if character not in parser.get_allowed_characters():
                return False
            parser = parser.add_character(character)
        return parser.can_end()

    assert accepts(json.dumps(RECEIPT))
    assert not accepts(json.dumps({**RECEIPT, "total": 285.8}))
    assert not accepts(json.dumps({**RECEIPT, "extra": "x"}))
    assert not accepts('{"company":"ACME"}')
