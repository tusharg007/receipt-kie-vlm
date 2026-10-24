from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts/analyze_qwen_experiment.py"
    spec = importlib.util.spec_from_file_location("qwen_analysis", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row():
    truth = {"company": "A", "address": "1 Rd", "date": "01/01/2024", "total": "1.00"}
    return {
        "sample_id": "A",
        "image_sha256": "hash",
        "ground_truth": truth,
        "parsed_prediction": dict(truth),
        "valid_json": True,
    }


def test_comparison_rejects_missing_outputs_or_wrong_image_content():
    module = _module()
    with pytest.raises(ValueError, match="missing"):
        module.verify_alignment({"v2": [_row()], "qwen": []}, {"A": "hash"})
    with pytest.raises(ValueError, match="hash mismatch"):
        module.verify_alignment({"v2": [_row()], "qwen": [_row()]}, {"A": "different"})


def test_comparison_rejects_relabeling_and_keeps_invalid_results_in_denominator():
    module = _module()
    wrong = _row()
    wrong["ground_truth"] = {**wrong["ground_truth"], "total": "2.00"}
    with pytest.raises(ValueError, match="annotation mismatch"):
        module.verify_alignment({"v2": [_row()], "qwen": [wrong]}, {"A": "hash"})
    invalid = {**_row(), "valid_json": False, "parsed_prediction": None}
    macro, complete = module.correct_vectors([invalid], ["A"])
    assert macro.tolist() == complete.tolist() == [0.0]


def test_comparison_detects_saved_fields_that_disagree_with_model_output():
    module = _module()
    row = _row()
    row["raw_output"] = json.dumps(row["parsed_prediction"])
    module.verify_raw_predictions([row])
    row["raw_output"] = "```json\n" + row["raw_output"] + "\n```"
    with pytest.raises(ValueError, match="Saved parsing"):
        module.verify_raw_predictions([row])
    row["valid_json"], row["parsed_prediction"] = False, None
    module.verify_raw_predictions([row])
