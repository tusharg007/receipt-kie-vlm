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


def test_comparison_rejects_a_report_that_disagrees_with_recomputed_field_metrics():
    module = _module()
    actual = module.evaluate_predictions([_row()])
    reported = json.loads(json.dumps(actual))
    reported["per_field"]["address"]["normalized_exact_match"] = 0.5
    with pytest.raises(ValueError, match="Reported metrics differ"):
        module.verify_metric_report(actual, reported)


def test_fenced_json_is_recovered_for_field_accuracy_but_not_strict_schema():
    module = _module()
    row = _row()
    row["raw_output"] = "```json\n" + json.dumps(row["ground_truth"]) + "\n```"
    row["parsed_prediction"], row["valid_json"] = None, False
    compared, compliance = module.comparison_rows([row])
    macro, complete = module.correct_vectors(compared, ["A"])
    assert macro.tolist() == complete.tolist() == [1.0]
    assert compliance["strict_json_valid_count"] == 0
    assert compliance["recovered_by_tolerant_parser"] == 1
    invalid = {**row, "raw_output": "not json"}
    compared, compliance = module.comparison_rows([invalid])
    macro, complete = module.correct_vectors(compared, ["A"])
    assert macro.tolist() == complete.tolist() == [0.0]
    assert compliance["recovered_by_tolerant_parser"] == 0
