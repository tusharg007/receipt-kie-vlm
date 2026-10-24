from __future__ import annotations

import json
from pathlib import Path

import pytest

from receipt_kie.quality import FEATURE_NAMES, FEATURE_VERSION, QualityGate, quality_features


def row() -> dict:
    return {
        "raw_output": json.dumps(
            {"company": "ACME", "address": "1 Road", "date": "01/02/2024", "total": "12.50"}
        )
    }


def gate(threshold=0.4) -> QualityGate:
    return QualityGate(
        {
            "feature_version": FEATURE_VERSION,
            "feature_names": list(FEATURE_NAMES),
            "model_version": "test-only",
            "deployment_allowed": True,
            "threshold": threshold,
            "mean": [0.0] * len(FEATURE_NAMES),
            "scale": [1.0] * len(FEATURE_NAMES),
            "coefficients": [0.0] * len(FEATURE_NAMES),
            "intercept": 0.0,
        }
    )


def test_features_and_scores_cannot_read_labels_or_ids() -> None:
    first = row()
    second = {
        **first,
        "ground_truth": {"company": "WRONG"},
        "sample_id": "secret",
        "label": False,
        "entity_path": "must-not-be-read",
    }
    assert quality_features(first) == quality_features(second)
    assert gate().score(first) == gate().score(second) == 0.5


def test_gate_cannot_override_schema_or_review_rules() -> None:
    assert gate().assess(row())["action"] == "accept_candidate"
    invalid = {"raw_output": '{"company":"ACME"}'}
    assert gate().assess(invalid)["action"] == "review"
    assert gate(threshold=1.000001).assess(row())["action"] == "review"
    assert gate().assess({**row(), "generation_limit_hit": True})["action"] == "review"


def test_configuration_shift_forces_review() -> None:
    instance = gate()
    instance.artifact["extractor_configuration"] = {"size": {"longest_edge": 1536}}
    shifted = {**row(), "processor_configuration": {"size": {"longest_edge": 1024}}}
    assert "quality:extractor_configuration_mismatch" in instance.assess(shifted)["review_reasons"]


def test_high_model_score_cannot_override_company_address_contamination() -> None:
    prediction = {"company": "1 ROAD", "address": "1 Road", "date": "01/02/2024", "total": "12.50"}
    result = gate().assess({"raw_output": json.dumps(prediction)})
    assert result["action"] == "review"
    assert "quality:company_matches_address" in result["review_reasons"]


def test_feature_contract_and_numeric_parameters_are_checked() -> None:
    artifact = dict(gate().artifact)
    artifact["feature_names"] = ["ground_truth"]
    with pytest.raises(ValueError):
        QualityGate(artifact)
    artifact = dict(gate().artifact)
    artifact["coefficients"] = [float("nan")] * len(FEATURE_NAMES)
    with pytest.raises(ValueError):
        QualityGate(artifact)


def test_trained_artifact_is_disjoint_and_reports_retrospective_scope() -> None:
    root = Path(__file__).resolve().parents[1] / "artifacts/experiments/quality_gate"
    report = json.loads((root / "report.json").read_text(encoding="utf-8"))
    assert not set(report["fit_ids"]) & set(report["calibration_ids"])
    assert not (set(report["fit_ids"]) | set(report["calibration_ids"])) & set(
        report["evaluation_ids"]
    )
    assert report["original_validation_count"] == 63
    assert len(report["fit_ids"]) + len(report["calibration_ids"]) == 62
    assert {item["sample_id"] for item in report["validation_exclusions"]} == {"X51005663280"}
    assert "Retrospective" in report["evaluation_scope"]
    model = QualityGate.load(root / "quality_gate.json")
    assert 0 <= model.score(row()) <= 1
    assert not model.artifact["deployment_allowed"]
    assert model.assess(row())["action"] == "review"
    # Constant fit features need outer bins, so a new invalid-value flag can drift.
    reference = model.artifact["drift_reference"]["missing_count"]
    assert len(reference["cuts"]) >= 2
    assert sum(reference["counts"]) == len(report["fit_ids"])
