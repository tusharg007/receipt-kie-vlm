from __future__ import annotations

import pytest

from receipt_kie.monitoring import AuditStore, histogram, population_stability


def test_psi_detects_shift_and_handles_zero_bins() -> None:
    assert population_stability([30, 70], [30, 70]) == 0
    assert population_stability([100, 0], [0, 100]) > 0.2
    assert histogram([-5, 0.5, 2, 8], [0, 1]) == [1, 1, 2]
    with pytest.raises(ValueError):
        population_stability([1, 2], [1])


def test_feedback_is_immutable_idempotent_and_windowed(tmp_path) -> None:
    store = AuditStore(tmp_path / "audit.sqlite3")
    assessment = {
        "quality_model_version": "test",
        "action": "review",
        "quality_score": 0.2,
        "schema_valid": True,
    }
    fields = {"company": True, "address": True, "date": True, "total": True}
    with pytest.raises(KeyError):
        store.feedback("missing", fields)
    store.record("older", assessment, {"x": 0.0}, 0.1)
    assert store.feedback("older", fields) == "recorded"
    assert store.feedback("older", fields) == "already_recorded"
    with pytest.raises(ValueError):
        store.feedback("older", {**fields, "total": False})
    store.record("newer", assessment, {"x": 1.0}, 0.2)
    summary = store.summary({"x": {"cuts": [0.5], "counts": [10, 0]}}, window=1, minimum=1)
    assert summary["feedback_count"] == 0
    assert summary["labeled_complete_accuracy"] is None
    assert summary["drift_alerts"] == ["x"]
    assert store.export_feedback()[0]["complete_correct"]
