from __future__ import annotations

from receipt_kie.dataset import ReceiptRecord
from receipt_kie.highres_training import (
    ConsecutiveValidationEarlyStoppingCallback,
    deterministic_subset,
)


def _record(sample_id: str) -> ReceiptRecord:
    return ReceiptRecord(sample_id, "train", "image", "entity", {}, 1, 1)


def test_deterministic_subset_is_stable_and_sorted() -> None:
    records = [_record(str(index)) for index in range(10)]
    first = deterministic_subset(records, 4, seed=42)
    second = deterministic_subset(records, 4, seed=42)
    assert [row.sample_id for row in first] == [row.sample_id for row in second]
    assert [row.sample_id for row in first] == sorted(
        row.sample_id for row in first
    )


def test_early_stopping_requires_two_consecutive_non_improvements() -> None:
    callback = ConsecutiveValidationEarlyStoppingCallback(patience=2)
    state = type("State", (), {"global_step": 1, "epoch": 0.5})()
    control = type("Control", (), {"should_training_stop": False})()
    callback.on_evaluate(None, state, control, {"eval_loss": 1.0})
    callback.on_evaluate(None, state, control, {"eval_loss": 1.1})
    assert not control.should_training_stop
    callback.on_evaluate(None, state, control, {"eval_loss": 1.2})
    assert control.should_training_stop
