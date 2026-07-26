from __future__ import annotations

from receipt_kie.v2_final import success_gate


def _metrics(
    macro: float,
    complete: float,
    address: float,
    valid: float,
) -> dict:
    return {
        "macro_normalized_exact_match": macro,
        "complete_record_normalized_exact_match": complete,
        "address_similarity": address,
        "valid_json_rate": valid,
    }


def test_success_gate_requires_improvement_and_validity_floor() -> None:
    v1 = _metrics(0.40, 0.10, 0.60, 1.0)
    assert success_gate(v1, _metrics(0.45, 0.10, 0.60, 0.98))["passed"]
    assert not success_gate(v1, _metrics(0.46, 0.10, 0.60, 0.97))["passed"]
    assert not success_gate(v1, _metrics(0.44, 0.12, 0.64, 1.0))["passed"]
