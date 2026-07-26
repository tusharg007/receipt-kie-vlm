from __future__ import annotations

from receipt_kie.v2_evaluation import (
    build_adaptive_retry_rows,
    retry_reasons,
)


def _row(sample_id: str, *, valid: bool, limit: bool, repeat: bool) -> dict:
    return {
        "sample_id": sample_id,
        "raw_output": "{}",
        "valid_json": valid,
        "generation_limit_hit": limit,
        "repetition_failure": repeat,
        "latency_seconds": 1.0,
        "peak_gpu_memory_mib": 2.0,
        "visual_tile_count": 3,
    }


def test_retry_reasons_cover_all_fixed_triggers() -> None:
    row = _row("x", valid=False, limit=True, repeat=True)
    assert retry_reasons(row) == [
        "invalid_json",
        "generation_limit",
        "repetition_failure",
    ]


def test_adaptive_retry_uses_penalty_only_for_triggered_rows() -> None:
    initial = [
        _row("ok", valid=True, limit=False, repeat=False),
        _row("retry", valid=False, limit=True, repeat=False),
    ]
    penalty = [
        _row("ok", valid=True, limit=False, repeat=False),
        _row("retry", valid=True, limit=False, repeat=False),
    ]
    rows = build_adaptive_retry_rows(initial, penalty)
    assert not rows[0]["retry_performed"]
    assert rows[0]["latency_seconds"] == 1.0
    assert rows[1]["retry_performed"]
    assert rows[1]["latency_seconds"] == 2.0
    assert rows[1]["visual_tile_count"] == 6
