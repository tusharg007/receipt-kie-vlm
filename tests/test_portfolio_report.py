from __future__ import annotations

import importlib.util
from pathlib import Path


def test_outcome_counts_distinguish_equal_accuracy_from_identical_outputs(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "portfolio", root / "scripts/build_portfolio_report.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    truth = {"company": "A", "address": "1 Rd", "date": "01/01/2024", "total": "1.00"}

    def row(matches):
        return {
            "valid_json": True,
            "ground_truth": truth,
            "parsed_prediction": {
                field: truth[field] if matches[index] else "wrong"
                for index, field in enumerate(module.CANONICAL_FIELDS)
            },
        }

    before = dict(
        zip("ABCD", map(row, ([1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0])), strict=True)
    )
    after = dict(
        zip("ABCD", map(row, ([1, 1, 1, 1], [0, 0, 1, 1], [1, 1, 1, 0], [0, 0, 0, 0])), strict=True)
    )
    assert module.summarize_pair(before, after, list("ABCD")) == {
        "sample_count": 4,
        "improved": 1,
        "equal_count": 2,
        "regressed": 1,
        "identical_fields": 1,
        "complete_record_recovered": 1,
        "complete_record_lost": 1,
        "both_complete": 0,
        "both_incomplete": 2,
    }
    assert not module.eligible("unchanged_fields", before["B"], after["B"])
    assert module.eligible("regression", before["C"], after["C"])
