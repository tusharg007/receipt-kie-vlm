"""Recompute historical model comparisons after exact image-content leakage exclusions."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from receipt_kie.metrics import evaluate_predictions, normalize_field  # noqa: E402
from receipt_kie.prompts import CANONICAL_FIELDS  # noqa: E402


def main() -> int:
    audit_path = ROOT / "artifacts/experiments/receiptops/data_quality.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    selection = audit["historical_holdout_audit"]
    selected = set(selection["retained_ids"])
    rows_by_variant = {}
    metrics = {}
    source_hashes = {}
    for variant in ("base", "v1", "v2"):
        source = ROOT / f"artifacts/experiments/highres_training_v2/{variant}_predictions.jsonl"
        rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
        rows = sorted(
            (row for row in rows if row["sample_id"] in selected), key=lambda row: row["sample_id"]
        )
        if {row["sample_id"] for row in rows} != selected:
            raise ValueError(f"Missing content-disjoint predictions: {variant}")
        rows_by_variant[variant] = rows
        metrics[variant] = evaluate_predictions(rows)
        source_hashes[variant] = hashlib.sha256(source.read_bytes()).hexdigest()
    field_matches = {}
    for variant, rows in rows_by_variant.items():
        field_matches[variant] = np.array(
            [
                [
                    int(
                        bool(row["valid_json"])
                        and normalize_field(
                            field, (row.get("parsed_prediction") or {}).get(field, "")
                        )
                        == normalize_field(field, row["ground_truth"][field])
                    )
                    for field in CANONICAL_FIELDS
                ]
                for row in rows
            ]
        )
    rng = np.random.default_rng(42)
    intervals = {}
    for metric, per_record in (
        (
            "macro_exact_delta_v2_minus_v1",
            {key: values.mean(axis=1) for key, values in field_matches.items()},
        ),
        (
            "complete_record_delta_v2_minus_v1",
            {key: values.all(axis=1).astype(float) for key, values in field_matches.items()},
        ),
    ):
        differences = per_record["v2"] - per_record["v1"]
        draws = rng.choice(differences, size=(2000, len(differences)), replace=True).mean(axis=1)
        intervals[metric] = {
            "estimate": float(differences.mean()),
            "paired_bootstrap_95_interval": np.quantile(draws, [0.025, 0.975]).tolist(),
        }
    report = {
        "scope": selection["scope"],
        "selection": selection,
        "sample_count": len(selected),
        "source_sha256": source_hashes,
        "metrics": metrics,
        "paired_intervals": intervals,
        "notes": [
            "Historical adapters and prediction files are unchanged",
            "Exclusions fixed by content hashes, not correctness or model scores",
            "Exact-hash filtering does not rule out perceptual duplicates "
            "or pretraining contamination",
        ],
    }
    output = ROOT / "artifacts/experiments/receiptops/content_disjoint_results.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: {
                    metric: value[metric]
                    for metric in (
                        "sample_count",
                        "company_accuracy",
                        "address_similarity",
                        "date_accuracy",
                        "total_accuracy",
                        "macro_normalized_exact_match",
                        "complete_record_normalized_exact_match",
                    )
                }
                for key, value in metrics.items()
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
