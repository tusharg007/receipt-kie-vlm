"""Paired development-only pilot: saved 1024 px V2 vs actual CPU OCR + local Bolt."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from receipt_kie.metrics import evaluate_predictions  # noqa: E402
from receipt_kie.ocr_bolt import BoltOCRPredictor  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8081")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "artifacts/experiments/ocr_bolt_pilot"
    )
    args = parser.parse_args()
    source = ROOT / "artifacts/experiments/structured_decoding/pilot10_1024/unconstrained.jsonl"
    baseline = sorted(
        [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()],
        key=lambda row: row["sample_id"],
    )
    if not 1 <= args.limit <= len(baseline):
        parser.error("--limit must be between 1 and the saved baseline sample count")
    baseline = baseline[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "scope": (
            "First sorted original-validation pilot IDs; "
            "development comparison, not held-out evidence"
        ),
        "sample_ids": [row["sample_id"] for row in baseline],
        "baseline_source": str(source.relative_to(ROOT)),
        "status": "started",
        "notes": (
            "Same original images. Backend-specific preprocessing differs. "
            "No reference annotations enter OCR or Bolt context."
        ),
    }
    (args.output_dir / "comparison.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    predictor = BoltOCRPredictor(args.base_url)
    rows = []
    try:
        for previous in baseline:
            prediction = predictor.predict(ROOT / previous["image_path"])
            rows.append(
                {
                    "sample_id": previous["sample_id"],
                    "image_path": previous["image_path"],
                    "ground_truth": previous["ground_truth"],
                    **prediction,
                }
            )
            (args.output_dir / "ocr_bolt_predictions.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            print(
                f"OCR + Bolt: {len(rows)}/{len(baseline)}, "
                f"schema_valid={prediction['schema_valid']}",
                flush=True,
            )
    finally:
        predictor.close()
    report.update(
        status="complete",
        original_v2=evaluate_predictions(baseline),
        ocr_bolt=evaluate_predictions(rows),
    )
    (args.output_dir / "comparison.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
