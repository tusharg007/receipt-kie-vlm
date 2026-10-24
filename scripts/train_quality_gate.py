"""Train a small quality classifier on validation predictions and audit historical test data."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from receipt_kie.metrics import normalize_field  # noqa: E402
from receipt_kie.monitoring import histogram  # noqa: E402
from receipt_kie.prompts import CANONICAL_FIELDS  # noqa: E402
from receipt_kie.quality import (  # noqa: E402
    FEATURE_NAMES,
    FEATURE_VERSION,
    QualityGate,
    output_quality_issues,
    quality_features,
)


def _load(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    ids = [row["sample_id"] for row in rows]
    if not rows or len(ids) != len(set(ids)):
        raise ValueError(f"Empty data or duplicate IDs: {path}")
    return rows


def _labels(rows: list[dict]) -> np.ndarray:
    return np.array(
        [
            int(
                bool(row.get("valid_json"))
                and all(
                    normalize_field(field, (row.get("parsed_prediction") or {}).get(field, ""))
                    == normalize_field(field, row["ground_truth"][field])
                    for field in CANONICAL_FIELDS
                )
            )
            for row in rows
        ]
    )


def _matrix(rows: list[dict]) -> np.ndarray:
    features = [quality_features(row) for row in rows]
    return np.array([[row[key] for key in FEATURE_NAMES] for row in features])


def _threshold(
    y: np.ndarray,
    scores: np.ndarray,
    eligible: np.ndarray,
    target: float,
    minimum: int,
) -> float:
    candidates = []
    for threshold in np.unique(scores):
        chosen = (scores >= threshold) & eligible
        if chosen.sum() >= minimum and y[chosen].mean() >= target:
            candidates.append((int(chosen.sum()), float(threshold)))
    # Explicitly abstain on everything if calibration cannot support the policy.
    return max(candidates)[1] if candidates else 1.000001


def _metrics(y: np.ndarray, scores: np.ndarray, accepted: np.ndarray) -> dict:
    count = int(accepted.sum())
    return {
        "sample_count": len(y),
        "correct_records": int(y.sum()),
        "correct_record_prevalence": float(y.mean()),
        "average_precision": float(average_precision_score(y, scores)) if y.sum() else 0.0,
        "roc_auc": float(roc_auc_score(y, scores)) if len(np.unique(y)) == 2 else None,
        "coverage": float(accepted.mean()),
        "accepted_count": count,
        "accepted_precision": float(y[accepted].mean()) if count else None,
        "false_accept_count": int(((y == 0) & accepted).sum()),
        "correct_record_recall": float(((y == 1) & accepted).sum() / max(y.sum(), 1)),
        "confusion_matrix_reject_accept": confusion_matrix(y, accepted, labels=[0, 1]).tolist(),
    }


def _eligible(rows: list[dict]) -> np.ndarray:
    return np.array([not output_quality_issues(row) for row in rows])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = PROJECT_ROOT / "artifacts/experiments/highres_training_v2"
    parser.add_argument(
        "--validation",
        type=Path,
        default=source
        / "validation_predictions/checkpoint-210/always_repetition_penalty_1p08.jsonl",
    )
    parser.add_argument("--evaluation", type=Path, default=source / "v2_predictions.jsonl")
    parser.add_argument(
        "--audit-report",
        type=Path,
        default=PROJECT_ROOT / "artifacts/experiments/receiptops/data_quality.json",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "artifacts/experiments/quality_gate"
    )
    parser.add_argument("--target-precision", type=float, default=0.8)
    parser.add_argument("--minimum-calibration-accepted", type=int, default=3)
    args = parser.parse_args()
    if not 0 < args.target_precision <= 1 or args.minimum_calibration_accepted < 1:
        parser.error("Invalid precision target or minimum accepted count")
    validation, evaluation = _load(args.validation), _load(args.evaluation)
    audit = json.loads(args.audit_report.read_text(encoding="utf-8"))
    original_validation_count = len(validation)
    validation_exclusions = [
        issue
        for issue in audit["issues"]
        if issue["split"] == "train"
        and issue["sample_id"] in {row["sample_id"] for row in validation}
    ]
    invalid_validation_ids = {issue["sample_id"] for issue in validation_exclusions}
    validation = [row for row in validation if row["sample_id"] not in invalid_validation_ids]
    content_audit = audit["historical_holdout_audit"]
    retained_ids = set(content_audit["retained_ids"])
    evaluation = [row for row in evaluation if row["sample_id"] in retained_ids]
    if {row["sample_id"] for row in evaluation} != retained_ids:
        raise ValueError("Evaluation predictions do not match the audited content-disjoint IDs")
    validation_ids = {row["sample_id"] for row in validation}
    if validation_ids & {row["sample_id"] for row in evaluation}:
        raise ValueError("Quality training and evaluation IDs overlap")
    x_validation, y_validation = _matrix(validation), _labels(validation)
    y_test = _labels(evaluation)
    fit, calibration = train_test_split(
        np.arange(len(validation)),
        test_size=0.25,
        random_state=42,
        stratify=y_validation,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    drift_reference = {}
    for column, field in enumerate(FEATURE_NAMES):
        values = x_validation[fit, column]
        low, high = float(values.min()), float(values.max())
        margin = max(abs(low), abs(high), 1) * 1e-6
        interior = np.quantile(values, [0.25, 0.5, 0.75]).tolist() if high > low else []
        cuts = sorted(set([low - margin, *interior, high + margin]))
        drift_reference[field] = {"cuts": cuts, "counts": histogram(values.tolist(), cuts)}
    report = {
        "experiment": "complete_record_quality_gate",
        "seed": 42,
        "evaluation_scope": (
            "Retrospective content-disjoint subset of published V2 predictions; not a fresh holdout"
        ),
        "content_exclusions": content_audit["exclusions"],
        "original_validation_count": original_validation_count,
        "validation_exclusions": validation_exclusions,
        "training_scope": "Original V2 validation predictions; no extractor retraining",
        "fit_ids": [validation[i]["sample_id"] for i in fit],
        "calibration_ids": [validation[i]["sample_id"] for i in calibration],
        "evaluation_ids": [row["sample_id"] for row in evaluation],
        "target_precision": args.target_precision,
        "minimum_calibration_accepted": args.minimum_calibration_accepted,
        "sources": {
            str(path.relative_to(PROJECT_ROOT))
            if path.is_relative_to(PROJECT_ROOT)
            else str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (args.validation, args.evaluation, args.audit_report)
        },
        "versions": {name: importlib.metadata.version(name) for name in ("numpy", "scikit-learn")},
        "class_distribution": {
            "fit_correct": int(y_validation[fit].sum()),
            "fit_total": len(fit),
            "calibration_correct": int(y_validation[calibration].sum()),
            "calibration_total": len(calibration),
        },
        "models": {},
    }
    for name, weights in (("unweighted_logistic", None), ("balanced_logistic", "balanced")):
        estimator = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight=weights,
                random_state=42,
                max_iter=2000,
                C=1.0,
            ),
        )
        estimator.fit(x_validation[fit], y_validation[fit])
        calibration_scores = estimator.predict_proba(x_validation[calibration])[:, 1]
        eligible_calibration = _eligible([validation[i] for i in calibration])
        threshold = _threshold(
            y_validation[calibration],
            calibration_scores,
            eligible_calibration,
            args.target_precision,
            args.minimum_calibration_accepted,
        )
        scaler, classifier = estimator.steps[0][1], estimator.steps[1][1]
        artifact = {
            "model_version": f"{FEATURE_VERSION}-{name}",
            "deployment_allowed": False,
            "deployment_note": (
                "Training never authorizes automatic acceptance; "
                "candidate policy is evaluated separately"
            ),
            "feature_version": FEATURE_VERSION,
            "feature_names": list(FEATURE_NAMES),
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "coefficients": classifier.coef_[0].tolist(),
            "intercept": float(classifier.intercept_[0]),
            "threshold": threshold,
            "fit_ids": report["fit_ids"],
            "calibration_ids": report["calibration_ids"],
            "evaluation_scope": report["evaluation_scope"],
            "extractor_configuration": validation[0].get("processor_configuration"),
            "drift_reference": drift_reference,
        }
        model_hash = hashlib.sha256(json.dumps(artifact, sort_keys=True).encode()).hexdigest()[:12]
        artifact["model_version"] += "-" + model_hash
        gate = QualityGate(artifact)
        test_scores = np.array([gate.score(row) for row in evaluation])
        if not np.allclose(
            test_scores, estimator.predict_proba(_matrix(evaluation))[:, 1], atol=1e-12
        ):
            raise RuntimeError("Exported model does not match scikit-learn inference")
        assessments = [gate.assess(row, enforce_deployment=False) for row in evaluation]
        accepted = np.array([item["action"] == "accept_candidate" for item in assessments])
        report["models"][name] = {
            "model_version": artifact["model_version"],
            "threshold": threshold,
            "calibration": _metrics(
                y_validation[calibration],
                calibration_scores,
                (calibration_scores >= threshold) & eligible_calibration,
            ),
            "evaluation": _metrics(y_test, test_scores, accepted),
            "deployed_review_all_policy": _metrics(
                y_test, test_scores, np.zeros(len(y_test), dtype=bool)
            ),
            "coefficients": dict(zip(FEATURE_NAMES, classifier.coef_[0].tolist(), strict=True)),
        }
        if name == "balanced_logistic":
            (args.output_dir / "quality_gate.json").write_text(
                json.dumps(artifact, indent=2) + "\n",
                encoding="utf-8",
            )
            score_rows = [
                {"sample_id": row["sample_id"], "correct": int(label), **assessment}
                for row, label, assessment in zip(evaluation, y_test, assessments, strict=True)
            ]
            (args.output_dir / "evaluation_scores.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in score_rows),
                encoding="utf-8",
            )
            report["quality_slices"] = {}
            for key, predicate in (
                (
                    "company_contains_digits",
                    lambda row: quality_features(row)["company_digit_ratio"] > 0.1,
                ),
                (
                    "company_address_overlap",
                    lambda row: quality_features(row)["company_address_overlap"] > 0.25,
                ),
                (
                    "long_address",
                    lambda row: quality_features(row)["address_length_log"] > np.log1p(100),
                ),
            ):
                for flag in (False, True):
                    mask = np.array([bool(predicate(row)) == flag for row in evaluation])
                    if mask.any():
                        report["quality_slices"][f"{key}={flag}"] = _metrics(
                            y_test[mask],
                            test_scores[mask],
                            accepted[mask],
                        )
            eligible_test = _eligible(evaluation)
            report["risk_coverage_curve"] = [
                {
                    "threshold": float(threshold),
                    **_metrics(
                        y_test,
                        test_scores,
                        eligible_test & (test_scores >= threshold),
                    ),
                }
                for threshold in np.linspace(0, 1, 21)
            ]
    report["baselines"] = {
        "accept_all": _metrics(
            y_test, np.full(len(y_test), y_validation[fit].mean()), np.ones(len(y_test), dtype=bool)
        ),
        "review_all": _metrics(y_test, np.zeros(len(y_test)), np.zeros(len(y_test), dtype=bool)),
    }
    report["limitations"] = [
        "Quality score is not calibrated; empirical calibration precision is not a guarantee",
        "Small validation split; precision may shift across merchants and image conditions",
        "Observable quality slices are not demographic fairness certification",
        "No claims of receipt correction, production scale, or fresh held-out accuracy",
        "Candidate threshold evaluation is diagnostic; "
        "the shipped artifact disallows automatic acceptance",
    ]
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps({key: value["evaluation"] for key, value in report["models"].items()}, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
