"""Inference-only features and an inspectable complete-record quality classifier.

The score estimates whether all four extracted fields match the dataset's
normalization convention. It is not a calibrated probability or a receipt audit.
Ground truth, image IDs, and merchant identities are never input features.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from receipt_kie.prompts import CANONICAL_FIELDS
from receipt_kie.structured import review_issues, strict_receipt, structured_metadata

FEATURE_VERSION = "receipt-quality-v1"
FEATURE_NAMES = (
    "schema_valid",
    "missing_count",
    "date_flag",
    "amount_flag",
    "company_length_log",
    "address_length_log",
    "company_digit_ratio",
    "address_digit_ratio",
    "company_address_overlap",
    "company_address_length_ratio",
    "repeated_word_ratio",
    "token_limit_hit",
)


def quality_features(row: dict[str, Any]) -> dict[str, float]:
    """Read model outputs only; deliberately ignore any ground_truth in the row."""
    raw = row.get("raw_output", "")
    raw = raw if isinstance(raw, str) else ""
    value, _ = strict_receipt(raw)
    # Features must reflect strict raw output, not a repaired/coerced projection.
    value = value or dict.fromkeys(CANONICAL_FIELDS, "")
    issues = review_issues(value)
    company, address = value["company"], value["address"]
    company_words = set(re.findall(r"\w+", company.casefold()))
    address_words = set(re.findall(r"\w+", address.casefold()))
    words = re.findall(r"\w+", " ".join(value.values()).casefold())
    return dict(
        zip(
            FEATURE_NAMES,
            (
                float(strict_receipt(raw)[0] is not None),
                float(sum(not value[field].strip() for field in CANONICAL_FIELDS)),
                float("date:unrecognized_or_invalid" in issues),
                float("total:unrecognized_amount" in issues),
                math.log1p(len(company)),
                math.log1p(len(address)),
                sum(char.isdigit() for char in company) / max(len(company), 1),
                sum(char.isdigit() for char in address) / max(len(address), 1),
                len(company_words & address_words) / max(len(company_words | address_words), 1),
                len(company) / max(len(address), 1),
                1 - len(set(words)) / max(len(words), 1) if words else 0,
                float(bool(row.get("generation_limit_hit"))),
            ),
            strict=True,
        )
    )


def output_quality_issues(row: dict[str, Any]) -> list[str]:
    metadata = structured_metadata(
        row.get("raw_output", ""),
        limit_hit=bool(row.get("generation_limit_hit")),
    )
    reasons = list(metadata["review_issues"])
    value, _ = strict_receipt(row.get("raw_output", ""))
    if (
        value
        and value["company"].strip()
        and (
            re.sub(r"\W", "", value["company"].casefold())
            == re.sub(r"\W", "", value["address"].casefold())
        )
    ):
        reasons.append("quality:company_matches_address")
    return reasons


class QualityGate:
    """Load numeric parameters from JSON; never unpickle model artifacts."""

    def __init__(self, artifact: dict[str, Any]) -> None:
        if artifact.get("feature_version") != FEATURE_VERSION or artifact.get(
            "feature_names"
        ) != list(FEATURE_NAMES):
            raise ValueError("Quality feature contract does not match the trained artifact")
        self.artifact = artifact
        self.threshold = float(artifact["threshold"])
        for key in ("mean", "scale", "coefficients"):
            values = artifact[key]
            if len(values) != len(FEATURE_NAMES) or not all(math.isfinite(v) for v in values):
                raise ValueError(f"Invalid quality model parameters: {key}")
        if (
            not math.isfinite(self.threshold)
            or not math.isfinite(artifact["intercept"])
            or any(value <= 0 for value in artifact["scale"])
        ):
            raise ValueError("Invalid quality model intercept, scale, or threshold")

    @classmethod
    def load(cls, path: str | Path) -> QualityGate:
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def score(self, row: dict[str, Any]) -> float:
        features = quality_features(row)
        logit = float(self.artifact["intercept"])
        for field, mean, scale, coefficient in zip(
            FEATURE_NAMES,
            self.artifact["mean"],
            self.artifact["scale"],
            self.artifact["coefficients"],
            strict=True,
        ):
            logit += ((features[field] - mean) / scale) * coefficient
        return 1 / (1 + math.exp(-max(-700, min(700, logit))))

    def assess(self, row: dict[str, Any], *, enforce_deployment: bool = True) -> dict[str, Any]:
        metadata = structured_metadata(
            row.get("raw_output", ""),
            limit_hit=bool(row.get("generation_limit_hit")),
        )
        score = self.score(row)
        reasons = output_quality_issues(row)
        expected = self.artifact.get("extractor_configuration")
        if expected and row.get("processor_configuration") != expected:
            reasons.append("quality:extractor_configuration_mismatch")
        if score < self.threshold:
            reasons.append("quality:below_validation_threshold")
        if enforce_deployment and not self.artifact.get("deployment_allowed", False):
            reasons.append("quality:deployment_not_approved")
        return {
            "quality_score": score,
            "quality_model_version": self.artifact["model_version"],
            "threshold": self.threshold,
            "action": "review" if reasons else "accept_candidate",
            "review_reasons": reasons,
            "schema_valid": metadata["schema_valid"],
            "deployment_allowed": bool(self.artifact.get("deployment_allowed", False)),
        }
