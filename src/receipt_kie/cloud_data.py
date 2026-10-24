"""Strict, content-deduplicated and merchant-grouped cloud-training partitions."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit

from receipt_kie.metrics import normalize_field, normalize_text
from receipt_kie.prompts import CANONICAL_FIELDS
from receipt_kie.structured import validate_receipt


def prepare_cloud_data(root: Path, evaluation_ids: list[str], seed: int = 42) -> dict:
    """Only official-train examples enter fitting or checkpoint selection."""
    if len(evaluation_ids) != len(set(evaluation_ids)):
        raise ValueError("Evaluation IDs must be unique")
    issues, rows = [], {"train": [], "test": []}
    all_train_hashes = set()
    for split in rows:
        image_dir, label_dir = root / split / "img", root / split / "entities"
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(f"Missing SROIE directories for {split}")
        for path in sorted(image_dir.iterdir()):
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if split == "train":
                all_train_hashes.add(digest)
            try:
                target = json.loads((label_dir / f"{path.stem}.txt").read_text("utf-8-sig"))
                errors = validate_receipt(target)
                if errors or any(not target[field].strip() for field in CANONICAL_FIELDS):
                    raise ValueError("Incomplete or invalid four-field annotation")
            except (OSError, ValueError, TypeError, KeyError):
                issues.append({"split": split, "sample_id": path.stem, "reason": "annotation"})
                continue
            rows[split].append(
                {
                    "sample_id": path.stem,
                    "image_path": str(path),
                    "target": target,
                    "sha256": digest,
                    "merchant_group": normalize_text(target["company"]),
                }
            )
    grouped = defaultdict(list)
    for row in rows["train"]:
        grouped[row["sha256"]].append(row)
    unique_train = []
    for members in grouped.values():
        labels = {
            tuple(normalize_field(field, row["target"][field]) for field in CANONICAL_FIELDS)
            for row in members
        }
        if len(labels) != 1:
            issues.extend(
                {
                    "split": "train",
                    "sample_id": row["sample_id"],
                    "reason": "conflicting_duplicate_annotations",
                }
                for row in members
            )
            continue
        unique_train.append(members[0])
        issues.extend(
            {"split": "train", "sample_id": row["sample_id"], "reason": "duplicate_content"}
            for row in members[1:]
        )
    unique_train.sort(key=lambda row: row["sample_id"])
    groups = [row["merchant_group"] for row in unique_train]
    if len(set(groups)) < 2:
        raise ValueError("At least two merchant groups are required")
    fit, validation = next(
        GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed).split(
            unique_train, groups=groups
        )
    )
    training = [unique_train[index] for index in fit]
    validation_rows = [unique_train[index] for index in validation]
    test_by_id = {row["sample_id"]: row for row in rows["test"]}
    missing = set(evaluation_ids) - test_by_id.keys()
    if missing:
        raise ValueError(f"Evaluation examples missing or invalid: {sorted(missing)}")
    evaluation = [test_by_id[sample_id] for sample_id in evaluation_ids]
    hashes = [row["sha256"] for row in evaluation]
    if len(hashes) != len(set(hashes)) or set(hashes) & all_train_hashes:
        raise ValueError("Evaluation content repeats or overlaps official-train images")
    assert {row["merchant_group"] for row in training}.isdisjoint(
        row["merchant_group"] for row in validation_rows
    )
    return {
        "seed": seed,
        "train": training,
        "validation": validation_rows,
        "evaluation": evaluation,
        "quarantined_or_deduplicated": issues,
        "notes": (
            "Validation merchant grouping uses normalized annotated names; spelling variants "
            "and near-duplicate images may remain. Evaluation is the historical retrospective "
            "subset, not fresh data. Test labels do not select training hyperparameters."
        ),
    }
