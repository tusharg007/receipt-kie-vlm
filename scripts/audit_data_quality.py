"""Audit local receipt images, strict entity schemas, duplicates and cross-split content leakage."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from receipt_kie.structured import validate_receipt  # noqa: E402


def audit(dataset: Path) -> dict:
    report = {"dataset": str(dataset), "splits": {}, "duplicate_contents": [], "issues": []}
    hashes = defaultdict(list)
    for split in ("train", "test"):
        image_root, labels_root = dataset / split / "img", dataset / split / "entities"
        if not image_root.is_dir() or not labels_root.is_dir():
            raise FileNotFoundError(f"Dataset split is missing: {split}")
        images = {
            path.stem: path
            for path in sorted(image_root.iterdir())
            if path.suffix.lower() in (".jpg", ".jpeg", ".png")
        }
        labels = {path.stem: path for path in sorted(labels_root.glob("*.txt"))}
        ratios, areas = [], []
        schema_valid = 0
        for sample_id in sorted(set(images) | set(labels)):
            image_path, label_path = images.get(sample_id), labels.get(sample_id)
            if image_path is None or label_path is None:
                report["issues"].append(
                    {
                        "split": split,
                        "sample_id": sample_id,
                        "issue": "unpaired_image_or_annotation",
                    }
                )
                continue
            hashes[hashlib.sha256(image_path.read_bytes()).hexdigest()].append(
                f"{split}/{sample_id}"
            )
            try:
                with Image.open(image_path) as image:
                    ratios.append(image.height / image.width)
                    areas.append(image.width * image.height)
                    image.verify()
            except (OSError, ValueError) as exc:
                report["issues"].append(
                    {"split": split, "sample_id": sample_id, "issue": type(exc).__name__}
                )
            try:
                annotation = json.loads(label_path.read_text(encoding="utf-8-sig"))
                errors = validate_receipt(annotation)
                schema_valid += not errors
                if errors:
                    report["issues"].append(
                        {
                            "split": split,
                            "sample_id": sample_id,
                            "issue": "annotation_schema",
                            "details": errors,
                        }
                    )
            except (OSError, ValueError) as exc:
                report["issues"].append(
                    {"split": split, "sample_id": sample_id, "issue": type(exc).__name__}
                )
        report["splits"][split] = {
            "image_count": len(images),
            "annotation_count": len(labels),
            "strict_annotation_valid_count": schema_valid,
            "median_height_width_ratio": statistics.median(ratios) if ratios else None,
            "median_pixels": statistics.median(areas) if areas else None,
        }
    duplicates = [
        {"sha256": digest, "members": members}
        for digest, members in hashes.items()
        if len(members) > 1
    ]
    report["duplicate_contents"] = duplicates
    report["cross_split_duplicate_contents"] = [
        item for item in duplicates if len({member.split("/")[0] for member in item["members"]}) > 1
    ]
    report["notes"] = (
        "Exact file-hash leakage audit only; perceptual near-duplicates "
        "and pretraining contamination "
        "are not ruled out. An annotation issue does not silently modify training data."
    )
    holdout_path = ROOT / "artifacts/experiments/highres_training_v2/final_holdout_ids.json"
    if holdout_path.is_file():
        holdout = set(json.loads(holdout_path.read_text(encoding="utf-8")))
        exclusions = []
        for group in duplicates:
            candidates = sorted(
                member[5:]
                for member in group["members"]
                if member.startswith("test/") and member[5:] in holdout
            )
            if not candidates:
                continue
            if any(member.startswith("train/") for member in group["members"]):
                reason, removed = "duplicate_of_official_train_image", candidates
            elif any(
                member.startswith("test/") and member[5:] not in holdout
                for member in group["members"]
            ):
                reason, removed = "duplicate_of_previously_evaluated_test_image", candidates
            else:
                reason, removed = "repeated_content_within_historical_holdout", candidates[1:]
            exclusions.extend(
                {
                    "sample_id": sample_id,
                    "reason": reason,
                    "members": group["members"],
                    "sha256": group["sha256"],
                }
                for sample_id in removed
            )
        excluded_ids = {item["sample_id"] for item in exclusions}
        report["historical_holdout_audit"] = {
            "original_count": len(holdout),
            "exclusions": exclusions,
            "retained_ids": sorted(holdout - excluded_ids),
            "scope": "Content-disjoint retrospective subset, not a new unseen test set",
        }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data/raw/sroie/SROIE2019")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts/experiments/receiptops/data_quality.json"
    )
    args = parser.parse_args()
    report = audit(args.dataset_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "splits": report["splits"],
                "issues": len(report["issues"]),
                "cross_split_duplicates": len(report["cross_split_duplicate_contents"]),
            },
            indent=2,
        )
    )
    return 1 if report["cross_split_duplicate_contents"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
