"""SROIE image/entity discovery, validation, canonicalization, and auditing."""

from __future__ import annotations

import json
import logging
import random
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from receipt_kie.prompts import CANONICAL_FIELDS, canonical_json, canonical_target
from receipt_kie.utils import project_path, repository_relative, write_json

LOGGER = logging.getLogger(__name__)
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


@dataclass(frozen=True)
class ReceiptRecord:
    """One validated SROIE receipt and its structured entity target."""

    sample_id: str
    split: str
    image_path: str
    entity_path: str
    target: dict[str, str]
    width: int
    height: int

    @property
    def target_json(self) -> str:
        return canonical_json(self.target)

    def trainer_row(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "split": self.split,
            "image_path": self.image_path,
            "entity_path": self.entity_path,
            "target": self.target,
            "target_json": self.target_json,
        }


def _read_entity(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    for encoding in ("utf-8-sig", "utf-8", "cp1252"):
        try:
            text = path.read_text(encoding=encoding)
            payload = json.loads(text)
            if not isinstance(payload, dict):
                return None, "annotation_root_not_object"
            return payload, None
        except UnicodeDecodeError:
            continue
        except json.JSONDecodeError as exc:
            return None, f"invalid_json:{exc.msg}"
        except OSError as exc:
            return None, f"read_error:{exc}"
    return None, "unsupported_encoding"


def discover_split(
    dataset_root: str | Path,
    split: str,
) -> tuple[list[ReceiptRecord], list[dict[str, str]], Counter[str]]:
    """Pair and validate all images/entity annotations in one split."""
    root = project_path(dataset_root)
    image_dir = root / split / "img"
    entity_dir = root / split / "entities"
    if not image_dir.is_dir() or not entity_dir.is_dir():
        raise FileNotFoundError(
            f"Expected SROIE directories are missing for split={split}: "
            f"{image_dir} and {entity_dir}"
        )
    images = {
        path.stem: path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }
    entities = {path.stem: path for path in entity_dir.glob("*.txt") if path.is_file()}
    records: list[ReceiptRecord] = []
    exclusions: list[dict[str, str]] = []
    key_frequency: Counter[str] = Counter()
    for sample_id in sorted(set(images) | set(entities)):
        image_path = images.get(sample_id)
        entity_path = entities.get(sample_id)
        if image_path is None:
            exclusions.append({"sample_id": sample_id, "split": split, "reason": "missing_image"})
            continue
        if entity_path is None:
            exclusions.append({"sample_id": sample_id, "split": split, "reason": "missing_entity"})
            continue
        payload, error = _read_entity(entity_path)
        if payload is None:
            exclusions.append(
                {"sample_id": sample_id, "split": split, "reason": error or "invalid_entity"}
            )
            continue
        key_frequency.update(str(key).strip().lower() for key in payload)
        try:
            with Image.open(image_path) as image:
                image.verify()
            with Image.open(image_path) as image:
                width, height = image.size
        except (OSError, UnidentifiedImageError) as exc:
            exclusions.append(
                {
                    "sample_id": sample_id,
                    "split": split,
                    "reason": f"invalid_image:{exc}",
                }
            )
            continue
        records.append(
            ReceiptRecord(
                sample_id=sample_id,
                split=split,
                image_path=str(image_path.resolve()),
                entity_path=str(entity_path.resolve()),
                target=canonical_target(payload),
                width=width,
                height=height,
            )
        )
    LOGGER.info(
        "Discovered split=%s records=%d exclusions=%d", split, len(records), len(exclusions)
    )
    return records, exclusions, key_frequency


def load_records(
    dataset_root: str | Path,
    split: str,
    limit: int | None = None,
    seed: int = 42,
) -> list[ReceiptRecord]:
    """Load a deterministic full split or reproducible subset."""
    records, exclusions, _ = discover_split(dataset_root, split)
    if exclusions:
        LOGGER.warning("Split %s contains %d excluded samples", split, len(exclusions))
    if limit is None or limit >= len(records):
        return records
    rng = random.Random(seed)
    selected = rng.sample(records, limit)
    return sorted(selected, key=lambda row: row.sample_id)


def partition_train_validation(
    records: list[ReceiptRecord],
    validation_size: int,
    seed: int = 42,
    train_limit: int | None = None,
) -> tuple[list[ReceiptRecord], list[ReceiptRecord]]:
    """Create deterministic, disjoint train/validation subsets from one source split."""
    ordered = sorted(records, key=lambda row: row.sample_id)
    if validation_size <= 0:
        raise ValueError("validation_size must be positive")
    if validation_size >= len(ordered):
        raise ValueError(
            f"validation_size={validation_size} must be smaller than "
            f"the {len(ordered)} available records"
        )
    rng = random.Random(seed)
    validation_ids = {
        record.sample_id for record in rng.sample(ordered, validation_size)
    }
    validation = [record for record in ordered if record.sample_id in validation_ids]
    training = [record for record in ordered if record.sample_id not in validation_ids]
    if train_limit is not None and train_limit < len(training):
        if train_limit <= 0:
            raise ValueError("train_limit must be positive when provided")
        training = random.Random(seed + 1).sample(training, train_limit)
    return (
        sorted(training, key=lambda row: row.sample_id),
        sorted(validation, key=lambda row: row.sample_id),
    )


def build_audit(dataset_root: str | Path) -> dict[str, Any]:
    """Audit pairing, JSON validity, field completeness, and image dimensions."""
    all_records: list[ReceiptRecord] = []
    all_exclusions: list[dict[str, str]] = []
    key_frequency: Counter[str] = Counter()
    split_counts: dict[str, int] = {}
    for split in ("train", "test"):
        records, exclusions, keys = discover_split(dataset_root, split)
        all_records.extend(records)
        all_exclusions.extend(exclusions)
        key_frequency.update(keys)
        split_counts[split] = len(records)
    widths = [record.width for record in all_records]
    heights = [record.height for record in all_records]
    empty_counts = {
        field: sum(not record.target[field].strip() for record in all_records)
        for field in CANONICAL_FIELDS
    }
    field_frequency = {
        field: sum(bool(record.target[field].strip()) for record in all_records)
        for field in CANONICAL_FIELDS
    }
    exclusion_reasons = Counter(row["reason"].split(":", 1)[0] for row in all_exclusions)
    return {
        "dataset_root": repository_relative(dataset_root),
        "valid_pairs": len(all_records),
        "image_files": len(all_records)
        + sum(row["reason"] == "missing_entity" for row in all_exclusions),
        "annotation_files": len(all_records)
        + sum(row["reason"] == "missing_image" for row in all_exclusions),
        "split_counts": split_counts,
        "excluded_count": len(all_exclusions),
        "exclusion_reasons": dict(sorted(exclusion_reasons.items())),
        "exclusions": all_exclusions,
        "actual_annotation_key_frequency": dict(sorted(key_frequency.items())),
        "canonical_field_frequency": field_frequency,
        "empty_value_counts": empty_counts,
        "image_dimensions": {
            "width": _dimension_stats(widths),
            "height": _dimension_stats(heights),
        },
        "schema": list(CANONICAL_FIELDS),
        "canonicalization": {
            "key_order": list(CANONICAL_FIELDS),
            "missing_value": "",
            "value_type": "string",
            "encoding": "UTF-8",
        },
    }


def _dimension_stats(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"min": 0, "max": 0, "mean": 0.0, "median": 0.0}
    return {
        "min": min(values),
        "max": max(values),
        "mean": round(statistics.mean(values), 2),
        "median": round(statistics.median(values), 2),
    }


def write_audit(dataset_root: str | Path) -> dict[str, Any]:
    """Generate both machine-readable and Markdown dataset audit reports."""
    audit = build_audit(dataset_root)
    write_json("artifacts/reports/dataset_audit.json", audit)
    report_path = project_path("artifacts/reports/dataset_audit.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_audit_markdown(audit), encoding="utf-8")
    return audit


def _audit_markdown(audit: dict[str, Any]) -> str:
    rows = [
        "# SROIE Dataset Audit",
        "",
        f"- Valid image/entity pairs: **{audit['valid_pairs']}**",
        f"- Train pairs: **{audit['split_counts']['train']}**",
        f"- Test pairs: **{audit['split_counts']['test']}**",
        f"- Excluded samples: **{audit['excluded_count']}**",
        "",
        "## Canonical field coverage",
        "",
        "| Field | Non-empty | Empty |",
        "|---|---:|---:|",
    ]
    for field in CANONICAL_FIELDS:
        rows.append(
            f"| {field} | {audit['canonical_field_frequency'][field]} | "
            f"{audit['empty_value_counts'][field]} |"
        )
    rows.extend(
        [
            "",
            "## Image dimensions",
            "",
            f"- Width: {audit['image_dimensions']['width']}",
            f"- Height: {audit['image_dimensions']['height']}",
            "",
            "## Exclusions",
            "",
            f"`{audit['exclusion_reasons']}`",
            "",
            "Targets preserve strings and serialize in the stable order "
            "`company`, `address`, `date`, `total`. Missing values use an empty string.",
            "",
        ]
    )
    return "\n".join(rows)
