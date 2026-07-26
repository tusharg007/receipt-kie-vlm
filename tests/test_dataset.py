from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from receipt_kie.dataset import ReceiptRecord, discover_split, partition_train_validation


def _write_sample(root: Path, sample_id: str = "receipt-1") -> None:
    image_dir = root / "train" / "img"
    entity_dir = root / "train" / "entities"
    image_dir.mkdir(parents=True)
    entity_dir.mkdir(parents=True)
    Image.new("RGB", (20, 30), "white").save(image_dir / f"{sample_id}.jpg")
    (entity_dir / f"{sample_id}.txt").write_text(
        json.dumps(
            {
                "company": "Shop",
                "address": "Main Street",
                "date": "01/02/2024",
                "total": "12.30",
            }
        ),
        encoding="utf-8",
    )


def test_dataset_pairs_image_and_entity(tmp_path: Path) -> None:
    _write_sample(tmp_path)
    records, exclusions, keys = discover_split(tmp_path, "train")
    assert len(records) == 1
    assert exclusions == []
    assert records[0].sample_id == "receipt-1"
    assert records[0].target["total"] == "12.30"
    assert records[0].width == 20
    assert keys["company"] == 1


def test_dataset_reports_missing_pair(tmp_path: Path) -> None:
    image_dir = tmp_path / "train" / "img"
    entity_dir = tmp_path / "train" / "entities"
    image_dir.mkdir(parents=True)
    entity_dir.mkdir(parents=True)
    Image.new("RGB", (4, 4), "white").save(image_dir / "orphan.jpg")
    records, exclusions, _ = discover_split(tmp_path, "train")
    assert records == []
    assert exclusions[0]["reason"] == "missing_entity"


def test_dataset_missing_directories_raise(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        discover_split(tmp_path, "train")


def test_train_validation_partition_is_deterministic_and_disjoint() -> None:
    records = [
        ReceiptRecord(
            sample_id=f"receipt-{index:03d}",
            split="train",
            image_path=Path(f"receipt-{index:03d}.jpg"),
            entity_path=Path(f"receipt-{index:03d}.txt"),
            target={"company": "", "address": "", "date": "", "total": ""},
            width=10,
            height=10,
        )
        for index in range(20)
    ]
    train, validation = partition_train_validation(records, validation_size=4, seed=42)
    repeated_train, repeated_validation = partition_train_validation(
        records, validation_size=4, seed=42
    )
    assert len(train) == 16
    assert len(validation) == 4
    assert {row.sample_id for row in train}.isdisjoint(
        row.sample_id for row in validation
    )
    assert [row.sample_id for row in train] == [
        row.sample_id for row in repeated_train
    ]
    assert [row.sample_id for row in validation] == [
        row.sample_id for row in repeated_validation
    ]
