from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from receipt_kie.dataset import discover_split


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
