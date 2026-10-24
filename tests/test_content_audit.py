from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from PIL import Image


def test_content_audit_catches_leakage_despite_different_ids(tmp_path) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts/audit_data_quality.py"
    spec = importlib.util.spec_from_file_location("audit_data_quality", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = tmp_path
    holdout = tmp_path / "artifacts/experiments/highres_training_v2/final_holdout_ids.json"
    holdout.parent.mkdir(parents=True)
    holdout.write_text(json.dumps(["A", "B", "C", "D"]))
    dataset = tmp_path / "dataset"
    for split, name, color in (
        ("train", "T", "red"),
        ("test", "A", "red"),
        ("test", "B", "blue"),
        ("test", "C", "blue"),
        ("test", "D", "green"),
        ("test", "P", "green"),
    ):
        images, entities = dataset / split / "img", dataset / split / "entities"
        images.mkdir(parents=True, exist_ok=True)
        entities.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (4, 4), color).save(images / f"{name}.png")
        annotation = {
            "company": "ACME",
            "address": "1 Road",
            "date": "01/02/2024",
            "total": "12.50",
        }
        (entities / f"{name}.txt").write_text(json.dumps(annotation))
    report = module.audit(dataset)
    assert len(report["cross_split_duplicate_contents"]) == 1
    selected = report["historical_holdout_audit"]
    assert selected["retained_ids"] == ["B"]
    assert {item["sample_id"] for item in selected["exclusions"]} == {"A", "C", "D"}
    assert not report["issues"]
