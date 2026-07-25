from __future__ import annotations

import hashlib
import json

from PIL import Image

from receipt_kie.config import load_config
from receipt_kie.utils import PROJECT_ROOT, repository_relative


def test_committed_adapter_matches_training_metadata() -> None:
    adapter_dir = PROJECT_ROOT / "models" / "receipt-kie-lora"
    adapter_path = adapter_dir / "adapter_model.safetensors"
    metadata = json.loads((adapter_dir / "training_metadata.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256(adapter_path.read_bytes()).hexdigest()
    assert adapter_path.stat().st_size == metadata["adapter_size_bytes"]
    assert digest == metadata["adapter_sha256"]
    assert not adapter_path.read_bytes()[:128].startswith(
        b"version https://git-lfs.github.com/spec/v1"
    )
    assert load_config("configs/evaluate.yaml")["paths"]["adapter_path"] == (
        "models/receipt-kie-lora"
    )


def test_synthetic_demo_is_dataset_independent() -> None:
    image_path = PROJECT_ROOT / "assets" / "demo" / "synthetic_receipt.png"
    expected_path = PROJECT_ROOT / "assets" / "demo" / "expected_output.json"
    with Image.open(image_path) as image:
        image.verify()
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    assert expected == {
        "company": "DEMO MART",
        "address": "12 SAMPLE ROAD",
        "date": "12/07/2026",
        "total": "42.50",
    }
    assert repository_relative(image_path) == "assets/demo/synthetic_receipt.png"
    demo_source = (PROJECT_ROOT / "scripts" / "demo_inference.py").read_text(encoding="utf-8")
    assert "data/raw" not in demo_source
