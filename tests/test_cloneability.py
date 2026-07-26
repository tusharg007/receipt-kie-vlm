from __future__ import annotations

import hashlib
import importlib.util
import json

from PIL import Image

from receipt_kie.config import load_config
from receipt_kie.utils import PROJECT_ROOT, repository_relative


def _demo_module():
    path = PROJECT_ROOT / "scripts" / "demo_inference.py"
    spec = importlib.util.spec_from_file_location("demo_inference", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_committed_adapters_match_training_metadata() -> None:
    for directory in ("receipt-kie-lora", "receipt-kie-lora-v2-highres"):
        adapter_dir = PROJECT_ROOT / "models" / directory
        adapter_path = adapter_dir / "adapter_model.safetensors"
        metadata = json.loads(
            (adapter_dir / "training_metadata.json").read_text(encoding="utf-8")
        )
        digest = hashlib.sha256(adapter_path.read_bytes()).hexdigest()
        assert adapter_path.stat().st_size == metadata["adapter_size_bytes"]
        assert digest == metadata["adapter_sha256"]
        assert not adapter_path.read_bytes()[:128].startswith(
            b"version https://git-lfs.github.com/spec/v1"
        )
    assert load_config("configs/evaluate.yaml")["paths"]["adapter_path"] == (
        "models/receipt-kie-lora-v2-highres"
    )
    assert load_config("configs/evaluate.yaml")["model"]["image_longest_edge"] == 1536
    assert load_config("configs/evaluate.yaml")["generation"] == {
        "max_new_tokens": 256,
        "do_sample": False,
        "repetition_penalty": 1.08,
    }


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
    assert 'default="v2"' in demo_source
    assert '"v1": {' in demo_source
    assert '"v2": {' in demo_source
    assert "refusing base-model fallback" in demo_source


def test_demo_formats_cpu_and_gpu_memory() -> None:
    demo = _demo_module()
    assert demo._format_peak_memory(None) == "not available (CPU)"
    assert demo._format_peak_memory(1877.3) == "1877.30 MiB"
