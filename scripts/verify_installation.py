"""Verify a clone has everything required for ReceiptKIE-VLM inference."""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path
from typing import Callable

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
ADAPTER_DIR = PROJECT_ROOT / "models" / "receipt-kie-lora"
DEMO_IMAGE = PROJECT_ROOT / "assets" / "demo" / "synthetic_receipt.png"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    results: list[tuple[str, bool, str]] = []

    def check(name: str, operation: Callable[[], str]) -> None:
        try:
            detail = operation()
        except Exception as exc:
            results.append((name, False, str(exc)))
        else:
            results.append((name, True, detail))

    check(
        "Python version",
        lambda: (
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            if sys.version_info >= (3, 12)
            else (_ for _ in ()).throw(RuntimeError("Python 3.12 or newer is required"))
        ),
    )
    for module_name in (
        "torch",
        "torchvision",
        "transformers",
        "accelerate",
        "peft",
        "trl",
        "PIL",
        "yaml",
        "safetensors",
        "receipt_kie",
    ):
        check(
            f"Import {module_name}",
            lambda module_name=module_name: str(importlib.import_module(module_name).__version__)
            if hasattr(importlib.import_module(module_name), "__version__")
            else "available",
        )

    import torch
    from peft import PeftConfig

    check("PyTorch available", lambda: torch.__version__)
    results.append(
        (
            "CUDA availability",
            True,
            (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else "CPU fallback available"
            ),
        )
    )
    metadata_path = ADAPTER_DIR / "training_metadata.json"
    adapter_path = ADAPTER_DIR / "adapter_model.safetensors"
    config_path = ADAPTER_DIR / "adapter_config.json"
    check("Training metadata exists", lambda: str(metadata_path.relative_to(PROJECT_ROOT)))
    check("Adapter weights exist", lambda: str(adapter_path.relative_to(PROJECT_ROOT)))
    check("Adapter config exists", lambda: str(config_path.relative_to(PROJECT_ROOT)))
    metadata: dict[str, object] = {}
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if adapter_path.is_file() and metadata:
        check(
            "Adapter checksum",
            lambda: _sha256(adapter_path)
            if _sha256(adapter_path) == metadata["adapter_sha256"]
            else (_ for _ in ()).throw(RuntimeError("SHA-256 does not match training metadata")),
        )
        check(
            "Adapter is a real binary",
            lambda: "safetensors binary"
            if not adapter_path.read_bytes()[:128].startswith(
                b"version https://git-lfs.github.com/spec/v1"
            )
            else (_ for _ in ()).throw(RuntimeError("Adapter is a Git LFS pointer")),
        )
    check(
        "PEFT adapter configuration",
        lambda: (
            f"{PeftConfig.from_pretrained(ADAPTER_DIR).peft_type.value} for "
            f"{PeftConfig.from_pretrained(ADAPTER_DIR).base_model_name_or_path}"
        ),
    )
    check(
        "Synthetic demo image",
        lambda: _verify_image(DEMO_IMAGE),
    )
    results.append(
        (
            "Raw dataset requirement",
            True,
            "not required for inference; the demo uses assets/demo/synthetic_receipt.png",
        )
    )
    print("ReceiptKIE-VLM installation verification")
    print("=" * 44)
    for name, passed, detail in results:
        print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")
    failed = [name for name, passed, _ in results if not passed]
    print("=" * 44)
    print("OVERALL: " + ("PASS" if not failed else "FAIL"))
    return 0 if not failed else 1


def _verify_image(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as image:
        image.verify()
    return str(path.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
