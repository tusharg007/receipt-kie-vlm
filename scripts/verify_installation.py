"""Verify a clone has both ReceiptKIE-VLM adapters and inference dependencies."""

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
DEMO_IMAGE = PROJECT_ROOT / "assets" / "demo" / "synthetic_receipt.png"
ADAPTERS = {
    "V1": {
        "path": PROJECT_ROOT / "models" / "receipt-kie-lora",
        "sha256": (
            "94ba0038153ea1aacb12dbcc80f1edf01d31a6309ea56919684e8cb8bbe90b28"
        ),
        "size": 10_956_944,
    },
    "V2": {
        "path": PROJECT_ROOT / "models" / "receipt-kie-lora-v2-highres",
        "sha256": (
            "3e0e5a88c36f0d6a0db6baf2a3b521e40be4ef84b212ed2eafecab431604bf79"
        ),
        "size": 10_956_944,
    },
}


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
            f"{sys.version_info.major}.{sys.version_info.minor}."
            f"{sys.version_info.micro}"
            if sys.version_info >= (3, 11)
            else (_ for _ in ()).throw(
                RuntimeError("Python 3.11 or newer is required")
            )
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
            lambda module_name=module_name: (
                str(importlib.import_module(module_name).__version__)
                if hasattr(importlib.import_module(module_name), "__version__")
                else "available"
            ),
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
    for version, expected in ADAPTERS.items():
        adapter_dir = expected["path"]
        metadata_path = adapter_dir / "training_metadata.json"
        adapter_path = adapter_dir / "adapter_model.safetensors"
        config_path = adapter_dir / "adapter_config.json"
        check(
            f"{version} training metadata",
            lambda path=metadata_path: str(path.relative_to(PROJECT_ROOT))
            if path.is_file()
            else (_ for _ in ()).throw(FileNotFoundError(path)),
        )
        check(
            f"{version} adapter files",
            lambda weights=adapter_path, config=config_path: (
                f"{weights.relative_to(PROJECT_ROOT)}, "
                f"{config.relative_to(PROJECT_ROOT)}"
                if weights.is_file() and config.is_file()
                else (_ for _ in ()).throw(
                    FileNotFoundError(f"{weights} or {config}")
                )
            ),
        )
        if metadata_path.is_file() and adapter_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            check(
                f"{version} adapter checksum",
                lambda path=adapter_path, metadata=metadata, expected=expected: (
                    _sha256(path)
                    if _sha256(path)
                    == metadata["adapter_sha256"]
                    == expected["sha256"]
                    else (_ for _ in ()).throw(
                        RuntimeError("SHA-256 does not match metadata/release value")
                    )
                ),
            )
            check(
                f"{version} adapter size",
                lambda path=adapter_path, metadata=metadata, expected=expected: (
                    f"{path.stat().st_size} bytes"
                    if path.stat().st_size
                    == metadata["adapter_size_bytes"]
                    == expected["size"]
                    else (_ for _ in ()).throw(
                        RuntimeError("Size does not match metadata/release value")
                    )
                ),
            )
            check(
                f"{version} adapter binary",
                lambda path=adapter_path: (
                    "safetensors binary"
                    if not path.read_bytes()[:128].startswith(
                        b"version https://git-lfs.github.com/spec/v1"
                    )
                    else (_ for _ in ()).throw(
                        RuntimeError("Adapter is a Git LFS pointer")
                    )
                ),
            )
        check(
            f"{version} PEFT configuration",
            lambda path=adapter_dir: (
                f"{PeftConfig.from_pretrained(path).peft_type.value} for "
                f"{PeftConfig.from_pretrained(path).base_model_name_or_path}"
            ),
        )
    check("Synthetic demo image", lambda: _verify_image(DEMO_IMAGE))
    results.extend(
        (
            (
                "Raw dataset requirement",
                True,
                "not required; demo uses assets/demo/synthetic_receipt.png",
            ),
            (
                "Kaggle credential requirement",
                True,
                "not required for installation or demo inference",
            ),
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
