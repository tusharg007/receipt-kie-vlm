"""Run the committed ReceiptKIE-VLM V1 or V2 adapter on one receipt image."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from peft import PeftModel  # noqa: E402

from receipt_kie.config import load_config  # noqa: E402
from receipt_kie.inference import ReceiptKIEPredictor  # noqa: E402

DEFAULT_IMAGE = PROJECT_ROOT / "assets" / "demo" / "synthetic_receipt.png"
MODEL_VERSIONS: dict[str, dict[str, Any]] = {
    "v1": {
        "adapter_path": "models/receipt-kie-lora",
        "image_longest_edge": 512,
        "max_image_patch_edge": 512,
        "do_image_splitting": True,
        "max_new_tokens": 128,
        "do_sample": False,
        "repetition_penalty": None,
    },
    "v2": {
        "adapter_path": "models/receipt-kie-lora-v2-highres",
        "image_longest_edge": 1536,
        "max_image_patch_edge": 512,
        "do_image_splitting": True,
        "max_new_tokens": 256,
        "do_sample": False,
        "repetition_penalty": 1.08,
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_image(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Image does not exist: {path}")
    try:
        with Image.open(path) as image:
            image.verify()
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(f"Image is invalid or unreadable: {path}") from exc


def _resolved_version(
    version: str,
    base_config: dict[str, Any],
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    settings = deepcopy(MODEL_VERSIONS[version])
    model_config = deepcopy(base_config["model"])
    for key in (
        "image_longest_edge",
        "max_image_patch_edge",
        "do_image_splitting",
    ):
        model_config[key] = settings[key]
    adapter_path = PROJECT_ROOT / settings["adapter_path"]
    return model_config, adapter_path, settings


def _verify_adapter(adapter_path: Path) -> tuple[str, dict[str, Any]]:
    required = (
        adapter_path / "adapter_model.safetensors",
        adapter_path / "adapter_config.json",
        adapter_path / "training_metadata.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Committed LoRA adapter is incomplete: " + ", ".join(missing)
        )
    metadata = json.loads(required[2].read_text(encoding="utf-8"))
    checksum = _sha256(required[0])
    if checksum != metadata.get("adapter_sha256"):
        raise RuntimeError(
            f"Adapter checksum mismatch: {checksum} != "
            f"{metadata.get('adapter_sha256')}"
        )
    if required[0].read_bytes()[:128].startswith(
        b"version https://git-lfs.github.com/spec/v1"
    ):
        raise RuntimeError("Adapter weights are a Git LFS pointer")
    return checksum, metadata


def _format_peak_memory(value: float | None) -> str:
    if value is None:
        return "not available (CPU)"
    return f"{float(value):.2f} MiB"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument(
        "--model-version",
        choices=sorted(MODEL_VERSIONS),
        default="v2",
        help="V2 is recommended and selected by default; V1 is retained for reproduction.",
    )
    args = parser.parse_args()
    image_path = args.image.resolve()
    predictor: ReceiptKIEPredictor | None = None
    try:
        _valid_image(image_path)
        base_config = load_config(PROJECT_ROOT / "configs" / "evaluate.yaml")
        model_config, adapter_path, settings = _resolved_version(
            args.model_version,
            base_config,
        )
        checksum, _ = _verify_adapter(adapter_path)
        predictor = ReceiptKIEPredictor(
            model_config,
            base_config["paths"]["hf_cache"],
            adapter_path=adapter_path,
        )
        if not isinstance(predictor.model, PeftModel) or not predictor.model.peft_config:
            raise RuntimeError(
                "Requested adapter did not load as an active PEFT model; "
                "refusing base-model fallback"
            )
        prediction = predictor.predict(
            image_path,
            max_new_tokens=int(settings["max_new_tokens"]),
            do_sample=bool(settings["do_sample"]),
            repetition_penalty=settings["repetition_penalty"],
        )
    except Exception as exc:
        print(f"ERROR: demo failed closed: {exc}", file=sys.stderr)
        return 1
    finally:
        if predictor is not None:
            predictor.close()

    processor = prediction["processor_configuration"]
    decoding = {
        "max_new_tokens": settings["max_new_tokens"],
        "do_sample": settings["do_sample"],
        "repetition_penalty": settings["repetition_penalty"],
    }
    print(f"Selected version: {args.model_version}")
    print(f"Device: {predictor.device}")
    print(f"Adapter path: {adapter_path.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"Adapter SHA-256: {checksum}")
    print(f"Processor size: {processor['size']}")
    print(f"Patch size: {processor['max_image_size']}")
    print(f"Image splitting: {processor['do_image_splitting']}")
    print(f"Visual tile count: {prediction['visual_tile_count']}")
    print(f"Decoding parameters: {json.dumps(decoding, sort_keys=True)}")
    print(f"Inference latency: {prediction['latency_seconds']:.3f} seconds")
    print(
        "Peak GPU memory: "
        f"{_format_peak_memory(prediction['peak_gpu_memory_mib'])}"
    )
    print("Raw model output:")
    print(prediction["raw_output"])
    print("Parsed JSON:")
    print(json.dumps(prediction["parsed_prediction"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
