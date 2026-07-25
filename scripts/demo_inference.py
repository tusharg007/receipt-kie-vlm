"""Run the committed ReceiptKIE-VLM adapter on one receipt image."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from PIL import Image, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from receipt_kie.config import load_config  # noqa: E402
from receipt_kie.inference import ReceiptKIEPredictor  # noqa: E402

DEFAULT_IMAGE = PROJECT_ROOT / "assets" / "demo" / "synthetic_receipt.png"
ADAPTER_PATH = PROJECT_ROOT / "models" / "receipt-kie-lora"


def _valid_image(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Image does not exist: {path}")
    try:
        with Image.open(path) as image:
            image.verify()
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(f"Image is invalid or unreadable: {path}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    args = parser.parse_args()
    image_path = args.image.resolve()
    required_adapter_files = (
        ADAPTER_PATH / "adapter_model.safetensors",
        ADAPTER_PATH / "adapter_config.json",
    )
    missing = [str(path) for path in required_adapter_files if not path.is_file()]
    if missing:
        print("ERROR: committed LoRA adapter is incomplete: " + ", ".join(missing), file=sys.stderr)
        return 1
    try:
        _valid_image(image_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    predictor: ReceiptKIEPredictor | None = None
    try:
        config = load_config(PROJECT_ROOT / "configs" / "evaluate.yaml")
        predictor = ReceiptKIEPredictor(
            config["model"],
            config["paths"]["hf_cache"],
            adapter_path=ADAPTER_PATH,
        )
        prediction = predictor.predict(
            image_path,
            max_new_tokens=int(config["generation"]["max_new_tokens"]),
            do_sample=False,
        )
    except Exception as exc:
        print(
            "ERROR: unable to load the base model and committed LoRA adapter "
            f"or run inference: {exc}",
            file=sys.stderr,
        )
        return 1
    finally:
        if predictor is not None:
            predictor.close()
    print(f"Device: {predictor.device}")
    print(f"Adapter path: {ADAPTER_PATH.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"Inference latency: {prediction['latency_seconds']:.3f} seconds")
    print("Raw model output:")
    print(prediction["raw_output"])
    print("Parsed JSON:")
    print(json.dumps(prediction["parsed_prediction"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
