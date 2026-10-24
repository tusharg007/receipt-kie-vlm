"""Run the experimental CPU-OCR + local Bolt receipt challenger on one image."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from receipt_kie.ocr_bolt import BoltOCRPredictor  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=ROOT / "assets/demo/synthetic_receipt.png")
    parser.add_argument("--base-url", default="http://127.0.0.1:8081")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    predictor = BoltOCRPredictor(args.base_url, timeout=args.timeout)
    try:
        result = predictor.predict(args.image)
    finally:
        predictor.close()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "ocr_lines"}, indent=2))
    return 0 if result["valid_json"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
