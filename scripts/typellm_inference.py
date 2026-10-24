"""Extract a receipt with native TypeLLM and a separately served SGLang VLM."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from receipt_kie.typellm_inference import TypeLLMReceiptPredictor  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--model", help="Checkpoint served by SGLang; discovered if omitted.")
    parser.add_argument("--thinking-budget", type=int, default=0)
    parser.add_argument("--text-max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    try:
        predictor = TypeLLMReceiptPredictor(
            base_url=args.base_url,
            model=args.model,
            thinking_budget=args.thinking_budget,
            timeout=args.timeout,
            text_max_tokens=args.text_max_tokens,
        )
        result = predictor.predict(args.image)
    except Exception as exc:
        print(f"ERROR: typed extraction failed closed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
