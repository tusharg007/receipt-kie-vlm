"""Run the optional fixed-subset LoRA robustness benchmark."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from receipt_kie.evaluate import run_robustness_evaluation  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/evaluate.yaml")
    args = parser.parse_args()
    results = run_robustness_evaluation(args.config)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
