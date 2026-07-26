"""Command-line entry point for ReceiptKIE LoRA training."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from receipt_kie.train import run_training  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_lora.yaml")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--validation-size", type=int)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    overrides = {
        key: value
        for key, value in {
            "training.max_steps": args.max_steps,
            "training.train_limit": args.train_limit,
            "training.validation_size": args.validation_size,
            "paths.output_dir": args.output_dir,
        }.items()
        if value is not None
    }
    summary = run_training(args.config, overrides)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
