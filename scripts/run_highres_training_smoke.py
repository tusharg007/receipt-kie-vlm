"""Run a leakage-controlled continued-training memory smoke test."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from receipt_kie.highres_training import run_highres_training_smoke  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/highres_training_smoke_2048.yaml",
    )
    args = parser.parse_args()
    report = run_highres_training_smoke(args.config)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
