"""Run the frozen Base/V1/V2 comparison on the never-evaluated holdout."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from receipt_kie.v2_final import run_frozen_holdout_evaluation  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/highres_training_v2_holdout.yaml",
    )
    args = parser.parse_args()
    results = run_frozen_holdout_evaluation(args.config)
    print(json.dumps(results["success_gate"], indent=2, default=str))


if __name__ == "__main__":
    main()
