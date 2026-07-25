"""Discover SROIE and generate JSON/Markdown audit reports."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))

from receipt_kie.dataset import write_audit  # noqa: E402
from receipt_kie.utils import setup_logging  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        default="data/raw/sroie/SROIE2019",
        help="SROIE root containing train/ and test/.",
    )
    args = parser.parse_args()
    setup_logging("artifacts/logs/dataset_audit.log")
    audit = write_audit(args.dataset_root)
    print(
        f"Validated {audit['valid_pairs']} pairs; "
        f"excluded {audit['excluded_count']} malformed or missing samples."
    )


if __name__ == "__main__":
    main()
