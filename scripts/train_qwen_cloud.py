"""Run the isolated Qwen3-VL Kaggle protocol; requires requirements-cloud.txt."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

print("Importing the cloud trainer and numerical libraries", flush=True)

from receipt_kie.qwen_cloud import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
