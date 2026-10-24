"""Export numeric features plus human correctness labels; no receipt text is stored."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from receipt_kie.monitoring import AuditStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database", type=Path, default=ROOT / ".cache/receipt-service/audit.sqlite3"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.database.is_file():
        parser.error("Audit database does not exist")
    rows = AuditStore(args.database).export_feedback()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    print(f"Exported {len(rows)} labeled feature records; split before retraining.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
