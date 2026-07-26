"""Build the V2 official-test usage and never-evaluated holdout manifest."""

from __future__ import annotations

import json

from receipt_kie.test_usage import build_test_usage_manifest


def main() -> None:
    manifest = build_test_usage_manifest(
        "data/raw/sroie/SROIE2019",
        "artifacts/experiments/highres_training_v2/test_usage_manifest.json",
    )
    print(json.dumps(manifest["counts"], indent=2))
    print(json.dumps(manifest["overlap_counts"], indent=2))


if __name__ == "__main__":
    main()
