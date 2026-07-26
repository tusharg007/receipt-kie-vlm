"""Run or resume the fixed-subset high-resolution inference ablation."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from receipt_kie.ablation import (  # noqa: E402
    SIZE_VARIANTS,
    fixed_test_records,
    initialize_results,
    record_variant_result,
    refresh_result_artifacts,
    repetition_variant_for,
    run_single_pass_variant,
    run_two_pass_variant,
    select_winning_highres_variant,
    two_pass_variant_for,
)
from receipt_kie.config import load_config  # noqa: E402
from receipt_kie.utils import setup_logging  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/highres_ablation.yaml")
    parser.add_argument(
        "--phase",
        choices=("sizes", "repetition", "crops", "followups", "finalize", "all"),
        default="all",
    )
    parser.add_argument("--variant", choices=tuple(SIZE_VARIANTS))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    setup_logging("artifacts/logs/highres_ablation.log")
    results = initialize_results(config)

    if args.phase == "finalize":
        refresh_result_artifacts(config, results)
        return
    records = fixed_test_records(config)
    if args.variant:
        _run_size(config, records, results, args.variant, args.force)
        return
    if args.phase in {"sizes", "all"}:
        for variant_name in SIZE_VARIANTS:
            _run_size(config, records, results, variant_name, args.force)
    if args.phase in {"repetition", "followups", "all"}:
        winner = select_winning_highres_variant(results)
        variant = repetition_variant_for(winner, config)
        _run_single(config, records, results, variant, args.force)
    if args.phase in {"crops", "followups", "all"}:
        winner = select_winning_highres_variant(results)
        variant = two_pass_variant_for(winner)
        if args.force or variant.name not in results["variants"]:
            _, result = run_two_pass_variant(config, variant, records)
            record_variant_result(config, results, result)


def _run_size(
    config: dict,
    records: list,
    results: dict,
    variant_name: str,
    force: bool,
) -> None:
    _run_single(config, records, results, SIZE_VARIANTS[variant_name], force)


def _run_single(
    config: dict,
    records: list,
    results: dict,
    variant,
    force: bool,
) -> None:
    if not force and variant.name in results["variants"]:
        return
    _, result = run_single_pass_variant(config, variant, records)
    record_variant_result(config, results, result)


if __name__ == "__main__":
    main()
