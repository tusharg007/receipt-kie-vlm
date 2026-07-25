"""Build concise derived reports and refresh curated prediction examples."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from receipt_kie.evaluate import _copy_examples  # noqa: E402
from receipt_kie.utils import write_json  # noqa: E402


def _read_json(relative_path: str) -> dict[str, Any]:
    with (PROJECT_ROOT / relative_path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(relative_path: str) -> list[dict[str, Any]]:
    with (PROJECT_ROOT / relative_path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    training = _read_json("artifacts/checkpoints/receipt-kie-lora/training_summary.json")
    audit = _read_json("artifacts/reports/dataset_audit.json")
    base = _read_json("artifacts/reports/base_metrics.json")
    lora = _read_json("artifacts/reports/lora_metrics.json")
    robustness = _read_json("artifacts/reports/robustness_metrics.json")
    base_rows = _read_jsonl("artifacts/predictions/base_predictions.jsonl")
    lora_rows = _read_jsonl("artifacts/predictions/lora_predictions.jsonl")
    _copy_examples({"base": base_rows, "lora": lora_rows})
    summary = {
        "dataset": {
            "valid_pairs": audit["valid_pairs"],
            "train_samples": training["train_samples"],
            "test_samples": lora["sample_count"],
            "excluded_samples": audit["excluded_count"],
        },
        "training": {
            "global_steps": training["global_steps"],
            "duration_seconds": training["duration_seconds"],
            "trainable_parameters": training["parameter_statistics"]["trainable_parameters"],
            "trainable_percentage": training["parameter_statistics"]["trainable_percentage"],
            "peak_gpu_memory_mib": training["peak_gpu_memory_mib"],
            "lora_parameter_changed": training["lora_parameter_changed"],
        },
        "base": base,
        "lora": lora,
        "absolute_improvements": {
            "valid_json_rate": lora["valid_json_rate"] - base["valid_json_rate"],
            "company_accuracy": lora["company_accuracy"] - base["company_accuracy"],
            "address_similarity": lora["address_similarity"] - base["address_similarity"],
            "date_accuracy": lora["date_accuracy"] - base["date_accuracy"],
            "total_accuracy": lora["total_accuracy"] - base["total_accuracy"],
            "complete_record_normalized_exact_match": (
                lora["complete_record_normalized_exact_match"]
                - base["complete_record_normalized_exact_match"]
            ),
        },
        "robustness_pilot": robustness,
    }
    write_json("artifacts/reports/results_summary.json", summary)
    lines = [
        "# Generated Results Summary",
        "",
        "All values below were read from the current training and prediction artifacts.",
        "",
        "| Metric | Base | LoRA | Absolute change |",
        "|---|---:|---:|---:|",
    ]
    for label, key in (
        ("Valid JSON", "valid_json_rate"),
        ("Company accuracy", "company_accuracy"),
        ("Address similarity", "address_similarity"),
        ("Date accuracy", "date_accuracy"),
        ("Total accuracy", "total_accuracy"),
        ("Complete-record accuracy", "complete_record_normalized_exact_match"),
    ):
        base_value = float(base[key])
        lora_value = float(lora[key])
        lines.append(
            f"| {label} | {base_value:.1%} | {lora_value:.1%} | "
            f"{lora_value - base_value:+.1%} |"
        )
    lines.extend(
        [
            "",
            f"- Training receipts: {training['train_samples']}",
            f"- Held-out generation receipts: {lora['sample_count']}",
            f"- Optimizer steps: {training['global_steps']}",
            f"- Training duration: {training['duration_seconds'] / 60:.2f} minutes",
            f"- Trainable parameters: "
            f"{training['parameter_statistics']['trainable_parameters']:,} "
            f"({training['parameter_statistics']['trainable_percentage']:.3f}%)",
        ]
    )
    (PROJECT_ROOT / "artifacts/reports/results_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
