"""Base-versus-LoRA evaluation and artifact generation."""

from __future__ import annotations

import csv
import logging
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from receipt_kie.config import load_config
from receipt_kie.dataset import load_records
from receipt_kie.inference import ReceiptKIEPredictor
from receipt_kie.metrics import evaluate_predictions, normalize_field
from receipt_kie.prompts import CANONICAL_FIELDS
from receipt_kie.utils import project_path, setup_logging, write_json, write_jsonl

LOGGER = logging.getLogger(__name__)


def run_evaluation(config_path: str | Path) -> dict[str, Any]:
    """Evaluate base and newly trained adapter on the exact same test records."""
    config = load_config(config_path)
    setup_logging("artifacts/logs/evaluation.log")
    records = load_records(
        config["paths"]["dataset_root"],
        config["evaluation"]["split"],
        config["evaluation"].get("sample_limit"),
        int(config["evaluation"]["seed"]),
    )
    results: dict[str, Any] = {}
    variant_rows: dict[str, list[dict[str, Any]]] = {}
    for variant, adapter in (
        ("base", None),
        ("lora", project_path(config["paths"]["adapter_path"])),
    ):
        LOGGER.info("Evaluating variant=%s samples=%d", variant, len(records))
        predictor = ReceiptKIEPredictor(
            config["model"],
            config["paths"]["hf_cache"],
            adapter_path=adapter,
        )
        rows: list[dict[str, Any]] = []
        for index, record in enumerate(records, start=1):
            prediction = predictor.predict(
                record.image_path,
                max_new_tokens=int(config["generation"]["max_new_tokens"]),
                do_sample=bool(config["generation"]["do_sample"]),
            )
            rows.append(
                {
                    "sample_id": record.sample_id,
                    "image_path": record.image_path,
                    "ground_truth": record.target,
                    **prediction,
                }
            )
            if index % 10 == 0:
                LOGGER.info("%s inference %d/%d", variant, index, len(records))
        predictor.close()
        metrics = evaluate_predictions(rows)
        write_json(f"artifacts/reports/{variant}_metrics.json", metrics)
        write_jsonl(f"artifacts/predictions/{variant}_predictions.jsonl", rows)
        variant_rows[variant] = rows
        results[variant] = metrics
    _write_comparison(results)
    _plot_comparison(results)
    _copy_examples(variant_rows)
    return results


def _write_comparison(results: dict[str, Any]) -> None:
    output = project_path("artifacts/reports/model_comparison.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    metric_rows = [
        ("valid_json_rate", results["base"]["valid_json_rate"], results["lora"]["valid_json_rate"]),
        (
            "complete_record_normalized_exact_match",
            results["base"]["complete_record_normalized_exact_match"],
            results["lora"]["complete_record_normalized_exact_match"],
        ),
    ]
    for field in CANONICAL_FIELDS:
        for metric in ("raw_exact_match", "normalized_exact_match", "normalized_similarity"):
            metric_rows.append(
                (
                    f"{field}_{metric}",
                    results["base"]["per_field"][field][metric],
                    results["lora"]["per_field"][field][metric],
                )
            )
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "base", "lora", "absolute_improvement"])
        for name, base, lora in metric_rows:
            writer.writerow([name, base, lora, lora - base])


def _plot_comparison(results: dict[str, Any]) -> None:
    fields = list(CANONICAL_FIELDS)
    base = [results["base"]["per_field"][field]["normalized_exact_match"] for field in fields]
    lora = [results["lora"]["per_field"][field]["normalized_exact_match"] for field in fields]
    x = list(range(len(fields)))
    figures = project_path("artifacts/figures")
    figures.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4.5))
    plt.bar([value - 0.18 for value in x], base, width=0.36, label="Base")
    plt.bar([value + 0.18 for value in x], lora, width=0.36, label="LoRA")
    plt.xticks(x, fields)
    plt.ylim(0, 1)
    plt.ylabel("Normalized exact-match accuracy")
    plt.title("Base vs LoRA field accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures / "field_accuracy.png", dpi=160)
    plt.close()
    metrics = [
        results["base"]["valid_json_rate"],
        results["lora"]["valid_json_rate"],
        results["base"]["complete_record_normalized_exact_match"],
        results["lora"]["complete_record_normalized_exact_match"],
    ]
    plt.figure(figsize=(7, 4.5))
    plt.bar(["Base JSON", "LoRA JSON", "Base record", "LoRA record"], metrics)
    plt.ylim(0, 1)
    plt.ylabel("Rate")
    plt.title("ReceiptKIE-VLM base vs LoRA")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(figures / "base_vs_lora.png", dpi=160)
    plt.close()


def _copy_examples(rows: dict[str, list[dict[str, Any]]]) -> None:
    base_by_id = {row["sample_id"]: row for row in rows["base"]}
    lora_by_id = {row["sample_id"]: row for row in rows["lora"]}
    successful: list[str] = []
    failures: list[str] = []
    improvements: list[str] = []
    both_fail: list[str] = []
    for sample_id, lora in lora_by_id.items():
        base = base_by_id[sample_id]
        lora_score = _record_score(lora)
        base_score = _record_score(base)
        if lora_score == len(CANONICAL_FIELDS) and lora["valid_json"]:
            successful.append(sample_id)
        else:
            failures.append(sample_id)
        if lora_score > base_score:
            improvements.append(sample_id)
        if lora_score == 0 and base_score == 0:
            both_fail.append(sample_id)
    selected = {
        "lora_success": successful[:5],
        "lora_failure": failures[:5],
        "lora_improves": improvements[:3],
        "both_fail": both_fail[:3],
    }
    output_dir = project_path("artifacts/predictions/examples")
    output_dir.mkdir(parents=True, exist_ok=True)
    for category, sample_ids in selected.items():
        for sample_id in sample_ids:
            source = Path(lora_by_id[sample_id]["image_path"])
            shutil.copy2(source, output_dir / f"{category}__{sample_id}{source.suffix.lower()}")
    write_json("artifacts/predictions/examples/index.json", selected)


def _record_score(row: dict[str, Any]) -> int:
    prediction = row.get("parsed_prediction") or {}
    truth = row["ground_truth"]
    return sum(
        normalize_field(field, str(prediction.get(field, "")))
        == normalize_field(field, str(truth.get(field, "")))
        for field in CANONICAL_FIELDS
    )
