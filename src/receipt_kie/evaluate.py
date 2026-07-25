"""Base-versus-LoRA evaluation and artifact generation."""

from __future__ import annotations

import csv
import logging
import shutil
from pathlib import Path
from typing import Any

import matplotlib
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from receipt_kie.config import load_config
from receipt_kie.corruptions import (
    gaussian_blur,
    jpeg_compression,
    reduced_brightness,
    small_rotation,
)
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


def run_robustness_evaluation(config_path: str | Path) -> dict[str, Any]:
    """Run a fixed-subset pilot benchmark under four mild image corruptions."""
    config = load_config(config_path)
    setup_logging("artifacts/logs/robustness.log")
    seed = int(config["evaluation"]["seed"])
    records = load_records(
        config["paths"]["dataset_root"],
        config["evaluation"]["split"],
        int(config["evaluation"]["robustness_sample_limit"]),
        seed,
    )
    predictor = ReceiptKIEPredictor(
        config["model"],
        config["paths"]["hf_cache"],
        adapter_path=project_path(config["paths"]["adapter_path"]),
    )
    variants = ("clean", "gaussian_blur", "jpeg_compression", "reduced_brightness", "rotation")
    results: dict[str, Any] = {}
    output_root = project_path("data/processed/robustness")
    for variant in variants:
        rows: list[dict[str, Any]] = []
        LOGGER.info("Robustness variant=%s samples=%d", variant, len(records))
        for index, record in enumerate(records):
            prediction_path = Path(record.image_path)
            if variant != "clean":
                prediction_path = _write_corrupted_image(
                    record.image_path,
                    output_root / variant / f"{record.sample_id}.jpg",
                    variant,
                    seed + index,
                )
            prediction = predictor.predict(
                prediction_path,
                max_new_tokens=int(config["generation"]["max_new_tokens"]),
                do_sample=bool(config["generation"]["do_sample"]),
            )
            rows.append(
                {
                    "sample_id": record.sample_id,
                    "image_path": str(prediction_path),
                    "ground_truth": record.target,
                    **prediction,
                }
            )
        metrics = evaluate_predictions(rows)
        results[variant] = metrics
        write_jsonl(f"artifacts/predictions/robustness_{variant}.jsonl", rows)
    predictor.close()
    write_json("artifacts/reports/robustness_metrics.json", results)
    _write_robustness_csv(results)
    _plot_robustness(results)
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
        if lora_score >= 1 and lora["valid_json"]:
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


def _write_corrupted_image(
    source_path: str,
    output_path: Path,
    variant: str,
    seed: int,
) -> Path:
    with Image.open(source_path) as opened:
        image = opened.convert("RGB")
        if variant == "gaussian_blur":
            transformed = gaussian_blur(image)
        elif variant == "jpeg_compression":
            transformed = jpeg_compression(image)
        elif variant == "reduced_brightness":
            transformed = reduced_brightness(image)
        elif variant == "rotation":
            transformed = small_rotation(image, seed=seed)
        else:
            raise ValueError(f"Unknown corruption variant: {variant}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    transformed.save(output_path, format="JPEG", quality=95)
    return output_path


def _write_robustness_csv(results: dict[str, Any]) -> None:
    output = project_path("artifacts/reports/robustness_metrics.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = (
        "valid_json_rate",
        "company_accuracy",
        "address_similarity",
        "date_accuracy",
        "total_accuracy",
        "complete_record_normalized_exact_match",
        "average_inference_latency_seconds",
    )
    clean = results["clean"]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["variant", "sample_count", *columns, "mean_field_score_delta"])
        for variant, metrics in results.items():
            mean_score = sum(float(metrics[name]) for name in columns[1:5]) / 4
            clean_mean = sum(float(clean[name]) for name in columns[1:5]) / 4
            writer.writerow(
                [
                    variant,
                    metrics["sample_count"],
                    *(metrics[name] for name in columns),
                    mean_score - clean_mean,
                ]
            )


def _plot_robustness(results: dict[str, Any]) -> None:
    variants = list(results)
    labels = [name.replace("_", " ").title() for name in variants]
    values = [
        sum(
            (
                results[name]["company_accuracy"],
                results[name]["address_similarity"],
                results[name]["date_accuracy"],
                results[name]["total_accuracy"],
            )
        )
        / 4
        for name in variants
    ]
    plt.figure(figsize=(9, 4.8))
    plt.bar(labels, values)
    plt.ylim(0, 1)
    plt.ylabel("Mean field score")
    plt.title("Pilot robustness benchmark (20 fixed receipts)")
    plt.xticks(rotation=18)
    plt.tight_layout()
    plt.savefig(project_path("artifacts/figures/robustness_results.png"), dpi=160)
    plt.close()


def _record_score(row: dict[str, Any]) -> int:
    prediction = row.get("parsed_prediction") or {}
    truth = row["ground_truth"]
    return sum(
        normalize_field(field, str(prediction.get(field, "")))
        == normalize_field(field, str(truth.get(field, "")))
        for field in CANONICAL_FIELDS
    )
