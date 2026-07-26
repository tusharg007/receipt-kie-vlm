"""Controlled high-resolution inference ablation for the committed LoRA adapter."""

from __future__ import annotations

import csv
import json
import logging
import statistics
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from receipt_kie.dataset import ReceiptRecord, load_records
from receipt_kie.inference import ReceiptKIEPredictor
from receipt_kie.metrics import evaluate_predictions, has_repetition_failure
from receipt_kie.prompts import canonical_target
from receipt_kie.utils import (
    project_path,
    repository_relative,
    seed_everything,
    write_json,
    write_jsonl,
)

LOGGER = logging.getLogger(__name__)

PASS_A_PROMPT = (
    "Extract company and address from the top of this receipt. Return valid JSON "
    'only with exactly these keys in this order: company, address. Use an empty '
    "string when a field is not visible."
)
PASS_B_PROMPT = (
    "The first image is the full receipt and the second is its lower region. "
    "Extract date and total. Return valid JSON only with exactly these keys in "
    "this order: date, total. Use an empty string when a field is not visible."
)


@dataclass(frozen=True)
class VariantConfiguration:
    """One deterministic inference configuration."""

    name: str
    image_longest_edge: int
    max_new_tokens: int
    max_image_patch_edge: int = 512
    do_image_splitting: bool = True
    repetition_penalty: float | None = None
    strategy: str = "single_pass"


SIZE_VARIANTS = {
    variant.name: variant
    for variant in (
        VariantConfiguration("baseline_512_128", 512, 128),
        VariantConfiguration("token_control_512_256", 512, 256),
        VariantConfiguration("highres_1024_256", 1024, 256),
        VariantConfiguration("highres_1536_256", 1536, 256),
        VariantConfiguration("default_2048_256", 2048, 256),
    )
}


def fixed_test_records(config: dict[str, Any]) -> list[ReceiptRecord]:
    """Select and persist the deterministic 30-receipt official-test subset."""
    evaluation = config["evaluation"]
    records = load_records(
        config["paths"]["dataset_root"],
        str(evaluation["split"]),
        int(evaluation["sample_limit"]),
        int(evaluation["seed"]),
    )
    expected_ids = [record.sample_id for record in records]
    ids_path = _output_root(config) / "test_ids.json"
    if ids_path.is_file():
        existing_ids = json.loads(ids_path.read_text(encoding="utf-8"))
        if existing_ids != expected_ids:
            raise ValueError(
                "Existing ablation IDs do not match the deterministic seed-42 selection"
            )
    else:
        write_json(ids_path, expected_ids)
    return records


def initialize_results(config: dict[str, Any]) -> dict[str, Any]:
    """Load partial results or initialize stable experiment metadata."""
    path = _output_root(config) / "results.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "experiment": "high_resolution_inference_ablation",
        "source_split": str(config["evaluation"]["split"]),
        "sample_count": int(config["evaluation"]["sample_limit"]),
        "seed": int(config["evaluation"]["seed"]),
        "adapter_path": str(config["paths"]["adapter_path"]),
        "deterministic_decoding": True,
        "variants": {},
    }


def run_single_pass_variant(
    config: dict[str, Any],
    variant: VariantConfiguration,
    records: list[ReceiptRecord],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evaluate one single-pass configuration on the fixed records."""
    seed_everything(int(config["evaluation"]["seed"]))
    predictor = _predictor(config, variant)
    rows: list[dict[str, Any]] = []
    prediction_path = _prediction_path(config, variant.name)
    try:
        for index, record in enumerate(records, start=1):
            prediction = predictor.predict(
                record.image_path,
                max_new_tokens=variant.max_new_tokens,
                do_sample=False,
                repetition_penalty=variant.repetition_penalty,
            )
            prediction["repetition_failure"] = has_repetition_failure(
                prediction["raw_output"]
            )
            rows.append(_prediction_row(record, prediction))
            write_jsonl(prediction_path, rows)
            LOGGER.info(
                "variant=%s sample=%d/%d id=%s valid=%s limit=%s "
                "repetition=%s tiles=%s latency=%.3f",
                variant.name,
                index,
                len(records),
                record.sample_id,
                prediction["valid_json"],
                prediction["generation_limit_hit"],
                prediction["repetition_failure"],
                prediction["visual_tile_count"],
                prediction["latency_seconds"],
            )
        processor_configuration = predictor.processor_configuration
    finally:
        predictor.close()
    return rows, _variant_result(
        variant,
        rows,
        processor_configuration,
        prediction_path,
    )


def run_two_pass_variant(
    config: dict[str, Any],
    variant: VariantConfiguration,
    records: list[ReceiptRecord],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run top/full/lower crop inference and merge unedited parsed fields."""
    seed_everything(int(config["evaluation"]["seed"]))
    predictor = _predictor(config, variant)
    rows: list[dict[str, Any]] = []
    prediction_path = _prediction_path(config, variant.name)
    top_fraction = float(config["two_pass"]["top_fraction"])
    lower_fraction = float(config["two_pass"]["lower_fraction"])
    try:
        for index, record in enumerate(records, start=1):
            with Image.open(record.image_path) as opened:
                full = opened.convert("RGB").copy()
            width, height = full.size
            top = full.crop((0, 0, width, max(1, round(height * top_fraction))))
            lower_start = max(0, round(height * (1.0 - lower_fraction)))
            lower = full.crop((0, lower_start, width, height))
            pass_a = predictor.predict_images(
                [top],
                PASS_A_PROMPT,
                max_new_tokens=variant.max_new_tokens,
                do_sample=False,
                repetition_penalty=variant.repetition_penalty,
            )
            pass_b = predictor.predict_images(
                [full, lower],
                PASS_B_PROMPT,
                max_new_tokens=variant.max_new_tokens,
                do_sample=False,
                repetition_penalty=variant.repetition_penalty,
            )
            parsed_a = canonical_target(pass_a.get("parsed_prediction") or {})
            parsed_b = canonical_target(pass_b.get("parsed_prediction") or {})
            merged = {
                "company": parsed_a["company"],
                "address": parsed_a["address"],
                "date": parsed_b["date"],
                "total": parsed_b["total"],
            }
            repetition_failure = has_repetition_failure(
                pass_a["raw_output"]
            ) or has_repetition_failure(pass_b["raw_output"])
            prediction = {
                "raw_output": json.dumps(merged, ensure_ascii=False),
                "valid_json": True,
                "parsed_prediction": merged,
                "parse_method": "deterministic_two_pass_merge",
                "parse_error": None,
                "latency_seconds": (
                    float(pass_a["latency_seconds"])
                    + float(pass_b["latency_seconds"])
                ),
                "peak_gpu_memory_mib": _maximum_optional(
                    pass_a["peak_gpu_memory_mib"],
                    pass_b["peak_gpu_memory_mib"],
                ),
                "generated_token_count": (
                    int(pass_a["generated_token_count"])
                    + int(pass_b["generated_token_count"])
                ),
                "generation_limit_hit": bool(
                    pass_a["generation_limit_hit"]
                    or pass_b["generation_limit_hit"]
                ),
                "repetition_failure": repetition_failure,
                "visual_tile_count": (
                    int(pass_a["visual_tile_count"])
                    + int(pass_b["visual_tile_count"])
                ),
                "pixel_values_shape": [
                    pass_a["pixel_values_shape"],
                    pass_b["pixel_values_shape"],
                ],
                "visual_tile_shape": pass_a["visual_tile_shape"],
                "processor_configuration": predictor.processor_configuration,
                "subpass_valid_json": {
                    "pass_a": bool(pass_a["valid_json"]),
                    "pass_b": bool(pass_b["valid_json"]),
                },
                "subpass_generation_limit_hit": {
                    "pass_a": bool(pass_a["generation_limit_hit"]),
                    "pass_b": bool(pass_b["generation_limit_hit"]),
                },
                "subpass_raw_output": {
                    "pass_a": pass_a["raw_output"],
                    "pass_b": pass_b["raw_output"],
                },
            }
            rows.append(_prediction_row(record, prediction))
            write_jsonl(prediction_path, rows)
            LOGGER.info(
                "variant=%s sample=%d/%d id=%s subpass_valid=%s "
                "limit=%s repetition=%s tiles=%s latency=%.3f",
                variant.name,
                index,
                len(records),
                record.sample_id,
                prediction["subpass_valid_json"],
                prediction["generation_limit_hit"],
                prediction["repetition_failure"],
                prediction["visual_tile_count"],
                prediction["latency_seconds"],
            )
        processor_configuration = predictor.processor_configuration
    finally:
        predictor.close()
    return rows, _variant_result(
        variant,
        rows,
        processor_configuration,
        prediction_path,
    )


def select_winning_highres_variant(results: dict[str, Any]) -> str:
    """Rank high-resolution sizes by exact, similarity, validity, then latency."""
    candidates = [
        name
        for name in ("highres_1024_256", "highres_1536_256", "default_2048_256")
        if name in results["variants"]
    ]
    if len(candidates) != 3:
        raise ValueError("All three high-resolution size variants must finish first")

    def rank(name: str) -> tuple[float, float, float, float]:
        metrics = results["variants"][name]["metrics"]
        return (
            float(metrics["macro_normalized_exact_match"]),
            float(metrics["macro_normalized_similarity"]),
            float(metrics["valid_json_rate"]),
            -float(metrics["average_inference_latency_seconds"]),
        )

    return max(candidates, key=rank)


def repetition_variant_for(
    winner_name: str,
    config: dict[str, Any],
) -> VariantConfiguration:
    winner = SIZE_VARIANTS[winner_name]
    return VariantConfiguration(
        name=f"{winner.name}_rep1p08",
        image_longest_edge=winner.image_longest_edge,
        max_new_tokens=winner.max_new_tokens,
        repetition_penalty=float(config["generation"]["repetition_penalty"]),
    )


def two_pass_variant_for(winner_name: str) -> VariantConfiguration:
    winner = SIZE_VARIANTS[winner_name]
    return VariantConfiguration(
        name=f"two_pass_{winner.image_longest_edge}_256",
        image_longest_edge=winner.image_longest_edge,
        max_new_tokens=winner.max_new_tokens,
        strategy="two_pass_top55_full_lower70",
    )


def record_variant_result(
    config: dict[str, Any],
    results: dict[str, Any],
    result: dict[str, Any],
) -> None:
    """Persist one completed result plus updated CSV and figure."""
    results["variants"][result["name"]] = result
    try:
        winner = select_winning_highres_variant(results)
    except ValueError:
        winner = None
    results["winning_high_resolution_size_variant"] = winner
    refresh_result_artifacts(config, results)


def refresh_result_artifacts(
    config: dict[str, Any],
    results: dict[str, Any],
) -> None:
    """Regenerate aggregate JSON, CSV, and figure without rerunning inference."""
    write_json(_output_root(config) / "results.json", results)
    _write_results_csv(config, results)
    _plot_results(config, results)


def _predictor(
    config: dict[str, Any],
    variant: VariantConfiguration,
) -> ReceiptKIEPredictor:
    model_config = deepcopy(config["model"])
    model_config.update(
        {
            "image_longest_edge": variant.image_longest_edge,
            "max_image_patch_edge": variant.max_image_patch_edge,
            "do_image_splitting": variant.do_image_splitting,
        }
    )
    return ReceiptKIEPredictor(
        model_config,
        config["paths"]["hf_cache"],
        adapter_path=project_path(config["paths"]["adapter_path"]),
    )


def _prediction_row(
    record: ReceiptRecord,
    prediction: dict[str, Any],
) -> dict[str, Any]:
    return {
        "sample_id": record.sample_id,
        "image_path": repository_relative(record.image_path),
        "ground_truth": record.target,
        **prediction,
    }


def _variant_result(
    variant: VariantConfiguration,
    rows: list[dict[str, Any]],
    processor_configuration: dict[str, Any],
    prediction_path: Path,
) -> dict[str, Any]:
    metrics = evaluate_predictions(rows)
    metrics.update(
        {
            "invalid_json_count": sum(not row["valid_json"] for row in rows),
            "generation_limit_hit_count": sum(
                bool(row["generation_limit_hit"]) for row in rows
            ),
            "repetition_failure_count": sum(
                bool(row["repetition_failure"]) for row in rows
            ),
            "average_visual_tile_count": statistics.mean(
                float(row["visual_tile_count"]) for row in rows
            ),
        }
    )
    subpass_validity = [
        validity
        for row in rows
        if (validity := row.get("subpass_valid_json")) is not None
    ]
    if subpass_validity:
        metrics["invalid_subpass_count"] = sum(
            not bool(value)
            for row in subpass_validity
            for value in row.values()
        )
    return {
        "name": variant.name,
        "configuration": asdict(variant),
        "processor_configuration": processor_configuration,
        "metrics": metrics,
        "predictions_path": repository_relative(prediction_path),
    }


def _write_results_csv(config: dict[str, Any], results: dict[str, Any]) -> None:
    output = _output_root(config) / "results.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = (
        "valid_json_rate",
        "company_accuracy",
        "address_similarity",
        "date_accuracy",
        "total_accuracy",
        "complete_record_normalized_exact_match",
        "macro_normalized_exact_match",
        "macro_normalized_similarity",
        "invalid_json_count",
        "generation_limit_hit_count",
        "repetition_failure_count",
        "average_inference_latency_seconds",
        "median_inference_latency_seconds",
        "peak_gpu_memory_mib",
        "average_visual_tile_count",
    )
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["variant", *columns])
        for name in _ordered_variant_names(results):
            result = results["variants"][name]
            metrics = result["metrics"]
            writer.writerow([name, *(metrics[column] for column in columns)])


def _plot_results(config: dict[str, Any], results: dict[str, Any]) -> None:
    if not results["variants"]:
        return
    names = _ordered_variant_names(results)
    labels = [name.replace("_", "\n") for name in names]
    metrics = [results["variants"][name]["metrics"] for name in names]
    x = list(range(len(names)))
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    width = 0.25
    for offset, key, label, color in (
        (-width, "valid_json_rate", "Valid JSON", "#4c78a8"),
        (0.0, "macro_normalized_exact_match", "Macro exact", "#f58518"),
        (width, "macro_normalized_similarity", "Macro similarity", "#54a24b"),
    ):
        axes[0].bar(
            [value + offset for value in x],
            [float(metric[key]) for metric in metrics],
            width=width,
            label=label,
            color=color,
        )
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Score")
    axes[0].set_title("Accuracy and JSON validity")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()
    latency_bars = axes[1].bar(
        x,
        [float(metric["average_inference_latency_seconds"]) for metric in metrics],
        color="#7f7f7f",
        label="Average latency",
    )
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("Seconds per receipt")
    axes[1].set_title("Latency, VRAM, and visual-tile trade-off")
    axes[1].grid(axis="y", alpha=0.25)
    memory_axis = axes[1].twinx()
    memory_axis.plot(
        x,
        [float(metric["peak_gpu_memory_mib"] or 0.0) for metric in metrics],
        marker="o",
        color="#b279a2",
        linewidth=2,
        label="Peak GPU memory",
    )
    memory_axis.set_ylabel("Peak allocated GPU memory (MiB)")
    for bar, metric in zip(latency_bars, metrics, strict=True):
        axes[1].annotate(
            f"{metric['average_visual_tile_count']:.1f} tiles",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    handles_a, labels_a = axes[1].get_legend_handles_labels()
    handles_b, labels_b = memory_axis.get_legend_handles_labels()
    axes[1].legend(handles_a + handles_b, labels_a + labels_b, loc="upper left")
    fig.suptitle("ReceiptKIE-VLM high-resolution inference ablation (30 fixed test receipts)")
    fig.tight_layout()
    output = project_path(config["paths"]["figure_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _output_root(config: dict[str, Any]) -> Path:
    return project_path(config["paths"]["output_root"])


def _ordered_variant_names(results: dict[str, Any]) -> list[str]:
    preferred = (
        "baseline_512_128",
        "token_control_512_256",
        "highres_1024_256",
        "highres_1536_256",
        "default_2048_256",
        "default_2048_256_rep1p08",
        "two_pass_2048_256",
    )
    variants = results["variants"]
    return [
        *[name for name in preferred if name in variants],
        *[name for name in variants if name not in preferred],
    ]


def _prediction_path(config: dict[str, Any], variant_name: str) -> Path:
    return _output_root(config) / "predictions" / f"{variant_name}.jsonl"


def _maximum_optional(*values: float | None) -> float | None:
    present = [float(value) for value in values if value is not None]
    return max(present) if present else None
