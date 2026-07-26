"""Frozen never-evaluated holdout comparison and V2 experiment reporting."""

from __future__ import annotations

import csv
import json
import logging
import statistics
import textwrap
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from rapidfuzz.fuzz import ratio

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from receipt_kie.config import load_config
from receipt_kie.dataset import ReceiptRecord, load_records
from receipt_kie.highres_training import sha256_file
from receipt_kie.inference import ReceiptKIEPredictor
from receipt_kie.metrics import (
    CANONICAL_FIELDS,
    evaluate_predictions,
    has_repetition_failure,
    normalize_field,
)
from receipt_kie.utils import (
    project_path,
    repository_relative,
    seed_everything,
    setup_logging,
    write_json,
)

LOGGER = logging.getLogger(__name__)
VARIANT_ADAPTERS = {
    "base": None,
    "v1": "models/receipt-kie-lora",
    "v2": "models/receipt-kie-lora-v2-highres",
}
CI_METRICS = (
    "valid_json_rate",
    "company_accuracy",
    "address_similarity",
    "date_accuracy",
    "total_accuracy",
    "complete_record_normalized_exact_match",
    "macro_normalized_exact_match",
    "macro_normalized_similarity",
)


def run_frozen_holdout_evaluation(config_path: str | Path) -> dict[str, Any]:
    """Evaluate Base, V1, and validation-selected V2 on never-seen test IDs."""
    config = load_config(config_path)
    setup_logging(config["paths"]["holdout_log_file"])
    seed = int(config["project"]["seed"])
    seed_everything(seed)
    selection = json.loads(
        project_path(config["paths"]["frozen_selection_path"]).read_text(
            encoding="utf-8"
        )
    )
    assert_frozen_configuration(config, selection)
    records, holdout_ids = load_never_evaluated_records(config)
    write_json(config["paths"]["final_holdout_ids_path"], holdout_ids)
    rows_by_variant: dict[str, list[dict[str, Any]]] = {}
    for variant, adapter_path in VARIANT_ADAPTERS.items():
        predictor = ReceiptKIEPredictor(
            deepcopy(config["model"]),
            config["paths"]["hf_cache"],
            adapter_path=(
                None if adapter_path is None else project_path(adapter_path)
            ),
        )
        try:
            rows_by_variant[variant] = run_or_resume_holdout_variant(
                config,
                variant,
                predictor,
                records,
                project_path(config["paths"][f"{variant}_predictions_path"]),
            )
        finally:
            predictor.close()
    results = build_holdout_results(config, selection, rows_by_variant)
    write_json(config["paths"]["holdout_results_path"], results)
    write_results_csv(config, results)
    plot_comparison(config, results)
    qualitative = build_qualitative_panels(config, rows_by_variant)
    results["qualitative_results"] = qualitative
    write_json(config["paths"]["holdout_results_path"], results)
    write_report(config, results)
    return results


def assert_frozen_configuration(
    config: dict[str, Any],
    selection: dict[str, Any],
) -> None:
    """Fail closed if final-evaluation configuration differs from selection."""
    checks = {
        "resolution": int(config["model"]["image_longest_edge"]),
        "max_image_patch_edge": int(config["model"]["max_image_patch_edge"]),
        "do_image_splitting": bool(config["model"]["do_image_splitting"]),
        "max_new_tokens": int(config["generation"]["max_new_tokens"]),
        "generation_policy": str(config["generation"]["policy"]),
        "repetition_penalty": float(config["generation"]["repetition_penalty"]),
    }
    expected = {
        key: selection[key]
        for key in (
            "resolution",
            "max_image_patch_edge",
            "do_image_splitting",
            "max_new_tokens",
            "generation_policy",
            "repetition_penalty",
        )
    }
    if checks != expected:
        raise ValueError(
            f"Final configuration differs from frozen selection: {checks} != {expected}"
        )


def load_never_evaluated_records(
    config: dict[str, Any],
) -> tuple[list[ReceiptRecord], list[str]]:
    """Resolve all and only IDs declared never evaluated in the usage manifest."""
    manifest = json.loads(
        project_path(config["paths"]["test_usage_manifest"]).read_text(
            encoding="utf-8"
        )
    )
    holdout_ids = [
        str(value) for value in manifest["remaining_never_evaluated_ids"]
    ]
    if len(holdout_ids) != manifest["counts"]["remaining_never_evaluated_ids"]:
        raise ValueError("Holdout count does not match usage manifest")
    records = load_records(
        config["paths"]["dataset_root"],
        "test",
        limit=None,
        seed=int(config["project"]["seed"]),
    )
    by_id = {record.sample_id: record for record in records}
    missing = sorted(set(holdout_ids) - set(by_id))
    if missing:
        raise ValueError(f"Never-evaluated IDs missing from official test: {missing}")
    return [by_id[sample_id] for sample_id in holdout_ids], holdout_ids


def run_or_resume_holdout_variant(
    config: dict[str, Any],
    variant: str,
    predictor: ReceiptKIEPredictor,
    records: list[ReceiptRecord],
    output_path: Path,
) -> list[dict[str, Any]]:
    """Checkpoint one frozen variant after every unseen receipt."""
    if output_path.is_file():
        rows = [
            json.loads(line)
            for line in output_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        rows = []
    expected_prefix = [record.sample_id for record in records[: len(rows)]]
    if [row["sample_id"] for row in rows] != expected_prefix:
        raise ValueError(f"Holdout resume IDs do not match: {output_path}")
    for index, record in enumerate(records[len(rows) :], start=len(rows) + 1):
        prediction = predictor.predict(
            record.image_path,
            max_new_tokens=int(config["generation"]["max_new_tokens"]),
            do_sample=False,
            repetition_penalty=float(
                config["generation"]["repetition_penalty"]
            ),
        )
        prediction["repetition_failure"] = has_repetition_failure(
            prediction["raw_output"]
        )
        row = {
            "sample_id": record.sample_id,
            "image_path": repository_relative(record.image_path),
            "ground_truth": record.target,
            **prediction,
        }
        append_jsonl_row_with_retry(output_path, row)
        rows.append(row)
        LOGGER.info(
            "holdout variant=%s sample=%d/%d id=%s valid=%s limit=%s "
            "repetition=%s latency=%.3f",
            variant,
            index,
            len(records),
            record.sample_id,
            prediction["valid_json"],
            prediction["generation_limit_hit"],
            prediction["repetition_failure"],
            prediction["latency_seconds"],
        )
    return rows


def append_jsonl_row_with_retry(
    output_path: Path,
    row: dict[str, Any],
    attempts: int = 5,
) -> None:
    """Append one checkpoint row and tolerate transient Windows file-open errors."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        row,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    for attempt in range(1, attempts + 1):
        try:
            with output_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(serialized + "\n")
                handle.flush()
            return
        except OSError:
            if _last_jsonl_sample_id(output_path) == row["sample_id"]:
                return
            if attempt == attempts:
                raise
            LOGGER.warning(
                "Transient JSONL append failure path=%s sample=%s attempt=%d/%d",
                output_path,
                row["sample_id"],
                attempt,
                attempts,
            )
            time.sleep(0.25 * attempt)


def _last_jsonl_sample_id(path: Path) -> str | None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        if not lines:
            return None
        return str(json.loads(lines[-1])["sample_id"])
    except (OSError, json.JSONDecodeError, KeyError):
        return None


def build_holdout_results(
    config: dict[str, Any],
    selection: dict[str, Any],
    rows_by_variant: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Calculate exact metrics, bootstrap intervals, paired deltas, and gate."""
    variants = {}
    for index, (name, rows) in enumerate(rows_by_variant.items()):
        metrics = extended_metrics(rows)
        variants[name] = {
            "adapter_path": VARIANT_ADAPTERS[name],
            "adapter_sha256": (
                None
                if VARIANT_ADAPTERS[name] is None
                else sha256_file(
                    project_path(VARIANT_ADAPTERS[name])
                    / "adapter_model.safetensors"
                )
            ),
            "predictions_path": config["paths"][f"{name}_predictions_path"],
            "metrics": metrics,
            "bootstrap_95_percent_ci": bootstrap_metrics(
                rows,
                int(config["bootstrap"]["resamples"]),
                int(config["project"]["seed"]) + index,
            ),
        }
    paired = paired_bootstrap_deltas(
        rows_by_variant["v1"],
        rows_by_variant["v2"],
        int(config["bootstrap"]["resamples"]),
        int(config["project"]["seed"]) + 100,
    )
    gate = success_gate(variants["v1"]["metrics"], variants["v2"]["metrics"])
    return {
        "experiment": "high_resolution_training_v2_final_unseen_holdout",
        "source_split": "official_test",
        "holdout_definition": (
            "Every official-test ID absent from V1 evaluation, robustness, "
            "qualitative selection, and high-resolution inference development."
        ),
        "sample_count": len(rows_by_variant["v2"]),
        "frozen_selection": selection,
        "identical_evaluation_controls": {
            "resolution": selection["resolution"],
            "max_image_patch_edge": selection["max_image_patch_edge"],
            "do_image_splitting": selection["do_image_splitting"],
            "max_new_tokens": selection["max_new_tokens"],
            "generation_policy": selection["generation_policy"],
            "repetition_penalty": selection["repetition_penalty"],
            "do_sample": False,
            "prompt": "receipt_kie.prompts.build_messages",
            "parser": selection["parsing_behavior"],
            "metric_code": "receipt_kie.metrics.evaluate_predictions",
        },
        "variants": variants,
        "paired_v2_minus_v1_bootstrap_95_percent_ci": paired,
        "success_gate": gate,
    }


def extended_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = evaluate_predictions(rows)
    sample_count = len(rows)
    metrics.update(
        {
            "generation_limit_hit_count": sum(
                bool(row["generation_limit_hit"]) for row in rows
            ),
            "repetition_failure_count": sum(
                bool(row["repetition_failure"]) for row in rows
            ),
            "invalid_json_count": sum(not row["valid_json"] for row in rows),
            "average_visual_tile_count": statistics.mean(
                float(row["visual_tile_count"]) for row in rows
            ),
        }
    )
    metrics["generation_limit_rate"] = (
        metrics["generation_limit_hit_count"] / sample_count
    )
    metrics["repetition_failure_rate"] = (
        metrics["repetition_failure_count"] / sample_count
    )
    return metrics


def bootstrap_metrics(
    rows: list[dict[str, Any]],
    resamples: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    samples = {name: [] for name in CI_METRICS}
    for _ in range(resamples):
        indices = rng.integers(0, len(rows), size=len(rows))
        metrics = evaluate_predictions([rows[int(index)] for index in indices])
        for name in CI_METRICS:
            samples[name].append(float(metrics[name]))
    return {
        name: {
            "low": float(np.percentile(values, 2.5)),
            "high": float(np.percentile(values, 97.5)),
        }
        for name, values in samples.items()
    }


def paired_bootstrap_deltas(
    v1_rows: list[dict[str, Any]],
    v2_rows: list[dict[str, Any]],
    resamples: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    if [row["sample_id"] for row in v1_rows] != [
        row["sample_id"] for row in v2_rows
    ]:
        raise ValueError("Paired bootstrap requires identical ordered IDs")
    rng = np.random.default_rng(seed)
    values = {name: [] for name in CI_METRICS}
    for _ in range(resamples):
        indices = rng.integers(0, len(v1_rows), size=len(v1_rows))
        first = evaluate_predictions([v1_rows[int(index)] for index in indices])
        second = evaluate_predictions([v2_rows[int(index)] for index in indices])
        for name in CI_METRICS:
            values[name].append(float(second[name]) - float(first[name]))
    return {
        name: {
            "point_delta": float(
                evaluate_predictions(v2_rows)[name]
                - evaluate_predictions(v1_rows)[name]
            ),
            "low": float(np.percentile(deltas, 2.5)),
            "high": float(np.percentile(deltas, 97.5)),
        }
        for name, deltas in values.items()
    }


def success_gate(
    v1_metrics: dict[str, Any],
    v2_metrics: dict[str, Any],
) -> dict[str, Any]:
    deltas = {
        "macro_normalized_exact_match": float(
            v2_metrics["macro_normalized_exact_match"]
            - v1_metrics["macro_normalized_exact_match"]
        ),
        "complete_record_normalized_exact_match": float(
            v2_metrics["complete_record_normalized_exact_match"]
            - v1_metrics["complete_record_normalized_exact_match"]
        ),
        "address_similarity": float(
            v2_metrics["address_similarity"] - v1_metrics["address_similarity"]
        ),
        "valid_json_rate": float(
            v2_metrics["valid_json_rate"] - v1_metrics["valid_json_rate"]
        ),
    }
    epsilon = 1e-12
    improvement = (
        deltas["macro_normalized_exact_match"] + epsilon >= 0.05
        or deltas["complete_record_normalized_exact_match"] + epsilon >= 0.03
        or deltas["address_similarity"] + epsilon >= 0.05
    )
    validity_ok = deltas["valid_json_rate"] + epsilon >= -0.02
    passed = improvement and validity_ok
    return {
        "passed": passed,
        "thresholds": {
            "macro_normalized_exact_match": 0.05,
            "complete_record_normalized_exact_match": 0.03,
            "address_similarity": 0.05,
            "minimum_valid_json_delta": -0.02,
        },
        "v2_minus_v1": deltas,
        "recommendation_if_failed": (
            None
            if passed
            else "Preserve V1 and use an OCR + KIE multi-task curriculum for Iteration 3."
        ),
    }


def write_results_csv(config: dict[str, Any], results: dict[str, Any]) -> None:
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
    path = project_path(config["paths"]["holdout_results_csv_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["variant", *columns])
        for name in ("base", "v1", "v2"):
            metrics = results["variants"][name]["metrics"]
            writer.writerow([name, *(metrics[column] for column in columns)])


def plot_comparison(config: dict[str, Any], results: dict[str, Any]) -> None:
    variants = ("base", "v1", "v2")
    labels = ("Base", "V1", "V2 high-res")
    metric_specs = (
        ("company_accuracy", "Company exact"),
        ("address_similarity", "Address similarity"),
        ("date_accuracy", "Date exact"),
        ("total_accuracy", "Total exact"),
        ("complete_record_normalized_exact_match", "Complete exact"),
        ("macro_normalized_exact_match", "Macro exact"),
    )
    x = np.arange(len(metric_specs))
    width = 0.25
    fig, axis = plt.subplots(figsize=(12, 6.5))
    for offset, variant, label, color in zip(
        (-width, 0.0, width),
        variants,
        labels,
        ("#4c78a8", "#f58518", "#54a24b"),
        strict=True,
    ):
        values = [
            float(results["variants"][variant]["metrics"][key])
            for key, _ in metric_specs
        ]
        bars = axis.bar(x + offset, values, width, label=label, color=color)
        for bar, value in zip(bars, values, strict=True):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.012,
                f"{value:.0%}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    axis.set_xticks(x, [label for _, label in metric_specs])
    axis.set_ylim(0, 1.08)
    axis.set_ylabel("Score")
    axis.set_title(
        f"Frozen high-resolution comparison on {results['sample_count']} "
        "never-evaluated receipts"
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    path = project_path(config["paths"]["comparison_figure_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def build_qualitative_panels(
    config: dict[str, Any],
    rows_by_variant: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    v1_by_id = {row["sample_id"]: row for row in rows_by_variant["v1"]}
    v2_by_id = {row["sample_id"]: row for row in rows_by_variant["v2"]}
    ordered_ids = [row["sample_id"] for row in rows_by_variant["v2"]]
    selections = {
        "v2_complete_record_success": next(
            sample_id
            for sample_id in ordered_ids
            if complete_record_match(v2_by_id[sample_id])
        ),
        "v2_improvement_over_v1": next(
            sample_id
            for sample_id in ordered_ids
            if record_exact_count(v2_by_id[sample_id])
            > record_exact_count(v1_by_id[sample_id])
        ),
        "v2_failure_or_regression": next(
            (
                sample_id
                for sample_id in ordered_ids
                if record_exact_count(v2_by_id[sample_id])
                < record_exact_count(v1_by_id[sample_id])
            ),
            next(
                sample_id
                for sample_id in ordered_ids
                if not complete_record_match(v2_by_id[sample_id])
            ),
        ),
    }
    output_root = project_path(config["paths"]["qualitative_figure_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for category, sample_id in selections.items():
        output_path = output_root / f"{category}.png"
        draw_qualitative_panel(
            category,
            v1_by_id[sample_id],
            v2_by_id[sample_id],
            output_path,
        )
        manifest[category] = {
            "sample_id": sample_id,
            "selection_rule": (
                "First sample ID in frozen holdout order satisfying category."
            ),
            "figure_path": repository_relative(output_path),
            "figure_sha256": sha256_file(output_path),
            "v1_exact_field_count": record_exact_count(v1_by_id[sample_id]),
            "v2_exact_field_count": record_exact_count(v2_by_id[sample_id]),
        }
    write_json(config["paths"]["qualitative_manifest_path"], manifest)
    return manifest


def draw_qualitative_panel(
    category: str,
    v1: dict[str, Any],
    v2: dict[str, Any],
    output_path: Path,
) -> None:
    image_path = project_path(v2["image_path"])
    with Image.open(image_path) as opened:
        image = opened.convert("RGB").copy()
    fig, axes = plt.subplots(1, 4, figsize=(18, 6))
    axes[0].imshow(image)
    axes[0].set_title(f"Receipt\nID: {v2['sample_id']}")
    axes[0].axis("off")
    draw_json_axis(axes[1], "Ground truth", v2["ground_truth"], None)
    draw_json_axis(
        axes[2],
        f"V1 prediction\nlatency {v1['latency_seconds']:.2f} s",
        v1.get("parsed_prediction") or {},
        v1["ground_truth"],
    )
    draw_json_axis(
        axes[3],
        f"V2 prediction\nlatency {v2['latency_seconds']:.2f} s",
        v2.get("parsed_prediction") or {},
        v2["ground_truth"],
    )
    fig.suptitle(category.replace("_", " ").title())
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def draw_json_axis(
    axis: Any,
    title: str,
    payload: dict[str, Any],
    truth: dict[str, Any] | None,
) -> None:
    axis.axis("off")
    axis.set_title(title)
    lines = []
    for field in CANONICAL_FIELDS:
        value = str(payload.get(field, ""))
        wrapped = "\n".join(textwrap.wrap(value, width=38)) or "<empty>"
        if truth is None:
            indicator = ""
        else:
            predicted_normalized = normalize_field(field, value)
            truth_normalized = normalize_field(field, str(truth[field]))
            exact = predicted_normalized == truth_normalized
            similarity = ratio(predicted_normalized, truth_normalized) / 100.0
            indicator = f" | exact={exact} sim={similarity:.2f}"
        lines.append(f"{field}{indicator}\n{wrapped}")
    axis.text(
        0.0,
        1.0,
        "\n\n".join(lines),
        ha="left",
        va="top",
        fontsize=9,
        family="monospace",
        transform=axis.transAxes,
    )


def record_exact_count(row: dict[str, Any]) -> int:
    prediction = row.get("parsed_prediction") or {}
    if not row.get("valid_json"):
        return 0
    return sum(
        normalize_field(field, str(prediction.get(field, "")))
        == normalize_field(field, str(row["ground_truth"][field]))
        for field in CANONICAL_FIELDS
    )


def complete_record_match(row: dict[str, Any]) -> bool:
    return record_exact_count(row) == len(CANONICAL_FIELDS)


def write_report(config: dict[str, Any], results: dict[str, Any]) -> None:
    smoke_2048 = json.loads(
        project_path(config["paths"]["smoke_2048_path"]).read_text(
            encoding="utf-8"
        )
    )
    smoke_1536 = json.loads(
        project_path(config["paths"]["smoke_1536_path"]).read_text(
            encoding="utf-8"
        )
    )
    training = json.loads(
        project_path(config["paths"]["training_summary_path"]).read_text(
            encoding="utf-8"
        )
    )
    selection = results["frozen_selection"]
    gate = results["success_gate"]
    lines = [
        "# High-Resolution Continued Training V2",
        "",
        "## Decision",
        "",
        (
            f"**Success gate: {'PASS' if gate['passed'] else 'FAIL'}.** "
            + (
                "V2 met the predeclared improvement threshold."
                if gate["passed"]
                else "V2 remains a preserved negative result; V1 is not replaced."
            )
        ),
        "",
        "The final comparison uses every official-test receipt that was absent from "
        "all prior V1, robustness, qualitative, and inference-development artifacts. "
        "No holdout result was inspected before resolution, checkpoint, token budget, "
        "penalty policy, and parsing behavior were frozen on validation data.",
        "",
        "## Memory gate",
        "",
        "| Resolution | Status | Peak allocated | Peak reserved | Seconds/step | Safe |",
        "|---:|---|---:|---:|---:|---|",
        (
            f"| 2048 | {smoke_2048['status']} | "
            f"{smoke_2048['peak_allocated_gpu_memory_mib']:.2f} MiB | "
            f"{smoke_2048['peak_reserved_gpu_memory_mib']:.2f} MiB | "
            f"{smoke_2048['seconds_per_optimization_step']:.2f} | "
            f"{smoke_2048['safe_memory']} |"
        ),
        (
            f"| 1536 | {smoke_1536['status']} | "
            f"{smoke_1536['peak_allocated_gpu_memory_mib']:.2f} MiB | "
            f"{smoke_1536['peak_reserved_gpu_memory_mib']:.2f} MiB | "
            f"{smoke_1536['seconds_per_optimization_step']:.2f} | "
            f"{smoke_1536['safe_memory']} |"
        ),
        "",
        "2048 completed but exceeded the predeclared 3.7 GiB allocated-memory "
        "threshold. Full training therefore used 1536 without quantization.",
        "",
        "## Training",
        "",
        f"- Source: committed V1 adapter `{training['adapter_source_sha256']}`.",
        f"- Exact split: {training['train_sample_count']} train / "
        f"{training['validation_sample_count']} validation receipts.",
        f"- Duration: {training['duration_seconds']:.2f} seconds "
        f"({training['duration_seconds'] / 60:.2f} minutes).",
        f"- Optimizer steps: {training['completed_optimization_steps']}.",
        f"- Peak allocated/reserved: "
        f"{training['peak_allocated_gpu_memory_mib']:.2f} / "
        f"{training['peak_reserved_gpu_memory_mib']:.2f} MiB.",
        f"- Validation loss: {training['validation_losses'][0]['eval_loss']:.6f} "
        f"to {training['validation_losses'][-1]['eval_loss']:.6f}.",
        "- Early stopping did not trigger because every validation-loss evaluation improved.",
        "",
        "## Frozen validation selection",
        "",
        f"- Checkpoint: `{selection['checkpoint']}`.",
        f"- Adapter SHA-256: `{selection['checkpoint_adapter_sha256']}`.",
        f"- Policy: `{selection['generation_policy']}`.",
        f"- Selection score: {selection['selection_score']:.6f}.",
        f"- Formula: `{selection['selection_score_formula']}`.",
        f"- Validation macro exact/similarity: "
        f"{selection['validation_metrics']['macro_normalized_exact_match']:.1%} / "
        f"{selection['validation_metrics']['macro_normalized_similarity']:.1%}.",
        f"- Validation complete exact / valid JSON: "
        f"{selection['validation_metrics']['complete_record_normalized_exact_match']:.1%} / "
        f"{selection['validation_metrics']['valid_json_rate']:.1%}.",
        "",
        "## Final never-evaluated holdout",
        "",
        f"Holdout size: **{results['sample_count']} receipts**.",
        "",
        "| Variant | Valid JSON | Company | Address sim. | Date | Total | "
        "Complete | Macro exact | Macro sim. | Avg latency |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("base", "v1", "v2"):
        metrics = results["variants"][name]["metrics"]
        lines.append(
            f"| {name.upper()} | {metrics['valid_json_rate']:.1%} | "
            f"{metrics['company_accuracy']:.1%} | "
            f"{metrics['address_similarity']:.1%} | "
            f"{metrics['date_accuracy']:.1%} | "
            f"{metrics['total_accuracy']:.1%} | "
            f"{metrics['complete_record_normalized_exact_match']:.1%} | "
            f"{metrics['macro_normalized_exact_match']:.1%} | "
            f"{metrics['macro_normalized_similarity']:.1%} | "
            f"{metrics['average_inference_latency_seconds']:.2f} s |"
        )
    deltas = gate["v2_minus_v1"]
    lines.extend(
        [
            "",
            f"![Base vs V1 vs V2]({config['paths']['comparison_figure_path']})",
            "",
            "## Success gate",
            "",
            f"- Macro exact delta: {deltas['macro_normalized_exact_match'] * 100:+.2f} pp.",
            f"- Complete-record delta: "
            f"{deltas['complete_record_normalized_exact_match'] * 100:+.2f} pp.",
            f"- Address-similarity delta: {deltas['address_similarity'] * 100:+.2f} pp.",
            f"- Valid-JSON delta: {deltas['valid_json_rate'] * 100:+.2f} pp.",
            f"- Gate result: **{'PASS' if gate['passed'] else 'FAIL'}**.",
            "",
            (
                "V2 may remain a separate candidate; do not replace V1 or public "
                "README claims until this branch is reviewed."
                if gate["passed"]
                else "Do not replace V1. Preserve this result and use an OCR + KIE "
                "multi-task curriculum for Iteration 3."
            ),
            "",
            "## Qualitative panels",
            "",
        ]
    )
    for category, item in results["qualitative_results"].items():
        lines.extend(
            [
                f"### {category.replace('_', ' ').title()}",
                "",
                f"Sample `{item['sample_id']}` selected by the documented first-match rule.",
                "",
                f"![{category}]({item['figure_path']})",
                "",
            ]
        )
    lines.extend(
        [
            "## Limitations",
            "",
            "- The experiment is specific to SROIE and a 256M-parameter VLM.",
            "- Bootstrap intervals quantify sampling uncertainty, not dataset shift.",
            "- Windows WDDM reserved-memory readings can exceed dedicated physical VRAM.",
            "- The public README and production V1 claims remain intentionally unchanged.",
            "",
        ]
    )
    report_path = project_path(config["paths"]["report_path"])
    report_path.write_text("\n".join(lines), encoding="utf-8")
