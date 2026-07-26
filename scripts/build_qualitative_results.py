"""Build auditable qualitative figures from tracked predictions and example images."""

from __future__ import annotations

import hashlib
import json
import sys
import textwrap
from pathlib import Path
from typing import Any

import matplotlib
from PIL import Image
from rapidfuzz.fuzz import ratio

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from receipt_kie.config import load_config  # noqa: E402
from receipt_kie.metrics import normalize_field  # noqa: E402
from receipt_kie.prompts import CANONICAL_FIELDS  # noqa: E402
from receipt_kie.utils import repository_relative, write_json  # noqa: E402

PREFERRED_IMPROVEMENT = "X51005230605"
PREFERRED_FAILURE = "X51005301666"
COLORS = {
    "exact": "#d9ead3",
    "similar": "#fff2cc",
    "incorrect": "#f4cccc",
    "unavailable": "#e7e6e6",
    "truth": "#d9eaf7",
}


def _read_json(path: str) -> Any:
    with (PROJECT_ROOT / path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    with (PROJECT_ROOT / path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _field_status(field: str, predicted: str, truth: str) -> tuple[str, float]:
    normalized_prediction = normalize_field(field, predicted)
    normalized_truth = normalize_field(field, truth)
    similarity = ratio(normalized_prediction, normalized_truth) / 100.0
    if normalized_prediction == normalized_truth:
        return "exact", 1.0
    if similarity >= 0.65:
        return "similar", similarity
    return "incorrect", similarity


def _record_score(row: dict[str, Any]) -> int:
    if not row.get("valid_json") or not isinstance(row.get("parsed_prediction"), dict):
        return 0
    truth = row["ground_truth"]
    prediction = row["parsed_prediction"]
    return sum(
        normalize_field(field, str(prediction.get(field, "")))
        == normalize_field(field, str(truth.get(field, "")))
        for field in CANONICAL_FIELDS
    )


def _is_improvement(base: dict[str, Any], lora: dict[str, Any]) -> bool:
    return (
        not base.get("valid_json")
        and bool(lora.get("valid_json"))
        and isinstance(lora.get("parsed_prediction"), dict)
        and _record_score(lora) >= 1
        and _record_score(lora) > _record_score(base)
    )


def _select_improvement(
    base_by_id: dict[str, dict[str, Any]],
    lora_by_id: dict[str, dict[str, Any]],
    index: dict[str, list[str]],
) -> str:
    indexed = index.get("lora_improves", [])
    candidates = [
        sample_id
        for sample_id in indexed
        if sample_id in base_by_id
        and sample_id in lora_by_id
        and _is_improvement(base_by_id[sample_id], lora_by_id[sample_id])
    ]
    if PREFERRED_IMPROVEMENT in candidates:
        return PREFERRED_IMPROVEMENT
    if candidates:
        return max(candidates, key=lambda sample_id: _record_score(lora_by_id[sample_id]))
    raise RuntimeError(
        "No indexed example has invalid base JSON, valid LoRA JSON, and an exact field gain"
    )


def _select_failure(
    lora_by_id: dict[str, dict[str, Any]],
    index: dict[str, list[str]],
) -> str:
    candidates = [
        sample_id
        for sample_id in index.get("lora_failure", [])
        if sample_id in lora_by_id and not lora_by_id[sample_id].get("valid_json")
    ]
    if PREFERRED_FAILURE in candidates:
        return PREFERRED_FAILURE
    if candidates:
        return max(candidates, key=lambda sample_id: len(lora_by_id[sample_id]["raw_output"]))
    raise RuntimeError("No indexed invalid LoRA generation is available for failure analysis")


def _example_image(sample_id: str, preferred_category: str) -> Path:
    example_dir = PROJECT_ROOT / "artifacts" / "predictions" / "examples"
    preferred = sorted(example_dir.glob(f"{preferred_category}__{sample_id}.*"))
    candidates = preferred or sorted(example_dir.glob(f"*__{sample_id}.*"))
    if not candidates:
        raise FileNotFoundError(
            f"No tracked example image found for sample {sample_id}; run build_report.py first"
        )
    return candidates[0]


def _draw_receipt(axis: Any, image_path: Path, sample_id: str) -> None:
    with Image.open(image_path) as opened:
        axis.imshow(opened.convert("RGB"))
    axis.set_title(f"Receipt image\n{sample_id}", fontweight="bold")
    axis.axis("off")


def _draw_fields(
    axis: Any,
    title: str,
    values: dict[str, Any],
    truth: dict[str, Any] | None = None,
    latency: float | None = None,
) -> None:
    axis.axis("off")
    axis.set_title(title, fontweight="bold")
    y_positions = (0.82, 0.60, 0.36, 0.14)
    for field, y_position in zip(CANONICAL_FIELDS, y_positions, strict=True):
        value = str(values.get(field, ""))
        if truth is None:
            marker = ""
            background = COLORS["truth"]
        else:
            status, similarity = _field_status(field, value, str(truth[field]))
            marker = {
                "exact": "✓ exact",
                "similar": f"~ {similarity:.0%} similar",
                "incorrect": f"✕ {similarity:.0%} similar",
            }[status]
            background = COLORS[status]
        wrapped = textwrap.fill(value or "(empty)", width=31)
        axis.text(
            0.03,
            y_position,
            f"{field}: {wrapped}\n{marker}".rstrip(),
            transform=axis.transAxes,
            va="top",
            fontsize=9.2,
            bbox={"boxstyle": "round,pad=0.45", "facecolor": background, "edgecolor": "none"},
        )
    if latency is not None:
        axis.text(
            0.03,
            0.01,
            f"Inference latency: {latency:.3f} s",
            transform=axis.transAxes,
            fontsize=8.5,
            color="#444444",
        )


def _draw_invalid_output(axis: Any, title: str, row: dict[str, Any]) -> None:
    axis.axis("off")
    axis.set_title(title, fontweight="bold")
    axis.text(
        0.03,
        0.91,
        "INVALID JSON",
        transform=axis.transAxes,
        fontsize=10,
        fontweight="bold",
        color="#990000",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": COLORS["unavailable"],
            "edgecolor": "#999999",
        },
    )
    raw = str(row.get("raw_output", ""))
    display = raw if len(raw) <= 900 else raw[:897] + "..."
    axis.text(
        0.03,
        0.81,
        textwrap.fill(display, width=40),
        transform=axis.transAxes,
        va="top",
        family="monospace",
        fontsize=8.4,
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": COLORS["unavailable"],
            "edgecolor": "none",
        },
    )
    axis.text(
        0.03,
        0.01,
        f"Fields unavailable · latency {float(row['latency_seconds']):.3f} s",
        transform=axis.transAxes,
        fontsize=8.5,
        color="#444444",
    )


def _build_improvement_figure(
    sample_id: str,
    image_path: Path,
    base: dict[str, Any],
    lora: dict[str, Any],
) -> Path:
    output = PROJECT_ROOT / "artifacts" / "figures" / "qualitative_lora_improvement.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(
        1,
        4,
        figsize=(20, 7.5),
        gridspec_kw={"width_ratios": [1.05, 1.25, 1.25, 1.25]},
    )
    _draw_receipt(axes[0], image_path, sample_id)
    _draw_fields(axes[1], "Ground-truth JSON", lora["ground_truth"])
    if base.get("valid_json") and isinstance(base.get("parsed_prediction"), dict):
        _draw_fields(
            axes[2],
            "Base-model output\nVALID JSON",
            base["parsed_prediction"],
            base["ground_truth"],
            float(base["latency_seconds"]),
        )
    else:
        _draw_invalid_output(axes[2], "Base-model output", base)
    _draw_fields(
        axes[3],
        "LoRA output\nVALID JSON",
        lora["parsed_prediction"],
        lora["ground_truth"],
        float(lora["latency_seconds"]),
    )
    figure.suptitle(
        "LoRA improves schema adherence, but this remains a partial extraction success",
        fontsize=15,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output


def _build_failure_figure(
    sample_id: str,
    image_path: Path,
    lora: dict[str, Any],
    max_new_tokens: int,
) -> Path:
    output = PROJECT_ROOT / "artifacts" / "figures" / "qualitative_failure_analysis.png"
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(17, 7.2),
        gridspec_kw={"width_ratios": [1.0, 1.2, 1.65]},
    )
    _draw_receipt(axes[0], image_path, sample_id)
    _draw_fields(axes[1], "Ground-truth JSON", lora["ground_truth"])
    axes[2].axis("off")
    axes[2].set_title("Truncated LoRA generation", fontweight="bold")
    raw = str(lora["raw_output"])
    display = raw if len(raw) <= 1250 else raw[:1247] + "..."
    axes[2].text(
        0.02,
        0.92,
        "INVALID JSON",
        transform=axes[2].transAxes,
        color="#990000",
        fontweight="bold",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": COLORS["unavailable"],
            "edgecolor": "#999999",
        },
    )
    axes[2].text(
        0.02,
        0.82,
        textwrap.fill(display, width=72),
        transform=axes[2].transAxes,
        va="top",
        family="monospace",
        fontsize=8.2,
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": COLORS["unavailable"],
            "edgecolor": "none",
        },
    )
    axes[2].text(
        0.02,
        0.05,
        "Failure mode: repeated address-like text reaches the generation limit,\n"
        "leaving an incomplete JSON object.\n"
        f"Generation limit: {max_new_tokens} tokens · "
        f"latency: {float(lora['latency_seconds']):.3f} s",
        transform=axes[2].transAxes,
        fontsize=9.5,
        color="#660000",
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": COLORS["incorrect"],
            "edgecolor": "none",
        },
    )
    figure.suptitle(
        "Failure analysis: repetition and long-address generation",
        fontsize=15,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output


def _manifest_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": row["sample_id"],
        "ground_truth": row["ground_truth"],
        "raw_output": row["raw_output"],
        "valid_json": row["valid_json"],
        "parsed_prediction": row["parsed_prediction"],
        "latency_seconds": row["latency_seconds"],
    }


def main() -> None:
    base_rows = _read_jsonl("artifacts/predictions/base_predictions.jsonl")
    lora_rows = _read_jsonl("artifacts/predictions/lora_predictions.jsonl")
    index = _read_json("artifacts/predictions/examples/index.json")
    config = load_config(PROJECT_ROOT / "configs" / "evaluate.yaml")
    base_by_id = {row["sample_id"]: row for row in base_rows}
    lora_by_id = {row["sample_id"]: row for row in lora_rows}
    improvement_id = _select_improvement(base_by_id, lora_by_id, index)
    failure_id = _select_failure(lora_by_id, index)
    improvement_image = _example_image(improvement_id, "lora_improves")
    failure_image = _example_image(failure_id, "lora_failure")
    improvement_figure = _build_improvement_figure(
        improvement_id,
        improvement_image,
        base_by_id[improvement_id],
        lora_by_id[improvement_id],
    )
    max_new_tokens = int(config["generation"]["max_new_tokens"])
    failure_figure = _build_failure_figure(
        failure_id,
        failure_image,
        lora_by_id[failure_id],
        max_new_tokens,
    )
    write_json(
        "artifacts/reports/qualitative_results_manifest.json",
        {
            "improvement": {
                "sample_id": improvement_id,
                "image_path": repository_relative(improvement_image),
                "image_sha256": _sha256(improvement_image),
                "base": _manifest_row(base_by_id[improvement_id]),
                "lora": _manifest_row(lora_by_id[improvement_id]),
                "figure_path": repository_relative(improvement_figure),
                "figure_sha256": _sha256(improvement_figure),
            },
            "failure": {
                "sample_id": failure_id,
                "image_path": repository_relative(failure_image),
                "image_sha256": _sha256(failure_image),
                "lora": _manifest_row(lora_by_id[failure_id]),
                "generation_max_new_tokens": max_new_tokens,
                "figure_path": repository_relative(failure_figure),
                "figure_sha256": _sha256(failure_figure),
            },
        },
    )
    print(f"Improvement sample: {improvement_id}")
    print(f"Failure sample: {failure_id}")
    print(f"Wrote: {repository_relative(improvement_figure)}")
    print(f"Wrote: {repository_relative(failure_figure)}")


if __name__ == "__main__":
    main()
