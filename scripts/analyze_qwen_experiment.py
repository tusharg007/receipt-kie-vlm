"""Verify paired cloud outputs before publishing retrospective accuracy claims."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from receipt_kie.metrics import evaluate_predictions, extract_json, normalize_field  # noqa: E402
from receipt_kie.prompts import CANONICAL_FIELDS  # noqa: E402
from receipt_kie.structured import strict_receipt  # noqa: E402


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]


def verify_raw_predictions(rows: list[dict]) -> None:
    """Do not let stale parsed fields or validity flags change reported accuracy."""
    for row in rows:
        parsed, _ = strict_receipt(row["raw_output"])
        if row.get("parsed_prediction") != parsed or row.get("valid_json") != (parsed is not None):
            raise ValueError(f"Saved parsing differs from raw output for {row['sample_id']}")


def comparison_rows(rows: list[dict]) -> tuple[list[dict], dict]:
    """Apply the production tolerant parser to every model; track strict schema separately."""
    result_rows, methods = [], Counter()
    strict_valid_count = 0
    for row in rows:
        strict_value, _ = strict_receipt(row["raw_output"])
        if strict_value is not None:
            value, method = strict_value, "strict_json"
            strict_valid_count += 1
        else:
            parsed = extract_json(row["raw_output"])
            value, method = (parsed.value, parsed.method) if parsed.valid else (None, "invalid")
        methods[method] += 1
        result_rows.append(
            {
                **row,
                "parsed_prediction": value,
                "strict_schema_valid": strict_value is not None,
                "valid_json": value is not None,
            }
        )
    count = len(rows)
    return result_rows, {
        "sample_count": count,
        "strict_json_valid_count": strict_valid_count,
        "strict_json_valid_rate": strict_valid_count / count if count else 0.0,
        "recovered_by_tolerant_parser": count - strict_valid_count - methods["invalid"],
        "parse_methods": dict(sorted(methods.items())),
    }


def verify_metric_report(actual: dict, reported: dict) -> None:
    if actual.keys() != reported.keys():
        raise ValueError("Reported metrics differ: keys")
    for key, value in actual.items():
        expected = reported[key]
        if isinstance(value, dict):
            verify_metric_report(value, expected)
        elif isinstance(value, float) and isinstance(expected, (float, int)):
            if not math.isclose(value, expected, rel_tol=1e-10, abs_tol=1e-10):
                raise ValueError(f"Reported metrics differ: {key}")
        elif value != expected:
            raise ValueError(f"Reported metrics differ: {key}")


def verify_alignment(variants: dict[str, list[dict]], hashes: dict[str, str]) -> list[str]:
    expected = set(hashes)
    indexed = {}
    for name, rows in variants.items():
        indexed[name] = {row["sample_id"]: row for row in rows}
        if len(rows) != len(indexed[name]) or set(indexed[name]) != expected:
            raise ValueError(f"{name}: duplicate, missing or additional evaluation IDs")
        for row in rows:
            if name != "v2" and row.get("image_sha256") != hashes[row["sample_id"]]:
                raise ValueError(f"{name}: image hash mismatch")
    names = list(indexed)
    for sample_id in sorted(expected):
        reference = indexed[names[0]][sample_id]["ground_truth"]
        for name in names[1:]:
            other = indexed[name][sample_id]["ground_truth"]
            if any(
                normalize_field(field, reference[field]) != normalize_field(field, other[field])
                for field in CANONICAL_FIELDS
            ):
                raise ValueError(f"{name}: reference annotation mismatch for {sample_id}")
    return sorted(expected)


def correct_vectors(rows: list[dict], ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    indexed = {row["sample_id"]: row for row in rows}
    correct = np.array(
        [
            [
                bool(indexed[sample_id].get("valid_json"))
                and normalize_field(
                    field, (indexed[sample_id].get("parsed_prediction") or {}).get(field, "")
                )
                == normalize_field(field, indexed[sample_id]["ground_truth"][field])
                for field in CANONICAL_FIELDS
            ]
            for sample_id in ids
        ]
    )
    return correct.mean(axis=1), correct.all(axis=1).astype(float)


def plot_metrics(metrics: dict, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    names = ["v2", "qwen_base", "qwen_qlora"]
    labels = ["Historical V2", "Qwen3-VL\nNF4 base", "Qwen3-VL\nQLoRA"]
    for axis, key, title in zip(
        axes,
        ["macro_normalized_exact_match", "complete_record_normalized_exact_match"],
        ["Macro field exact match", "Complete-record exact match"],
        strict=True,
    ):
        values = [100 * metrics[name][key] for name in names]
        bars = axis.bar(labels, values, color=["#718096", "#7098c3", "#43856a"])
        axis.bar_label(bars, labels=[f"{value:.1f}%" for value in values], padding=4)
        axis.set(ylim=(0, 106), ylabel="Normalized accuracy (%)", title=title)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Receipt extraction: 239-record retrospective comparison", fontsize=13)
    figure.text(
        0.5,
        0.025,
        "Fields use the same tolerant parser; strict JSON validity is reported separately. "
        "V2 preprocessing differs. Paired intervals in report.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 0.95))
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def render_examples(variants: dict, ids: list[str], destination: Path) -> Path:
    from build_structured_comparison import render_pair

    indexed = {name: {row["sample_id"]: row for row in rows} for name, rows in variants.items()}
    selected, merchants = [], set()
    for sample_id in ids:
        merchant = normalize_field("company", indexed["v2"][sample_id]["ground_truth"]["company"])
        if merchant not in merchants:
            selected.append(sample_id)
            merchants.add(merchant)
        if len(selected) == 3:
            break
    destination.mkdir(parents=True, exist_ok=True)
    gallery = [
        "# Bill-by-bill Qwen comparison",
        "",
        "Selection: first three distinct normalized merchants in lexicographic evaluation-ID "
        "order, independent of model outcomes. These examples are not an accuracy estimate.",
        "",
        "Before = pinned NF4 base; after = selected QLoRA adapter. The image, prompt, "
        "pixel cap and decoding are identical. Colors compare normalized "
        "values with dataset annotations; annotations can contain errors.",
        "",
    ]
    manifest = []
    for number, sample_id in enumerate(selected, 1):
        image_path = indexed["v2"][sample_id]["image_path"]
        before = {**indexed["qwen_base"][sample_id], "image_path": image_path}
        after = {**indexed["qwen_qlora"][sample_id], "image_path": image_path}
        if hashlib.sha256((ROOT / image_path).read_bytes()).hexdigest() != before["image_sha256"]:
            raise ValueError(f"Comparison image changed: {sample_id}")
        output = destination / f"{number:02d}_{sample_id}.png"
        manifest.append(
            render_pair(
                before,
                after,
                output,
                number,
                len(selected),
                comparison_title="Before and after QLoRA fine-tuning",
                context="Same image, NF4, prompt, pixel cap and greedy decoding",
                before_heading="BEFORE\nQwen3-VL NF4 base",
                after_heading="AFTER\nQwen3-VL QLoRA",
                note="Selected validation-loss checkpoint; examples chosen by merchant/ID "
                "order, independently of whether fine-tuning helped.",
            )
        )
        gallery.extend([f"## Bill {number}: {sample_id}", "", f"![Comparison]({output.name})", ""])
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", "utf-8")
    report = destination / "README.md"
    report.write_text("\n".join(gallery), "utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "artifacts/experiments/qwen3_vl/full_run"
    )
    parser.add_argument("--report", type=Path, default=ROOT / "QWEN3_VL_RESULTS.md")
    parser.add_argument(
        "--render-examples", action="store_true", help="Requires local source images"
    )
    args = parser.parse_args()
    summary = json.loads((args.run_dir / "training_summary.json").read_text("utf-8"))
    if summary.get("status") != "evaluation_complete" or summary.get("global_steps", 0) <= 2:
        raise ValueError(
            "Require completed full training and evaluation, not a smoke or partial run"
        )
    hashes = json.loads(
        (ROOT / "artifacts/experiments/receiptops/evaluation_image_sha256.json").read_text("utf-8")
    )
    v2_path = ROOT / "artifacts/experiments/highres_training_v2/v2_predictions.jsonl"
    variants = {
        "v2": [row for row in read_rows(v2_path) if row["sample_id"] in hashes],
        "qwen_base": read_rows(args.run_dir / "base_predictions.jsonl"),
        "qwen_qlora": read_rows(args.run_dir / "adapted_predictions.jsonl"),
    }
    for name in ("qwen_base", "qwen_qlora"):
        verify_raw_predictions(variants[name])
    ids = verify_alignment(variants, hashes)
    frozen = json.loads((args.run_dir / "frozen_selection.json").read_text("utf-8"))
    if frozen.get("deployment_allowed") is not False:
        raise ValueError("A validation choice is not deployment approval")
    strict_metrics = {name: evaluate_predictions(rows) for name, rows in variants.items()}
    cloud_report = json.loads((args.run_dir / "comparison.json").read_text("utf-8"))
    for local, cloud in (("qwen_base", "qwen_quantized_base"), ("qwen_qlora", "qwen_qlora")):
        verify_metric_report(strict_metrics[local], cloud_report[cloud])
    comparison = {}
    schema_compliance = {}
    for name, rows in variants.items():
        comparison[name], schema_compliance[name] = comparison_rows(rows)
    metrics = {name: evaluate_predictions(rows) for name, rows in comparison.items()}
    previous_report = args.run_dir / "verified_comparison.json"
    if previous_report.is_file():
        verify_metric_report(metrics, json.loads(previous_report.read_text("utf-8"))["metrics"])
    manifest = json.loads((args.run_dir / "split_manifest.json").read_text("utf-8"))
    validation_rows = {
        "base": read_rows(args.run_dir / "base_validation_predictions.jsonl"),
        "adapted": read_rows(args.run_dir / "validation_predictions.jsonl"),
    }
    for rows in validation_rows.values():
        verify_raw_predictions(rows)
    validation_hashes = {row["sample_id"]: row["sha256"] for row in manifest["validation"]}
    references = [
        {
            "sample_id": row["sample_id"],
            "image_sha256": row["sha256"],
            "ground_truth": row["target"],
        }
        for row in manifest["validation"]
    ]
    verify_alignment({"reference": references, **validation_rows}, validation_hashes)
    validation_metrics = {
        name: evaluate_predictions(rows) for name, rows in validation_rows.items()
    }
    verify_metric_report(
        validation_metrics,
        json.loads((args.run_dir / "validation_metrics.json").read_text("utf-8")),
    )
    base, adapted = validation_metrics["base"], validation_metrics["adapted"]
    expected_selection = (
        "qwen_qlora"
        if adapted["macro_normalized_exact_match"] > base["macro_normalized_exact_match"]
        and adapted["complete_record_normalized_exact_match"]
        >= base["complete_record_normalized_exact_match"]
        else "qwen_quantized_base"
    )
    if frozen["selected"] != expected_selection:
        raise ValueError("Frozen selection disagrees with validation-only criterion")
    vectors = {name: correct_vectors(rows, ids) for name, rows in comparison.items()}
    rng = np.random.default_rng(42)
    resamples = rng.integers(0, len(ids), size=(2000, len(ids)))
    references = {row["sample_id"]: row["ground_truth"] for row in variants["v2"]}
    merchant_keys = [
        normalize_field("company", references[sample_id]["company"]) for sample_id in ids
    ]
    unique_merchants = sorted(set(merchant_keys))
    membership = np.array(
        [[key == merchant for key in merchant_keys] for merchant in unique_merchants]
    )
    merchant_sizes = membership.sum(axis=1)
    cluster_draws = rng.integers(0, len(unique_merchants), size=(2000, len(unique_merchants)))
    intervals = {}
    for before, after in (("v2", "qwen_base"), ("v2", "qwen_qlora"), ("qwen_base", "qwen_qlora")):
        intervals[f"{after}_minus_{before}"] = {}
        for index, metric in enumerate(("macro_exact", "complete_record_exact")):
            difference = vectors[after][index] - vectors[before][index]
            lower, upper = np.quantile(difference[resamples].mean(axis=1), [0.025, 0.975])
            group_sums = membership @ difference
            cluster_gains = group_sums[cluster_draws].sum(axis=1) / merchant_sizes[
                cluster_draws
            ].sum(axis=1)
            cluster_lower, cluster_upper = np.quantile(cluster_gains, [0.025, 0.975])
            intervals[f"{after}_minus_{before}"][metric] = {
                "point_difference": float(difference.mean()),
                "paired_bootstrap_95_percent": [float(lower), float(upper)],
                "paired_merchant_cluster_bootstrap_95_percent": [
                    float(cluster_lower),
                    float(cluster_upper),
                ],
            }
    report = {
        "scope": "239-record content-disjoint retrospective comparison, not fresh test data",
        "sample_count": len(ids),
        "normalized_merchant_count": len(unique_merchants),
        "equal_merchant_weighted_macro_exact": {
            name: float(((membership @ value[0]) / merchant_sizes).mean())
            for name, value in vectors.items()
        },
        "schema_compliance": schema_compliance,
        "field_accuracy_parser": (
            "Strict four-field JSON first; then the existing tolerant receipt parser. "
            "Unparseable outputs remain incorrect across all fields."
        ),
        "metrics": metrics,
        "paired_bootstrap": intervals,
        "bootstrap_resamples": 2000,
        "bootstrap_seed": 42,
        "validation_choice": frozen,
        "training_summary": summary,
        "limitations": [
            "Exact-content checks do not rule out perceptual duplicates or pretraining exposure",
            "Historical V2 used different image preprocessing and parsing; timings are not paired",
            "Paired Qwen base/adapted predictions use the same NF4, images, prompt and decoding",
            "No default-model promotion, fresh-data accuracy or production scale claim",
            "Image bootstrap assumes independent records; merchant-cluster intervals are "
            "also reported, but normalized names may leave merchant aliases ungrouped",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "training_summary.json",
        "split_manifest.json",
        "validation_metrics.json",
        "frozen_selection.json",
        "comparison.json",
        "base_predictions.jsonl",
        "adapted_predictions.jsonl",
        "validation_predictions.jsonl",
        "base_validation_predictions.jsonl",
    ):
        shutil.copy2(args.run_dir / name, args.output_dir / name)
    (args.output_dir / "verified_comparison.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    plot_metrics(metrics, args.output_dir / "accuracy_comparison.png")
    labels = {
        "v2": "Historical V2",
        "qwen_base": "Qwen3-VL NF4 base",
        "qwen_qlora": "Qwen3-VL QLoRA",
    }
    lines = [
        "# Qwen3-VL: verified Kaggle experiment",
        "",
        report["scope"] + ". All invalid predictions remain in the denominator.",
        "",
        "| Model | Macro field exact | Complete-record exact | Parseable | Strict JSON |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, values in metrics.items():
        lines.append(
            f"| {labels[name]} | {values['macro_normalized_exact_match']:.1%} | "
            f"{values['complete_record_normalized_exact_match']:.1%} | "
            f"{values['valid_json_rate']:.1%} | "
            f"{schema_compliance[name]['strict_json_valid_rate']:.1%} |"
        )
    lines.extend(
        [
            "",
            f"Validation-only choice before evaluation: `{frozen['selected']}`.",
            "",
            "This does not automatically replace V2 or approve the Qwen backend for deployment.",
            "",
            "Field accuracy uses the same tolerant parser for every model. Strict JSON "
            "compliance is separate; Qwen base code-fence recovery does not count as strict JSON.",
            "",
            f"Training: {summary['global_steps']} optimizer steps on {summary['gpu']}, "
            f"{summary['dtype']}, peak allocated {summary['peak_allocated_gpu_mib']:.0f} MiB.",
            "",
            f"Exported adapter: `{Path(summary['best_checkpoint']).name}`, selected by "
            "minimum validation loss; total run steps are not the selected checkpoint step.",
            "",
            "The code and weights are pinned; the training summary records software versions.",
            "",
            "## Paired uncertainty",
            "",
        ]
    )
    for name, values in intervals.items():
        item = values["macro_exact"]
        lower, upper = item["paired_bootstrap_95_percent"]
        cluster_lower, cluster_upper = item["paired_merchant_cluster_bootstrap_95_percent"]
        lines.append(
            f"- `{name}` macro gain: {item['point_difference'] * 100:+.2f} pp; "
            f"paired image-level 95% interval [{lower * 100:+.2f}, {upper * 100:+.2f}] pp; "
            f"merchant-cluster interval [{cluster_lower * 100:+.2f}, "
            f"{cluster_upper * 100:+.2f}] pp."
        )
    lines.extend(
        [
            "",
            f"The evaluation contains {len(unique_merchants)} normalized merchant groups.",
            "Equal-merchant-weighted macro exact (each observed merchant gets equal weight):",
        ]
    )
    for name, value in report["equal_merchant_weighted_macro_exact"].items():
        lines.append(f"- {labels[name]}: {value:.1%}.")
    lines.extend(["", "## Strict output format", ""])
    for name, details in schema_compliance.items():
        recovered = details["recovered_by_tolerant_parser"]
        lines.append(
            f"- {labels[name]}: {details['strict_json_valid_count']}/{details['sample_count']} "
            f"strict JSON; {recovered} additional output(s) parsed by the tolerant parser; "
            f"methods: {details['parse_methods']}."
        )
    lines.extend(["", "## Limits", ""] + [f"- {item}." for item in report["limitations"]])
    chart_link = Path(
        os.path.relpath(args.output_dir / "accuracy_comparison.png", args.report.parent)
    ).as_posix()
    lines.extend(["", f"![Retrospective accuracy comparison]({chart_link})", ""])
    if args.render_examples:
        gallery = render_examples(comparison, ids, args.output_dir / "image_comparisons")
        gallery_link = Path(os.path.relpath(gallery, args.report.parent)).as_posix()
        lines.extend([f"[Bill-by-bill before/after examples]({gallery_link})", ""])
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"metrics": metrics, "paired_bootstrap": intervals}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
