"""Build a portfolio gallery and outcome analysis from verified saved predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from analyze_qwen_experiment import (
    ROOT,
    comparison_rows,
    evaluate_predictions,
    normalize_field,
    read_rows,
    verify_alignment,
    verify_metric_report,
    verify_raw_predictions,
)
from build_structured_comparison import render_pair

from receipt_kie.prompts import CANONICAL_FIELDS

RUN = ROOT / "artifacts/experiments/qwen3_vl/full_run"
LABELS = {"v2": "Legacy V2", "qwen_base": "Qwen NF4 base", "qwen_qlora": "Latest Qwen QLoRA"}
CATEGORIES = (
    ("multi_field_recovery", "Several fields recovered"),
    ("total_correction", "Total corrected"),
    ("date_correction", "Date corrected"),
    ("company_correction", "Merchant corrected"),
    ("address_correction", "Address corrected"),
    ("unchanged_fields", "Extracted fields unchanged"),
    ("regression", "Fewer correct fields after the upgrade"),
)
ANNOTATION_NOTES = {
    "X51005705729": (
        "Visual annotation discrepancy: the printed address appears to say SECTION 26, "
        "while the reference says SECTION 25. The adapter's address matches the reference, "
        "but the base matches the visible text. Green cells mean reference agreement. "
        "The 89.26 to 94.60 total correction is visible on the bill. Original references "
        "are preserved and benchmark scores are unchanged."
    )
}


def field_matches(row: dict) -> list[bool]:
    prediction = row.get("parsed_prediction") or {}
    return [
        bool(row.get("valid_json"))
        and normalize_field(field, prediction.get(field, ""))
        == normalize_field(field, row["ground_truth"][field])
        for field in CANONICAL_FIELDS
    ]


def same_fields(before: dict, after: dict) -> bool:
    return all(
        normalize_field(field, (before.get("parsed_prediction") or {}).get(field, ""))
        == normalize_field(field, (after.get("parsed_prediction") or {}).get(field, ""))
        for field in CANONICAL_FIELDS
    )


def summarize_pair(before: dict[str, dict], after: dict[str, dict], ids: list[str]) -> dict:
    counts = dict.fromkeys(("improved", "equal_count", "regressed", "identical_fields"), 0)
    transitions = dict.fromkeys(
        ("complete_record_recovered", "complete_record_lost", "both_complete", "both_incomplete"), 0
    )
    for sample_id in ids:
        a, b = field_matches(before[sample_id]), field_matches(after[sample_id])
        key = "improved" if sum(b) > sum(a) else "regressed" if sum(b) < sum(a) else "equal_count"
        counts[key] += 1
        counts["identical_fields"] += int(same_fields(before[sample_id], after[sample_id]))
        transition = (
            "both_complete"
            if all(a) and all(b)
            else "complete_record_recovered"
            if all(b)
            else "complete_record_lost"
            if all(a)
            else "both_incomplete"
        )
        transitions[transition] += 1
    return {"sample_count": len(ids), **counts, **transitions}


def eligible(category: str, before: dict, after: dict) -> bool:
    a, b = field_matches(before), field_matches(after)
    if category == "multi_field_recovery":
        return all(b) and sum(a) <= 2
    if category == "regression":
        return sum(b) < sum(a)
    if category == "unchanged_fields":
        return same_fields(before, after)
    field = category.removesuffix("_correction")
    index = CANONICAL_FIELDS.index(field)
    return not a[index] and b[index] and sum(b) > sum(a)


def select_cases(indexed: dict[str, dict[str, dict]], ids: list[str]) -> list[dict]:
    """Use declared outcome buckets, sorted IDs, no repeated bill, and merchant diversity."""
    cases, used_ids = [], set()
    for before_name in ("v2", "qwen_base"):
        used_merchants = set()
        for category, title in CATEGORIES:
            candidates = [
                sample_id
                for sample_id in sorted(ids)
                if eligible(
                    category, indexed[before_name][sample_id], indexed["qwen_qlora"][sample_id]
                )
            ]
            available = [
                sample_id
                for sample_id in candidates
                if sample_id not in used_ids
                and normalize_field(
                    "company", indexed["qwen_qlora"][sample_id]["ground_truth"]["company"]
                )
                not in used_merchants
            ]
            if not available:
                raise ValueError(f"No distinct bill/merchant for {before_name}: {category}")
            sample_id = available[0]
            used_ids.add(sample_id)
            used_merchants.add(
                normalize_field(
                    "company", indexed["qwen_qlora"][sample_id]["ground_truth"]["company"]
                )
            )
            cases.append(
                {
                    "before_model": before_name,
                    "category": category,
                    "title": title,
                    "sample_id": sample_id,
                    "eligible_count_before_diversity_filter": len(candidates),
                }
            )
    return cases


def verify_gallery(destination: Path, summary: dict, indexed: dict, hashes: dict) -> None:
    published = json.loads((destination / "summary.json").read_text("utf-8"))
    if published != summary:
        raise ValueError("Published portfolio summary differs from saved prediction outcomes")
    manifest = json.loads((destination / "manifest.json").read_text("utf-8"))
    if len(manifest) != len(summary["selection"]):
        raise ValueError("Portfolio gallery has missing or additional examples")
    for case, panel in zip(summary["selection"], manifest, strict=True):
        sid = case["sample_id"]
        before, after = indexed[case["before_model"]][sid], indexed["qwen_qlora"][sid]
        expected = {
            **case,
            "before": before["parsed_prediction"] or {},
            "after": after["parsed_prediction"] or {},
            "reference": before["ground_truth"],
            "raw_output_before": before["raw_output"],
            "raw_output_after": after["raw_output"],
            "image_sha256": hashes[sid],
            "matched_fields_before": [
                f for f, match in zip(CANONICAL_FIELDS, field_matches(before), strict=True) if match
            ],
            "matched_fields_after": [
                f for f, match in zip(CANONICAL_FIELDS, field_matches(after), strict=True) if match
            ],
        }
        if sid in ANNOTATION_NOTES:
            expected["annotation_note"] = ANNOTATION_NOTES[sid]
        if any(panel.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Portfolio panel differs from source predictions: {sid}")
        actual_hash = hashlib.sha256((destination / panel["figure"]).read_bytes()).hexdigest()
        if actual_hash != panel["figure_sha256"]:
            raise ValueError(f"Portfolio figure changed: {sid}")


def plot_diagnostics(metrics: dict, outcomes: dict, destination: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figure, axis = plt.subplots(figsize=(10.5, 4.8))
    positions = np.arange(4)
    for offset, name, color in zip(
        (-0.24, 0, 0.24), LABELS, ("#8290a6", "#6595c1", "#31856b"), strict=True
    ):
        values = [
            100 * metrics[name]["per_field"][f]["normalized_exact_match"] for f in CANONICAL_FIELDS
        ]
        bars = axis.bar(positions + offset, values, 0.23, label=LABELS[name], color=color)
        axis.bar_label(bars, labels=[f"{v:.1f}" for v in values], fontsize=8, padding=3)
    axis.set(
        xticks=positions,
        xticklabels=[f.title() for f in CANONICAL_FIELDS],
        ylabel="Normalized exact match (%)",
        ylim=(0, 112),
        title="Where extraction improved: all 239 retrospective receipts",
    )
    axis.legend(loc="upper left", bbox_to_anchor=(0, 1.14), ncol=3, frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    figure.text(
        0.5,
        0.015,
        "Common tolerant parser; strict JSON is a separate metric. "
        "V2 preprocessing differs from Qwen.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    figure.savefig(destination / "field_accuracy.png", dpi=170)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10.5, 3.8))
    labels = ["Legacy V2 → latest Qwen", "Qwen base → QLoRA"]
    left = np.zeros(2)
    for key, label, color in (
        ("improved", "More correct fields", "#31856b"),
        ("equal_count", "Same number correct", "#9aa8bc"),
        ("regressed", "Fewer correct fields", "#c76565"),
    ):
        values = np.array([outcomes[name][key] for name in ("v2", "qwen_base")])
        bars = axis.barh(labels, values, left=left, label=label, color=color, height=0.48)
        axis.bar_label(bars, labels=[str(v) for v in values], label_type="center", fontsize=11)
        left += values
    axis.set(
        xlim=(0, 239),
        xlabel="Receipts (same 239 images for each comparison)",
        title="Accuracy gains are not uniform across bills",
    )
    axis.invert_yaxis()
    axis.legend(loc="upper left", bbox_to_anchor=(0, 1.25), ncol=3, frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    figure.text(
        0.5,
        0.018,
        "Equal correct-field counts can hide changes in which fields "
        "are wrong. Buckets use normalized field correctness, not schema validity.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.08, 1, 0.96))
    figure.savefig(destination / "outcome_transitions.png", dpi=170)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=RUN / "portfolio")
    parser.add_argument(
        "--verify-only", action="store_true", help="Audit published files; no raw images"
    )
    args = parser.parse_args()
    expected_hashes = json.loads(
        (ROOT / "artifacts/experiments/receiptops/evaluation_image_sha256.json").read_text("utf-8")
    )
    variants = {
        "v2": [
            r
            for r in read_rows(
                ROOT / "artifacts/experiments/highres_training_v2/v2_predictions.jsonl"
            )
            if r["sample_id"] in expected_hashes
        ],
        "qwen_base": read_rows(RUN / "base_predictions.jsonl"),
        "qwen_qlora": read_rows(RUN / "adapted_predictions.jsonl"),
    }
    for name in ("qwen_base", "qwen_qlora"):
        verify_raw_predictions(variants[name])
    ids = verify_alignment(variants, expected_hashes)
    variants = {name: comparison_rows(rows)[0] for name, rows in variants.items()}
    metrics = {name: evaluate_predictions(rows) for name, rows in variants.items()}
    verified = json.loads((RUN / "verified_comparison.json").read_text("utf-8"))
    verify_metric_report(metrics, verified["metrics"])
    indexed = {name: {r["sample_id"]: r for r in rows} for name, rows in variants.items()}
    outcomes = {
        name: summarize_pair(indexed[name], indexed["qwen_qlora"], ids)
        for name in ("v2", "qwen_base")
    }
    cases = select_cases(indexed, ids)
    summary = {
        "sample_count": len(ids),
        "scope": verified["scope"],
        "outcomes": outcomes,
        "case_count": len(cases),
        "selection": cases,
    }
    if args.verify_only:
        verify_gallery(args.output_dir, summary, indexed, expected_hashes)
        print("Portfolio gallery verified: 14 cases and all-record counts match saved predictions")
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_diagnostics(metrics, outcomes, args.output_dir)
    lines = [
        "# ReceiptOps output differences: 14 documented cases",
        "",
        "Original receipt → saved baseline output → latest Qwen QLoRA output → dataset reference.",
        "",
        "## How to read this gallery",
        "",
        "These are outcome-selected examples, not a representative sample or an accuracy "
        "estimate. Each comparison has seven declared buckets: several fields recovered, "
        "total/date/merchant/address correction, unchanged fields, and a regression. Within "
        "each bucket select the first lexicographic evaluation ID that has not been used; "
        "keep distinct normalized merchants within each comparison. All 14 bills are unique.",
        "",
        "V2 versus Qwen compares project versions with different preprocessing and prompts. "
        "Qwen base versus QLoRA keeps images, NF4, prompt, pixel cap and decoding identical. "
        "All panels use the same tolerant parser for fields; strict schema is labeled separately. "
        "Reference labels can contain mistakes.",
        "",
        "## Outcomes across all 239 receipts",
        "",
        "| Comparison | More correct fields | Equal correct-field count | Fewer correct fields |",
        "|---|---:|---:|---:|",
    ]
    for name, details in outcomes.items():
        lines.append(
            f"| {LABELS[name]} → QLoRA | {details['improved']} | "
            f"{details['equal_count']} | {details['regressed']} |"
        )
    lines.extend(
        [
            "",
            "Equal counts do not imply identical predictions. The two comparisons "
            f"have respectively {outcomes['v2']['identical_fields']} and "
            f"{outcomes['qwen_base']['identical_fields']} bills with identical "
            "normalized field values.",
            "",
            "![All-record outcome changes](outcome_transitions.png)",
            "",
            "![Exact match by field](field_accuracy.png)",
            "",
        ]
    )
    manifest = []
    for number, case in enumerate(cases, 1):
        sid, name = case["sample_id"], case["before_model"]
        before = indexed[name][sid]
        image_path = indexed["v2"][sid]["image_path"]
        before = {**before, "image_path": image_path}
        after = {**indexed["qwen_qlora"][sid], "image_path": image_path}
        if hashlib.sha256((ROOT / image_path).read_bytes()).hexdigest() != expected_hashes[sid]:
            raise ValueError(f"Source image changed: {sid}")
        output = args.output_dir / f"{number:02d}_{name}_{case['category']}_{sid}.png"
        item = render_pair(
            before,
            after,
            output,
            number,
            len(cases),
            comparison_title=case["title"],
            context=(
                "Same receipt; V2 and Qwen use different preprocessing and prompts"
                if name == "v2"
                else "Same image, prompt, NF4, pixel cap and greedy decoding"
            ),
            before_heading=f"BEFORE\n{LABELS[name]}",
            after_heading="AFTER\nLatest Qwen3-VL QLoRA",
            note=f"Outcome-selected case: {case['category']}. First eligible ID after "
            "bill/merchant diversity filters; not an accuracy estimate.",
        )
        item.update(case)
        item.update(raw_output_before=before["raw_output"], raw_output_after=after["raw_output"])
        if sid in ANNOTATION_NOTES:
            item["annotation_note"] = ANNOTATION_NOTES[sid]
        item["figure_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
        manifest.append(item)
        a, b = len(item["matched_fields_before"]), len(item["matched_fields_after"])
        lines.extend(
            [
                f"## {number:02d}. {LABELS[name]} → QLoRA: {case['title']}",
                "",
                f"Receipt `{sid}`. Normalized correct fields: **{a}/4 → {b}/4**. "
                f"Changed fields: {', '.join(item['normalized_changed_fields']) or 'none'}. "
                "Eligible cases before diversity filtering: "
                f"{case['eligible_count_before_diversity_filter']}.",
                "",
                f"![{case['title']}]({output.name})",
                "",
            ]
        )
        if sid in ANNOTATION_NOTES:
            lines.extend([ANNOTATION_NOTES[sid], ""])
    (args.output_dir / "README.md").write_text("\n".join(lines).rstrip() + "\n", "utf-8")
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", "utf-8")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", "utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
