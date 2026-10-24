"""Render image-by-image comparisons from saved paired decoding predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import textwrap
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import matplotlib
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from receipt_kie.metrics import normalize_field  # noqa: E402
from receipt_kie.prompts import CANONICAL_FIELDS  # noqa: E402


def _read_rows(path: Path) -> dict[str, dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    indexed = {row["sample_id"]: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"Duplicate sample IDs in {path}")
    return indexed


def _matches(prediction: dict, truth: dict) -> list[str]:
    return [
        field
        for field in CANONICAL_FIELDS
        if normalize_field(field, prediction.get(field, "")) == normalize_field(field, truth[field])
    ]


def _schema_valid(row: dict) -> bool:
    return bool(
        row.get("strict_schema_valid", row.get("schema_valid", row.get("valid_json", False)))
    )


def render_pair(
    before: dict,
    after: dict,
    destination: Path,
    position: int,
    count: int,
    *,
    comparison_title: str = "Before and after typed decoding",
    context: str = "Same V2 adapter, same receipt, 1024 px, greedy decoding",
    before_heading: str = "BEFORE\nOriginal V2 model",
    after_heading: str = "AFTER\nTypeLLM-inspired constraints",
    note: str = "Comparison: original SmolVLM V2 vs local schema constraints inspired by "
    "TypeLLM. Native TypeLLM/SGLang was not run.",
) -> dict:
    sample_id = before["sample_id"]
    if before["ground_truth"] != after["ground_truth"]:
        raise ValueError(f"Ground truth differs for {sample_id}")
    if before["image_path"] != after["image_path"]:
        raise ValueError(f"Receipt image differs for {sample_id}")
    truth = before["ground_truth"]
    original = before["parsed_prediction"] or {}
    constrained = after["parsed_prediction"] or {}
    correct_before = _matches(original, truth)
    correct_after = _matches(constrained, truth)
    changed = [field for field in CANONICAL_FIELDS if original.get(field) != constrained.get(field)]
    normalized_changed = [
        field
        for field in CANONICAL_FIELDS
        if normalize_field(field, original.get(field, ""))
        != normalize_field(field, constrained.get(field, ""))
    ]
    image_path = PROJECT_ROOT / before["image_path"]
    fig = plt.figure(figsize=(18, 10.5), facecolor="#f6f8fb")
    fig.text(
        0.035,
        0.95,
        f"Bill {position:02d} / {count:02d}  |  {comparison_title}",
        fontsize=23,
        weight="bold",
        color="#152238",
    )
    fig.text(
        0.035,
        0.91,
        f"{sample_id}   |   {context}",
        fontsize=12,
        color="#48566c",
    )
    image_axis = fig.add_axes((0.025, 0.125, 0.245, 0.74))
    with Image.open(image_path) as image:
        image_axis.imshow(image.convert("RGB"))
    image_axis.set_axis_off()
    image_axis.set_title("Original bill", fontsize=14, loc="left", pad=12)

    headings = (
        before_heading,
        after_heading,
        "REFERENCE\nDataset annotation",
    )
    columns = (0.30, 0.53, 0.76)
    width = 0.215
    for column, heading in zip(columns, headings, strict=True):
        fig.text(
            column, 0.84, heading, fontsize=13, weight="bold", color="#152238", linespacing=1.5
        )
    for column, row in ((columns[0], before), (columns[1], after)):
        valid = _schema_valid(row)
        fig.text(
            column,
            0.785,
            "Schema: valid" if valid else "Schema: invalid; raw response saved separately",
            fontsize=8,
            color="#48566c",
        )
    for index, field in enumerate(CANONICAL_FIELDS):
        y = 0.735 - index * 0.155
        fig.text(0.30, y + 0.033, field.upper(), fontsize=10, weight="bold", color="#48566c")
        for column, values, correct in (
            (columns[0], original, field in correct_before),
            (columns[1], constrained, field in correct_after),
            (columns[2], truth, None),
        ):
            fill = "#e9f0fa" if correct is None else "#e8f3ec" if correct else "#f9eaea"
            fig.add_artist(
                FancyBboxPatch(
                    (column - 0.006, y - 0.107),
                    width,
                    0.128,
                    boxstyle="round,pad=0.006,rounding_size=0.006",
                    facecolor=fill,
                    linewidth=0,
                    transform=fig.transFigure,
                    zorder=0,
                )
            )
            value = (
                "(no schema-valid field object)"
                if correct is not None and not values
                else str(values.get(field, "")) or "(empty)"
            )
            lines = textwrap.wrap(value, width=28, break_long_words=True) or ["(empty)"]
            # All text is retained; shrink only for exceptionally long merchant strings.
            font_size = 11.5 if len(lines) <= 4 else 10
            fig.text(
                column,
                y + 0.006,
                "\n".join(lines),
                fontsize=font_size,
                va="top",
                color="#152238",
                linespacing=1.3,
            )
            status = (
                "Dataset reference"
                if correct is None
                else "MATCH"
                if correct
                else "DIFFERS FROM REFERENCE"
            )
            fig.text(
                column,
                y - 0.092,
                status,
                fontsize=8.5,
                color="#48566c" if correct is None else "#176239" if correct else "#923f3f",
            )

    fig.text(
        0.30,
        0.139,
        f"Normalized exact matches: {len(correct_before)}/4 before   |   "
        f"{len(correct_after)}/4 after",
        fontsize=13,
        weight="bold",
        color="#152238",
    )
    if not changed:
        change_note = "Extracted field values are identical."
    elif not normalized_changed:
        change_note = "Only punctuation/formatting changed; normalized correctness is unchanged."
    else:
        change_note = "Changed fields: " + ", ".join(changed)
    fig.text(0.30, 0.104, change_note, fontsize=11.5, color="#48566c")
    fig.text(
        0.035,
        0.067,
        note,
        fontsize=11,
        color="#48566c",
    )
    fig.text(
        0.035,
        0.038,
        "Match = exact after existing evaluation normalization. "
        "A valid schema does not establish that a value is correct.",
        fontsize=11,
        color="#48566c",
    )
    fig.savefig(destination, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    return {
        "sample_id": sample_id,
        "image_path": before["image_path"],
        "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
        "figure": destination.name,
        "before": original,
        "after": constrained,
        "reference": truth,
        "matched_fields_before": correct_before,
        "matched_fields_after": correct_after,
        "changed_fields": changed,
        "normalized_changed_fields": normalized_changed,
        "before_schema_valid": _schema_valid(before),
        "after_schema_valid": _schema_valid(after),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts/experiments/structured_decoding/pilot10_1024",
    )
    args = parser.parse_args()
    root = args.predictions_dir.resolve()
    before = _read_rows(root / "unconstrained.jsonl")
    after = _read_rows(root / "constrained.jsonl")
    if not before or set(before) != set(after):
        raise ValueError("Expected a nonempty, completely paired prediction set")
    destination = root / "image_comparisons"
    destination.mkdir(parents=True, exist_ok=True)
    manifest = []
    gallery = [
        "# Bill-by-bill comparison",
        "",
        "Original V2 model versus local schema constraints inspired by TypeLLM. "
        "Native TypeLLM/SGLang inference was not run. Same images, adapter, and 1024 px settings.",
        "",
        "Colors show normalized exact matches against dataset annotations. "
        "This ten-receipt pilot showed no normalized OCR accuracy improvement.",
        "",
    ]
    archive_readme = gallery.copy()
    for number, sample_id in enumerate(before, start=1):
        output = destination / f"{number:02d}_{sample_id}.png"
        entry = render_pair(before[sample_id], after[sample_id], output, number, len(before))
        manifest.append(entry)
        title = f"## Bill {number:02d}: {sample_id}"
        gallery.extend([title, "", f"![Comparison for {sample_id}]({output.as_posix()})", ""])
        archive_readme.extend([title, "", f"![Comparison for {sample_id}]({output.name})", ""])
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (destination / "comparison_report.md").write_text("\n".join(gallery), encoding="utf-8")
    (destination / "README.md").write_text("\n".join(archive_readme), encoding="utf-8")
    with ZipFile(destination / "all_bill_comparisons.zip", "w", ZIP_DEFLATED) as archive:
        for path in sorted(destination.glob("*.png")):
            archive.write(path, path.name)
        for name in ("README.md", "manifest.json"):
            archive.write(destination / name, name)
    print(f"Rendered {len(manifest)} receipt comparisons in {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
