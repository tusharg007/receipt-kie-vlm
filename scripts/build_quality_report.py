"""Plot the frozen quality ranker, class imbalance and observable error slices."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
from sklearn.metrics import precision_recall_curve

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    folder = ROOT / "artifacts/experiments/quality_gate"
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in (folder / "evaluation_scores.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    labels = np.array([row["correct"] for row in rows])
    scores = np.array([row["quality_score"] for row in rows])
    precision, recall, _ = precision_recall_curve(labels, scores)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].bar(
        ["All fields\ncorrect", "At least one\nfield differs"],
        [int(labels.sum()), int((1 - labels).sum())],
        color=["#3d8f79", "#cc795a"],
    )
    axes[0].set_ylabel("Receipts")
    axes[0].set_title(f"Class imbalance: {int(labels.sum())} / {int((1 - labels).sum())}")
    ap = report["models"]["balanced_logistic"]["evaluation"]["average_precision"]
    axes[1].plot(recall, precision, label=f"Class-weighted logistic (AP {ap:.3f})", color="#3269a8")
    axes[1].axhline(
        labels.mean(),
        color="#777777",
        linestyle="--",
        label=f"Prevalence baseline ({labels.mean():.3f})",
    )
    axes[1].set(
        xlabel="Recall of fully correct records", ylabel="Precision", xlim=(0, 1), ylim=(0, 1)
    )
    axes[1].set_title("Quality ranking, not OCR correction")
    axes[1].legend(fontsize=8)
    slices = report["quality_slices"]
    keys = list(slices)
    axes[2].barh(
        np.arange(len(keys)),
        [1 - slices[key]["correct_record_prevalence"] for key in keys],
        color="#637b9e",
    )
    axes[2].set_yticks(
        np.arange(len(keys)),
        [
            key.replace("company_contains_digits", "Digit-heavy merchant")
            .replace("company_address_overlap", "Merchant/address overlap")
            .replace("long_address", "Long predicted address")
            + f" (n={slices[key]['sample_count']})"
            for key in keys
        ],
        fontsize=8,
    )
    axes[2].set(xlabel="Fraction with at least one differing field", xlim=(0, 1))
    axes[2].set_title("Observable output slices")
    fig.suptitle(
        f"ReceiptOps: retrospective, content-disjoint analysis of {len(rows)} V2 predictions",
        fontsize=14,
    )
    fig.text(
        0.5,
        0.012,
        "Small-data pilot. Candidate threshold fails evaluation precision; "
        "deployment remains review-only. "
        "Slices do not establish demographic fairness.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    output = folder / "quality_analysis.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
