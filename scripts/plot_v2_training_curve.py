"""Render the V2 training/validation curve from the committed trainer summary."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUMMARY = (
    PROJECT_ROOT
    / "artifacts"
    / "experiments"
    / "highres_training_v2"
    / "training_summary.json"
)
OUTPUT = (
    PROJECT_ROOT
    / "artifacts"
    / "figures"
    / "highres_training_v2_loss.png"
)


def main() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    training = summary["training_losses"]
    validation = summary["validation_losses"]
    figure, axis = plt.subplots(figsize=(10, 5.5))
    axis.plot(
        [row["step"] for row in training],
        [row["loss"] for row in training],
        color="#4c78a8",
        linewidth=1.8,
        label="Training loss",
    )
    axis.scatter(
        [row["step"] for row in validation],
        [row["eval_loss"] for row in validation],
        color="#e45756",
        marker="D",
        s=45,
        zorder=3,
        label="Validation loss",
    )
    axis.plot(
        [row["step"] for row in validation],
        [row["eval_loss"] for row in validation],
        color="#e45756",
        linewidth=1.2,
        alpha=0.75,
    )
    axis.set(
        title="V2 continued high-resolution adaptation",
        xlabel="Optimizer step",
        ylabel="Assistant-token cross-entropy loss",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=180)
    plt.close(figure)
    print(OUTPUT.relative_to(PROJECT_ROOT).as_posix())


if __name__ == "__main__":
    main()
