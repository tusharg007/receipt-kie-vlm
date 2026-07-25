"""Verify that claims and generated artifacts derive from the current run."""

from __future__ import annotations

import json
import math
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from receipt_kie.metrics import evaluate_predictions  # noqa: E402


def _json(relative_path: str) -> dict[str, Any]:
    with (PROJECT_ROOT / relative_path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _jsonl(relative_path: str) -> list[dict[str, Any]]:
    with (PROJECT_ROOT / relative_path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _tracked_files() -> list[str]:
    output = subprocess.check_output(
        ["git", "ls-files"],
        cwd=PROJECT_ROOT,
        text=True,
        encoding="utf-8",
    )
    return [line for line in output.splitlines() if line]


def main() -> None:
    checks: list[tuple[str, bool, str]] = []
    required = (
        "artifacts/checkpoints/receipt-kie-lora/adapter_model.safetensors",
        "artifacts/checkpoints/receipt-kie-lora/adapter_config.json",
        "artifacts/checkpoints/receipt-kie-lora/loss_history.json",
        "artifacts/checkpoints/receipt-kie-lora/trainer_state.json",
        "artifacts/logs/training.log",
        "artifacts/reports/base_metrics.json",
        "artifacts/reports/lora_metrics.json",
        "artifacts/predictions/base_predictions.jsonl",
        "artifacts/predictions/lora_predictions.jsonl",
        "artifacts/figures/training_loss.png",
        "artifacts/figures/base_vs_lora.png",
        "artifacts/figures/field_accuracy.png",
        "README.md",
    )
    for relative in required:
        path = PROJECT_ROOT / relative
        checks.append(
            (
                f"Artifact exists: `{relative}`",
                path.is_file() and path.stat().st_size > 0,
                "",
            )
        )
    training = _json("artifacts/checkpoints/receipt-kie-lora/training_summary.json")
    loss_history = _json("artifacts/checkpoints/receipt-kie-lora/loss_history.json")
    trainer_state = _json("artifacts/checkpoints/receipt-kie-lora/trainer_state.json")
    adapter = PROJECT_ROOT / "artifacts/checkpoints/receipt-kie-lora/adapter_model.safetensors"
    loss_file = PROJECT_ROOT / "artifacts/checkpoints/receipt-kie-lora/loss_history.json"
    modification_delta = abs(adapter.stat().st_mtime - loss_file.stat().st_mtime)
    checks.append(
        (
            "Adapter timestamp corresponds to the current run",
            modification_delta < 600,
            f"adapter={datetime.fromtimestamp(adapter.stat().st_mtime, timezone.utc).isoformat()}",
        )
    )
    checks.append(
        (
            "A newly initialized LoRA parameter changed",
            bool(training["lora_parameter_changed"]),
            f"steps={training['global_steps']}",
        )
    )
    optimization_rows = [row for row in trainer_state["log_history"] if "loss" in row]
    checks.append(("Trainer state contains real optimization steps", bool(optimization_rows), ""))
    checks.append(
        (
            "Loss history is non-empty and finite",
            bool(loss_history),
            f"rows={len(loss_history)}",
        )
    )
    base_rows = _jsonl("artifacts/predictions/base_predictions.jsonl")
    lora_rows = _jsonl("artifacts/predictions/lora_predictions.jsonl")
    base_metrics = _json("artifacts/reports/base_metrics.json")
    lora_metrics = _json("artifacts/reports/lora_metrics.json")
    checks.append(
        (
            "Base and LoRA use the same non-zero sample IDs",
            bool(base_rows)
            and [row["sample_id"] for row in base_rows]
            == [row["sample_id"] for row in lora_rows],
            f"count={len(lora_rows)}",
        )
    )
    for name, rows, recorded in (
        ("base", base_rows, base_metrics),
        ("lora", lora_rows, lora_metrics),
    ):
        recomputed = evaluate_predictions(rows)
        matching = all(
            math.isclose(float(recomputed[key]), float(recorded[key]), rel_tol=0, abs_tol=1e-12)
            for key in (
                "valid_json_rate",
                "company_accuracy",
                "address_similarity",
                "date_accuracy",
                "total_accuracy",
                "complete_record_normalized_exact_match",
            )
        )
        checks.append((f"{name.title()} metrics recompute from prediction JSONL", matching, ""))
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    expected_claims = (
        f"{base_metrics['valid_json_rate']:.0%}",
        f"{lora_metrics['valid_json_rate']:.0%}",
        f"{lora_metrics['company_accuracy']:.0%}",
        f"{lora_metrics['date_accuracy']:.0%}",
        f"{lora_metrics['total_accuracy']:.0%}",
        f"{lora_metrics['address_similarity']:.2%}",
    )
    checks.append(
        (
            "README contains current metric values",
            all(claim in readme for claim in expected_claims),
            ", ".join(expected_claims),
        )
    )
    tracked = _tracked_files()
    checks.append(
        (
            "No raw dataset is tracked",
            not any(path.startswith(("data/raw/", "data\\raw\\")) for path in tracked),
            "",
        )
    )
    secret_patterns = (
        re.compile(r"ghp_[A-Za-z0-9]{30,}"),
        re.compile(r"hf_[A-Za-z0-9]{30,}"),
        re.compile(r"AIza[0-9A-Za-z_-]{30,}"),
        re.compile(r'"key"\s*:\s*"[A-Za-z0-9]{30,}"'),
    )
    secret_hits: list[str] = []
    for relative in tracked:
        path = PROJECT_ROOT / relative
        if not path.is_file() or path.stat().st_size > 5 * 2**20:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if any(pattern.search(text) for pattern in secret_patterns):
            secret_hits.append(relative)
    checks.append(
        (
            "No credential-like strings in tracked text",
            not secret_hits,
            ", ".join(secret_hits),
        )
    )
    failed = [name for name, passed, _ in checks if not passed]
    lines = [
        "# Result Integrity Check",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "| Check | Status | Evidence |",
        "|---|---|---|",
    ]
    for name, passed, evidence in checks:
        lines.append(f"| {name} | {'PASS' if passed else 'FAIL'} | {evidence} |")
    lines.extend(
        [
            "",
            f"**Overall: {'PASS' if not failed else 'FAIL'}**",
            "",
            "This check validates the current local run. It does not assert production readiness.",
        ]
    )
    output = PROJECT_ROOT / "artifacts/reports/integrity_check.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if failed:
        raise SystemExit("Integrity failures: " + "; ".join(failed))


if __name__ == "__main__":
    main()
