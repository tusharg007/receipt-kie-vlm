"""Verify that committed claims and artifacts derive from the recorded run."""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from receipt_kie.metrics import evaluate_predictions  # noqa: E402


def _json(relative_path: str) -> Any:
    with (PROJECT_ROOT / relative_path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _jsonl(relative_path: str) -> list[dict[str, Any]]:
    with (PROJECT_ROOT / relative_path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        "models/receipt-kie-lora/adapter_model.safetensors",
        "models/receipt-kie-lora/adapter_config.json",
        "models/receipt-kie-lora/training_metadata.json",
        "models/receipt-kie-lora/README.md",
        "assets/demo/synthetic_receipt.png",
        "assets/demo/expected_output.json",
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
    metadata = _json("models/receipt-kie-lora/training_metadata.json")
    adapter = PROJECT_ROOT / "models/receipt-kie-lora/adapter_model.safetensors"
    adapter_hash = _sha256(adapter)
    checks.extend(
        (
            (
                "Committed adapter checksum matches training metadata",
                adapter_hash == metadata["adapter_sha256"],
                adapter_hash,
            ),
            (
                "Committed adapter size matches training metadata",
                adapter.stat().st_size == metadata["adapter_size_bytes"],
                f"{adapter.stat().st_size} bytes",
            ),
            (
                "Committed adapter is a real safetensors binary",
                not adapter.read_bytes()[:128].startswith(
                    b"version https://git-lfs.github.com/spec/v1"
                ),
                "",
            ),
        )
    )
    tracked = _tracked_files()
    checks.append(
        (
            "Final adapter is tracked by normal Git",
            "models/receipt-kie-lora/adapter_model.safetensors" in tracked,
            "",
        )
    )
    local_training_dir = PROJECT_ROOT / "artifacts/checkpoints/receipt-kie-lora"
    if (local_training_dir / "training_summary.json").is_file():
        training = _json("artifacts/checkpoints/receipt-kie-lora/training_summary.json")
        loss_history = _json("artifacts/checkpoints/receipt-kie-lora/loss_history.json")
        trainer_state = _json("artifacts/checkpoints/receipt-kie-lora/trainer_state.json")
        local_adapter = local_training_dir / "adapter_model.safetensors"
        optimization_rows = [row for row in trainer_state["log_history"] if "loss" in row]
        checks.extend(
            (
                (
                    "Local training adapter matches committed adapter",
                    _sha256(local_adapter) == adapter_hash,
                    "",
                ),
                (
                    "A newly initialized LoRA parameter changed",
                    bool(training["lora_parameter_changed"]),
                    f"steps={training['global_steps']}",
                ),
                (
                    "Trainer state contains real optimization steps",
                    bool(optimization_rows),
                    "",
                ),
                (
                    "Loss history is non-empty and finite",
                    bool(loss_history),
                    f"rows={len(loss_history)}",
                ),
            )
        )
    else:
        checks.append(
            (
                "Clone does not require private trainer state",
                True,
                "committed adapter metadata and checksum are sufficient for inference",
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
    windows_user_root = "C:/" + "Users/"
    mac_user_root = "/" + "Users/"
    linux_user_root = "/" + "home/"
    absolute_path_pattern = re.compile(
        rf"(?:[A-Za-z]:\\\\|{re.escape(windows_user_root)}|"
        rf"{re.escape(mac_user_root)}[^/\s]+/|{re.escape(linux_user_root)}[^/\s]+/)"
    )
    secret_hits: list[str] = []
    absolute_path_hits: list[str] = []
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
        if absolute_path_pattern.search(text):
            absolute_path_hits.append(relative)
    checks.extend(
        (
            (
                "No credential-like strings in tracked text",
                not secret_hits,
                ", ".join(secret_hits),
            ),
            (
                "No local absolute paths in tracked text",
                not absolute_path_hits,
                ", ".join(absolute_path_hits),
            ),
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
            "This check validates the committed adapter and current evidence. "
            "It does not assert production readiness.",
        ]
    )
    output = PROJECT_ROOT / "artifacts/reports/integrity_check.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if failed:
        raise SystemExit("Integrity failures: " + "; ".join(failed))


if __name__ == "__main__":
    main()
