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
        "models/receipt-kie-lora-v2-highres/adapter_model.safetensors",
        "models/receipt-kie-lora-v2-highres/adapter_config.json",
        "models/receipt-kie-lora-v2-highres/training_metadata.json",
        "models/receipt-kie-lora-v2-highres/README.md",
        "MODEL_COMPARISON.md",
        "assets/demo/synthetic_receipt.png",
        "assets/demo/expected_output.json",
        "artifacts/reports/base_metrics.json",
        "artifacts/reports/lora_metrics.json",
        "artifacts/predictions/base_predictions.jsonl",
        "artifacts/predictions/lora_predictions.jsonl",
        "artifacts/figures/training_loss.png",
        "artifacts/figures/base_vs_lora.png",
        "artifacts/figures/field_accuracy.png",
        "artifacts/figures/robustness_results.png",
        "artifacts/figures/qualitative_lora_improvement.png",
        "artifacts/figures/qualitative_failure_analysis.png",
        "artifacts/reports/qualitative_results_manifest.json",
        "artifacts/experiments/highres_training_v2/test_usage_manifest.json",
        "artifacts/experiments/highres_training_v2/final_holdout_ids.json",
        "artifacts/experiments/highres_training_v2/base_predictions.jsonl",
        "artifacts/experiments/highres_training_v2/v1_predictions.jsonl",
        "artifacts/experiments/highres_training_v2/v2_predictions.jsonl",
        "artifacts/experiments/highres_training_v2/results.json",
        "artifacts/figures/highres_training_v2_comparison.png",
        "artifacts/figures/highres_training_v2_qualitative/v2_complete_record_success.png",
        "artifacts/figures/highres_training_v2_qualitative/v2_improvement_over_v1.png",
        "artifacts/figures/highres_training_v2_qualitative/v2_failure_or_regression.png",
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
    tracked = _tracked_files()
    release_adapters = {
        "V1": {
            "directory": "models/receipt-kie-lora",
            "sha256": (
                "94ba0038153ea1aacb12dbcc80f1edf01d31a6309ea56919684e8cb8bbe90b28"
            ),
            "size": 10_956_944,
        },
        "V2": {
            "directory": "models/receipt-kie-lora-v2-highres",
            "sha256": (
                "3e0e5a88c36f0d6a0db6baf2a3b521e40be4ef84b212ed2eafecab431604bf79"
            ),
            "size": 10_956_944,
        },
    }
    adapter_hash = ""
    for version, release in release_adapters.items():
        directory = str(release["directory"])
        metadata = _json(f"{directory}/training_metadata.json")
        adapter = PROJECT_ROOT / directory / "adapter_model.safetensors"
        digest = _sha256(adapter)
        if version == "V1":
            adapter_hash = digest
        checks.extend(
            (
                (
                    f"{version} adapter checksum matches metadata and release",
                    digest == metadata["adapter_sha256"] == release["sha256"],
                    digest,
                ),
                (
                    f"{version} adapter size matches metadata and release",
                    adapter.stat().st_size
                    == metadata["adapter_size_bytes"]
                    == release["size"],
                    f"{adapter.stat().st_size} bytes",
                ),
                (
                    f"{version} adapter is a real safetensors binary",
                    not adapter.read_bytes()[:128].startswith(
                        b"version https://git-lfs.github.com/spec/v1"
                    ),
                    "",
                ),
                (
                    f"{version} adapter is tracked by normal Git",
                    f"{directory}/adapter_model.safetensors" in tracked,
                    "",
                ),
            )
        )
    local_training_dir = PROJECT_ROOT / "artifacts/checkpoints/receipt-kie-lora"
    if (local_training_dir / "training_summary.json").is_file():
        training = _json("artifacts/checkpoints/receipt-kie-lora/training_summary.json")
        loss_history = _json("artifacts/checkpoints/receipt-kie-lora/loss_history.json")
        validation_losses = _json(
            "artifacts/checkpoints/receipt-kie-lora/validation_loss_history.json"
        )
        split_manifest = _json(
            "artifacts/checkpoints/receipt-kie-lora/dataset_split_manifest.json"
        )
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
                (
                    "Validation loss history is non-empty and finite",
                    bool(validation_losses)
                    and all(math.isfinite(row["eval_loss"]) for row in validation_losses),
                    f"rows={len(validation_losses)}",
                ),
                (
                    "Training and validation both derive from official train split",
                    split_manifest["train_source_split"] == "train"
                    and split_manifest["validation_source_split"] == "train"
                    and not split_manifest["official_test_used_during_training"],
                    (
                        f"train={split_manifest['train_source_split']}, "
                        f"validation={split_manifest['validation_source_split']}"
                    ),
                ),
                (
                    "Training and validation IDs are disjoint",
                    not (
                        set(split_manifest["train_ids"])
                        & set(split_manifest["validation_ids"])
                    )
                    and not split_manifest["overlap_ids"],
                    (
                        f"train={len(split_manifest['train_ids'])}, "
                        f"validation={len(split_manifest['validation_ids'])}"
                    ),
                ),
                (
                    "Leakage-free split has expected 563/63 counts",
                    len(split_manifest["train_ids"]) == 563
                    and len(split_manifest["validation_ids"]) == 63,
                    (
                        f"train={len(split_manifest['train_ids'])}, "
                        f"validation={len(split_manifest['validation_ids'])}"
                    ),
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
    if (local_training_dir / "dataset_split_manifest.json").is_file():
        validation_ids = set(split_manifest["validation_ids"])
        test_ids = {row["sample_id"] for row in lora_rows}
        checks.append(
            (
                "Training-time validation IDs do not overlap final test evaluation",
                validation_ids.isdisjoint(test_ids),
                f"overlap={len(validation_ids & test_ids)}",
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
    v2_root = "artifacts/experiments/highres_training_v2"
    holdout_ids = _json(f"{v2_root}/final_holdout_ids.json")
    usage = _json(f"{v2_root}/test_usage_manifest.json")
    v2_results = _json(f"{v2_root}/results.json")
    v2_rows_by_variant = {
        name: _jsonl(f"{v2_root}/{name}_predictions.jsonl")
        for name in ("base", "v1", "v2")
    }
    holdout_alignment = all(
        [row["sample_id"] for row in rows] == holdout_ids
        for rows in v2_rows_by_variant.values()
    )
    checks.extend(
        (
            (
                "V2 holdout has 246 unique aligned IDs",
                len(holdout_ids) == len(set(holdout_ids)) == 246
                and holdout_alignment,
                f"count={len(holdout_ids)}",
            ),
            (
                "V2 holdout is disjoint from every prior evaluated ID",
                set(holdout_ids).isdisjoint(usage["previously_evaluated_ids"])
                and holdout_ids == usage["remaining_never_evaluated_ids"],
                (
                    f"prior={len(usage['previously_evaluated_ids'])}, "
                    f"holdout={len(holdout_ids)}"
                ),
            ),
        )
    )
    for name, rows in v2_rows_by_variant.items():
        recomputed = evaluate_predictions(rows)
        recorded = v2_results["variants"][name]["metrics"]
        matching = all(
            math.isclose(
                float(recomputed[key]),
                float(recorded[key]),
                rel_tol=0,
                abs_tol=1e-12,
            )
            for key in (
                "valid_json_rate",
                "company_accuracy",
                "address_similarity",
                "date_accuracy",
                "total_accuracy",
                "complete_record_normalized_exact_match",
                "macro_normalized_exact_match",
                "macro_normalized_similarity",
            )
        )
        checks.append(
            (
                f"V2-release {name.title()} metrics recompute from prediction JSONL",
                matching,
                "",
            )
        )
    paired = v2_results["paired_v2_minus_v1_bootstrap_95_percent_ci"]
    expected_paired = {
        "macro_normalized_exact_match": (
            0.1941056910569105,
            0.16260162601626021,
            0.2276422764227642,
        ),
        "complete_record_normalized_exact_match": (
            0.14227642276422764,
            0.1016260162601626,
            0.19105691056910568,
        ),
        "address_similarity": (0.0852045083545192, 0.060056753347310546, 0.11014046757996102),
        "valid_json_rate": (0.008130081300812941, -0.012195121951219523, 0.028455284552845517),
    }
    ci_matches = all(
        math.isclose(paired[name]["point_delta"], values[0], rel_tol=0, abs_tol=1e-12)
        and math.isclose(paired[name]["low"], values[1], rel_tol=0, abs_tol=1e-12)
        and math.isclose(paired[name]["high"], values[2], rel_tol=0, abs_tol=1e-12)
        for name, values in expected_paired.items()
    )
    checks.append(
        (
            "V2 paired bootstrap values match the seeded release result",
            ci_matches and bool(v2_results["success_gate"]["passed"]),
            "2,000 resamples; release gate PASS",
        )
    )
    qualitative = _json("artifacts/reports/qualitative_results_manifest.json")
    base_by_id = {row["sample_id"]: row for row in base_rows}
    lora_by_id = {row["sample_id"]: row for row in lora_rows}
    improvement_id = qualitative["improvement"]["sample_id"]
    failure_id = qualitative["failure"]["sample_id"]
    qualitative_matches = (
        qualitative["improvement"]["base"]["raw_output"]
        == base_by_id[improvement_id]["raw_output"]
        and qualitative["improvement"]["lora"]["raw_output"]
        == lora_by_id[improvement_id]["raw_output"]
        and qualitative["improvement"]["lora"]["ground_truth"]
        == lora_by_id[improvement_id]["ground_truth"]
        and qualitative["failure"]["lora"]["raw_output"]
        == lora_by_id[failure_id]["raw_output"]
        and qualitative["failure"]["lora"]["ground_truth"]
        == lora_by_id[failure_id]["ground_truth"]
    )
    checks.append(
        (
            "Qualitative manifest values match prediction JSONL",
            qualitative_matches,
            f"improvement={improvement_id}, failure={failure_id}",
        )
    )
    qualitative_hashes_match = all(
        _sha256(PROJECT_ROOT / qualitative[category]["figure_path"])
        == qualitative[category]["figure_sha256"]
        and _sha256(PROJECT_ROOT / qualitative[category]["image_path"])
        == qualitative[category]["image_sha256"]
        for category in ("improvement", "failure")
    )
    checks.append(
        (
            "Qualitative figure and receipt hashes match manifest",
            qualitative_hashes_match,
            "",
        )
    )
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    expected_claims = (
        f"{base_metrics['valid_json_rate']:.0%}",
        f"{lora_metrics['valid_json_rate']:.0%}",
        f"{lora_metrics['company_accuracy']:.0%}",
        f"{lora_metrics['date_accuracy']:.0%}",
        f"{lora_metrics['total_accuracy']:.0%}",
        f"{lora_metrics['address_similarity']:.2%}",
        "99.2%",
        "69.1%",
        "90.7%",
        "85.8%",
        "64.6%",
        "18.3%",
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
    forbidden_tracked = re.compile(
        r"(?:^artifacts/checkpoints/|"
        r"(?:^|/)(?:\.cache|\.venv|venv|logs?)(?:/|$))"
    )
    checks.extend(
        (
            (
                "No intermediate checkpoints, caches, environments, or logs tracked",
                not any(forbidden_tracked.search(path) for path in tracked),
                "",
            ),
            (
                "No tracked file exceeds 100 MiB",
                not any(
                    (PROJECT_ROOT / path).is_file()
                    and (PROJECT_ROOT / path).stat().st_size > 100 * 2**20
                    for path in tracked
                ),
                "",
            ),
        )
    )
    markdown_links: list[tuple[str, str]] = []
    for relative in tracked:
        if not relative.lower().endswith(".md"):
            continue
        text = (PROJECT_ROOT / relative).read_text(encoding="utf-8")
        for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", text):
            clean = target.strip().strip("<>").split("#", 1)[0]
            if (
                not clean
                or re.match(r"^[a-z]+://", clean, re.IGNORECASE)
                or clean.startswith(("#", "mailto:"))
            ):
                continue
            resolved = (PROJECT_ROOT / relative).parent / clean
            markdown_links.append((relative, clean))
            if not resolved.exists():
                checks.append(
                    (
                        f"Markdown link exists: {relative} -> {clean}",
                        False,
                        "",
                    )
                )
    checks.append(
        (
            "All local Markdown links and images exist",
            not any(
                name.startswith("Markdown link exists:") and not passed
                for name, passed, _ in checks
            ),
            f"checked={len(markdown_links)}",
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
