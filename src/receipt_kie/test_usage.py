"""Auditable accounting of official-test receipts used by prior experiments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from receipt_kie.dataset import load_records
from receipt_kie.utils import project_path, repository_relative, write_json

PRE_ABLATION_ARTIFACTS = {
    "v1_base_predictions": (
        "artifacts/predictions/base_predictions.jsonl",
        "jsonl",
    ),
    "v1_lora_predictions": (
        "artifacts/predictions/lora_predictions.jsonl",
        "jsonl",
    ),
    "robustness_clean": (
        "artifacts/predictions/robustness_clean.jsonl",
        "jsonl",
    ),
    "robustness_gaussian_blur": (
        "artifacts/predictions/robustness_gaussian_blur.jsonl",
        "jsonl",
    ),
    "robustness_jpeg_compression": (
        "artifacts/predictions/robustness_jpeg_compression.jsonl",
        "jsonl",
    ),
    "robustness_reduced_brightness": (
        "artifacts/predictions/robustness_reduced_brightness.jsonl",
        "jsonl",
    ),
    "robustness_rotation": (
        "artifacts/predictions/robustness_rotation.jsonl",
        "jsonl",
    ),
    "qualitative_example_index": (
        "artifacts/predictions/examples/index.json",
        "nested_id_lists",
    ),
    "qualitative_results_manifest": (
        "artifacts/reports/qualitative_results_manifest.json",
        "qualitative_manifest",
    ),
}
HIGHRES_ARTIFACT = (
    "artifacts/experiments/highres_ablation/test_ids.json",
    "id_list",
)


def build_test_usage_manifest(
    dataset_root: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Calculate prior-use unions, overlaps, unseen IDs, and stable list hashes."""
    all_official_ids = sorted(
        record.sample_id
        for record in load_records(dataset_root, "test", limit=None, seed=42)
    )
    all_official = set(all_official_ids)
    artifact_ids = {
        name: _read_ids(path, format_name)
        for name, (path, format_name) in PRE_ABLATION_ARTIFACTS.items()
    }
    highres_path, highres_format = HIGHRES_ARTIFACT
    highres_ids = _read_ids(highres_path, highres_format)
    artifact_ids["high_resolution_ablation"] = highres_ids
    outside_official = {
        name: sorted(set(ids) - all_official)
        for name, ids in artifact_ids.items()
        if set(ids) - all_official
    }
    if outside_official:
        raise ValueError(
            f"Prior artifacts contain IDs outside the official test split: {outside_official}"
        )

    pre_ablation = set().union(
        *(set(ids) for name, ids in artifact_ids.items() if name != "high_resolution_ablation")
    )
    highres = set(highres_ids)
    previously_evaluated = pre_ablation | highres
    remaining = all_official - previously_evaluated
    ordered_lists = {
        "all_official_test_ids": all_official_ids,
        "pre_ablation_evaluated_ids": sorted(pre_ablation),
        "previously_evaluated_ids": sorted(previously_evaluated),
        "high_resolution_development_ids": sorted(highres),
        "remaining_never_evaluated_ids": sorted(remaining),
    }
    artifact_summary = {}
    for name, ids in artifact_ids.items():
        values = set(ids)
        artifact_summary[name] = {
            "path": (
                repository_relative(HIGHRES_ARTIFACT[0])
                if name == "high_resolution_ablation"
                else repository_relative(PRE_ABLATION_ARTIFACTS[name][0])
            ),
            "count": len(values),
            "ids": sorted(values),
            "sha256": hash_id_list(sorted(values)),
            "overlap_with_pre_ablation_evaluated_count": len(values & pre_ablation),
            "overlap_with_high_resolution_development_count": len(values & highres),
            "overlap_with_all_previously_evaluated_count": len(
                values & previously_evaluated
            ),
        }
    pairwise = {
        left: {
            right: len(set(artifact_ids[left]) & set(artifact_ids[right]))
            for right in artifact_ids
        }
        for left in artifact_ids
    }
    manifest = {
        "schema_version": 1,
        "source_split": "official_test",
        "dataset_root": repository_relative(dataset_root),
        "definitions": {
            "pre_ablation_evaluated_ids": (
                "Union of V1 base/LoRA predictions, robustness predictions, "
                "and qualitative selections."
            ),
            "previously_evaluated_ids": (
                "Union of all pre-ablation artifacts and the 30-receipt "
                "high-resolution inference-development ablation."
            ),
            "remaining_never_evaluated_ids": (
                "Official-test IDs absent from every listed prior artifact."
            ),
            "list_sha256": (
                "SHA-256 of compact UTF-8 JSON serialization of each sorted ID list."
            ),
        },
        "counts": {name: len(ids) for name, ids in ordered_lists.items()},
        **ordered_lists,
        "overlap_counts": {
            "highres_with_pre_ablation": len(highres & pre_ablation),
            "highres_not_in_pre_ablation": len(highres - pre_ablation),
            "pre_ablation_not_in_highres": len(pre_ablation - highres),
        },
        "artifacts": artifact_summary,
        "pairwise_overlap_counts_by_artifact": pairwise,
        "list_sha256": {
            name: hash_id_list(ids) for name, ids in ordered_lists.items()
        },
        "selection_policy": {
            "selection_source": "63 validation receipts from official train split",
            "official_test_selection_forbidden_until_frozen": True,
        },
    }
    write_json(output_path, manifest)
    return manifest


def hash_id_list(ids: list[str]) -> str:
    """Hash an ordered ID list using an explicit stable serialization."""
    payload = json.dumps(ids, ensure_ascii=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def _read_ids(path: str | Path, format_name: str) -> list[str]:
    resolved = project_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Required prior-use artifact is missing: {resolved}")
    if format_name == "jsonl":
        rows = [
            json.loads(line)
            for line in resolved.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        values = [str(row["sample_id"]) for row in rows]
    else:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        if format_name == "id_list":
            values = [str(value) for value in payload]
        elif format_name == "nested_id_lists":
            values = [
                str(value)
                for category_values in payload.values()
                for value in category_values
            ]
        elif format_name == "qualitative_manifest":
            values = [
                str(section["sample_id"])
                for section in payload.values()
                if isinstance(section, dict) and "sample_id" in section
            ]
        else:
            raise ValueError(f"Unsupported ID artifact format: {format_name}")
    if len(values) != len(set(values)) and format_name not in {
        "nested_id_lists",
        "qualitative_manifest",
    }:
        raise ValueError(f"Duplicate sample IDs in {resolved}")
    return sorted(set(values))
