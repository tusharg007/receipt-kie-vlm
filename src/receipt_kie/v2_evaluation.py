"""Validation selection and frozen unseen evaluation for high-resolution V2."""

from __future__ import annotations

import json
import logging
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any

from receipt_kie.config import load_config
from receipt_kie.highres_training import (
    load_v1_split_records,
    sha256_file,
)
from receipt_kie.inference import ReceiptKIEPredictor
from receipt_kie.metrics import evaluate_predictions, has_repetition_failure
from receipt_kie.utils import (
    project_path,
    repository_relative,
    seed_everything,
    setup_logging,
    write_json,
    write_jsonl,
)

LOGGER = logging.getLogger(__name__)
POLICY_NO_PENALTY = "no_repetition_penalty"
POLICY_ALWAYS_PENALTY = "always_repetition_penalty_1p08"
POLICY_ADAPTIVE_RETRY = "adaptive_retry_1p08"


def run_validation_selection(config_path: str | Path) -> dict[str, Any]:
    """Score every epoch candidate and decoding policy on all 63 validation IDs."""
    config = load_config(config_path)
    setup_logging(config["paths"]["selection_log_file"])
    seed = int(config["project"]["seed"])
    seed_everything(seed)
    _, validation_records = load_v1_split_records(config)
    expected_ids = [record.sample_id for record in validation_records]
    training_summary = json.loads(
        project_path(config["paths"]["training_summary_path"]).read_text(
            encoding="utf-8"
        )
    )
    candidates = training_summary["candidate_checkpoints"]
    results_path = project_path(config["paths"]["validation_results_path"])
    if results_path.is_file():
        results = json.loads(results_path.read_text(encoding="utf-8"))
    else:
        results = {
            "experiment": "highres_training_v2_validation_selection",
            "source_split": "official_train_validation",
            "sample_count": len(validation_records),
            "validation_ids": expected_ids,
            "resolution": int(config["model"]["image_longest_edge"]),
            "max_new_tokens": int(config["generation"]["max_new_tokens"]),
            "deterministic_decoding": True,
            "score_formula": selection_formula(config),
            "score_weights": config["selection"]["weights"],
            "candidates": {},
        }
    if results["validation_ids"] != expected_ids:
        raise ValueError("Existing validation results use different validation IDs")
    for candidate in candidates:
        candidate_name = Path(candidate["path"]).name
        candidate_path = project_path(candidate["path"])
        candidate_result = run_candidate_validation(
            config,
            candidate_name,
            candidate_path,
            validation_records,
        )
        results["candidates"][candidate_name] = candidate_result
        write_json(results_path, results)
    selection = select_best_validation_candidate(config, results)
    results["selection"] = selection
    write_json(results_path, results)
    write_json(config["paths"]["frozen_selection_path"], selection)
    promote_selected_adapter(config, selection, training_summary)
    return results


def run_candidate_validation(
    config: dict[str, Any],
    candidate_name: str,
    candidate_path: Path,
    records: list[Any],
) -> dict[str, Any]:
    """Run both deterministic policies once, then construct adaptive retry."""
    model_config = deepcopy(config["model"])
    predictor = ReceiptKIEPredictor(
        model_config,
        config["paths"]["hf_cache"],
        adapter_path=candidate_path,
    )
    try:
        no_penalty_path = validation_prediction_path(
            config,
            candidate_name,
            POLICY_NO_PENALTY,
        )
        no_penalty = run_or_resume_predictions(
            predictor,
            records,
            no_penalty_path,
            max_new_tokens=int(config["generation"]["max_new_tokens"]),
            repetition_penalty=None,
        )
        penalty_path = validation_prediction_path(
            config,
            candidate_name,
            POLICY_ALWAYS_PENALTY,
        )
        penalty = run_or_resume_predictions(
            predictor,
            records,
            penalty_path,
            max_new_tokens=int(config["generation"]["max_new_tokens"]),
            repetition_penalty=float(config["generation"]["repetition_penalty"]),
        )
    finally:
        predictor.close()
    adaptive = build_adaptive_retry_rows(no_penalty, penalty)
    adaptive_path = validation_prediction_path(
        config,
        candidate_name,
        POLICY_ADAPTIVE_RETRY,
    )
    write_jsonl(adaptive_path, adaptive)
    policy_rows = {
        POLICY_NO_PENALTY: no_penalty,
        POLICY_ALWAYS_PENALTY: penalty,
        POLICY_ADAPTIVE_RETRY: adaptive,
    }
    policies = {}
    for policy_name, rows in policy_rows.items():
        metrics = policy_metrics(rows)
        metrics["selection_score"] = validation_score(config, metrics)
        policies[policy_name] = {
            "metrics": metrics,
            "predictions_path": repository_relative(
                validation_prediction_path(config, candidate_name, policy_name)
            ),
        }
    return {
        "checkpoint": candidate_name,
        "checkpoint_path": repository_relative(candidate_path),
        "adapter_sha256": sha256_file(
            candidate_path / "adapter_model.safetensors"
        ),
        "policies": policies,
    }


def run_or_resume_predictions(
    predictor: ReceiptKIEPredictor,
    records: list[Any],
    output_path: Path,
    max_new_tokens: int,
    repetition_penalty: float | None,
) -> list[dict[str, Any]]:
    """Checkpoint deterministic validation inference after every receipt."""
    if output_path.is_file():
        rows = [
            json.loads(line)
            for line in output_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        rows = []
    expected_prefix = [record.sample_id for record in records[: len(rows)]]
    if [row["sample_id"] for row in rows] != expected_prefix:
        raise ValueError(f"Prediction resume IDs do not match: {output_path}")
    for index, record in enumerate(records[len(rows) :], start=len(rows) + 1):
        prediction = predictor.predict(
            record.image_path,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=repetition_penalty,
        )
        prediction["repetition_failure"] = has_repetition_failure(
            prediction["raw_output"]
        )
        rows.append(
            {
                "sample_id": record.sample_id,
                "image_path": repository_relative(record.image_path),
                "ground_truth": record.target,
                **prediction,
            }
        )
        write_jsonl(output_path, rows)
        LOGGER.info(
            "validation checkpoint=%s policy_penalty=%s sample=%d/%d id=%s "
            "valid=%s limit=%s repetition=%s",
            output_path.parent.name,
            repetition_penalty,
            index,
            len(records),
            record.sample_id,
            prediction["valid_json"],
            prediction["generation_limit_hit"],
            prediction["repetition_failure"],
        )
    return rows


def build_adaptive_retry_rows(
    initial_rows: list[dict[str, Any]],
    penalty_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Use the penalty result only when the deterministic first attempt fails."""
    penalty_by_id = {row["sample_id"]: row for row in penalty_rows}
    adaptive = []
    for initial in initial_rows:
        reasons = retry_reasons(initial)
        if not reasons:
            row = deepcopy(initial)
            row.update(
                {
                    "retry_performed": False,
                    "retry_reasons": [],
                    "initial_latency_seconds": initial["latency_seconds"],
                }
            )
        else:
            retry = deepcopy(penalty_by_id[initial["sample_id"]])
            retry.update(
                {
                    "retry_performed": True,
                    "retry_reasons": reasons,
                    "initial_raw_output": initial["raw_output"],
                    "initial_valid_json": initial["valid_json"],
                    "initial_generation_limit_hit": initial[
                        "generation_limit_hit"
                    ],
                    "initial_repetition_failure": initial[
                        "repetition_failure"
                    ],
                    "initial_latency_seconds": initial["latency_seconds"],
                    "latency_seconds": float(initial["latency_seconds"])
                    + float(retry["latency_seconds"]),
                    "peak_gpu_memory_mib": max(
                        float(initial["peak_gpu_memory_mib"] or 0.0),
                        float(retry["peak_gpu_memory_mib"] or 0.0),
                    ),
                    "visual_tile_count": int(initial["visual_tile_count"])
                    + int(retry["visual_tile_count"]),
                }
            )
            row = retry
        adaptive.append(row)
    return adaptive


def retry_reasons(row: dict[str, Any]) -> list[str]:
    reasons = []
    if not row["valid_json"]:
        reasons.append("invalid_json")
    if row["generation_limit_hit"]:
        reasons.append("generation_limit")
    if row["repetition_failure"]:
        reasons.append("repetition_failure")
    return reasons


def policy_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = evaluate_predictions(rows)
    sample_count = len(rows)
    limit_hits = sum(bool(row["generation_limit_hit"]) for row in rows)
    repetitions = sum(bool(row["repetition_failure"]) for row in rows)
    retries = sum(bool(row.get("retry_performed")) for row in rows)
    metrics.update(
        {
            "generation_limit_hit_count": limit_hits,
            "generation_limit_rate": limit_hits / sample_count,
            "repetition_failure_count": repetitions,
            "repetition_failure_rate": repetitions / sample_count,
            "retry_count": retries,
            "retry_rate": retries / sample_count,
        }
    )
    return metrics


def validation_score(config: dict[str, Any], metrics: dict[str, Any]) -> float:
    weights = config["selection"]["weights"]
    return (
        float(weights["macro_normalized_exact_match"])
        * float(metrics["macro_normalized_exact_match"])
        + float(weights["macro_normalized_similarity"])
        * float(metrics["macro_normalized_similarity"])
        + float(weights["complete_record_normalized_exact_match"])
        * float(metrics["complete_record_normalized_exact_match"])
        + float(weights["valid_json_rate"])
        * float(metrics["valid_json_rate"])
        - float(weights["generation_limit_rate"])
        * float(metrics["generation_limit_rate"])
        - float(weights["repetition_failure_rate"])
        * float(metrics["repetition_failure_rate"])
    )


def selection_formula(config: dict[str, Any]) -> str:
    weights = config["selection"]["weights"]
    return (
        f"{weights['macro_normalized_exact_match']} * macro_normalized_exact_match "
        f"+ {weights['macro_normalized_similarity']} * macro_normalized_similarity "
        f"+ {weights['complete_record_normalized_exact_match']} * "
        "complete_record_normalized_exact_match "
        f"+ {weights['valid_json_rate']} * valid_json_rate "
        f"- {weights['generation_limit_rate']} * generation_limit_rate "
        f"- {weights['repetition_failure_rate']} * repetition_failure_rate"
    )


def select_best_validation_candidate(
    config: dict[str, Any],
    results: dict[str, Any],
) -> dict[str, Any]:
    choices = []
    for candidate_name, candidate in results["candidates"].items():
        for policy_name, policy in candidate["policies"].items():
            choices.append(
                (
                    float(policy["metrics"]["selection_score"]),
                    float(policy["metrics"]["macro_normalized_exact_match"]),
                    float(
                        policy["metrics"][
                            "complete_record_normalized_exact_match"
                        ]
                    ),
                    float(policy["metrics"]["valid_json_rate"]),
                    candidate_name,
                    policy_name,
                    candidate,
                    policy,
                )
            )
    winner = max(choices, key=lambda row: row[:4])
    _, _, _, _, candidate_name, policy_name, candidate, policy = winner
    return {
        "frozen": True,
        "selection_source": "63 validation receipts from official train split",
        "official_test_inspected_during_selection": False,
        "resolution": int(config["model"]["image_longest_edge"]),
        "max_image_patch_edge": int(config["model"]["max_image_patch_edge"]),
        "do_image_splitting": bool(config["model"]["do_image_splitting"]),
        "max_new_tokens": int(config["generation"]["max_new_tokens"]),
        "checkpoint": candidate_name,
        "checkpoint_path": candidate["checkpoint_path"],
        "checkpoint_adapter_sha256": candidate["adapter_sha256"],
        "generation_policy": policy_name,
        "repetition_penalty": (
            float(config["generation"]["repetition_penalty"])
            if policy_name != POLICY_NO_PENALTY
            else None
        ),
        "adaptive_retry_trigger": (
            ["invalid_json", "generation_limit", "repetition_failure"]
            if policy_name == POLICY_ADAPTIVE_RETRY
            else []
        ),
        "parsing_behavior": "receipt_kie.metrics.extract_json",
        "selection_score_formula": selection_formula(config),
        "selection_score": policy["metrics"]["selection_score"],
        "validation_metrics": policy["metrics"],
    }


def promote_selected_adapter(
    config: dict[str, Any],
    selection: dict[str, Any],
    training_summary: dict[str, Any],
) -> None:
    """Copy only the validation-selected candidate into the new V2 model path."""
    source = project_path(selection["checkpoint_path"])
    destination = project_path(config["paths"]["final_adapter_path"])
    destination.mkdir(parents=True, exist_ok=True)
    source_adapter = source / "adapter_model.safetensors"
    source_config = source / "adapter_config.json"
    target_adapter = destination / "adapter_model.safetensors"
    target_config = destination / "adapter_config.json"
    if target_adapter.exists() and sha256_file(target_adapter) != sha256_file(
        source_adapter
    ):
        raise FileExistsError(
            f"Refusing to overwrite a different V2 adapter: {target_adapter}"
        )
    shutil.copy2(source_adapter, target_adapter)
    shutil.copy2(source_config, target_config)
    metadata = {
        "base_model_id": config["model"]["model_id"],
        "continued_from_v1_adapter_sha256": training_summary[
            "adapter_source_sha256"
        ],
        "adapter_sha256": sha256_file(target_adapter),
        "adapter_size_bytes": target_adapter.stat().st_size,
        "resolution": selection["resolution"],
        "max_image_patch_edge": selection["max_image_patch_edge"],
        "do_image_splitting": selection["do_image_splitting"],
        "training_sample_count": training_summary["train_sample_count"],
        "validation_sample_count": training_summary["validation_sample_count"],
        "official_test_used_during_training_or_selection": False,
        "optimizer_step_count": training_summary["completed_optimization_steps"],
        "checkpoint_selected": selection["checkpoint"],
        "checkpoint_selection_source": selection["selection_source"],
        "generation_policy_selected": selection["generation_policy"],
        "selection_score": selection["selection_score"],
        "selection_score_formula": selection["selection_score_formula"],
        "lora_rank": training_summary["adapter_structure"]["rank"],
        "lora_alpha": training_summary["adapter_structure"]["alpha"],
        "lora_dropout": training_summary["adapter_structure"]["dropout"],
        "target_modules": training_summary["adapter_structure"][
            "target_modules"
        ],
    }
    write_json(destination / "training_metadata.json", metadata)


def validation_prediction_path(
    config: dict[str, Any],
    candidate_name: str,
    policy_name: str,
) -> Path:
    return (
        project_path(config["paths"]["validation_predictions_root"])
        / candidate_name
        / f"{policy_name}.jsonl"
    )
