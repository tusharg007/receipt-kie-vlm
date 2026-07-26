"""LoRA supervised fine-tuning for structured receipt extraction."""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from datasets import Dataset
from trl import SFTConfig, SFTTrainer

from receipt_kie.collator import ReceiptKIECollator
from receipt_kie.config import load_config
from receipt_kie.dataset import load_records, partition_train_validation
from receipt_kie.model import attach_lora, load_base_model, load_processor
from receipt_kie.utils import (
    ensure_artifact_directories,
    package_versions,
    project_path,
    repository_relative,
    seed_everything,
    setup_logging,
    write_json,
)

LOGGER = logging.getLogger(__name__)


def run_training(
    config_path: str | Path,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train and save a newly initialized LoRA adapter."""
    config = load_config(config_path)
    _apply_overrides(config, overrides or {})
    ensure_artifact_directories()
    setup_logging(config["paths"]["log_file"])
    seed = int(config["project"]["seed"])
    seed_everything(seed)
    dataset_root = config["paths"]["dataset_root"]
    train_limit = config["training"].get("train_limit")
    validation_size = int(config["training"]["validation_size"])
    source_records = load_records(dataset_root, "train", None, seed)
    train_records, eval_records = partition_train_validation(
        source_records,
        validation_size=validation_size,
        seed=seed,
        train_limit=train_limit,
    )
    train_dataset = Dataset.from_list([record.trainer_row() for record in train_records])
    eval_dataset = Dataset.from_list([record.trainer_row() for record in eval_records])
    processor = load_processor(config["model"], config["paths"]["hf_cache"])
    model, runtime = load_base_model(config["model"], config["paths"]["hf_cache"], training=True)
    model, parameter_stats = attach_lora(model, config["lora"])
    if config["training"].get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    collator = ReceiptKIECollator(processor)
    output_dir = project_path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    training_args = _training_arguments(config, output_dir, runtime.dtype_name)
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        tokenizer=processor.tokenizer,
    )
    trainer.can_return_loss = True
    first_trainable = next(parameter for parameter in model.parameters() if parameter.requires_grad)
    initial_checksum = float(first_trainable.detach().float().sum().cpu())
    started = time.perf_counter()
    train_result = trainer.train()
    duration = time.perf_counter() - started
    final_checksum = float(first_trainable.detach().float().sum().cpu())
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))
    trainer.save_state()
    log_history = trainer.state.log_history
    losses = [
        {"step": int(row["step"]), "loss": float(row["loss"])}
        for row in log_history
        if "loss" in row and math.isfinite(float(row["loss"]))
    ]
    validation_losses = [
        {
            "step": int(row["step"]),
            "epoch": float(row["epoch"]),
            "eval_loss": float(row["eval_loss"]),
        }
        for row in log_history
        if "eval_loss" in row and math.isfinite(float(row["eval_loss"]))
    ]
    summary = {
        "config_path": repository_relative(config_path),
        "output_dir": repository_relative(output_dir),
        "runtime": runtime.to_dict(),
        "parameter_statistics": parameter_stats,
        "train_samples": len(train_records),
        "eval_samples": len(eval_records),
        "validation_samples": len(eval_records),
        "train_source_split": "train",
        "validation_source_split": "train",
        "official_test_used_during_training": False,
        "global_steps": trainer.state.global_step,
        "duration_seconds": duration,
        "seconds_per_optimization_step": duration / max(trainer.state.global_step, 1),
        "peak_gpu_memory_mib": (
            torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else None
        ),
        "finite_loss_count": len(losses),
        "finite_validation_loss_count": len(validation_losses),
        "final_validation_loss": (
            validation_losses[-1]["eval_loss"] if validation_losses else None
        ),
        "lora_parameter_changed": not math.isclose(
            initial_checksum, final_checksum, rel_tol=0.0, abs_tol=1e-9
        ),
        "train_metrics": train_result.metrics,
        "masking_report_last_batch": collator.last_masking_report,
        "seed": seed,
    }
    write_json(output_dir / "resolved_config.json", config)
    write_json(output_dir / "package_versions.json", package_versions())
    write_json(
        output_dir / "dataset_split_manifest.json",
        {
            "train_ids": [record.sample_id for record in train_records],
            "validation_ids": [record.sample_id for record in eval_records],
            "eval_ids": [record.sample_id for record in eval_records],
            "train_source_split": "train",
            "validation_source_split": "train",
            "official_test_used_during_training": False,
            "overlap_ids": sorted(
                {record.sample_id for record in train_records}
                & {record.sample_id for record in eval_records}
            ),
            "seed": seed,
        },
    )
    write_json(output_dir / "loss_history.json", losses)
    write_json(output_dir / "validation_loss_history.json", validation_losses)
    write_json(output_dir / "training_summary.json", summary)
    _plot_losses(losses, validation_losses)
    LOGGER.info("Training complete: %s", json.dumps(summary, default=str))
    return summary


def _training_arguments(
    config: dict[str, Any],
    output_dir: Path,
    dtype_name: str,
) -> SFTConfig:
    values = config["training"]
    max_steps = values.get("max_steps")
    return SFTConfig(
        output_dir=str(output_dir),
        logging_dir=str(project_path("artifacts/logs/tensorboard")),
        num_train_epochs=float(values["num_train_epochs"]),
        max_steps=-1 if max_steps is None else int(max_steps),
        per_device_train_batch_size=int(values["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(values["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(values["gradient_accumulation_steps"]),
        learning_rate=float(values["learning_rate"]),
        weight_decay=float(values["weight_decay"]),
        warmup_ratio=float(values["warmup_ratio"]),
        logging_steps=int(values["logging_steps"]),
        save_strategy=str(values["save_strategy"]),
        save_steps=int(values["save_steps"]),
        eval_strategy=str(values["eval_strategy"]),
        eval_steps=int(values["eval_steps"]),
        save_total_limit=int(values["save_total_limit"]),
        gradient_checkpointing=bool(values["gradient_checkpointing"]),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=dtype_name == "bf16",
        fp16=dtype_name == "fp16",
        optim="adamw_torch_fused",
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=int(values["dataloader_num_workers"]),
        dataloader_pin_memory=False,
        max_seq_length=int(values["max_seq_length"]),
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        save_safetensors=True,
        seed=int(config["project"]["seed"]),
    )


def _plot_losses(
    losses: list[dict[str, float]],
    validation_losses: list[dict[str, float]] | None = None,
) -> None:
    if not losses:
        return
    output = project_path("artifacts/figures/training_loss.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4.8))
    plt.plot(
        [row["step"] for row in losses],
        [row["loss"] for row in losses],
        marker="o",
        markersize=4,
        linewidth=1.8,
        label="Training loss",
    )
    if validation_losses:
        plt.scatter(
            [row["step"] for row in validation_losses],
            [row["eval_loss"] for row in validation_losses],
            marker="D",
            s=58,
            color="#d55e00",
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
            label="Validation loss",
        )
    plt.xlabel("Optimization step")
    plt.ylabel("Loss")
    plt.title("ReceiptKIE-VLM LoRA training loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def _apply_overrides(config: dict[str, Any], overrides: dict[str, Any]) -> None:
    for dotted_key, value in overrides.items():
        target = config
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
