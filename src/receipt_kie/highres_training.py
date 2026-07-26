"""Leakage-controlled continued high-resolution LoRA training utilities."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import TrainerCallback
from trl import SFTConfig, SFTTrainer

from receipt_kie.collator import ReceiptKIECollator
from receipt_kie.config import load_config
from receipt_kie.dataset import ReceiptRecord, load_records
from receipt_kie.model import load_base_model, load_processor, parameter_statistics
from receipt_kie.test_usage import hash_id_list
from receipt_kie.utils import (
    clear_cuda,
    package_versions,
    project_path,
    repository_relative,
    seed_everything,
    setup_logging,
    write_json,
)

LOGGER = logging.getLogger(__name__)


def run_highres_training_smoke(config_path: str | Path) -> dict[str, Any]:
    """Continue the V1 adapter for five steps and record the memory safety gate."""
    config = load_config(config_path)
    setup_logging(config["paths"]["log_file"])
    seed = int(config["project"]["seed"])
    seed_everything(seed)
    resolution = int(config["model"]["image_longest_edge"])
    report_path = project_path(config["paths"]["report_path"])
    report: dict[str, Any] = {
        "experiment": "high_resolution_continued_training_smoke",
        "config_path": repository_relative(config_path),
        "resolution": resolution,
        "status": "started",
        "success": False,
        "oom": False,
        "safe_memory": False,
    }
    model: Any | None = None
    trainer: SFTTrainer | None = None
    try:
        train_records, validation_records = load_v1_split_records(config)
        smoke_train = deterministic_subset(
            train_records,
            int(config["smoke"]["train_sample_count"]),
            seed,
        )
        smoke_validation = deterministic_subset(
            validation_records,
            int(config["smoke"]["validation_sample_count"]),
            seed + 1,
        )
        processor = load_processor(config["model"], config["paths"]["hf_cache"])
        base_model, runtime = load_base_model(
            config["model"],
            config["paths"]["hf_cache"],
            training=True,
        )
        model = PeftModel.from_pretrained(
            base_model,
            str(project_path(config["paths"]["v1_adapter_path"])),
            is_trainable=True,
        )
        adapter_structure = validate_continued_adapter(model, config["lora"])
        if bool(config["training"]["gradient_checkpointing"]):
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            model.enable_input_require_grads()
        collator = ReceiptKIECollator(
            processor,
            image_longest_edge_override=resolution,
        )
        profile = profile_records(
            collator,
            [*smoke_train, *smoke_validation],
        )
        train_dataset = Dataset.from_list(
            [record.trainer_row() for record in smoke_train]
        )
        eval_dataset = Dataset.from_list(
            [record.trainer_row() for record in smoke_validation]
        )
        output_dir = project_path(config["paths"]["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        training_args = smoke_training_arguments(
            config,
            output_dir,
            runtime.dtype_name,
        )
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=collator,
            tokenizer=processor.tokenizer,
        )
        trainer.can_return_loss = True
        initial_lora_sha256 = hash_trainable_tensors(model)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        started = time.perf_counter()
        train_result = trainer.train()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        duration = time.perf_counter() - started
        final_lora_sha256 = hash_trainable_tensors(model)
        trainer.save_model(str(output_dir))
        processor.save_pretrained(str(output_dir))
        trainer.save_state()
        log_history = trainer.state.log_history
        training_losses = [
            {
                "step": int(row["step"]),
                "loss": float(row["loss"]),
            }
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
        peak_allocated = (
            torch.cuda.max_memory_allocated() / 2**20
            if torch.cuda.is_available()
            else None
        )
        peak_reserved = (
            torch.cuda.max_memory_reserved() / 2**20
            if torch.cuda.is_available()
            else None
        )
        expected_steps = int(config["training"]["max_steps"])
        safe_threshold = float(config["memory"]["unsafe_allocated_mib"])
        safe_memory = bool(
            peak_allocated is None or peak_allocated <= safe_threshold
        )
        report.update(
            {
                "status": "success",
                "success": trainer.state.global_step == expected_steps,
                "runtime": runtime.to_dict(),
                "train_ids": [record.sample_id for record in smoke_train],
                "validation_ids": [
                    record.sample_id for record in smoke_validation
                ],
                "train_sample_count": len(smoke_train),
                "validation_sample_count": len(smoke_validation),
                "official_test_used": False,
                "adapter_source": repository_relative(
                    config["paths"]["v1_adapter_path"]
                ),
                "adapter_source_sha256": sha256_file(
                    project_path(config["paths"]["v1_adapter_path"])
                    / "adapter_model.safetensors"
                ),
                "adapter_structure": adapter_structure,
                "parameter_statistics": parameter_statistics(model),
                "precision": runtime.dtype_name,
                "gradient_checkpointing": bool(
                    config["training"]["gradient_checkpointing"]
                ),
                "batch_size": int(
                    config["training"]["per_device_train_batch_size"]
                ),
                "gradient_accumulation_steps": int(
                    config["training"]["gradient_accumulation_steps"]
                ),
                "optimization_steps": trainer.state.global_step,
                "duration_seconds": duration,
                "seconds_per_optimization_step": duration
                / max(trainer.state.global_step, 1),
                "peak_allocated_gpu_memory_mib": peak_allocated,
                "peak_reserved_gpu_memory_mib": peak_reserved,
                "unsafe_allocated_threshold_mib": safe_threshold,
                "safe_memory_margin_mib": (
                    safe_threshold - peak_allocated
                    if peak_allocated is not None
                    else None
                ),
                "safe_memory": safe_memory,
                "data_profile": profile,
                "training_losses": training_losses,
                "validation_losses": validation_losses,
                "final_training_loss": (
                    training_losses[-1]["loss"] if training_losses else None
                ),
                "final_validation_loss": (
                    validation_losses[-1]["eval_loss"]
                    if validation_losses
                    else None
                ),
                "initial_lora_tensor_sha256": initial_lora_sha256,
                "final_lora_tensor_sha256": final_lora_sha256,
                "lora_tensors_changed": initial_lora_sha256
                != final_lora_sha256,
                "train_metrics": train_result.metrics,
                "output_dir": repository_relative(output_dir),
                "package_versions": package_versions(),
            }
        )
        del trainer
        trainer = None
        del model
        model = None
        del processor
        clear_cuda()
        reload_verified = verify_adapter_reload(config, output_dir)
        report["adapter_reload_verified"] = reload_verified
        report["saved_adapter_sha256"] = sha256_file(
            output_dir / "adapter_model.safetensors"
        )
        report["passed"] = bool(
            report["success"]
            and report["lora_tensors_changed"]
            and reload_verified
            and training_losses
            and validation_losses
        )
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        if not _is_oom(exc):
            report.update(
                {
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "passed": False,
                }
            )
            write_json(report_path, report)
            raise
        report.update(
            {
                "status": "oom",
                "oom": True,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "peak_allocated_gpu_memory_mib": (
                    torch.cuda.max_memory_allocated() / 2**20
                    if torch.cuda.is_available()
                    else None
                ),
                "peak_reserved_gpu_memory_mib": (
                    torch.cuda.max_memory_reserved() / 2**20
                    if torch.cuda.is_available()
                    else None
                ),
                "passed": False,
            }
        )
    finally:
        if trainer is not None:
            del trainer
        if model is not None:
            del model
        clear_cuda()
        write_json(report_path, report)
    return report


def run_highres_continued_training(config_path: str | Path) -> dict[str, Any]:
    """Continue V1 on the exact 563/63 split and preserve epoch candidates."""
    config = load_config(config_path)
    setup_logging(config["paths"]["log_file"])
    seed = int(config["project"]["seed"])
    seed_everything(seed)
    train_records, validation_records = load_v1_split_records(config)
    output_dir = project_path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    processor = load_processor(config["model"], config["paths"]["hf_cache"])
    base_model, runtime = load_base_model(
        config["model"],
        config["paths"]["hf_cache"],
        training=True,
    )
    model = PeftModel.from_pretrained(
        base_model,
        str(project_path(config["paths"]["v1_adapter_path"])),
        is_trainable=True,
    )
    adapter_structure = validate_continued_adapter(model, config["lora"])
    if bool(config["training"]["gradient_checkpointing"]):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()
    collator = ReceiptKIECollator(
        processor,
        image_longest_edge_override=int(config["model"]["image_longest_edge"]),
    )
    trainer = SFTTrainer(
        model=model,
        args=continued_training_arguments(config, output_dir, runtime.dtype_name),
        train_dataset=Dataset.from_list(
            [record.trainer_row() for record in train_records]
        ),
        eval_dataset=Dataset.from_list(
            [record.trainer_row() for record in validation_records]
        ),
        data_collator=collator,
        tokenizer=processor.tokenizer,
        callbacks=[
            ConsecutiveValidationEarlyStoppingCallback(
                patience=int(config["training"]["early_stopping_patience"]),
                min_delta=float(config["training"]["early_stopping_min_delta"]),
            )
        ],
    )
    trainer.can_return_loss = True
    callback = next(
        item
        for item in trainer.callback_handler.callbacks
        if isinstance(item, ConsecutiveValidationEarlyStoppingCallback)
    )
    initial_lora_sha256 = hash_trainable_tensors(model)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    train_result = trainer.train()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    duration = time.perf_counter() - started
    final_lora_sha256 = hash_trainable_tensors(model)
    last_dir = output_dir / "last"
    trainer.save_model(str(last_dir))
    processor.save_pretrained(str(last_dir))
    trainer.save_state()
    log_history = trainer.state.log_history
    training_losses = [
        {
            "step": int(row["step"]),
            "epoch": float(row["epoch"]),
            "loss": float(row["loss"]),
        }
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
    candidates = sorted(
        (
            path
            for path in output_dir.glob("checkpoint-*")
            if (path / "adapter_model.safetensors").is_file()
        ),
        key=lambda path: int(path.name.rsplit("-", 1)[-1]),
    )
    if not candidates:
        raise RuntimeError("Full V2 training did not preserve any epoch checkpoints")
    peak_allocated = (
        torch.cuda.max_memory_allocated() / 2**20
        if torch.cuda.is_available()
        else None
    )
    peak_reserved = (
        torch.cuda.max_memory_reserved() / 2**20
        if torch.cuda.is_available()
        else None
    )
    summary = {
        "experiment": "high_resolution_continued_training_v2",
        "config_path": repository_relative(config_path),
        "output_dir": repository_relative(output_dir),
        "adapter_source": repository_relative(config["paths"]["v1_adapter_path"]),
        "adapter_source_sha256": sha256_file(
            project_path(config["paths"]["v1_adapter_path"])
            / "adapter_model.safetensors"
        ),
        "resolution": int(config["model"]["image_longest_edge"]),
        "max_image_patch_edge": int(config["model"]["max_image_patch_edge"]),
        "do_image_splitting": bool(config["model"]["do_image_splitting"]),
        "runtime": runtime.to_dict(),
        "adapter_structure": adapter_structure,
        "train_sample_count": len(train_records),
        "validation_sample_count": len(validation_records),
        "official_test_used_during_training": False,
        "train_ids_sha256": hash_id_list(
            [record.sample_id for record in train_records]
        ),
        "validation_ids_sha256": hash_id_list(
            [record.sample_id for record in validation_records]
        ),
        "completed_optimization_steps": trainer.state.global_step,
        "completed_epochs": float(trainer.state.epoch or 0.0),
        "duration_seconds": duration,
        "seconds_per_optimization_step": duration
        / max(trainer.state.global_step, 1),
        "peak_allocated_gpu_memory_mib": peak_allocated,
        "peak_reserved_gpu_memory_mib": peak_reserved,
        "initial_lora_tensor_sha256": initial_lora_sha256,
        "final_lora_tensor_sha256": final_lora_sha256,
        "lora_tensors_changed": initial_lora_sha256 != final_lora_sha256,
        "average_observed_visual_tiles": statistics.mean(
            collator.observed_visual_tile_counts
        ),
        "average_observed_assistant_tokens": statistics.mean(
            collator.observed_assistant_token_counts
        ),
        "training_losses": training_losses,
        "validation_losses": validation_losses,
        "early_stopping": {
            "patience": callback.patience,
            "min_delta": callback.min_delta,
            "history": callback.history,
            "triggered": callback.triggered,
        },
        "candidate_checkpoints": [
            {
                "path": repository_relative(path),
                "step": int(path.name.rsplit("-", 1)[-1]),
                "adapter_sha256": sha256_file(
                    path / "adapter_model.safetensors"
                ),
            }
            for path in candidates
        ],
        "last_adapter_path": repository_relative(last_dir),
        "last_adapter_sha256": sha256_file(
            last_dir / "adapter_model.safetensors"
        ),
        "train_metrics": train_result.metrics,
        "package_versions": package_versions(),
    }
    manifest = {
        "train_ids": [record.sample_id for record in train_records],
        "validation_ids": [record.sample_id for record in validation_records],
        "train_source_split": "train",
        "validation_source_split": "train",
        "official_test_used_during_training": False,
        "overlap_ids": [],
        "train_ids_sha256": summary["train_ids_sha256"],
        "validation_ids_sha256": summary["validation_ids_sha256"],
        "source_manifest": repository_relative(
            config["paths"]["v1_split_manifest"]
        ),
        "seed": seed,
    }
    write_json(config["paths"]["training_summary_path"], summary)
    write_json(config["paths"]["split_manifest_copy_path"], manifest)
    LOGGER.info("V2 continued training complete: %s", json.dumps(summary))
    del trainer
    del model
    del processor
    clear_cuda()
    return summary


class ConsecutiveValidationEarlyStoppingCallback(TrainerCallback):
    """Stop after consecutive evaluations without a validation-loss improvement."""

    def __init__(self, patience: int, min_delta: float = 0.0) -> None:
        if patience <= 0:
            raise ValueError("Early-stopping patience must be positive")
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = math.inf
        self.non_improving_evaluations = 0
        self.triggered = False
        self.history: list[dict[str, Any]] = []

    def on_evaluate(
        self,
        args: Any,
        state: Any,
        control: Any,
        metrics: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> Any:
        del args, kwargs
        eval_loss = None if metrics is None else metrics.get("eval_loss")
        if eval_loss is None or not math.isfinite(float(eval_loss)):
            return control
        loss = float(eval_loss)
        improved = loss < self.best_loss - self.min_delta
        if improved:
            self.best_loss = loss
            self.non_improving_evaluations = 0
        else:
            self.non_improving_evaluations += 1
        if self.non_improving_evaluations >= self.patience:
            control.should_training_stop = True
            self.triggered = True
        self.history.append(
            {
                "step": int(state.global_step),
                "epoch": float(state.epoch or 0.0),
                "eval_loss": loss,
                "improved": improved,
                "consecutive_non_improvements": self.non_improving_evaluations,
                "stop": self.triggered,
            }
        )
        return control


def load_v1_split_records(
    config: dict[str, Any],
) -> tuple[list[ReceiptRecord], list[ReceiptRecord]]:
    """Load the exact original 563/63 split from its immutable manifest."""
    manifest_path = project_path(config["paths"]["v1_split_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_ids = [str(value) for value in manifest["train_ids"]]
    validation_ids = [str(value) for value in manifest["validation_ids"]]
    if len(train_ids) != 563 or len(validation_ids) != 63:
        raise ValueError(
            "V1 split manifest must contain exactly 563 train and 63 validation IDs"
        )
    if set(train_ids) & set(validation_ids):
        raise ValueError("V1 train and validation IDs overlap")
    source = load_records(
        config["paths"]["dataset_root"],
        "train",
        limit=None,
        seed=int(config["project"]["seed"]),
    )
    by_id = {record.sample_id: record for record in source}
    missing = sorted((set(train_ids) | set(validation_ids)) - set(by_id))
    if missing:
        raise ValueError(f"V1 split IDs missing from official train split: {missing}")
    return (
        [by_id[sample_id] for sample_id in train_ids],
        [by_id[sample_id] for sample_id in validation_ids],
    )


def deterministic_subset(
    records: list[ReceiptRecord],
    count: int,
    seed: int,
) -> list[ReceiptRecord]:
    """Select a stable sample without changing the source ordering."""
    import random

    if count <= 0 or count > len(records):
        raise ValueError(f"Invalid deterministic subset size: {count}")
    selected = random.Random(seed).sample(records, count)
    return sorted(selected, key=lambda record: record.sample_id)


def validate_continued_adapter(
    model: PeftModel,
    expected: dict[str, Any],
) -> dict[str, Any]:
    """Fail closed unless the loaded V1 adapter has the requested structure."""
    config = model.peft_config["default"]
    actual_targets = sorted(str(value) for value in config.target_modules)
    expected_targets = sorted(str(value) for value in expected["target_modules"])
    checks = {
        "rank": int(config.r),
        "alpha": int(config.lora_alpha),
        "dropout": float(config.lora_dropout),
        "target_modules": actual_targets,
    }
    expected_values = {
        "rank": int(expected["rank"]),
        "alpha": int(expected["alpha"]),
        "dropout": float(expected["dropout"]),
        "target_modules": expected_targets,
    }
    if checks != expected_values:
        raise ValueError(
            f"V1 adapter structure mismatch: actual={checks} expected={expected_values}"
        )
    return checks


def profile_records(
    collator: ReceiptKIECollator,
    records: list[ReceiptRecord],
) -> dict[str, Any]:
    """Measure tile and assistant-token counts before the timed training run."""
    rows = []
    for record in records:
        batch = collator([record.trainer_row()])
        masking = collator.last_masking_report[0]
        rows.append(
            {
                "sample_id": record.sample_id,
                "visual_tile_count": collator.last_visual_tile_counts[0],
                "assistant_token_count": masking["assistant_tokens"],
            }
        )
        del batch
    tile_counts = [row["visual_tile_count"] for row in rows]
    assistant_counts = [row["assistant_token_count"] for row in rows]
    clear_cuda()
    return {
        "records": rows,
        "average_visual_tiles": statistics.mean(tile_counts),
        "median_visual_tiles": statistics.median(tile_counts),
        "total_assistant_tokens": sum(assistant_counts),
        "average_assistant_tokens": statistics.mean(assistant_counts),
    }


def smoke_training_arguments(
    config: dict[str, Any],
    output_dir: Path,
    dtype_name: str,
) -> SFTConfig:
    values = config["training"]
    return SFTConfig(
        output_dir=str(output_dir),
        logging_dir=str(project_path(config["paths"]["tensorboard_dir"])),
        num_train_epochs=float(values["num_train_epochs"]),
        max_steps=int(values["max_steps"]),
        per_device_train_batch_size=int(values["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(values["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(values["gradient_accumulation_steps"]),
        learning_rate=float(values["learning_rate"]),
        weight_decay=float(values["weight_decay"]),
        warmup_ratio=float(values["warmup_ratio"]),
        logging_steps=int(values["logging_steps"]),
        save_strategy="no",
        eval_strategy="steps",
        eval_steps=int(values["eval_steps"]),
        gradient_checkpointing=bool(values["gradient_checkpointing"]),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=dtype_name == "bf16",
        fp16=dtype_name == "fp16",
        optim="adamw_torch_fused",
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        max_seq_length=int(values["max_seq_length"]),
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        save_safetensors=True,
        seed=int(config["project"]["seed"]),
    )


def continued_training_arguments(
    config: dict[str, Any],
    output_dir: Path,
    dtype_name: str,
) -> SFTConfig:
    """Build the fixed V2 full-training arguments."""
    values = config["training"]
    return SFTConfig(
        output_dir=str(output_dir),
        logging_dir=str(project_path(config["paths"]["tensorboard_dir"])),
        num_train_epochs=float(values["num_train_epochs"]),
        max_steps=-1,
        per_device_train_batch_size=int(values["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(values["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(values["gradient_accumulation_steps"]),
        learning_rate=float(values["learning_rate"]),
        weight_decay=float(values["weight_decay"]),
        warmup_ratio=float(values["warmup_ratio"]),
        logging_steps=int(values["logging_steps"]),
        save_strategy="epoch",
        save_total_limit=int(values["save_total_limit"]),
        eval_strategy="steps",
        eval_steps=int(values["eval_steps"]),
        gradient_checkpointing=bool(values["gradient_checkpointing"]),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=dtype_name == "bf16",
        fp16=dtype_name == "fp16",
        optim="adamw_torch_fused",
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        max_seq_length=int(values["max_seq_length"]),
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        save_safetensors=True,
        seed=int(config["project"]["seed"]),
    )


def hash_trainable_tensors(model: Any) -> str:
    """Hash trainable tensor names, shapes, dtypes, and bytes deterministically."""
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        tensor = parameter.detach().float().contiguous().cpu()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_adapter_reload(config: dict[str, Any], adapter_path: Path) -> bool:
    """Reload a saved adapter trainably and validate its exact LoRA structure."""
    base_model, _ = load_base_model(
        config["model"],
        config["paths"]["hf_cache"],
        training=False,
    )
    reloaded = PeftModel.from_pretrained(
        base_model,
        str(adapter_path),
        is_trainable=False,
    )
    validate_continued_adapter(reloaded, config["lora"])
    del reloaded
    clear_cuda()
    return True


def _is_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(
        exc
    ).casefold()
