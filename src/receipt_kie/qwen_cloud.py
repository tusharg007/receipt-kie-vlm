"""Isolated Qwen3-VL QLoRA experiment for a real Kaggle CUDA worker."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import time
from pathlib import Path

import torch
from PIL import Image

from receipt_kie.cloud_data import prepare_cloud_data
from receipt_kie.metrics import evaluate_predictions
from receipt_kie.prompts import SYSTEM_PROMPT, USER_PROMPT, canonical_json
from receipt_kie.structured import strict_receipt, structured_metadata

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
MODEL_REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"


def messages(target: dict | None = None) -> list[dict]:
    result = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": USER_PROMPT}]},
    ]
    if target is not None:
        result.append({"role": "assistant", "content": canonical_json(target)})
    return result


class QwenReceiptCollator:
    """One image per batch; no truncation of visual tokens or supervised targets."""

    def __init__(self, processor, max_pixels: int, max_length: int = 2048):
        self.processor, self.max_pixels, self.max_length = processor, max_pixels, max_length
        self.processor.tokenizer.padding_side = "right"

    def __call__(self, examples: list[dict]) -> dict:
        if len(examples) != 1:
            raise ValueError("This single-GPU protocol requires batch size one")
        row = examples[0]
        with Image.open(row["image_path"]) as image:
            image = image.convert("RGB").copy()
        processor = self.processor
        kwargs = {
            "images": [image],
            "return_tensors": "pt",
            "padding": False,
            "truncation": False,
            "max_pixels": self.max_pixels,
            "min_pixels": 65536,
        }
        prompt = processor.apply_chat_template(
            messages(), tokenize=False, add_generation_prompt=True
        )
        full = processor.apply_chat_template(
            messages(row["target"]), tokenize=False, add_generation_prompt=False
        )
        batch = processor(text=[full], **kwargs)
        prefix = processor(text=[prompt], **kwargs)["input_ids"]
        length = prefix.shape[1]
        if batch["input_ids"].shape[1] > self.max_length:
            raise ValueError("Token budget exceeded; revise the declared protocol, never truncate")
        if not torch.equal(batch["input_ids"][:, :length], prefix):
            raise ValueError("Prompt is not an exact prefix of the training conversation")
        labels = batch["input_ids"].clone()
        labels[:, :length] = -100
        labels[batch["attention_mask"] == 0] = -100
        for token in ("<|image_pad|>", "<|vision_start|>", "<|vision_end|>"):
            labels[batch["input_ids"] == processor.tokenizer.convert_tokens_to_ids(token)] = -100
        if not torch.any(labels != -100):
            raise ValueError("No supervised assistant tokens")
        batch["labels"] = labels
        return dict(batch)


def _save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def predict_rows(model, processor, rows: list[dict], output: Path, max_pixels: int) -> list[dict]:
    results = []
    model.eval()
    for index, row in enumerate(rows):
        with Image.open(row["image_path"]) as image:
            image = image.convert("RGB").copy()
        prompt = processor.apply_chat_template(
            messages(), tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[prompt],
            images=[image],
            return_tensors="pt",
            padding=False,
            max_pixels=max_pixels,
            min_pixels=65536,
        ).to(model.device)
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        torch.cuda.synchronize()
        new_tokens = generated[:, inputs["input_ids"].shape[1] :]
        raw = processor.batch_decode(
            new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        parsed, errors = strict_receipt(raw)
        limit_hit = new_tokens.shape[1] >= 256
        results.append(
            {
                "sample_id": row["sample_id"],
                "image_sha256": row["sha256"],
                "ground_truth": row["target"],
                "raw_output": raw,
                "parsed_prediction": parsed,
                "valid_json": parsed is not None,
                "parse_error": "; ".join(errors) or None,
                "latency_seconds": time.perf_counter() - start,
                "generation_limit_hit": limit_hit,
                **structured_metadata(raw, limit_hit=limit_hit),
            }
        )
        with output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(results[-1]) + "\n")
        print(f"{output.stem}: {index + 1}/{len(rows)}", flush=True)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    # Import only in the isolated cloud environment, never upgrade the local V2 stack.
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoProcessor,
        BitsAndBytesConfig,
        Qwen3VLForConditionalGeneration,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("A real CUDA worker is required; CPU fallback is not a training result")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one CUDA device; two T4s do not pool their VRAM")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Use a fresh output directory to keep experiment records immutable")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    set_seed(protocol["seed"])
    split = prepare_cloud_data(args.dataset_root, protocol["evaluation_ids"], protocol["seed"])
    observed_hashes = {row["sample_id"]: row["sha256"] for row in split["evaluation"]}
    if observed_hashes != protocol["evaluation_image_sha256"]:
        raise ValueError("Cloud evaluation images differ from the audited local image hashes")
    _save(args.output_dir / "split_manifest.json", split)
    max_pixels = protocol["max_pixels"]
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )
    started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        quantization_config=quantization,
        device_map={"": 0},
        dtype=dtype,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    suffixes = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    targets = [
        name
        for name, _ in model.named_modules()
        if "language_model" in name and name.rsplit(".", 1)[-1] in suffixes
    ]
    if not targets:
        raise ValueError("Expected language-model projection modules were not found")
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            target_modules=targets,
            task_type="CAUSAL_LM",
        ),
    )
    reference_name, reference_parameter = next(
        (name, value) for name, value in model.named_parameters() if "lora_B" in name
    )
    before = reference_parameter.detach().float().cpu().clone()
    training = split["train"]
    if args.smoke:
        # Largest source images exercise the declared visual token cap.
        training = sorted(training, key=lambda row: Path(row["image_path"]).stat().st_size)[-4:]
    collator = QwenReceiptCollator(processor, max_pixels, protocol["max_sequence_length"])
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(args.output_dir / "checkpoints"),
            num_train_epochs=protocol["epochs"],
            max_steps=2 if args.smoke else -1,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=1 if args.smoke else 8,
            learning_rate=protocol["learning_rate"],
            weight_decay=0.01,
            warmup_ratio=0 if args.smoke else 0.05,
            optim="adamw_torch",
            fp16=dtype == torch.float16,
            bf16=dtype == torch.bfloat16,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            eval_strategy="no" if args.smoke else "epoch",
            save_strategy="no" if args.smoke else "epoch",
            save_total_limit=2,
            load_best_model_at_end=not args.smoke,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            logging_steps=1 if args.smoke else 10,
            dataloader_num_workers=0,
            remove_unused_columns=False,
            report_to="none",
            seed=protocol["seed"],
            data_seed=protocol["seed"],
        ),
        train_dataset=training,
        eval_dataset=split["validation"],
        data_collator=collator,
    )
    torch.cuda.reset_peak_memory_stats()
    trained = trainer.train()
    losses = [item["loss"] for item in trainer.state.log_history if "loss" in item]
    delta = float((reference_parameter.detach().float().cpu() - before).abs().max())
    finite = bool(losses) and all(math.isfinite(loss) for loss in losses)
    finite_weights = all(
        torch.isfinite(parameter).all().item()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    if not finite or not finite_weights or not math.isfinite(delta) or delta <= 0:
        raise RuntimeError("Training must have finite loss and actually change LoRA parameters")
    summary = {
        "status": "smoke_passed" if args.smoke else "trained_pending_evaluation",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "protocol_sha256": hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
        "gpu": torch.cuda.get_device_name(),
        "dtype": str(dtype),
        "quantization": "NF4",
        "max_pixels": max_pixels,
        "global_steps": trainer.state.global_step,
        "train_metrics": trained.metrics,
        "losses": losses,
        "checked_parameter": reference_name,
        "checked_parameter_max_change": delta,
        "all_trainable_weights_finite": finite_weights,
        "peak_allocated_gpu_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_gpu_mib": torch.cuda.max_memory_reserved() / 2**20,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "checkpoint_selection": "Minimum official-train merchant-disjoint validation loss",
        "elapsed_seconds": time.perf_counter() - started,
        "versions": {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "torchvision",
                "transformers",
                "peft",
                "accelerate",
                "bitsandbytes",
            )
        },
        "deployment_allowed": False,
    }
    _save(args.output_dir / "training_summary.json", summary)
    if args.smoke:
        print(json.dumps(summary, indent=2), flush=True)
        return 0
    model.save_pretrained(args.output_dir / "adapter")
    processor.save_pretrained(args.output_dir / "adapter")
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    validation = predict_rows(
        model,
        processor,
        split["validation"],
        args.output_dir / "validation_predictions.jsonl",
        max_pixels,
    )
    validation_metrics = evaluate_predictions(validation)
    with model.disable_adapter():
        base_validation = predict_rows(
            model,
            processor,
            split["validation"],
            args.output_dir / "base_validation_predictions.jsonl",
            max_pixels,
        )
    base_validation_metrics = evaluate_predictions(base_validation)
    _save(
        args.output_dir / "validation_metrics.json",
        {
            "adapted": validation_metrics,
            "base": base_validation_metrics,
        },
    )
    selected = (
        "qwen_qlora"
        if validation_metrics["macro_normalized_exact_match"]
        > base_validation_metrics["macro_normalized_exact_match"]
        and validation_metrics["complete_record_normalized_exact_match"]
        >= base_validation_metrics["complete_record_normalized_exact_match"]
        else "qwen_quantized_base"
    )
    _save(
        args.output_dir / "frozen_selection.json",
        {
            "selected": selected,
            "criterion": "Higher validation macro exact with no complete-record exact regression",
            "based_on": "Merchant-disjoint validation only, before evaluation predictions",
            "deployment_allowed": False,
        },
    )
    adapted = predict_rows(
        model,
        processor,
        split["evaluation"],
        args.output_dir / "adapted_predictions.jsonl",
        max_pixels,
    )
    with model.disable_adapter():
        baseline = predict_rows(
            model,
            processor,
            split["evaluation"],
            args.output_dir / "base_predictions.jsonl",
            max_pixels,
        )
    comparison = {
        "scope": "239-record retrospective comparison; no fresh-data generalization claim",
        "protocol": protocol,
        "qwen_quantized_base": evaluate_predictions(baseline),
        "qwen_qlora": evaluate_predictions(adapted),
        "notes": "Both Qwen variants use the same images, NF4, prompt, strict parser and decoding. "
        "Historical V2 results used different resolution and parsing; latency is not comparable.",
    }
    _save(args.output_dir / "comparison.json", comparison)
    summary.update(status="evaluation_complete", elapsed_seconds=time.perf_counter() - started)
    _save(args.output_dir / "training_summary.json", summary)
    print(json.dumps(comparison, indent=2), flush=True)
    return 0
