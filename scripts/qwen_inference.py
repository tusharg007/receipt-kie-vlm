"""Run the optional pinned Qwen base or exported QLoRA adapter on one local image."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--adapter", type=Path)
    mode.add_argument("--base", action="store_true")
    parser.add_argument("--max-pixels", type=int, default=1048576)
    parser.add_argument("--cpu", action="store_true", help="Explicit unquantized FP32 CPU mode")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".cache/huggingface-qwen")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error("Receipt image does not exist")
    if not 65536 <= args.max_pixels <= 1048576:
        parser.error("--max-pixels must be between 65536 and the declared training cap 1048576")
    import torch
    from peft import PeftModel, prepare_model_for_kbit_training
    from PIL import Image
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    from receipt_kie.qwen_cloud import MODEL_ID, MODEL_REVISION, messages, native_training_dtype
    from receipt_kie.structured import strict_receipt, structured_metadata

    adapter_sha256 = None
    if args.adapter:
        config = json.loads((args.adapter / "adapter_config.json").read_text("utf-8"))
        if (
            config.get("base_model_name_or_path") != MODEL_ID
            or config.get("revision") != MODEL_REVISION
        ):
            raise ValueError("Adapter must record the exact supported Qwen base and revision")
        adapter_sha256 = hashlib.sha256(
            (args.adapter / "adapter_model.safetensors").read_bytes()
        ).hexdigest()
    if not args.cpu and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable; explicitly choose --cpu if sufficient RAM is available"
        )
    dtype = (
        torch.float32
        if args.cpu
        else native_training_dtype(
            torch.cuda.get_device_capability(), torch.cuda.is_bf16_supported()
        )
    )
    kwargs = {
        "revision": MODEL_REVISION,
        "cache_dir": str(args.cache_dir),
        "device_map": {"": "cpu" if args.cpu else 0},
        "dtype": dtype,
        "attn_implementation": "sdpa",
    }
    if not args.cpu:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
    started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, cache_dir=str(args.cache_dir)
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL_ID, **kwargs)
    if not args.cpu:
        # Match the frozen base/norm casting used for the cloud NF4 evaluation.
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    if args.adapter:
        model = PeftModel.from_pretrained(model, str(args.adapter), is_trainable=False)
    model.eval()
    model.config.use_cache = True
    startup_seconds = time.perf_counter() - started
    with Image.open(args.image) as image:
        image = image.convert("RGB").copy()
    prompt = processor.apply_chat_template(messages(), tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[prompt],
        images=[image],
        return_tensors="pt",
        padding=False,
        max_pixels=args.max_pixels,
        min_pixels=65536,
    ).to(model.device)
    if not args.cpu:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    if not args.cpu:
        torch.cuda.synchronize()
    completion = generated[:, inputs["input_ids"].shape[1] :]
    raw = processor.batch_decode(
        completion, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    parsed, errors = strict_receipt(raw)
    metadata = structured_metadata(raw, limit_hit=completion.shape[1] >= 256)
    metadata["review_required"] = True
    metadata["review_issues"].append("backend:experimental_qwen")
    result = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "mode": "qlora" if args.adapter else "base",
        "adapter_sha256": adapter_sha256,
        "device": str(model.device),
        "dtype": str(dtype),
        "quantization": "none" if args.cpu else "NF4",
        "max_pixels": args.max_pixels,
        "raw_output": raw,
        "parsed_prediction": parsed,
        "parse_errors": errors,
        "startup_seconds": startup_seconds,
        "generation_seconds": time.perf_counter() - started,
        "peak_allocated_gpu_mib": None if args.cpu else torch.cuda.max_memory_allocated() / 2**20,
        "deployment_allowed": False,
        **metadata,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if parsed is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
