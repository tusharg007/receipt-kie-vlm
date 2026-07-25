"""Deterministic base and LoRA-adapted SmolVLM inference."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from PIL import Image

from receipt_kie.metrics import ParsedOutput, extract_json
from receipt_kie.model import load_base_model, load_processor
from receipt_kie.prompts import build_messages
from receipt_kie.utils import clear_cuda


class ReceiptKIEPredictor:
    """Load one model variant and generate structured receipt predictions."""

    def __init__(
        self,
        model_config: dict[str, Any],
        hf_cache: str,
        adapter_path: str | Path | None = None,
    ) -> None:
        self.processor = load_processor(model_config, hf_cache)
        base_model, self.runtime = load_base_model(model_config, hf_cache, training=False)
        if adapter_path is not None:
            self.model = PeftModel.from_pretrained(base_model, str(adapter_path))
            self.model.eval()
        else:
            self.model = base_model
        self.device = self.runtime.device
        self.dtype = self.runtime.torch_dtype

    def predict(
        self,
        image_path: str | Path,
        max_new_tokens: int = 128,
        do_sample: bool = False,
    ) -> dict[str, Any]:
        """Generate one raw output and a resilient parsed representation."""
        image_path = Path(image_path)
        messages = build_messages(str(image_path))
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        with Image.open(image_path) as opened:
            image = opened.convert("RGB").copy()
        inputs = self.processor(
            text=[text],
            images=[[image]],
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        inputs = inputs.to(self.device, dtype=self.dtype)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latency = time.perf_counter() - started
        trimmed = generated[:, inputs["input_ids"].shape[1] :]
        raw_output = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        parsed: ParsedOutput = extract_json(raw_output)
        peak_mib = torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else None
        return {
            "raw_output": raw_output,
            "valid_json": parsed.valid,
            "parsed_prediction": parsed.value,
            "parse_method": parsed.method,
            "parse_error": parsed.error,
            "latency_seconds": latency,
            "peak_gpu_memory_mib": peak_mib,
        }

    def close(self) -> None:
        del self.model
        del self.processor
        clear_cuda()
