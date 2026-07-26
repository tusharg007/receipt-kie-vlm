"""Deterministic base and LoRA-adapted SmolVLM inference."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from PIL import Image

from receipt_kie.metrics import ParsedOutput, extract_json
from receipt_kie.model import (
    load_base_model,
    load_processor,
    processor_image_configuration,
)
from receipt_kie.prompts import SYSTEM_PROMPT, build_messages
from receipt_kie.utils import clear_cuda

LOGGER = logging.getLogger(__name__)


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
        configured_edge = model_config.get("image_longest_edge")
        self.image_longest_edge_override = (
            int(configured_edge) if configured_edge is not None else None
        )
        self.processor_configuration = processor_image_configuration(
            self.processor.image_processor
        )

    def predict(
        self,
        image_path: str | Path,
        max_new_tokens: int = 128,
        do_sample: bool = False,
        repetition_penalty: float | None = None,
    ) -> dict[str, Any]:
        """Generate one raw output and a resilient parsed representation."""
        image_path = Path(image_path)
        messages = build_messages(str(image_path))
        with Image.open(image_path) as opened:
            image = opened.convert("RGB").copy()
        return self._generate(
            [image],
            messages,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
        )

    def predict_images(
        self,
        images: list[Image.Image],
        user_prompt: str,
        max_new_tokens: int = 128,
        do_sample: bool = False,
        repetition_penalty: float | None = None,
    ) -> dict[str, Any]:
        """Generate from one or more in-memory images with an explicit prompt."""
        if not images:
            raise ValueError("At least one image is required")
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    *({"type": "image"} for _ in images),
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]
        return self._generate(
            [image.convert("RGB").copy() for image in images],
            messages,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
        )

    def _generate(
        self,
        images: list[Image.Image],
        messages: list[dict[str, Any]],
        max_new_tokens: int,
        do_sample: bool,
        repetition_penalty: float | None,
    ) -> dict[str, Any]:
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        processor_images = images
        processor_kwargs: dict[str, Any] = {}
        if self.image_longest_edge_override is not None:
            processor_images = [
                resize_to_longest_edge(image, self.image_longest_edge_override)
                for image in images
            ]
            # Transformers 4.46.3 rejects longest_edge > 1820 before its
            # splitting stage. Explicit pre-resizing is tensor-identical at
            # the 512 baseline and permits controlled 2048-pixel ablations.
            processor_kwargs["do_resize"] = False
        inputs = self.processor(
            text=[text],
            images=[processor_images],
            return_tensors="pt",
            padding=True,
            truncation=False,
            **processor_kwargs,
        )
        pixel_values_shape = [int(value) for value in inputs["pixel_values"].shape]
        visual_tile_count = (
            pixel_values_shape[1] if len(pixel_values_shape) >= 5 else 1
        )
        visual_tile_shape = pixel_values_shape[-3:]
        LOGGER.info(
            "Processor output configuration=%s visual_tiles=%d "
            "pixel_values_shape=%s tile_shape=%s",
            self.processor_configuration,
            visual_tile_count,
            pixel_values_shape,
            visual_tile_shape,
        )
        inputs = inputs.to(self.device, dtype=self.dtype)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        started = time.perf_counter()
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }
        if repetition_penalty is not None:
            generation_kwargs["repetition_penalty"] = repetition_penalty
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                **generation_kwargs,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latency = time.perf_counter() - started
        trimmed = generated[:, inputs["input_ids"].shape[1] :]
        generated_token_count = int(trimmed.shape[1])
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
            "generated_token_count": generated_token_count,
            "generation_limit_hit": generated_token_count >= max_new_tokens,
            "visual_tile_count": visual_tile_count,
            "pixel_values_shape": pixel_values_shape,
            "visual_tile_shape": visual_tile_shape,
            "processor_configuration": self.processor_configuration,
        }

    def close(self) -> None:
        del self.model
        del self.processor
        clear_cuda()


def resize_to_longest_edge(image: Image.Image, longest_edge: int) -> Image.Image:
    """Match Idefics3's explicit longest-edge LANCZOS resize deterministically."""
    if longest_edge <= 0:
        raise ValueError("longest_edge must be positive")
    width, height = image.size
    aspect_ratio = width / height
    if width >= height:
        output_width = longest_edge
        output_height = int(output_width / aspect_ratio)
        if output_height % 2:
            output_height += 1
    else:
        output_height = longest_edge
        output_width = int(output_height * aspect_ratio)
        if output_width % 2:
            output_width += 1
    return image.resize(
        (max(output_width, 1), max(output_height, 1)),
        resample=Image.Resampling.LANCZOS,
    )
