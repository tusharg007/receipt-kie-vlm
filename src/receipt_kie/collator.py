"""Multimodal SFT collation with assistant-only label masking."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from PIL import Image

from receipt_kie.prompts import build_messages


class ReceiptKIECollator:
    """Build SmolVLM batches while applying loss only to assistant tokens."""

    def __init__(self, processor: Any) -> None:
        self.processor = processor
        self.processor.tokenizer.padding_side = "right"
        self.image_token_id = self._image_token_id()
        self.last_masking_report: list[dict[str, int]] = []

    def _image_token_id(self) -> int:
        tokenizer = self.processor.tokenizer
        try:
            index = tokenizer.additional_special_tokens.index("<image>")
        except ValueError as exc:
            raise ValueError(
                "Processor tokenizer does not expose the required <image> token"
            ) from exc
        return int(tokenizer.additional_special_tokens_ids[index])

    def __call__(self, examples: Sequence[dict[str, Any]]) -> dict[str, torch.Tensor]:
        if not examples:
            raise ValueError("Cannot collate an empty batch")
        images = [self._load_image(str(example["image_path"])) for example in examples]
        image_batches = [[image] for image in images]
        full_texts = [
            self.processor.apply_chat_template(
                build_messages(str(example["image_path"]), example["target"]),
                tokenize=False,
                add_generation_prompt=False,
            )
            for example in examples
        ]
        prompt_texts = [
            self.processor.apply_chat_template(
                build_messages(str(example["image_path"])),
                tokenize=False,
                add_generation_prompt=True,
            )
            for example in examples
        ]
        full_batch = self.processor(
            text=full_texts,
            images=image_batches,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        prompt_batch = self.processor(
            text=prompt_texts,
            images=image_batches,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        labels = full_batch["input_ids"].clone()
        self.last_masking_report = []
        for index in range(len(examples)):
            full_length = int(full_batch["attention_mask"][index].sum().item())
            prompt_length = int(prompt_batch["attention_mask"][index].sum().item())
            full_ids = full_batch["input_ids"][index, :full_length]
            prompt_ids = prompt_batch["input_ids"][index, :prompt_length]
            common_prefix = _common_prefix_length(full_ids, prompt_ids)
            if common_prefix < prompt_length - 1:
                raise ValueError(
                    "Prompt tokens are not a prefix of the full conversation: "
                    f"sample={examples[index].get('sample_id')} "
                    f"prompt_length={prompt_length} common_prefix={common_prefix}"
                )
            labels[index, :common_prefix] = -100
            self.last_masking_report.append(
                {
                    "full_tokens": full_length,
                    "prompt_tokens": common_prefix,
                    "assistant_tokens": full_length - common_prefix,
                }
            )
        labels[full_batch["attention_mask"] == 0] = -100
        labels[full_batch["input_ids"] == self.image_token_id] = -100
        if not torch.any(labels != -100):
            raise ValueError("All labels are masked; no assistant tokens remain for loss")
        full_batch["labels"] = labels
        return dict(full_batch)

    @staticmethod
    def _load_image(path: str) -> Image.Image:
        with Image.open(path) as image:
            return image.convert("RGB").copy()


def _common_prefix_length(left: torch.Tensor, right: torch.Tensor) -> int:
    length = min(left.numel(), right.numel())
    if length == 0:
        return 0
    equal = left[:length].eq(right[:length])
    mismatch = (~equal).nonzero(as_tuple=False)
    return int(mismatch[0].item()) if mismatch.numel() else length
