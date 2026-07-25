from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from receipt_kie.collator import ReceiptKIECollator


class _Tokenizer:
    additional_special_tokens = ["<image>"]
    additional_special_tokens_ids = [99]
    padding_side = "left"


class _Processor:
    tokenizer = _Tokenizer()

    @staticmethod
    def apply_chat_template(messages, tokenize, add_generation_prompt):
        del messages, tokenize
        return "prompt" if add_generation_prompt else "full"

    @staticmethod
    def __call__(text, images, return_tensors, padding, truncation):
        del images, return_tensors, padding, truncation
        rows = [[1, 99, 2, 4, 5] if value == "full" else [1, 99, 2, 3] for value in text]
        width = max(len(row) for row in rows)
        input_ids = torch.tensor([row + [0] * (width - len(row)) for row in rows])
        attention_mask = input_ids.ne(0).long()
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def test_collator_masks_prompt_padding_and_image_tokens(tmp_path: Path) -> None:
    image_path = tmp_path / "receipt.png"
    Image.new("RGB", (8, 8), color="white").save(image_path)
    collator = ReceiptKIECollator(_Processor())

    batch = collator(
        [
            {
                "sample_id": "sample",
                "image_path": str(image_path),
                "target": {
                    "company": "A",
                    "address": "B",
                    "date": "01/01/2024",
                    "total": "1.00",
                },
            }
        ]
    )

    assert batch["labels"].tolist() == [[-100, -100, -100, 4, 5]]
    assert collator.last_masking_report == [
        {"full_tokens": 5, "prompt_tokens": 3, "assistant_tokens": 2}
    ]
