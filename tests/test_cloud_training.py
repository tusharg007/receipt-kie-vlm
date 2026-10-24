from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from receipt_kie.cloud_data import prepare_cloud_data
from receipt_kie.qwen_cloud import QwenReceiptCollator, native_training_dtype


def test_t4_never_selects_emulated_bf16_even_when_torch_reports_support():
    assert native_training_dtype((7, 5), True) == torch.float16
    assert native_training_dtype((8, 0), True) == torch.bfloat16
    assert native_training_dtype((9, 0), False) == torch.float16


def _image(root, split, sample_id, color, company="ACME"):
    image_dir, labels = root / split / "img", root / split / "entities"
    image_dir.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color).save(image_dir / f"{sample_id}.png")
    (labels / f"{sample_id}.txt").write_text(
        json.dumps(
            {"company": company, "address": "1 Road", "date": "01/02/2024", "total": "12.50"}
        )
    )


def test_cloud_split_deduplicates_and_never_splits_a_merchant(tmp_path):
    for number in range(12):
        _image(tmp_path, "train", str(number), (number, 100, 100), f"Merchant {number // 2}")
    _image(tmp_path, "train", "duplicate", (0, 100, 100), "Merchant 0")
    _image(tmp_path, "test", "T", "white")
    split = prepare_cloud_data(tmp_path, ["T"])
    assert len(split["train"]) + len(split["validation"]) == 12
    assert {row["merchant_group"] for row in split["train"]}.isdisjoint(
        row["merchant_group"] for row in split["validation"]
    )
    assert any(row["reason"] == "duplicate_content" for row in split["quarantined_or_deduplicated"])
    assert [row["sample_id"] for row in split["evaluation"]] == ["T"]


def test_cloud_split_rejects_test_content_in_training_even_with_different_ids(tmp_path):
    for number, color in enumerate(["red", "green", "blue"]):
        _image(tmp_path, "train", str(number), color, f"Merchant {number}")
    _image(tmp_path, "test", "DIFFERENT_ID", "red")
    with pytest.raises(ValueError, match="overlaps"):
        prepare_cloud_data(tmp_path, ["DIFFERENT_ID"])


def test_conflicting_duplicate_labels_are_quarantined_together(tmp_path):
    for number, color in enumerate(["red", "green", "blue", "yellow"]):
        _image(tmp_path, "train", str(number), color, f"Merchant {number}")
    _image(tmp_path, "train", "conflict", "red", "Contradictory merchant")
    _image(tmp_path, "test", "T", "white")
    split = prepare_cloud_data(tmp_path, ["T"])
    kept = {row["sample_id"] for name in ["train", "validation"] for row in split[name]}
    assert kept == {"1", "2", "3"}
    assert {row["sample_id"] for row in split["quarantined_or_deduplicated"]} == {"0", "conflict"}


def test_bundle_is_private_and_embeds_only_allowlisted_source(tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts/build_kaggle_training.py"
    spec = importlib.util.spec_from_file_location("kaggle_builder", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    metadata = module.build("test_account", True, tmp_path)
    assert metadata["is_private"] is True and metadata["enable_gpu"] is True
    assert metadata["dataset_sources"] == ["urbikn/sroie-datasetv2"]
    protocol = json.loads((tmp_path / "protocol.json").read_text())
    assert len(set(protocol["evaluation_ids"])) == 239
    assert all(name.startswith(("src/", "scripts/", "requirements-")) for name in module.SOURCES)
    code = (tmp_path / metadata["code_file"]).read_text()
    compile(code, metadata["code_file"], "exec")
    assert "CUDA_VISIBLE_DEVICES" in code
    assert "system-site-packages" in code


class _Tokenizer:
    padding_side = "left"

    @staticmethod
    def convert_tokens_to_ids(token):
        return {"<|image_pad|>": 90, "<|vision_start|>": 91, "<|vision_end|>": 92}[token]


class _Processor:
    tokenizer = _Tokenizer()

    def __init__(self, full=None):
        self.full = full or [1, 91, 90, 92, 2, 7, 8]

    @staticmethod
    def apply_chat_template(messages, tokenize, add_generation_prompt):
        return "prompt" if add_generation_prompt else "full"

    def __call__(self, text, **kwargs):
        ids = self.full if text == ["full"] else [1, 91, 90, 92, 2]
        return {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones(1, len(ids))}


def _example(tmp_path):
    image = tmp_path / "bill.png"
    Image.new("RGB", (8, 8), "white").save(image)
    return {"image_path": str(image), "target": {"company": "A"}}


def test_qwen_labels_include_only_the_assistant_completion(tmp_path):
    batch = QwenReceiptCollator(_Processor(), 1048576)([_example(tmp_path)])
    assert batch["labels"].tolist() == [[-100, -100, -100, -100, -100, 7, 8]]


def test_qwen_collator_refuses_non_prefix_conversations_and_silent_truncation(tmp_path):
    with pytest.raises(ValueError, match="exact prefix"):
        QwenReceiptCollator(_Processor([1, 3, 2, 7, 8]), 1048576)([_example(tmp_path)])
    with pytest.raises(ValueError, match="never truncate"):
        QwenReceiptCollator(_Processor(), 1048576, max_length=6)([_example(tmp_path)])
