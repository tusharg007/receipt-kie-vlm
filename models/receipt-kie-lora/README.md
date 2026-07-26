# ReceiptKIE-VLM LoRA Adapter

This directory contains the minimum inference artifacts for the newly trained
ReceiptKIE-VLM adapter:

- `adapter_model.safetensors`: LoRA weights.
- `adapter_config.json`: PEFT configuration and base-model reference.
- `training_metadata.json`: training provenance, checksum, and evaluation metadata.

The tokenizer, processor, and frozen base weights are loaded from the public
`HuggingFaceTB/SmolVLM-256M-Instruct` repository on first use. They are not
duplicated here.

Adapter SHA-256:

```text
94ba0038153ea1aacb12dbcc80f1edf01d31a6309ea56919684e8cb8bbe90b28
```

Run from the repository root:

```bash
python scripts/verify_installation.py
python scripts/demo_inference.py
```

This adapter was trained on 563 SROIE official-train annotations and validated
on a disjoint 63-receipt partition from that same official train split. The
official test split was used only for final evaluation. SmolVLM and SROIE are
external projects and are not owned by this repository.
