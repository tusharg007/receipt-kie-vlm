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
7fc369231dd37e064bb9a18777f66ba418d1dd7ae9720db71ba7d2183e6e4ba4
```

Run from the repository root:

```bash
python scripts/verify_installation.py
python scripts/demo_inference.py
```

This adapter was trained on SROIE entity annotations. SmolVLM and SROIE are
external projects and are not owned by this repository.
